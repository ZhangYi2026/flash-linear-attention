# GDP 四项前向失败定位

## 结论

冻结基线 `98193113ecb7835143252a786e06933e371b67cc` 与 ReWY PR `7608606ab33ee4fc570dd3999053a34851761636` 的同四项失败来自共享 `solve_tril` 的 SM120 自动调优布局差异，不是 ReWY backward。四项均在 dense/packed hidden-state 对比处、调用 backward 前失败；默认 GDP 配置为 ungated、num_householder=1、K256/V512，本来不进入 ReWY 后端。

同一输入矩阵下，dense 选择 num_warps=2，packed 选择 num_warps=4。FP32 规约次序随布局变化，造成 16900 个 FP32 元素的细小差异（最大 1.4901161e-8）。其中两个数跨过 BF16 舍入边界，形成两个 BF16 差异；随后 WY、状态递推与层堆叠放大差异。没有证据表明这是越界或 Triton 错编译，不能与 #1658 的 GB200/Triton3.3 状态存储错编译混称。

## 排除与最小复现

- IEEE 设置必须与原整文件测试一致：test_gdp 会设置 `TRITON_F32_DEFAULT=ieee`。未对齐设置的首个孤立运行误差 0.007507，显式 IEEE 后恢复原误差 0.006866；前者不混入原对照。
- 最小模型 L3/B2/T1024 仍失败，L1/L2 或 B1 不触发原门槛。它只用于定位，不作为官方整网性能配置。
- 第一层逐位一致；第二层 norm、q/k/v 投影、短卷积、beta、原始算子输入、l2norm、FP32 KKT 均逐位一致。
- `solve_tril` 首先出现两个 BF16 差异；u 有 2 个差异、状态 h 有 105 个、v_new 有 48 个、GDP 输出有 66 个。参见 operator-stage-trace.json。
- 分别强制两种布局使用 w2、w4、w8，各自的 dense/packed 输出都逐位一致。w2/w4 配合 stages2/3/4/5 全部逐位一致。说明 warp 数不同是该样本差异的原因，流水线级数不是。
- CPU FP64 三角求解核验两个跨界元素，FP64 舍入到 BF16 均与 w2 结果一致。此诊断参考不替代、也不修改上游测试参考。

## 修复与回归

候选 `034c3d41a18f1c6a941918c39396ef004ec757ca`：仅 SM120、BT64 共享求逆固定 num_warps=2，其余硬件、BT16/32、四种 stages 均保持。既有 FP32 运算、输入/输出类型、存储、保存策略及测试参考和门槛不变；没有按 benchmark case 分流。

先以未修改基线的诊断配置验证原 GDP 模型完整文件：10 passed、2 个已有 Hopper-only skip；原阈值仍是 0.001。最终组合的原四项比较另存完整 FP32 比较：max_abs_diff=0，bitwise_equal=true。最终共享调用方选择 212 passed、2 个原 skip；ReWY 组合选择 235 passed、50 个原 skip、0 failed。精确命令、节点与原始日志见 README.md，独立修复和组合结果分开说明。

当前上游 `0a649e4c`（包含 #1658）仍复现四项失败。solver-only 实测版本 `44db92fd` 整理为最终单提交 `8ee08891`；最终组合实测 `2d2eef81` 整理为 `f616ab69`。完整 Git tree 均保持相同，见 `submission-history-equivalence.json`。验证中另外隔离了 Oja 共享内存配置、KDA cache 测试输入溢出与可选 TMA 坐标类型三个独立问题，均保留原失败证据；不能把组合通过归为 solver-only 或 ReWY-only 的结果。

此次修复与 ReWY 算法独立。按 CONTRIBUTING.md 的独立修复拆分要求，单独提交修复 PR，再更新 #1703 的证据与依赖说明；不会为清除 CI 元数据红灯而虚勾 checklist。当前 fork 无法提供跨仓库 native stack，不将两个并列 PR 伪称为 stack。
