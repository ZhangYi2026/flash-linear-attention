"""Update the unique project ledger with final results and publication state."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
project = root.parent.parent
published = json.loads((root / 'publication-final.json').read_text(encoding='utf-8'))
solver = json.loads((root / 'benchmark-summary.json').read_text())
oja = json.loads((root / 'oja-benchmark-summary.json').read_text())
tma = json.loads((root / 'tma-default-benchmark-summary.json').read_text())
urls = {r['kind']: r['url'] for r in published['prs']}
ledger = project / 'EXPERIMENT_LEDGER.zh-CN.md'
text = ledger.read_text(encoding='utf-8')
anchor = '## 2026-11-09 FLA GDP SM120 前向一致性修复'
assert anchor in text
prefix, history = text.split(anchor, 1)
record = f'''## 2026-11-13 FLA GDP SM120 前向一致性修复（最终复测更新）

- **算子／方案／设备**：共享 `solve_tril` BT64 在 RTX 5070 Ti Laptop SM120 统一两 warps，保留四种 stages，其他硬件与 BT16/32 不改。用户本轮明确续授本机 GPU；不是未经授权的本机替代实验。
- **推导／成立条件／精度**：GDP dense/packed 自动选择 w2/w4 导致 FP32 规约次序差异，最大 1.49e-8 使两个元素跨 BF16 舍入边界，随后层叠放大。同 warp 与各 stage 对照及 CPU FP64 舍入参考已定位首个分歧。仅统一 SM120 布局，不改公式、FP32 运算、输入输出或中间量精度；不消除新的代数计算，不按 case 分流，不计独立算法成果。
- **对照／保存策略／计时边界**：当前 upstream `0a649e4c` 对已测 `44db92fd`，最终单提交 `8ee08891` 与其完整 Git tree 相同。保存策略不变。公共求逆**前向** CUDA Graph，含输出初始化，不含 Python dispatch 或 backward；严格 IEEE/TF32 override=0，同卡 ABBA 全部样本汇总，8 个 FP32 输入、FP32/BF16 输出 dense/packed 工作负载。
- **最终汇总**：几何平均 baseline/candidate **{solver['gm_baseline_over_candidate']:.4f}×**；最大耗时增加 **{solver['worst_latency_change_pct']:.2f}%**，最大进程间中位数波动 **{solver['max_process_spread_pct']:.2f}%**。仅修复调度成本，不能宣称显著加速、完整 B/FB 或整网收益。
- **正确性／保留状态**：原四个 GDP 比较在原 0.001 阈值下通过，逐位相等。共享调用方原选择 212 passed/2 既有 skips（210 SM120 kernel/model 节点加 4 个 A800 纯 Torch 参考）；最终含独立 KDA/Oja/TMA 修复的 ReWY 组合 235 passed/50 既有 skips/0 failed。此为组合验收，不能归给 solver 单分支。原巨大 batch 测试超过本机显存，仍未运行，未伪记 pass/skip。保留独立 [solver Draft PR]({urls['solver']})，明确等待外部修复前置合入。
- **来源**：[最终报告](output/fla_gdp_forward_fix_20261108/README.md)、[汇总](output/fla_gdp_forward_fix_20261108/benchmark-summary.json)、[最终门槛](output/fla_gdp_forward_fix_20261108/FINAL_GATE.json)、[单提交源码等价核验](output/fla_gdp_forward_fix_20261108/submission-history-equivalence.json)。所有原失败、OOM、未完成记录均保留。

## 2026-11-13 KDA cache 测试初始化修复（最终更新）

- **算子／方案／设备／对照**：原 KDA cache 测试 Gaussian key 使 32768-token 递推在 CPU FP64 中达约 3.029e204；RTX SM120 上冻结及当前 upstream 均复现 NaN。仅归一化测试 key 并增加有限值断言；原 seed、长度、尺寸、cache 检查和 1e-4 容差不变。
- **精度／保存／计时／结论**：未修改生产算子、运算或保存精度，无性能计时或加速比。正确性为修正后的完整 cache 文件 **8 passed**，不宣称原失稳输入不变通过。已测 `6aa27ca0` 与最终单提交 `7ef2a96c` 完整源码树一致；[PR #1726]({urls['kda']}) 保留独立审阅。
- **来源**：[KDA 根因](output/fla_gdp_forward_fix_20261108/KDA_CACHE_ROOT_CAUSE.md)、`latest-fixed-kda.*`、`kda-diagnosis-20261113.json`、最终报告。旧失败证据保留。

## 2026-11-13 SM120 Oja 反向资源兼容性（最终复测更新）

- **算子／方案／设备**：Oja/Oja2 共享反向状态核仅 SM120 保留原有单 stage 候选；RTX 5070 Ti Laptop、Torch 2.9.1+cu128、Triton 3.5.1。原 BF16 q/k、FP32 value、K=M128 用例请求 425984 bytes shared memory，超过 101376 bytes，upstream 独立复现。单 stage 避免流水化状态块的共享内存压力，原 V>128 FP32 分支不动；其他架构候选策略不变。
- **对照／保存／精度**：`0a649e4c` 对最终单提交 `b6172a05`；原 kernel 正文、FP32 累加、输入输出与存储精度、默认重算保持。属于硬件统一候选策略，未引入 case 特化或新代数成果。
- **计时边界／最终汇总**：公共 autograd 完整 **B**，含默认重算、状态与归一化梯度，不含前向和 Python dispatch；4 个 FP16/BF16 q/k dense/packed 工作负载同卡严格 ABBA，几何平均 **{oja['gm_baseline_over_candidate']:.4f}×**，最差耗时变化 **{oja['worst_latency_change_pct']:+.2f}%**，最大进程间波动 **{oja['max_process_spread_pct']:.2f}%**。范围限已可运行的 K=M64 小工作负载；原不能编译的 K128 不定义加速比，不外推数据中心、FB 或整网收益。
- **正确性／保留**：独立 Oja24 + Oja2 33 原 kernel 节点全部通过，原输出/状态/七项梯度门槛不变；4 个额外纯 Torch 参考沿用已核验 A800 结果。保留独立 [Oja PR]({urls['oja']})。首次计时因默认流与 Graph capture 冲突失败，未产出有效样本；只修正测量脚本流同步，完整新 ABBA 替代，不选择性保留旧样本。
- **来源**：[资源根因](output/fla_gdp_forward_fix_20261108/OJA_RESOURCE_ROOT_CAUSE.md)、[最终汇总](output/fla_gdp_forward_fix_20261108/oja-benchmark-summary.json)、原失败及 `oja-benchmark-stream-*` 全部新样本。

## 2026-11-13 solve_tril TMA 坐标编译兼容性

- **算子／方案／设备／对照**：共享 BT16/32/64 求逆在描述符 load/store 入口采用 API 要求的 int32 相对坐标，全局地址保持 int64。RTX SM120、Triton 3.5.1；`0a649e4c` 对最终单提交 `28e74f8`。原版和 solver-only 版在相同两个 BT64 TMA 节点均编译失败，错误一致；不是 ReWY 引入。
- **保存／精度／正确性**：浮点运算、存储与保存策略保持，非 TMA 特化 AST 与原版一致。独立默认/TMA 各 10 原测试通过，组合 TMA 10 原测试通过；dense/packed 在 TMA off/on 的真实 >2^31 元素偏移公共求逆均通过。大偏移探针只验两个非零晚块及三个 identity 对照，不冒充原巨大 batch 全量测试或所有输出元素检查。
- **计时边界／最终汇总**：公共求逆**前向** CUDA Graph 含输出初始化。TMA 关闭的同卡 ABBA 8 工作负载 baseline/candidate 几何平均 **{tma['gm_baseline_over_candidate']:.4f}×**，最差耗时变化 **{tma['worst_latency_change_pct']:+.2f}%**，进程间波动 **{tma['max_process_spread_pct']:.2f}%**；仅默认路径成本核验。TMA 开启的原版不能编译，无可定义的前后加速比；修复版绝对计时另存，不计新算法收益。
- **保留／局限／来源**：独立 [TMA PR]({urls['tma']})；其他 TMA 架构未执行验收。见[最终报告](output/fla_gdp_forward_fix_20261108/README.md)、`tma-default-benchmark-summary.json`、`benchmark-TMA-enabled.json`、`baseline-tma-20261113.*`、`solver-fix-tma.*`、`submission-*`。

## 本轮 PR 发布结论

- [ReWY #1703]({urls['rewy']}) 已更新独立修复依赖和最终组合证据，按用户明确指示保留 Draft，仍缺数据中心 SM90+ 性能与上游前置合入；本机/A800 不替代该项。
- 当前这组 PR 每个只保留一个最终提交；历史整理前后完整 Git tree 一致。原始失败及实验数据未删除，来源映射见 `submission-history-equivalence.json`。
- 用户 fork 的 GitHub Actions 曾因账户 billing lock 在启动前失败，lint 未运行；不是代码 lint 失败或付费算力购买。仓库公开且使用标准 ubuntu-latest。实际发布后检查以 `publication-final.json` 为准，本地静态检查与 GPU 验证分开报告。

<details>
<summary>本条目此前的更正与交接记录（保留历史，最终结论以上文为准）</summary>

{anchor}{history}

</details>
'''
ledger.write_text(prefix + record, encoding='utf-8', newline='\n')
status = root / 'STATUS.zh-CN.md'
old = status.read_text(encoding='utf-8')
status.write_text(f'''# 最终发布状态（2026-11-13，Asia/Shanghai）

本轮目标已按用户指示完成：独立修复提交、证据发布、#1703 Draft 更新；每个 PR 一个最终提交，完整源码树与已测版本一致。发布结果和检查见 `publication-final.json`，证据 commit `{published['evidence_commit']}`。最终组合 235 passed/50 原 skips，四项 GDP 比较逐位相等；共享选择 212 passed/2 原 skips。没有将原失败改写为通过，也没有修改原容差或添加 skip。

用户已明确续授本机 GPU 验证，并选择先更新 Draft，不等待 SM90+。GitHub fork billing lock 导致部分 fork Actions 未启动；未进行收费服务开通、预算或付款设置操作。

所有本轮 GPU 任务已结束。不要重新执行会碰到旧输出的队列；复测使用新结果 stem。

## 以下为历史交接状态

{old}''', encoding='utf-8', newline='\n')
print('Updated the unique experiment ledger and final handoff state.')
