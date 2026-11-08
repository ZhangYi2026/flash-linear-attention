import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
p.add_argument('--oja', type=int)
p.add_argument('--tma', type=int)
p.add_argument('--solver', type=int)
a = p.parse_args()
root = Path(__file__).resolve().parent
assert json.loads((root / 'FINAL_GATE.json').read_text())['passed']
evidence = f'https://github.com/ZhangYi2026/flash-linear-attention/blob/{a.evidence}/gdp-forward-fix/README.md'
old_evidence = f'https://github.com/ZhangYi2026/flash-linear-attention/blob/{a.evidence}/README.md'
env = 'RTX 5070 Ti Laptop (SM120, 12 GiB, driver 595.97), Python 3.12.3, PyTorch 2.9.1+cu128, Triton 3.5.1; model tests use Transformers 5.20.2. `FLA_CI_ENV=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee`; correctness retains `FLA_CACHE_RESULTS=1`.'


def checklist(tests, bench):
    return f'''## Checklist

- [x] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](../AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](../.agents/skills).
- [{'x' if tests else ' '}] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [{'x' if bench else ' '}] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.
'''


def cost(name):
    d = json.loads((root / name).read_text())
    return f"Largest measured latency change: {d['worst_latency_change_pct']:+.2f}% (positive means slower); largest between-process median spread: {d['max_process_spread_pct']:.2f}%. These affected-device results describe scheduling cost, not a datacenter speedup or training-throughput claim."


oja_body = f'''## Summary

Allows Oja/Oja2 backward on SM120 to retain existing single-stage configurations when pipelined candidates exceed shared memory. An unchanged Oja2 case with BF16 q/k, FP32 values and K=M=128 fails on upstream with 425984 bytes required versus the device's 101376-byte limit; a single-stage configuration passes its original output, state and all seven gradient checks.

Only the SM120 pruning policy changes. The existing restriction for larger FP32 states remains, and other architectures, kernel arithmetic, tensor precision and saved values keep their existing behavior. Found during #1703 validation; the failure reproduces independently of ReWY and the solver-layout fix.

## Test plan

Baseline `0a649e4c`; candidate `b6172a05`. {env} `FLA_REWY=0`.

All 57 original Oja/Oja2 kernel cases passed (24 Oja, 33 Oja2), with original inputs, references, tolerances and skips unchanged. The four remaining pure-PyTorch Oja reference cases passed separately on A800 and do not exercise the modified SM120 kernel. Only Oja/Oja2 call this pruner; the GSA2 helpers in the same file are unchanged. Pre-commit, headers and whitespace checks pass.

[Exact pytest commands, failed baseline control, source hashes, caller audit and complete results]({evidence}). The later combined result is reported separately from this standalone validation.

## Benchmark / NCU

Same SM120 GPU/software, baseline `0a649e4c` versus `b6172a05`, fresh ABBA processes, five 100 ms CUDA Graph samples per process/case. Public **backward**, including default recomputation and state/normalization gradients; excludes forward and Python dispatch. Shape B,T,H,K,M; q/k dtype shown, values/gates/state FP32. Packed lengths [1,64,191]. All timed inputs pass the original output/state/seven-gradient thresholds. Timing alone disables disk tuning-result reuse; tuning precedes timing.

{(root / 'OJA_BENCHMARK_TABLE.md').read_text()}
{cost('oja-benchmark-summary.json')} No ratio is defined for the original uncompilable K128 input. NCU was not run.

## Breaking changes

None. Public APIs, supported gradients, precision, storage and checkpoints are unchanged.

{checklist(True, True)}'''
(root / 'OJA_PR_BODY.md').write_text(oja_body, encoding='utf-8', newline='\n')

tma_rows = json.loads((root / 'benchmark-TMA-enabled.json').read_text())['rows']
enabled = '\n'.join('| ' + ','.join(map(str, r['shape'])) + f" | {r['layout']} | {r['output_dtype'].split('.')[-1]} | compile failure | {r['median_us']:.3f} |" for r in tma_rows)
tma_body = f'''## Summary

Fixes triangular-solve compilation with `FLA_USE_TMA=1` when int64 chunk indices reach Triton's descriptor load/store coordinates. Triton 3.5.1 rejects these coordinates because the descriptor API requires int32. The three solve kernels now cast descriptor-relative coordinates to int32 while preserving int64 global pointer arithmetic, including large batch and packed-sequence offsets.

The non-TMA specialization is AST-identical to upstream. Floating-point arithmetic, tensor types, autotune policy and public interfaces are unchanged. This is independent of the SM120 solver-layout and ReWY changes.

## Test plan

Baseline `0a649e4c`; candidate `28e74f8`. {env} `FLA_REWY=0`.

The same two original BT64 dense/packed tests fail on both upstream and the solver-layout-only branch. On this standalone fix, all ten ordinary original `test_solve_tril.py` cases pass with TMA off and again with TMA on, covering BT16/32/64 and tails; inputs, references and 0.0001 thresholds are unchanged. These existing tests supply the regression coverage when run with `FLA_USE_TMA=1`. The combined fix also passes these ten TMA cases and real dense/packed solves above 2^31 element offsets with TMA off/on.

The original huge-batch test exceeds the available 12 GiB and remains unrun; the smaller 8.06 GiB probe checks two late nonzero blocks and three identity controls, not every element or the original huge-batch test. Other TMA-capable GPU architectures have not been executed here. Default-path shared caller coverage is reusable because its specialization is unchanged. Pre-commit, headers and whitespace checks pass.

[Commands, source hashes, preserved baseline failures, final results and addressing checks]({evidence}).

## Benchmark / NCU

Same SM120 GPU/software; public solver **forward** including zero-initialized output, excluding Python dispatch and backward. FP32 input, shape B,T,H,BT, output dtype shown. Fresh ABBA processes and all ten 100 ms CUDA Graph samples per revision/case; tuning is excluded. The default path comparison below uses TMA disabled in both revisions (`0a649e4c` -> `28e74f8`).

{(root / 'TMA_BENCHMARK_TABLE.md').read_text()}
{cost('tma-default-benchmark-summary.json')}

The enabled path did not compile before this patch, so a TMA speedup ratio is undefined. Absolute measurements for the repaired path (five samples, same shapes and reference checks):

| B,T,H,BT | Layout | Output | Before TMA | After TMA (us) |
| --- | --- | --- | --- | ---: |
{enabled}

NCU was not run. These consumer-device measurements are supplementary.

## Breaking changes

None. Descriptor coordinates use the API's required width; base-pointer addressing stays int64. No change to mathematical precision or returned structures.

{checklist(True, True)}'''
(root / 'TMA_PR_BODY.md').write_text(tma_body, encoding='utf-8', newline='\n')

if a.oja and a.tma:
    solver_body = f'''## Summary

Fixes four strict BF16 GatedDeltaProduct dense/packed forward comparisons on SM120 by keeping the same two-warp layout for the shared BT64 triangular inverse. Independently tuned w2/w4 reductions differed by at most 1.49e-8, crossing two BF16 rounding boundaries and amplifying through later model layers. Original errors were 0.006866 twice, 0.003204 and 0.003738 against the unchanged 0.001 threshold.

Only the SM120 BT64 warp candidates change; all pipeline stages, FP32 arithmetic, tensor types, addressing, interfaces and saved values remain. Other architectures and BT16/32 retain their policy. This is independent of ReWY #1703 and the state-kernel fix #1658, which is present in the tested current upstream.

This remains Draft with explicit external prerequisites: #{a.oja} fixes an existing Oja shared-memory failure in the caller suite; #{a.tma} fixes the existing optional TMA coordinate compilation failure. They reproduce on upstream, and the combined validation below includes their fixes. Cross-fork native stacks are unsupported.

## Test plan

Baseline `0a649e4c`; solver-only published head `8ee08891` (tree-identical to tested `44db92fd`); published integration `f616ab69` (tree-identical to tested `2d2eef81`). {env}

Current upstream still has 4 failed/6 passed/2 skipped in the complete GDP model file. The original four comparisons pass in the combination without changing inputs or tolerances; exact output differences are zero. Same-warp/stage controls and a CPU FP64 oracle isolate the rounding boundary.

The exact shared-caller selection is 212 passed/2 existing skips: 210 kernel/model nodes on SM120 plus four unchanged pure-PyTorch reference nodes on A800. The final combination passes 235 tests with 50 existing hardware/backend/upstream skips in the ReWY integration selection, and ten original solver cases with TMA enabled. Dense/packed public solves above 2^31 elements pass with TMA off/on. The original huge-batch solver test remains unrun because it exceeds 12 GiB; the smaller probe does not replace it.

These are **combined** results, not a claim that this standalone head fixes Oja, KDA or TMA. The test checklist stays unchecked until its prerequisites are integrated. No original tolerance or test skip was changed; KDA's separately disclosed fixture correction is #1726. Pre-commit, headers and whitespace checks pass.

[Exact commands, source hashes, caller audit, baseline failures and all combined results]({evidence}).

## Benchmark / NCU

Baseline `0a649e4c` versus solver-only `44db92fd`, same SM120 GPU/software. Public solver **forward** including output zero-initialization, excluding Python dispatch and backward. FP32 input, BT64; shape B,T,H. Packed cases concatenate equal-length sequences. Fresh ABBA processes, all ten 100 ms CUDA Graph samples per revision/case, retuning outside timing. CPU FP64 references verify FP32 output at 0.0001; original operator/model gates cover BF16 output.

{(root / 'BENCHMARK_TABLE.md').read_text()}
{cost('benchmark-summary.json')} Datacenter configurations do not change. NCU was not run.

## Breaking changes

No interface, dtype, state or checkpoint changes. SM120 BT64 rounding can differ from a previously tuned warp layout; this restores the existing model-test tolerance. Other architectures and BT16/32 keep their original calculation and tuning policy.

{checklist(False, True)}'''
    (root / 'SOLVER_PR_BODY.md').write_text(solver_body, encoding='utf-8', newline='\n')

if a.oja and a.tma and a.solver:
    old = (root / 'REWY_PR_BODY_LIVE.md').read_text(encoding='utf-8')
    start = old.index('## Benchmark / NCU')
    end = old.index('## Breaking changes')
    benchmark = old[start:end].replace('https://github.com/ZhangYi2026/flash-linear-attention/blob/2d52ce9d51726c975a798f7435a7a3dd187320dc/README.md', old_evidence)
    body = f'''## Summary

Adds a default-off ReWY backend for Gated Delta Rule backward, replacing the final input-gradient assembly with the residual WY pullback and Gram-free gate reduction. Related to #1653; paper: [ReWY](https://arxiv.org/abs/2611.01940).

`FLA_REWY=1` enables matching FP16/BF16 inputs on NVIDIA GPUs with equal key/value head counts, V <= 128 and chunk sizes 16/32/64. Dense/packed sequences and both state layouts share the implementation; context-parallel and unsupported calls retain the existing path. GDP can reach it through shared GDN backward. Forward computation, state scans, saved tensors, interfaces and defaults remain unchanged.

This remains **Draft**. Independent upstream defects found during validation are now isolated in #{a.solver} (GDP forward solver layout), #1726 (overflowing KDA cache fixture), #{a.oja} (SM120 Oja resource pruning) and #{a.tma} (optional TMA coordinates). The tested combination passes below. These are external prerequisites, not a cross-fork native stack; the unchanged head of this PR alone is not claimed to have resolved them. Datacenter SM90+ performance evidence is also still missing.

## Test plan

Published ReWY head: `7608606ab33ee4fc570dd3999053a34851761636`. [Original frozen-baseline evidence]({old_evidence}) compares `98193113` with this head, with original references/tolerances and strict FP32 settings. Full GDN results were 100/114 passed plus 25 skips on RTX, and 96/110 passed plus 29 skips on A800. Eight excluded grouped-head cases passed separately for both revisions on each device. Four GDP forward failures reproduced identically on the baseline and ReWY revisions; their failed records remain preserved.

[New prerequisite and integration evidence]({evidence}) uses current upstream `0a649e4c`, including #1658, which still reproduces the same four GDP failures. The tested combination `2d2eef814c0fe14346dfdf13ee5b82724ee8d090` contains this ReWY head, current main and the four independent fixes. Its published single-commit snapshot is `f616ab692567014e12341547934c627a5af3c352`; complete Git trees are identical. The solver and KDA PRs likewise retain one final commit (`8ee08891`, `7ef2a96c`) with tree identity recorded in the evidence.

| Final check | Result |
| --- | --- |
| Complete GDN/GDN-kernel/GDP files plus CP, layer and model integration selection | 235 passed, 50 existing skips, 0 failed |
| Original four GDP dense/packed forward comparisons | Original 0.001 threshold passes; exact FP32 max difference zero and bitwise equal |
| Exact shared-solver caller selection | 212 passed, 2 existing skips |
| Independent Oja/Oja2 kernel selection | 24 + 33 passed |
| Corrected KDA cache file | 8 passed |
| Original ordinary solver cases, standalone default/TMA and combined TMA | 10 passed in each run |
| Real dense/packed offsets beyond 2^31 elements, TMA off/on | Passed |

{env} Integration uses `FLA_REWY=1`. All affected kernels ran on SM120; four unchanged pure-PyTorch Oja reference cases in the shared selection ran separately on A800 and do not validate the changed SM120 kernels. KDA's test-only fix normalizes numerically unstable keys and adds finite checks, retaining seed, sequence length, dimensions, cache assertions and tolerances. No GDP input or threshold changed.

The 50 skips comprise grouped-head exclusions covered separately in the original evidence, unavailable FlashQLA/TileLang backends, multi-device CP, upstream-disabled MoM cases and Hopper-only model cases. The original huge-batch solver test exceeds local memory and remains unrun; the smaller address probe is supplemental. Full logs account for each collected node. Earlier OOM, KDA, Oja and TMA failures are retained, not converted to passes. Pre-commit, headers and whitespace checks pass.

The 14 added ReWY cases cover recurrence references, outputs/final states/all six gradients, dispatch, tails, asymmetric dimensions, zero beta, both dtypes/layouts and bitwise rejected/disabled fallback. Original captured-production-kernel memcheck found zero errors. Full reference-process memcheck had numerical passes but two cuBLAS API errors also reproduced by standalone `torch.bmm`; it is not reported as clean. Offline SM80/SM90 compilation is not execution evidence.

{benchmark}## Breaking changes

None. Public signatures, defaults, returned structures, supported gradients and checkpoints are unchanged. ReWY is explicitly enabled and falls back for unsupported calls.

{checklist(False, False)}'''
    (root / 'REWY_PR_BODY_UPDATED.md').write_text(body, encoding='utf-8', newline='\n')
print('Prepared PR descriptions with evidence commit ' + a.evidence)
