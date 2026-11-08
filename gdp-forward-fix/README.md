# GDP forward consistency and independent prerequisite fixes

These results resolve the four GDP forward comparisons found while validating ReWY PR #1703. ReWY's published head remains separate from the fixes and the tested integration. No original tolerance is relaxed and no new test skip is introduced.

## Revisions

| Revision | Commit | Scope |
| --- | --- | --- |
| Current upstream control | `0a649e4c7201f0edcba734832773fe8ab371367a` | Includes upstream #1658; still reproduces all four GDP failures |
| ReWY PR #1703 | `7608606ab33ee4fc570dd3999053a34851761636` | Opt-in backward only |
| SM120 solver layout | `44db92fd7c343544c58c505edf217dceb1931636` | Two warps for BT64; other architectures unchanged |
| SM120 Oja resource fix | `b6172a05f2405a73c96ec190ee41f4083faa31a8` | Retain existing single-stage candidates |
| KDA fixture PR #1726 | `6aa27ca0e095ef91f74b6f068c5ec8de07d96970` | Normalize unstable cache-test keys; no runtime change |
| TMA coordinate fix | `28e74f8ba0c8ebb0a0edd11c42dc7ef628b27cae` | Int32 descriptor coordinates, int64 global addresses |
| Previous combined validation | `7d7b00ca4e638d1a77aea9976e945e6f122b6c2d` | ReWY + solver + Oja + KDA + current upstream |
| Final combined validation | `2d2eef814c0fe14346dfdf13ee5b82724ee8d090` | Previous combination plus TMA coordinate fix |

`submission-source-manifest.json`, the `*-verified.json` records, and `submission-source-review.json` identify extracted source snapshots and hashes. The final integration changes only the solver relative to the previous integration; specializing `USE_TMA=False` gives identical ASTs. The 214-node shared-caller result therefore still describes its default path. The final integration is additionally retested below.

## Single-commit publication cleanup

At the user's request each PR and the combined snapshot retain one final commit. `submission-history-equivalence.json` verifies complete Git tree identity, not just the touched files: KDA `6aa27ca0` -> `7ef2a96c`, solver `44db92fd` -> `8ee08891`, and integration `2d2eef81` -> `f616ab69`. ReWY `7608606a`, Oja `b6172a05` and TMA `28e74f8` were already single commits. The raw test records retain the revisions actually executed. The final published hashes reproduce exactly the same source trees without requiring a numerical rerun for metadata-only history cleanup.

## Environment and final correctness

RTX 5070 Ti Laptop, SM120, 12 GiB, driver 595.97; Ubuntu 24.04 WSL2; Python 3.12.3, PyTorch 2.9.1+cu128, Triton 3.5.1, Transformers 5.20.2. Strict settings: `FLA_CI_ENV=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee`. Correctness retains the default `FLA_CACHE_RESULTS=1`. Integration uses `FLA_REWY=1`; standalone fixes and timing use `FLA_REWY=0`. TMA is explicitly enabled only for named TMA checks. The pytest audit plugin records node IDs, exit status, source hashes, selected configurations and memory without changing test inputs or assertions.

| Check | Source | Result |
| --- | --- | --- |
| Original GDP model file, current upstream | `0a649e4c` | 6 passed, 4 failed, 2 existing Hopper-only skips |
| Shared solver callers, exact 214-node selection | `7d7b00ca` | 212 passed, 2 existing skips |
| Complete original Oja/Oja2 kernel selection | `b6172a05` | 24 + 33 passed |
| Complete corrected KDA cache file | `6aa27ca0` | 8 passed |
| Solver default path, ordinary original cases | `28e74f8` | 10 passed |
| Solver TMA path, same original cases | `28e74f8` | 10 passed |
| Solver TMA path, final combination | `2d2eef81` | 10 passed |
| Complete ReWY integration selection | `2d2eef81` | 235 passed, 50 existing skips |
| Real offsets above 2^31 elements, dense/packed, TMA off/on | `2d2eef81` | Four public solves passed |

The shared selection includes 210 SM120 kernel/model nodes and four unchanged pure-PyTorch Oja reference nodes already passed on A800-SXM4-80GB. Those four do not exercise either modified SM120 kernel. The Oja standalone selection above excludes these same four pure-PyTorch references; it includes all 57 kernel cases. `current-shared-runs.json` accounts for the original selection exactly, without duplicates or unexecuted nodes being counted as passes.

The final ReWY selection includes full GDN, GDN kernels and GDP files plus CP, layer and model consumers. Its 50 skips are 8 upstream grouped-head exclusions (separately covered in the earlier ReWY evidence), 16 unavailable FlashQLA cases, 1 unavailable TileLang case, 12 multi-device CP cases, 11 upstream-disabled MoM cases and 2 Hopper-only GDP model cases. Skips remain gaps, not passes. The original huge-batch solver test does not fit 12 GiB and is explicitly deselected, not newly marked skipped; the ordinary solver selections contain ten nodes. The supplemental 8.06 GiB address probe checks two nonzero blocks beyond 2^31 elements and three identity controls; it does not replace the original huge-batch test or claim to inspect every output element.

All four original GDP output assertions pass at the unchanged 0.001 threshold. The observer additionally records exact FP32 maximum difference zero and bitwise equality in `submission-integration.model-comparisons.json`. This is the combined result, not a claim about the unchanged standalone #1703 head. Pre-commit, license headers and whitespace checks pass.

## Root causes and preserved failures

* **GDP:** dense/packed solver autotuning selected w2/w4. FP32 reductions differed by up to 1.49e-8 and crossed two BF16 rounding boundaries, amplified through subsequent layers. Same-warp and stage controls plus a CPU FP64 oracle isolate this boundary. The solver fix changes only the SM120 BT64 candidate list, not arithmetic, tensor types or saved values. [Detailed diagnosis](ROOT_CAUSE.zh-CN.md); [caller audit](CALLSITE_AUDIT.md).
* **KDA fixture:** unnormalized Gaussian keys make the actual 32768-token recurrence grow to about 3.029e204 in CPU FP64. The original cache test fails on both upstream and the earlier combination. The independent fixture correction preserves length, seed, dimensions, caching assertions and 1e-4 tolerances; it normalizes keys and asserts finite outputs. This is disclosed as a test-input correction. [Diagnosis](KDA_CACHE_ROOT_CAUSE.md).
* **Oja:** an original BF16-q/k, FP32-value Oja2 case requests 425984 bytes of shared memory versus SM120's 101376-byte limit. Upstream reproduces the same failure. Retaining existing single-stage candidates passes outputs, final state and all seven gradients. [Diagnosis](OJA_RESOURCE_ROOT_CAUSE.md).
* **TMA:** both upstream and the solver-only branch fail the same two original BT64 tests when `FLA_USE_TMA=1`; descriptor coordinates have int64 type, rejected by Triton 3.5.1. The independent fix casts only descriptor coordinates to int32 in all three solve kernels. Global pointer arithmetic stays int64 and the non-TMA specialization is AST-identical. The original failures are `baseline-tma-20261113.*` and `solver-fix-tma.*`; the final passes have new `submission-*` stems.

The first shared batch suffered a CUDA OOM and cascading context errors (console summary: 91 passed, 123 failed, 116 errors). Its original logs remain; it is not a successful gate. Concurrent GPU use was observed, and subsequent fresh-process baseline/fixed controls passed with zero allocator OOMs. The next recovery stopped at the independent Oja resource error (126 passed, 1 failed). The earlier ReWY combination stopped at KDA (232 passed, 1 failed, 50 skipped). The initial Oja timing harness also failed before producing samples because default-stream autograd conflicted with CUDA Graph capture; `oja-benchmark-A1.log` and `benchmark_oja_initial_stream.py` preserve that attempt. The corrected harness builds the forward on a non-default stream and explicitly synchronizes it, without changing kernels, inputs or reference checks. The complete new ABBA series uses `stream-A1/B1/B2/A2`. None of these failed or interrupted runs is relabeled as passing. See `test-summary.json` and the original logs.

## Supplementary scheduling cost

All comparisons use the same SM120 GPU, strict precision flags, CPU/fixed-seed identical inputs and fresh ABBA processes. Tuning is outside timed samples; only timing uses `FLA_CACHE_RESULTS=0` to avoid reusing a layout chosen under earlier resource contention. Compiled and in-process caches remain. Each row pools all ten samples per revision (five 100 ms CUDA Graph samples per process). Correctness precedes timing. No NCU or training-convergence claim is made.

### SM120 solver layout

Baseline `0a649e4c` versus `44db92fd`. Public solver **forward**, including output zero-initialization; excludes Python dispatch and backward. Input FP32, BT64; shape column is B,T,H,BT. Packed layouts concatenate equal-length sequences. CPU FP64 triangular reference verifies FP32 outputs at 0.0001; original model/operator gates cover BF16 output accuracy.

| Workload | Layout | Dtype | Before (us) | After (us) | Latency change |
| --- | --- | --- | ---: | ---: | ---: |
| 2,1024,4,64 | dense | float32 | 20.020 | 20.006 | -0.07% |
| 2,1024,4,64 | dense | bfloat16 | 19.088 | 19.061 | -0.14% |
| 2,1024,4,64 | packed | float32 | 20.763 | 20.537 | -1.09% |
| 2,1024,4,64 | packed | bfloat16 | 20.297 | 20.336 | +0.20% |
| 1,4096,8,64 | dense | float32 | 63.508 | 62.830 | -1.07% |
| 1,4096,8,64 | dense | bfloat16 | 59.041 | 58.373 | -1.13% |
| 1,4096,8,64 | packed | float32 | 64.236 | 63.508 | -1.13% |
| 1,4096,8,64 | packed | bfloat16 | 61.854 | 61.097 | -1.22% |

Geometric mean baseline/candidate: 1.0071x; largest latency change: +0.20% (positive means slower); largest between-process median spread: 5.51%. These are scheduling-cost observations, not a datacenter or training-throughput claim.

### SM120 Oja resource policy

Baseline `0a649e4c` versus `b6172a05`. Public autograd **backward** including default recomputation, state and normalization gradients; excludes forward and Python dispatch. Shape column B,T,H,K,M; values/gates/state remain FP32, q/k dtype is shown. Dense T256; packed lengths [1,64,191]. Each timed input passes the original output/state/all-seven-gradient reference thresholds. These are previously runnable inputs; no speedup is invented for the original uncompilable K128 case.

| Workload | Layout | Dtype | Before (us) | After (us) | Latency change |
| --- | --- | --- | ---: | ---: | ---: |
| 1,256,2,64,64 | dense | float16 | 210.296 | 114.898 | -45.36% |
| 1,256,2,64,64 | packed | float16 | 177.905 | 106.531 | -40.12% |
| 1,256,2,64,64 | dense | bfloat16 | 212.835 | 115.895 | -45.55% |
| 1,256,2,64,64 | packed | bfloat16 | 173.578 | 109.765 | -36.76% |

Geometric mean baseline/candidate: 1.7261x; largest latency change: -36.76% (positive means slower); largest between-process median spread: 3.49%. These are scheduling-cost observations, not a datacenter or training-throughput claim.

### TMA coordinates

Baseline `0a649e4c` versus `28e74f8`, **TMA disabled**: same public solver forward measurement. This measures the default path's cost; it is not a before/after speedup for the previously uncompilable TMA path.

| Workload | Layout | Dtype | Before (us) | After (us) | Latency change |
| --- | --- | --- | ---: | ---: | ---: |
| 2,1024,4,64 | dense | float32 | 20.361 | 20.031 | -1.62% |
| 2,1024,4,64 | dense | bfloat16 | 19.164 | 19.105 | -0.31% |
| 2,1024,4,64 | packed | float32 | 21.179 | 20.560 | -2.92% |
| 2,1024,4,64 | packed | bfloat16 | 20.613 | 20.218 | -1.92% |
| 1,4096,8,64 | dense | float32 | 65.373 | 63.397 | -3.02% |
| 1,4096,8,64 | dense | bfloat16 | 61.786 | 58.686 | -5.02% |
| 1,4096,8,64 | packed | float32 | 67.282 | 64.542 | -4.07% |
| 1,4096,8,64 | packed | bfloat16 | 64.147 | 61.347 | -4.37% |

Geometric mean baseline/candidate: 1.0300x; largest latency change: -0.31% (positive means slower); largest between-process median spread: 4.06%. These are scheduling-cost observations, not a datacenter or training-throughput claim.

The enabled path's eight absolute measurements are in `benchmark-TMA-enabled.json`, with the same shapes, layouts, dtypes and reference checks. Its original TMA baseline is a compile failure, so no TMA speedup ratio is defined.

## Reproduction and publication boundary

`finish_submission_20261113.py` contains exact test targets, flags, timing order and fail-stop gates. Replace its local snapshot/log directories with your own checkout locations and use a new output directory to preserve earlier evidence. `submission-run-status.json` preserves the first queue including its failed Oja timing attempt. `resume_submission_20261113.py` resumes only the missing measurements; `submission-final-runs.json` joins the completed initial gates with those new successful runs. `verify_final_gates.py` validates complete node counts, all final result stems, source hashes, exact GDP comparisons and offset probes. `FINAL_GATE.json` records both successes and unrun hardware checks.

Use the separate fix commits for independent review and the integration commit for the combined gate. Cross-fork GitHub native stacks are unsupported; these are explicit external prerequisites. #1703 remains Draft pending prerequisite integration and datacenter SM90+ performance evidence. Consumer/A800 measurements and offline compilation do not fill the latter gap.
