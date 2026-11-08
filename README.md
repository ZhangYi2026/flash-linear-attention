# ReWY GDN draft validation

## Current PR verification (November 18)

[Current paired verification and H100 reproduction instructions](rewy-resume-20261118/README.md) cover candidate `e9fc25f2714a653995120a751e5f9356fd8bab19` against main `cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b`. Only ReWY #1703 remains in scope. The remaining sections below preserve the earlier pinned revisions and raw experiments; their timings are historical supplementary evidence and do not measure the rebased head. Independent solver/Oja/TMA/KDA submissions were withdrawn and are not prerequisites.

This evidence accompanies the default-off ReWY GDN backward contribution linked to [RFC #1653](https://github.com/fla-org/flash-linear-attention/issues/1653).

- Baseline: `98193113ecb7835143252a786e06933e371b67cc`.
- Final candidate: `7608606ab33ee4fc570dd3999053a34851761636`.
- The numerical runs began at `161dfe02f6b1b814fdd9140e68c17da8e1a9f64e`; subsequent changes only shorten and align the new `ENVs.md` row while removing unrelated table padding. Runtime and test files are byte-identical. The source manifest records both revisions and the final file hashes, verified in both execution environments.
- Scope: residual WY pullback plus Gram-free gate reduction, after the existing reverse state scan. This is a combined implementation, not an isolation of residual WY alone. The existing forward, state scan and default recomputation policy remain unchanged.

## Environments and commands

Both GPU environments use Python 3.12.3, PyTorch 2.9.1+cu128 and Triton 3.5.1. The local environment also uses Transformers 5.20.2. GPU drivers: A800-SXM4-80GB 550.163.01; RTX 5070 Ti Laptop GPU 595.97 (WSL Ubuntu 24.04).

Set the chosen GPU explicitly with `CUDA_VISIBLE_DEVICES`. For the final A800 runs, explicitly set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`; the environment audit verifies `torch.backends.cuda.matmul.allow_tf32=False`. The local base environment has both TF32 override variables unset and the same false setting. Run from each checkout with `FLA_CI_ENV=0 FLA_FLASH_QLA=0 FLA_TILELANG=0`, and `FLA_REWY=0` for the baseline or `FLA_REWY=1` for the candidate. References, tolerances, and existing test parameter sets were unchanged.

```bash
python -m pytest tests/ops/test_gdn.py -q --tb=short -rs
# Candidate-only focused coverage:
python -m pytest tests/ops/test_gdn.py -k rewy -q --tb=short
# Existing grouped-head cases otherwise skipped by the upstream backend condition:
FLA_DISABLE_BACKEND_DISPATCH=1 python run_grouped_tests.py
```

The identical dependent selection was run on both checkouts:

```bash
python -m pytest \
  tests/ops/test_gdn_kernels.py::test_gdn_full_bwd \
  tests/ops/test_gdp.py \
  tests/layers/test_gated_deltanet.py \
  tests/layers/test_mom_shared_no_grad.py \
  tests/models/test_modeling_gated_deltanet.py \
  tests/models/test_modeling_gated_deltaproduct.py \
  tests/models/test_modeling_mom.py \
  tests/cp/test_gdn.py tests/cp/test_gdp.py \
  tests/cp/test_intracard.py::test_chunk_gdn_grouped_heads \
  -q --tb=short -rs
```

The GDN layer and GDP operator tests include eligible backward calls. Default full-model configurations have value dimensions above the opt-in limit, so those runs mainly check integration and fallback. They are not a training-throughput or convergence evaluation.


## Correctness results

| Selection                                        | Baseline                        | Candidate                       |
| ------------------------------------------------ | ------------------------------- | ------------------------------- |
| Full GDN, RTX 5070 Ti Laptop                     | 100 passed, 25 skipped          | 114 passed, 25 skipped          |
| Full GDN, A800, TF32 override disabled           | 96 passed, 29 skipped           | 110 passed, 29 skipped          |
| Eight original grouped-head cases, each GPU type | 8 passed                        | 8 passed                        |
| Dependent selection, RTX                         | 51 passed, 4 failed, 25 skipped | 51 passed, 4 failed, 25 skipped |

All 14 added tests passed on both GPU types. Registration/verifier CPU coverage passed (5 cases), repository pre-commit/whitespace/license checks passed, and six SM80/SM90 configurations compiled offline. Offline compilation does not establish runtime correctness or performance. Source hashes cover the final 497-line, six-file diff; final post-test edits only shorten/align the new environment-variable documentation row.

## Failures and validation gaps

The dependent selection has **51 passed, 4 failed, 25 skipped** on both revisions. All four failures are unchanged GDP variable-length forward comparisons, before backward. The failing nodes and logged errors match exactly; see `dependent-comparison.json`. Relative errors are 0.006866 (two cases), 0.003204 and 0.003738 against the unchanged 0.001 threshold. These are failures, not tolerance passes. No additional candidate failure was observed in that selection.

Its skips comprise two Hopper-only model cases, eleven MoM cases marked as known bugs by upstream, and twelve multi-device CP cases requiring 2/4/8 GPUs. The five warnings concern ShortConvolution configuration and CPU DTensor random support, not numerical comparison warnings. Multi-device CP is unvalidated and explicitly rejects the ReWY backend.

Full GDN runs skip eight original grouped-head cases under the generic backend condition; the unchanged cases are separately rerun with dispatch disabled. Sixteen FlashQLA tests and one TileLang test are unavailable. A800 additionally skips four Blackwell/SM120-only configuration guards. Those alternative-backend gaps are not reported as passes.

Datacenter SM90+ validation/performance evidence is still missing. A800 and consumer GPU results are supplementary under the repository's NVIDIA performance policy. Historical paper results and offline compilation do not fill this gap. NCU profiling and training convergence are not claimed.

## Addressing and memory checks

`check_large_offsets.py` checks real input, state, and output offsets of at least 2**31 elements with both state layouts and a partial final chunk on the A800. It compares the final finite chunk bitwise against an identical small allocation. This is an addressing invariant, not a separate numerical oracle.

```bash
python check_large_offsets.py --repo /path/to/candidate --output large-offsets.json
compute-sanitizer --tool memcheck --error-exitcode 1 \
  --kernel-name kns=chunk_gated_delta_rule_bwd_kernel_rewy \
  python -m pytest tests/ops/test_gdn.py -k test_rewy_chunk -q --tb=short
```

The large-allocation check passed for both layouts: T=16,777,279 and 2,147,483,648-element starting offsets for the final input/state chunk, with 52,076,908,544 allocated bytes.

The five numerical cases passed under both initial and final TF32-disabled reference-process memcheck invocations, but each **process exited nonzero with two `cuCtxGetLimit` API errors during cuBLAS initialization in the reference**. A standalone `torch.bmm` control, without importing FLA, reproduced the same API error (one occurrence) in both precision environments; see the retained full logs. This is not reported as a clean full-process sanitizer run.

The exact five production-kernel calls were then captured from the unchanged reference tests and replayed with NaN-filled output buffers, retaining all dtype, shape, packing and layout variants. Replay matched the captured outputs bitwise and memcheck reported **0 errors** for the ReWY kernel, both initially on GPU4 and in the final TF32-disabled replay on GPU3. The latter reuses the same captured production-kernel tensors; it is not a new independent-reference test. No sanitizer API error suppression was used. The captured tensors are reproducible with `replay_rewy_memcheck.py --capture`; they are retained locally with SHA256 hashes, rather than added as binary files to this evidence branch.

```bash
python replay_rewy_memcheck.py --repo /path/to/candidate --directory /new/path/to/inputs --capture
compute-sanitizer --tool memcheck --error-exitcode 1 \
  --kernel-name kns=chunk_gated_delta_rule_bwd_kernel_rewy \
  python replay_rewy_memcheck.py --repo /path/to/candidate --directory /path/to/inputs
```

## Precision-environment correction

The remote container inherited `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, discovered during the final environment audit. This could change PyTorch FP32 reference computations even though the ReWY Triton dots explicitly disallow TF32. Initial A800 logs and timing reports are retained under `a800/initial-tf32-override/` in this evidence branch and are superseded for the final precision/performance claims. The initial baseline full file finished; its duplicate candidate run was stopped after the fresh strict run reached the final fallback tests. A first strict repeat on GPU4 was stopped after discovering another task using that card; it produced no final result. No other task was interrupted. Final A800 correctness and complete ABBA timing were rerun with the override disabled on GPU3; local numerical results were already obtained without that override. No production code, reference, tolerance, or test matrix changed. The unchanged full GDN file also sets `TRITON_F32_DEFAULT=ieee` before its later cases, which makes the FP32 fallback compile different kernels from a focused-only run. The final timing harness explicitly sets this to `ieee` as well, alongside `FLA_TRIL_PRECISION=ieee`; focused-only results were not used to bypass the full-file gate.

## Supplementary timing method

`benchmark_rewy.py` checks finite outputs/states and all six gradients against the unchanged default at the original tolerances before timing each workload. It verifies actual ReWY selection for candidate measurements. Use separate processes in baseline/candidate/candidate/baseline order on one otherwise idle GPU, with identical software and input seeds.

```bash
python benchmark_rewy.py --repo /path/to/baseline --revision 98193113ecb7835143252a786e06933e371b67cc --rewy 0 --output baseline.json
python benchmark_rewy.py --repo /path/to/candidate --revision 7608606ab33ee4fc570dd3999053a34851761636 --rewy 1 --output candidate.json
```

B is the public autograd backward including the existing recomputation, input gradients and initial-state gradient. FB includes the public forward and backward. CUDA Graph timings exclude Python dispatch. The saving policy is the upstream default recomputation, with final-state output and initial-state gradients enabled. There are ten calls to `triton.testing.do_bench_cudagraph(rep=100, return_mode="median")` per mode in each of the two processes per revision. The 100 ms value is the helper's requested repetition target, not a claim that every captured graph lasts 100 ms; each stored sample is the helper's median of ten internal graph replays. Summaries use all twenty per-invocation medians, retain all four workloads, and disclose regressions and run-to-run spread. Autotune result persistence is disabled for these isolated measurement processes.

Memory is the eager peak allocated bytes above live inputs; B additionally excludes the already-built forward graph. Returned gradients are included. FB is measured after releasing that graph. These are operator timings, not a model or optimizer training-step claim. They are not interchangeable with the paper's lower-level backward timing boundaries.

The initial baseline timing attempt failed before any valid measurement because its forward graph was created on the legacy default stream. The harness now creates that graph on a non-default stream so autograd can participate in the timer's side-stream capture. Both revisions use the same corrected harness; the original error log is retained. No kernel or tolerance was changed for this correction.

## Supplementary A800 results

All four workloads use B=1, H=4, K=V. Dense FP16 is T=1024, D=64; dense BF16 is T=4096, D=128. Packed FP16 has lengths [73, 951], D=64; packed BF16 has lengths [1, 64, 511, 1024, 2549], D=128. These are operator microbenchmarks, not official-model throughput results.

| Workload    | Mode | Baseline (us) | ReWY (us) | Ratio  | Additional peak MiB, baseline -> ReWY |
| ----------- | ---- | ------------: | --------: | -----: | ------------------------------------: |
| dense-fp16  | B    | 167.11        | 146.13    | 1.144x | 5.61 -> 4.59                          |
| dense-fp16  | FB   | 220.90        | 199.50    | 1.107x | 6.69 -> 5.67                          |
| dense-bf16  | B    | 2578.83       | 2556.43   | 1.009x | 52.44 -> 44.38                        |
| dense-bf16  | FB   | 2745.75       | 2726.85   | 1.007x | 58.75 -> 50.69                        |
| packed-fp16 | B    | 161.57        | 145.24    | 1.112x | 5.73 -> 4.72                          |
| packed-fp16 | FB   | 214.92        | 199.90    | 1.075x | 6.88 -> 5.86                          |
| packed-bf16 | B    | 1691.40       | 1666.97   | 1.015x | 54.41 -> 46.24                        |
| packed-bf16 | FB   | 1822.99       | 1798.24   | 1.014x | 61.84 -> 53.67                        |

The four-workload geometric means are 1.0682x for B and 1.0499x for FB. No median regression was observed; the small BF16 gains are preliminary and should not be treated as robust performance claims. The largest between-process median spread, relative to the smaller of the two, is 6.470%. All process medians, sample ranges, and raw reports are in `a800/strict-fp32/`. Additional allocated peak reductions range from 15.0% to 18.1% for B and 13.2% to 15.2% for FB.

Final timing used A800 GPU UUID `GPU-3bffd5d2-3e59-3525-6211-22749909c412`. Final full-file correctness and strict memcheck replay used `GPU-3bffd5d2-3e59-3525-6211-22749909c412`. Each timing process checked for an otherwise idle GPU before and after execution; a read-only occupancy audit sampled compute-process identities every two seconds throughout each process and found no other process; full correctness finished and its processes exited before timing began. The benchmark harness asserts that PyTorch FP32 matmul TF32 is disabled and records all precision/backend flags.

Initial timing and large-allocation/memcheck checks used A800 GPU UUID `GPU-aa88f58e-182d-a91c-20a7-e1adb55e2158`. Initial timing reports are retained solely for provenance, not combined with or selected over the final reports. The large-offset check is independent of the cuBLAS reference setting and remains an addressing check.

## Independent cache-test fixture investigation

[KDA cache-test overflow and bounded input correction](kda-cache-inputs/README.md). This test-only correction is independent of the ReWY backend and shared-solver scheduling fix.

## Subsequent GDP forward investigation

[GDP SM120 forward root cause and independent scheduling fix](gdp-forward-fix/README.md). This evidence distinguishes the frozen ReWY head from the independent fix and their tested integration; earlier failed records remain unchanged.
