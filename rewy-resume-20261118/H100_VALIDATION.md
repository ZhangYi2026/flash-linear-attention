# ReWY H100 validation request

Only PR #1703 is in scope. The earlier solver/Oja/TMA/KDA proposals are closed and are not prerequisites.

The candidate is `e9fc25f2714a653995120a751e5f9356fd8bab19`, rebased onto `cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b`. The six-file ReWY patch is unchanged apart from upstream context (see `range-diff.txt`). Refresh the PR's exact base/head fields when dispatching the upstream workflow, because it checks both values.

## Existing upstream workflow

`.github/workflows/agent-h100-tests.yml` accepts `pr_number=1703`, exact base/head SHAs, `suite=ops`, `targets=tests/ops/test_gdn.py`, `dispatch=on`, and `source_ref=main`. A maintainer or authorized review service must dispatch it. The contributor account has only `pull` permission on the upstream repository; unlocking the fork's Actions billing does not grant access to the upstream H100 runner.

The four `test_rewy_*` functions collect 14 cases and explicitly enable or reject the opt-in backend while checking actual dispatch. `dispatch=on` alone does not enable ReWY for other tests: the backend is default-off. For the full existing suite with the backend enabled, also set `FLA_REWY=1`. A smoke run with `FLA_DISABLE_BACKEND_DISPATCH=1` cannot supply this evidence.

## Matched baseline/candidate run

Use separate checkouts of the two pinned revisions, on the same idle datacenter SM90+ GPU and software. Record the GPU model, driver, PyTorch, Triton and Python versions. Set these environment variables before importing Torch/FLA:

```bash
export CUDA_VISIBLE_DEVICES=0
export FLA_CI_ENV=0 FLA_FLASH_QLA=0 FLA_TILELANG=0
export FLA_DISABLE_BACKEND_DISPATCH=0 FLA_CACHE_RESULTS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee
# From the baseline checkout: FLA_REWY=0; from the candidate checkout: FLA_REWY=1.
FLA_REWY=1 python -m pytest tests/ops/test_gdn.py -q --tb=short -rs
FLA_REWY=1 python -m pytest \
  tests/ops/test_gdn_kernels.py::test_gdn_full_bwd tests/ops/test_gdp.py \
  tests/layers/test_gated_deltanet.py tests/layers/test_mom.py \
  tests/models/test_modeling_gated_deltanet.py tests/models/test_modeling_gated_deltaproduct.py \
  tests/models/test_modeling_mom.py tests/cp/test_gdn.py tests/cp/test_gdp.py \
  tests/cp/test_intracard.py::test_chunk_gdn_grouped_heads -q --tb=short -rs
```

Repeat the same targets with `FLA_REWY=0` in the baseline checkout. Keep original inputs, references, thresholds and skips. Current main replaced `test_mom_shared_no_grad.py` with coverage in `test_mom.py`; the removed path is not a runnable target. Report failing node IDs and messages on each revision, rather than treating baseline failures as passes.

## Timing after correctness acceptance

The evidence branch root contains `benchmark_rewy.py`, its commands and all original A800 samples. It verifies workload outputs, states and gradients, checks actual ReWY selection, and measures public backward (B) and forward+backward (FB) separately with the existing recomputation/save policy. Dense/packed and FP16/BF16 workloads are included.

Use that unchanged reproducer with `--repo CHECKOUT --revision COMMIT --rewy 0|1 --output NEW_FILE.json`. Run fresh baseline/candidate/candidate/baseline processes on the same idle H100, retain all samples, and use a fresh output filename for every process. The script sets strict math and dispatch flags itself. Do not run or claim a performance gate while relevant numerical failures remain unresolved; record any maintainer-approved baseline exception explicitly. Consumer SM120 diagnostics and historical A800 timings are supplementary, not H100 evidence.
