# KDA cache-test input overflow

The unchanged `tests/cp/test_intracard.py::test_chunk_kda_cache` fails independently on both frozen upstream `98193113ecb7835143252a786e06933e371b67cc` and the ReWY/solver integration. Both calls return NaNs, so the existing `allclose` correctly rejects them. See `kda-control-20261113-baseline.log` and `kda-control-20261113-solver-fix-rewy.log`.

This test uses seed 0, B1/T32768/H1/D32, BF16 Gaussian keys and sigmoid beta. In the recurrence, the update factor `I - beta * k k^T` has eigenvalue `1 - beta * ||k||^2` along k. Raw Gaussian keys can make its magnitude exceed one. The negative gate does not prevent growth for these actual inputs.

`diagnose_kda_20261113.py` captures the original test inputs and evaluates the documented recurrence in CPU FP64 across all 32768 tokens. The last state has maximum magnitude 4.916e204 and the last output 3.029e204, beyond the BF16/FP32 range. Stage tracing also observes nonfinite states and outputs in the native implementation. NaNs in unused triangular intermediates are not treated as proof of the causal first divergence. Wrapper instrumentation is diagnostic and does not replace the original pytest reproduction.

The independent test-only fix `37a80cd184d23e3f35a66f826031d4444cdd3b4d` normalizes keys in this cache fixture and requires both outputs to be finite. It retains the seed, sequence length, head dimensions, cache assertions and original `atol=rtol=1e-4`. No operator implementation, numerical tolerance or skip annotation changes. This is an explicit change to test input generation, not a claim that the original overflowing inputs now work.

On RTX 5070 Ti Laptop SM120, PyTorch 2.9.1+cu128 / Triton 3.5.1, the entire original test file with this fixture correction passes: **8 passed**, no skips. The NaN allocation guard remains enabled. `FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 FLA_CACHE_RESULTS=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee`.

Command and raw result: `run_kda_fixture_fix_20261113.sh`, `kda-fixture-fix-20261113.log/.xml`. Pre-commit for the changed file and `git diff --check` pass. Runtime kernels are identical to frozen upstream, so a performance comparison is inapplicable.

The four GDP forward regression inputs, references and tolerances remain unchanged. The KDA fixture correction is reviewed separately from both the solver scheduling fix and ReWY backward. Earlier failing integration logs remain unchanged; the new integration has its own revision, manifest and result names.
