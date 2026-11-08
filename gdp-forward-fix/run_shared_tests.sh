#!/usr/bin/env bash
set -uo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_CACHE_RESULTS=1
cd /home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix
export PYTHONPATH="$base:$PWD"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python -m pytest -p audit_collection tests/ops/test_solve_tril.py::test_solve_tril tests/ops/test_solve_tril.py::test_solve_tril_varlen tests/ops/test_delta.py tests/ops/test_delta_product.py tests/ops/test_gdp.py tests/ops/test_comba.py tests/ops/test_oja.py tests/ops/test_gated_oja_rule2.py tests/ops/test_precond_gated_delta.py tests/ops/test_path_attn.py tests/models/test_modeling_gated_deltaproduct.py -q --tb=short -rs --junitxml="$base/solver-fix-shared.xml" 2>&1 | tee "$base/solver-fix-shared.log"
