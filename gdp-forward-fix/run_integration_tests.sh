#!/usr/bin/env bash
set -uo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=1 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_CACHE_RESULTS=1
cd /home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix-rewy
export PYTHONPATH="$base:$PWD"
python=/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python
"$python" -m pytest -p audit_collection tests/ops/test_gdn.py tests/ops/test_gdn_kernels.py tests/ops/test_gdp.py tests/cp/test_gdn.py tests/cp/test_gdp.py tests/cp/test_intracard.py tests/layers/test_gated_deltanet.py tests/layers/test_mom_shared_no_grad.py tests/models/test_modeling_gated_deltanet.py tests/models/test_modeling_gated_deltaproduct.py tests/models/test_modeling_mom.py -q --tb=short -rs --junitxml="$base/solver-fix-rewy.xml" 2>&1 | tee "$base/solver-fix-rewy.log"
