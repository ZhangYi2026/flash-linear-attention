#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee FLA_CACHE_RESULTS=1 FLA_USE_TMA=1
root=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
cd /mnt/d/mycode/hw/gdn2/output/fla_solve_tril_fix_20261108
export PYTHONPATH="$root:$PWD"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python -m pytest -p audit_collection tests/ops/test_solve_tril.py -k 'not large_batch_offsets' -q --tb=short -rs --junitxml="$root/tma-coordinates-first.xml" > "$root/tma-coordinates-first.log" 2>&1
