#!/usr/bin/env bash
set -uo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/test_fixed_warps.py" 2>&1 | tee "$base/fixed-warps-models.log"
