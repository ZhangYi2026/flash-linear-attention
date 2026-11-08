#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python /mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108/trace_operator.py --repo /home/zhang_e/.cache/fla-rewy-pr-20261108/baseline "$@"
