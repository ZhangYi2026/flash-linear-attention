#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_CACHE_RESULTS=0 PYTHONPATH="$base"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/probe_oja2_single_stage.py" > "$base/oja2-single-stage-diagnostic.log" 2>&1
