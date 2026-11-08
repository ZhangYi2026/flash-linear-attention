#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_CACHE_RESULTS=0
for pair in latest-baseline-20261113:A1 latest-oja-20261113:B1 latest-oja-20261113:B2 latest-baseline-20261113:A2; do
 variant=${pair%:*}
 tag=${pair#*:}
 /home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/benchmark_oja.py" --repo "/home/zhang_e/.cache/fla-rewy-pr-20261108/$variant" --tag "$tag" > "$base/oja-benchmark-$tag.log" 2>&1
done
