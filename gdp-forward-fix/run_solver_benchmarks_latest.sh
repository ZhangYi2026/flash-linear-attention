#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee FLA_CACHE_RESULTS=1
python=/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python
FLA_USE_TMA=1 "$python" "$base/test_tma_solver.py" --repo /home/zhang_e/.cache/fla-rewy-pr-20261108/latest-solver-20261113 2>&1 | tee "$base/solver-fix-tma.log"
"$python" "$base/gate_xml.py" "$base/solver-fix-tma.xml"
"$python" "$base/check_solver_large_offset.py" --repo /home/zhang_e/.cache/fla-rewy-pr-20261108/latest-solver-20261113 2>&1 | tee "$base/solver-large-offset.log"
"$python" "$base/reproduce_solver_rounding.py" --repo /home/zhang_e/.cache/fla-rewy-pr-20261108/latest-baseline-20261113 2>&1 | tee "$base/minimal-solver-reproduction.log"
for pair in latest-baseline-20261113:A1 latest-solver-20261113:B1 latest-solver-20261113:B2 latest-baseline-20261113:A2; do
 variant=${pair%:*}
 tag=${pair#*:}
 FLA_CACHE_RESULTS=0 "$python" "$base/benchmark_solver.py" --repo "/home/zhang_e/.cache/fla-rewy-pr-20261108/$variant" --tag "$tag" 2>&1 | tee "$base/benchmark-$tag.log"
done
