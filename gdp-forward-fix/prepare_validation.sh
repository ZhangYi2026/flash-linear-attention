#!/usr/bin/env bash
set -euo pipefail
cache=/home/zhang_e/.cache/fla-rewy-pr-20261108
cp -a "$cache/baseline" "$cache/solver-fix"
cp -a "$cache/candidate" "$cache/solver-fix-rewy"
for variant in solver-fix solver-fix-rewy; do
 cp /mnt/d/mycode/hw/gdn2/output/fla_solve_tril_fix_20261108/fla/ops/utils/solve_tril.py "$cache/$variant/fla/ops/utils/solve_tril.py"
done
