#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
python=/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python
bash "$base/run_shared_tests.sh"
"$python" "$base/gate_xml.py" "$base/solver-fix-shared.xml"
bash "$base/run_integration_tests.sh"
"$python" "$base/gate_xml.py" "$base/solver-fix-rewy.xml"
bash "$base/run_benchmarks.sh"
"$python" "$base/summarize_results.py"
