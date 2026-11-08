#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_CACHE_RESULTS=1
cd /home/zhang_e/.cache/fla-rewy-pr-20261108/baseline
export PYTHONPATH="$base:$PWD"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python -m pytest -p audit_collection 'tests/ops/test_oja.py::test_naive_chunk_oja[B4-T2048-H8-D128-scale1-gate_logit_normalizer10-torch.float32]' -q --tb=short -rs --junitxml="$base/baseline-oja-fresh.xml" 2>&1 | tee "$base/baseline-oja-fresh.log"

/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/gate_xml.py" "$base/baseline-oja-fresh.xml"
cd /home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix
export PYTHONPATH="$base:$PWD"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python -m pytest -p audit_collection 'tests/ops/test_oja.py::test_naive_chunk_oja[B4-T2048-H8-D128-scale1-gate_logit_normalizer10-torch.float32]' -q --tb=short -rs --junitxml="$base/solver-fix-oja-fresh.xml" 2>&1 | tee "$base/solver-fix-oja-fresh.log"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/gate_xml.py" "$base/solver-fix-oja-fresh.xml"
bash "$base/run_integration_tests.sh"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/gate_xml.py" "$base/solver-fix-rewy.xml"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/run_shared_recovery.py"
bash "$base/run_benchmarks.sh"
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python "$base/summarize_benchmark.py"
