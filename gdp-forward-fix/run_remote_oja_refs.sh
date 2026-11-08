#!/usr/bin/env bash
set -euo pipefail
out=/home/zy/zy_delta_rule/fla_gdp_forward_fix_20261109
mkdir -p "$out"
common=(docker exec -e USER=zy -e LOGNAME=zy -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
 -e CUDA_VISIBLE_DEVICES=GPU-aa88f58e-182d-a91c-20a7-e1adb55e2158
 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e TRITON_F32_DEFAULT=ieee -e FLA_CI_ENV=0
 -e FLA_REWY=0 -e FLA_CACHE_RESULTS=1 -e FLA_FLASH_QLA=0 -e FLA_TILELANG=0
 -e PYTHONPATH=/workspace/fla_gdp_forward_fix_20261109:/workspace/fla_rewy_baseline_20261108
 -w /workspace/fla_rewy_baseline_20261108 zy_delta_rule /workspace/env/bin/python)
"${common[@]}" -m pytest -p audit_collection 'tests/ops/test_oja.py::test_naive_chunk_oja[B4-T2048-H8-D128-scale1-gate_logit_normalizer10-torch.float32]' -q --tb=short -rs --junitxml=/workspace/fla_gdp_forward_fix_20261109/a800-naive-oja-0.xml > "$out/a800-naive-oja-0.log" 2>&1
cat "$out/a800-naive-oja-0.log"
"${common[@]}" -m pytest -p audit_collection 'tests/ops/test_oja.py::test_naive_chunk_oja[B4-T1024-H4-D64-scale0.1-gate_logit_normalizer10-torch.float32]' -q --tb=short -rs --junitxml=/workspace/fla_gdp_forward_fix_20261109/a800-naive-oja-1.xml > "$out/a800-naive-oja-1.log" 2>&1
cat "$out/a800-naive-oja-1.log"
"${common[@]}" -m pytest -p audit_collection 'tests/ops/test_oja.py::test_naive_chunk_oja[B2-T1536-H4-D128-scale1-gate_logit_normalizer100-torch.float16]' -q --tb=short -rs --junitxml=/workspace/fla_gdp_forward_fix_20261109/a800-naive-oja-2.xml > "$out/a800-naive-oja-2.log" 2>&1
cat "$out/a800-naive-oja-2.log"
"${common[@]}" -m pytest -p audit_collection 'tests/ops/test_oja.py::test_naive_chunk_oja[B4-T2048-H4-D256-scale1-gate_logit_normalizer100-torch.float16]' -q --tb=short -rs --junitxml=/workspace/fla_gdp_forward_fix_20261109/a800-naive-oja-3.xml > "$out/a800-naive-oja-3.log" 2>&1
cat "$out/a800-naive-oja-3.log"
