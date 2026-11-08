#!/usr/bin/env bash
set -euo pipefail
cd /home/zhang_e/.cache/fla-rewy-pr-20261108/baseline
export CUDA_VISIBLE_DEVICES=0 FLA_CI_ENV=0 FLA_REWY=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
/home/zhang_e/.venvs/kda-residual-wy-20261023/bin/python -m pytest 'tests/models/test_modeling_gated_deltaproduct.py::test_modeling[L4-B4-T1024-H4-D64-l2False-bsNone-torch.bfloat16-varlen]' -q --tb=short -rs
