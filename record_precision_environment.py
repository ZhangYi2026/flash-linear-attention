"""Record precision-related settings without initializing a CUDA context."""
import json
import os
import platform
import torch
import triton
keys = ('CUDA_VISIBLE_DEVICES', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE',
        'FLA_TRIL_PRECISION', 'FLA_USE_FAST_OPS', 'FLA_USE_TMA', 'FLA_CACHE_MODE', 'FLA_CI_ENV',
        'FLA_FLASH_QLA', 'FLA_TILELANG', 'FLA_REWY', 'FLA_DISABLE_BACKEND_DISPATCH')
print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
    'triton': triton.__version__, 'cuda_runtime': torch.version.cuda,
    'torch_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
    'environment': {k: os.environ.get(k) for k in keys}}, indent=2))
