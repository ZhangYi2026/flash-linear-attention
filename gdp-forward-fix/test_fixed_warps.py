import importlib, json, os, sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/baseline')
sys.path.insert(0,str(repo));os.chdir(repo)
m=importlib.import_module('fla.ops.utils.solve_tril')
tuner=m.merge_16x16_to_64x64_inverse_kernel
while not hasattr(tuner,'configs'):tuner=tuner.fn
tuner.configs=[c for c in tuner.configs if c.num_warps==2]
tuner.cache.clear()
print('DIAGNOSTIC solver configurations:',tuner.configs,flush=True)
import pytest
sys.exit(pytest.main(['-q','tests/models/test_modeling_gated_deltaproduct.py','-k','test_modeling','--tb=short','--junitxml='+str(root/'fixed-warps-models.xml')]))
