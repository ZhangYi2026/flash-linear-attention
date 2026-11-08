import importlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
repo = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/latest-baseline-20261113')
sys.path.insert(0, str(repo))
os.chdir(repo)
import torch
import pytest

assert torch.cuda.get_device_capability() == (12, 0)
module = importlib.import_module('fla.ops.gated_oja_rule.chunk_h')
tuner = module.chunk_oja_bwd_kernel_dhu_blockdim64
while not hasattr(tuner, 'configs'):
    tuner = tuner.fn
original = tuner.early_config_prune
calls = []

def keep_declared_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    before = original(configs, named_args, **kwargs)
    calls.append({'V': args['V'], 'K': args['K'], 'vg_dtype': str(args['vg'].dtype), 'before': [str(c) for c in before], 'diagnostic_candidates': [str(c) for c in configs]})
    return configs

tuner.early_config_prune = keep_declared_configs
nodes = json.loads((root / 'recovery-06-test_gated_oja_rule2.nodes.json').read_text())
selected = [n for n in nodes if 'test_chunk[B2-T1000-H3-K128-M128-scale0.1-torch.bfloat16' in n]
assert len(selected) == 1
name = 'oja2-single-stage-diagnostic'
assert not (root / (name + '.xml')).exists()
status = pytest.main(['-p', 'audit_collection', *selected, '-q', '--tb=short', '-rs', '--junitxml=' + str(root / (name + '.xml'))])
report = {'exitcode': status, 'calls': calls, 'selected_configs': [str(c) for c in tuner.cache.values()], 'scope': 'diagnostic in-memory candidate pruning only; source, inputs, reference and tolerances unchanged; disk tuning cache disabled only in this probe'}
(root / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
sys.exit(status)
