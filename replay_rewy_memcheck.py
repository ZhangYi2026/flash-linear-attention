"""Capture the five unchanged numerical cases, then audit their exact kernel calls.

Capture runs the normal independent-reference tests. Replay avoids initializing
cuBLAS for the reference, and checks the production kernel under memcheck with
NaN-poisoned output buffers. It is a memory check, not another numerical oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--directory', type=Path, required=True)
parser.add_argument('--capture', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))

import torch
import fla.ops.gated_delta_rule.backends.rewy as module

kernel = module.chunk_gated_delta_rule_bwd_kernel_rewy
outputs = ('dq', 'dk', 'dv', 'db', 'dg')

if args.capture:
    import os
    import pytest

    args.directory.mkdir(parents=True, exist_ok=False)
    captured = []

    class Recorder:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(**kwargs):
                result = launch(**kwargs)
                snapshot = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                path = args.directory / f'case-{len(captured)}.pt'
                torch.save({'grid': grid, 'kwargs': snapshot}, path)
                captured.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                return result

            return run

    module.chunk_gated_delta_rule_bwd_kernel_rewy = Recorder()
    os.chdir(args.repo)
    code = pytest.main(['tests/ops/test_gdn.py', '-k', 'test_rewy_chunk', '-q', '--tb=short'])
    assert code == 0 and len(captured) == 5, (code, len(captured))
    (args.directory / 'manifest.json').write_text(json.dumps(captured, indent=2) + '\n')
else:
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    assert len(manifest) == 5
    for item in manifest:
        path = args.directory / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        snapshot = torch.load(path, weights_only=True)
        reference = snapshot['kwargs']
        kwargs = {}
        for key, value in reference.items():
            if key in outputs:
                kwargs[key] = torch.full_like(value, float('nan'), device='cuda')
            elif isinstance(value, torch.Tensor):
                kwargs[key] = value.cuda()
            else:
                kwargs[key] = value
        kernel[snapshot['grid']](**kwargs)
        torch.cuda.synchronize()
        for name in outputs:
            actual = kwargs[name].cpu()
            assert torch.isfinite(actual).all(), (item['file'], name)
            torch.testing.assert_close(actual, reference[name], rtol=0, atol=0)
        print(json.dumps({'case': item['file'], 'passed': True}), flush=True)
    print('Five exact production-kernel calls replayed successfully.')
