import os
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/baseline')
nodes = json.loads((root / 'recovery-06-test_gated_oja_rule2.nodes.json').read_text())
selected = [n for n in nodes if 'test_chunk[B2-T1000-H3-K128-M128-scale0.1-torch.bfloat16' in n]
assert len(selected) == 1
name = 'baseline-oja2-resource-20261113'
assert not (root / (name + '.xml')).exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', FLA_CI_ENV='0', FLA_REWY='0', FLA_FLASH_QLA='0', FLA_TILELANG='0', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', TRITON_F32_DEFAULT='ieee', FLA_CACHE_RESULTS='1', PYTHONPATH=str(root) + ':' + str(repo))
with (root / (name + '.log')).open('w') as output:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'audit_collection', *selected, '-q', '--tb=short', '-rs', '--junitxml=' + str(root / (name + '.xml'))], cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
print('Oja2 baseline exit:', result.returncode, flush=True)
print('\n'.join((root / (name + '.log')).read_text().splitlines()[-25:]), flush=True)
subprocess.run([sys.executable, str(root / 'run_latest_controls.py')], check=True)
