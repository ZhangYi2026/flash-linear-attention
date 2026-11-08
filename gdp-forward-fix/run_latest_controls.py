import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
runs = [('latest-baseline-20261113', 'latest-baseline-gdp', ['tests/models/test_modeling_gated_deltaproduct.py']), ('latest-baseline-20261113', 'latest-baseline-kda', ['tests/cp/test_intracard.py::test_chunk_kda_cache']), ('latest-kda-20261113', 'latest-fixed-kda', ['tests/cp/test_intracard.py'])]
results = []
for variant, name, targets in runs:
    repo = cache / variant
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', FLA_CI_ENV='0', FLA_REWY='0', FLA_FLASH_QLA='0', FLA_TILELANG='0', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', TRITON_F32_DEFAULT='ieee', FLA_CACHE_RESULTS='1', PYTHONPATH=str(root) + ':' + str(repo))
    assert not (root / (name + '.xml')).exists()
    print('Running', name, flush=True)
    with (root / (name + '.log')).open('w') as output:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'audit_collection', *targets, '-q', '--tb=short', '-rs', '--junitxml=' + str(root / (name + '.xml'))], cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
    results.append({'name': name, 'exitcode': result.returncode})
    print('\n'.join((root / (name + '.log')).read_text().splitlines()[-18:]), flush=True)
(root / 'latest-control-runs.json').write_text(json.dumps(results, indent=2) + '\n')
assert results[-1]['exitcode'] == 0
