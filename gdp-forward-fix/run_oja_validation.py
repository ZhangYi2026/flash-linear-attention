import os
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/latest-oja-20261113')
original = json.loads((root / 'solver-fix-shared.nodes.json').read_text())
remote = json.loads((root / 'oja-reference-nodes.json').read_text())
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', FLA_CI_ENV='0', FLA_REWY='0', FLA_FLASH_QLA='0', FLA_TILELANG='0', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', TRITON_F32_DEFAULT='ieee', FLA_CACHE_RESULTS='1', PYTHONPATH=str(root) + ':' + str(repo))
runs = []
for filename in ['test_oja', 'test_gated_oja_rule2']:
    selected = [n for n in original if n.startswith('tests/ops/' + filename + '.py::') and n not in remote]
    name = 'oja-fix-' + filename
    assert not (root / (name + '.xml')).exists()
    print('Running', name, len(selected), flush=True)
    with (root / (name + '.log')).open('w') as output:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'audit_collection', *selected, '-q', '--tb=short', '-rs', '--maxfail=1', '--junitxml=' + str(root / (name + '.xml'))], cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
    runs.append({'name': name, 'nodes': selected, 'exitcode': result.returncode, 'revision': 'b6172a05f2405a73c96ec190ee41f4083faa31a8'})
    (root / 'oja-fix-runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    print('\n'.join((root / (name + '.log')).read_text().splitlines()[-20:]), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    subprocess.run([sys.executable, str(root / 'gate_xml.py'), str(root / (name + '.xml'))], check=True)
assert sum(len(r['nodes']) for r in runs) == 57
