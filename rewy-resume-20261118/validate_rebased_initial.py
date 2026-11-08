"""Validate only ReWY against current upstream, preserving baseline failures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-resume-20261118')
sources = json.loads((root / 'sources.json').read_text())
cache.mkdir(exist_ok=True)
assert not any(cache.iterdir()), 'Use a fresh source cache for a new validation attempt'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', FLA_CI_ENV='0', FLA_FLASH_QLA='0',
           FLA_TILELANG='0', FLA_DISABLE_BACKEND_DISPATCH='0', FLA_CACHE_RESULTS='1',
           TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', TRITON_F32_DEFAULT='ieee', FLA_TRIL_PRECISION='ieee')
targets = ['tests/ops/test_gdn_kernels.py::test_gdn_full_bwd', 'tests/ops/test_gdp.py',
           'tests/layers/test_gated_deltanet.py', 'tests/layers/test_mom_shared_no_grad.py',
           'tests/models/test_modeling_gated_deltanet.py', 'tests/models/test_modeling_gated_deltaproduct.py',
           'tests/models/test_modeling_mom.py', 'tests/cp/test_gdn.py', 'tests/cp/test_gdp.py',
           'tests/cp/test_intracard.py::test_chunk_gdn_grouped_heads']
results = []
for source in sources:
    label = source['label']
    archive = root / (label + '.tar')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == source['archive_sha256']
    checkout = cache / label
    checkout.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(checkout, filter='data')
    active = dict(env, FLA_REWY='1' if label == 'candidate' else '0', PYTHONPATH=str(checkout))
    if label == 'candidate':
        dep = subprocess.run([sys.executable, 'scripts/find_dependent_tests.py',
                              'fla/ops/gated_delta_rule/chunk.py',
                              'fla/ops/gated_delta_rule/backends/rewy.py'],
                             cwd=checkout, env=active, text=True, capture_output=True, check=True)
        (root / 'dependent-discovery.txt').write_text(dep.stdout + dep.stderr)
    for selection, tests in [('gdn', ['tests/ops/test_gdn.py']), ('dependent', targets)]:
        name = label + '-' + selection
        output = root / (name + '.log')
        assert not output.exists()
        command = [sys.executable, '-m', 'pytest', *tests, '-q', '--tb=short', '-rs',
                   '--junitxml=' + str(root / (name + '.xml'))]
        print('Running ' + name, flush=True)
        with output.open('w') as stream:
            run = subprocess.run(command, cwd=checkout, env=active, stdout=stream, stderr=subprocess.STDOUT)
        cases = list(ET.parse(root / (name + '.xml')).getroot().iter('testcase'))
        failed = [{'node': c.attrib, 'failure': ET.tostring(c, encoding='unicode')} for c in cases
                  if c.find('failure') is not None or c.find('error') is not None]
        skipped = [c.attrib for c in cases if c.find('skipped') is not None]
        result = {'name': name, 'commit': source['commit'], 'exitcode': run.returncode,
                  'command': command, 'environment': {k: active[k] for k in env if k.startswith(('FLA_', 'TRITON_', 'TORCH_'))},
                  'rewy': active['FLA_REWY'], 'collected': len(cases), 'passed': len(cases)-len(failed)-len(skipped),
                  'failed': failed, 'skipped': skipped}
        results.append(result)
        (root / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ['failed', 'skipped', 'command', 'environment']}), flush=True)
        print('\n'.join(output.read_text().splitlines()[-7:]), flush=True)
        if selection == 'gdn' and (run.returncode or failed):
            raise SystemExit('GDN gate failed; stop before further validation')
        if selection == 'dependent' and (run.returncode not in (0, 1) or any(c.find('error') is not None for c in cases)):
            raise SystemExit('Dependent selection has collection/setup/runtime errors; investigate')

baseline, candidate = results[1], results[3]
def identities(rows):
    return sorted((r['node']['classname'], r['node']['name']) for r in rows)
assert identities(baseline['failed']) == identities(candidate['failed']), 'New candidate failure'
assert baseline['collected'] == candidate['collected']
assert baseline['passed'] == candidate['passed']
assert sorted((r['classname'], r['name']) for r in baseline['skipped']) == sorted((r['classname'], r['name']) for r in candidate['skipped'])
(root / 'comparison.json').write_text(json.dumps({'no_additional_candidate_failures': True,
    'shared_failed_nodes': identities(baseline['failed']), 'baseline_commit': sources[0]['commit'],
    'candidate_commit': sources[1]['commit'], 'candidate_gdn_passed': results[2]['passed'],
    'dependent_passed_each': baseline['passed'], 'dependent_failed_each': len(baseline['failed']),
    'dependent_skipped_each': len(baseline['skipped']), 'sm90_performance_still_missing': True}, indent=2) + '\n')
print('Rebased ReWY validation completed; baseline failures remain explicitly disclosed.', flush=True)
