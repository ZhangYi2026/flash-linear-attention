from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
rows = []
for name, commit in [('latest-oja-20261113', 'b6172a05f2405a73c96ec190ee41f4083faa31a8'), ('latest-integration-oja-20261113', '7d7b00ca4e638d1a77aea9976e945e6f122b6c2d')]:
    target = cache / name
    assert not target.exists()
    target.mkdir()
    with tarfile.open(root / (name + '.tar')) as archive:
        archive.extractall(target, filter='data')
    hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['fla', 'tests'] for p in (target / folder).rglob('*.py')}
    rows.append({'name': name, 'commit': commit, 'files': hashes})
(root / 'oja-candidate-source-hashes.json').write_text(json.dumps(rows, indent=2) + '\n')
base = cache / 'latest-baseline-20261113'
assert [f for f in rows[0]['files'] if (base / f).read_bytes() != (cache / rows[0]['name'] / f).read_bytes()] == ['fla/ops/gated_oja_rule/chunk_h.py']
for file in ['tests/ops/test_oja.py', 'tests/ops/test_gated_oja_rule2.py']:
    assert (base / file).read_bytes() == (cache / 'baseline' / file).read_bytes()
shell = (root / 'run_integration_tests.sh').read_text().replace('solver-fix-rewy', 'latest-integration-oja-20261113')
(root / 'run_latest_integration_oja.sh').write_text(shell, newline='\n')
print([(r['name'], r['commit'], r['files']['fla/ops/gated_oja_rule/chunk_h.py']) for r in rows])
