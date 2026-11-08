from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
rows = []
for name, commit in [('latest-solver-20261113', '44db92fd7c343544c58c505edf217dceb1931636'), ('latest-integration-20261113', '04041db08f438cc457a4d70828b1641ed0bf8a0b')]:
    target = cache / name
    assert not target.exists()
    target.mkdir()
    with tarfile.open(root / (name + '.tar')) as archive:
        archive.extractall(target, filter='data')
    hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['fla', 'tests'] for p in (target / folder).rglob('*.py')}
    rows.append({'name': name, 'commit': commit, 'files': hashes})
    assert hashes['fla/ops/utils/solve_tril.py'] == 'fba06478219c4086b9b461e508aa35ebe8656cc74237baab78dafeb71906a77d'
(root / 'latest-candidate-source-hashes.json').write_text(json.dumps(rows, indent=2) + '\n')
base = cache / 'latest-baseline-20261113'
changed = [f for f in rows[0]['files'] if (base / f).read_bytes() != (cache / rows[0]['name'] / f).read_bytes()]
assert changed == ['fla/ops/utils/solve_tril.py'], changed
shell = (root / 'run_integration_tests.sh').read_text().replace('solver-fix-rewy', 'latest-integration-20261113')
(root / 'run_latest_integration.sh').write_text(shell, newline='\n')
print([(r['name'], r['commit'], len(r['files'])) for r in rows])
