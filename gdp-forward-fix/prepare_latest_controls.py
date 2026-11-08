from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
rows = []
for name, commit in [('latest-baseline-20261113', '0a649e4c7201f0edcba734832773fe8ab371367a'), ('latest-kda-20261113', '6aa27ca0e095ef91f74b6f068c5ec8de07d96970')]:
    target = cache / name
    assert not target.exists()
    target.mkdir()
    with tarfile.open(root / (name + '.tar')) as archive:
        archive.extractall(target, filter='data')
    hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['fla', 'tests'] for p in (target / folder).rglob('*.py')}
    rows.append({'name': name, 'commit': commit, 'files': hashes})
(root / 'latest-control-source-hashes.json').write_text(json.dumps(rows, indent=2) + '\n')
print([(r['name'], r['commit'], len(r['files'])) for r in rows])
