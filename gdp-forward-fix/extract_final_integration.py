from pathlib import Path
import tarfile,hashlib,json
root=Path('/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108')
dest=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix-rewy-kda-20261113')
assert not dest.exists()
dest.mkdir()
with tarfile.open(root/'integration-20261113.tar') as t:
    t.extractall(dest,filter='data')
record=json.loads((root/'integration-20261113-source-hashes.json').read_text())
assert all(hashlib.sha256((dest/f).read_bytes()).hexdigest()==h for f,h in record['files'].items())
old=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix-rewy')
changed=[f for f in record['files'] if (old/f).read_bytes()!=(dest/f).read_bytes()]
assert changed==['tests/cp/test_intracard.py'],changed
(root/'integration-20261113-runtime-verification.json').write_text(json.dumps({'commit':record['commit'],'files_verified':len(record['files']),'changes_from_previous_integration':changed},indent=2)+'\n')
print('Verified',len(record['files']),'tracked runtime files at',record['commit'])
