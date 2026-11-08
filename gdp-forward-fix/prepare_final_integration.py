import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parent / 'fla_rewy_solver_integration_20261108'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert commit == 'e7e11699b40e337cabc11c8956e050eaae6809b9'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
archive = root / 'integration-20261113.tar'
assert not archive.exists()
subprocess.run(['git', 'archive', '--format=tar', '--output=' + str(archive), commit], cwd=repo, check=True)
files = subprocess.check_output(['git', 'ls-files', 'fla', 'tests'], cwd=repo, text=True).splitlines()
hashes = {f: hashlib.sha256(subprocess.check_output(['git', 'show', commit + ':' + f], cwd=repo)).hexdigest() for f in files}
(root / 'integration-20261113-source-hashes.json').write_text(json.dumps({'commit': commit, 'files': hashes}, indent=2) + '\n')
script = '''from pathlib import Path
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
(root/'integration-20261113-runtime-verification.json').write_text(json.dumps({'commit':record['commit'],'files_verified':len(record['files']),'changes_from_previous_integration':changed},indent=2)+'\\n')
print('Verified',len(record['files']),'tracked runtime files at',record['commit'])
'''
(root / 'extract_final_integration.py').write_text(script, newline='\n')
shell = (root / 'run_integration_tests.sh').read_text().replace('solver-fix-rewy', 'solver-fix-rewy-kda-20261113')
(root / 'run_final_integration.sh').write_text(shell, newline='\n')
print(commit)
