from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
cache=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
expected=json.loads((root/'solver-fix-source-hashes.json').read_text())
checked={f:hashlib.sha256((cache/'solver-fix'/f).read_bytes()).hexdigest() for f in expected}
bad=[f for f in checked if checked[f]!=expected[f]]
assert not bad,bad
integration=json.loads((root/'integration-source-hashes.json').read_text())
assert all(hashlib.sha256((cache/'solver-fix-rewy'/f).read_bytes()).hexdigest()==h for f,h in integration.items())
pairs=[]
for base,cand in [('baseline','solver-fix'),('candidate','solver-fix-rewy')]:
 files=sorted(p.relative_to(cache/base) for d in ['fla','tests'] for p in (cache/base/d).rglob('*.py'))
 diffs=[str(f) for f in files if (cache/base/f).read_bytes()!=(cache/cand/f).read_bytes()]
 assert diffs==['fla/ops/utils/solve_tril.py'],diffs
 pairs.append({'baseline':base,'candidate':cand,'changed_files':diffs,'solver_sha256':hashlib.sha256((cache/cand/diffs[0]).read_bytes()).hexdigest()})
report={'fix_commit':'034c3d41a18f1c6a941918c39396ef004ec757ca','integration_commit':'476bf096937afe433ff867ef073a2524e5144678','integration_files_verified':len(integration),'source_files_verified':len(checked),'pairs':pairs}
(root/'runtime-source-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
