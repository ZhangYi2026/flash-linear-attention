import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
file = 'fla/ops/gated_oja_rule/chunk_h.py'
before = (cache / 'latest-baseline-20261113' / file).read_bytes()
after = (cache / 'latest-oja-20261113' / file).read_bytes()
old = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(before).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
new = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(after).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
assert old.keys() == new.keys()
changed = [name for name in old if old[name] != new[name]]
assert changed == ['_prune_bwd_dhu_configs'], changed
record = {'baseline': '0a649e4c7201f0edcba734832773fe8ab371367a', 'candidate': 'b6172a05f2405a73c96ec190ee41f4083faa31a8', 'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': hashlib.sha256(after).hexdigest(), 'only_changed_function': changed, 'unchanged_functions': [name for name in old if name not in changed]}
(root / 'oja-algorithm-ast-verification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
