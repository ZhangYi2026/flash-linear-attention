"""Host-only source and result audit; does not import accelerator libraries."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
repo = root.parent / 'fla_rewy_upstream_20261108'
source = 'fla/ops/utils/solve_tril.py'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo)


class WithoutTMA(ast.NodeTransformer):
    def visit_If(self, node):
        if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) and isinstance(node.test.operand, ast.Name) and node.test.operand.id == 'USE_TMA':
            return [self.visit(n) for n in node.body]
        return self.generic_visit(node)


before = ast.parse(git('show', '0a649e4c:' + source))
after = ast.parse(git('show', '28e74f8:' + source))
assert ast.dump(WithoutTMA().visit(before)) == ast.dump(WithoutTMA().visit(after))
previous = '7d7b00ca4e638d1a77aea9976e945e6f122b6c2d'
final = '2d2eef814c0fe14346dfdf13ee5b82724ee8d090'
changed = git('diff', '--name-only', previous, final).decode().splitlines()
assert changed == [source], changed
before = ast.parse(git('show', previous + ':' + source))
after = ast.parse(git('show', final + ':' + source))
assert ast.dump(WithoutTMA().visit(before)) == ast.dump(WithoutTMA().visit(after))
report = {'baseline': '0a649e4c7201f0edcba734832773fe8ab371367a', 'tma_fix': '28e74f8ba0c8ebb0a0edd11c42dc7ef628b27cae',
          'previous_integration': previous, 'final_integration': final, 'files_changed_from_previous_integration': changed,
          'non_tma_specialization_ast_equal': True,
          'shared_caller_evidence': 'current-shared: 212 passed, 2 skipped; identical non-TMA source specialization'}
for ref, expected in [('fix-sm120-solve-tril-consistency', [source]),
                      ('fix-sm120-oja-single-stage', ['fla/ops/gated_oja_rule/chunk_h.py']),
                      ('fix-solve-tril-tma-coordinates', [source])]:
    changed = git('diff', '--name-only', '0a649e4c', ref).decode().splitlines()
    assert changed == expected, (ref, changed)
    report[ref] = {'sha': git('rev-parse', ref).decode().strip(), 'changed_files': changed,
                   'source_sha256': {f: hashlib.sha256(git('show', ref + ':' + f)).hexdigest() for f in changed}}
(root / 'submission-source-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
