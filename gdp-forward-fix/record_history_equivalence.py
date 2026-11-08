"""Verify that the user's single-commit cleanup changes no tested source tree."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = root.parent / 'fla_rewy_upstream_20261108'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()


rows = []
for branch, tested in [
    ('rewy-gdn-backward', '7608606ab33ee4fc570dd3999053a34851761636'),
    ('fix-kda-cache-test-inputs', '6aa27ca0e095ef91f74b6f068c5ec8de07d96970'),
    ('fix-sm120-solve-tril-consistency', '44db92fd7c343544c58c505edf217dceb1931636'),
    ('fix-sm120-oja-single-stage', 'b6172a05f2405a73c96ec190ee41f4083faa31a8'),
    ('fix-solve-tril-tma-coordinates', '28e74f8ba0c8ebb0a0edd11c42dc7ef628b27cae'),
    ('rewy-gdn-solver-integration', '2d2eef814c0fe14346dfdf13ee5b82724ee8d090'),
]:
    final = git('rev-parse', branch)
    tree = git('rev-parse', final + '^{tree}')
    assert tree == git('rev-parse', tested + '^{tree}'), branch
    assert int(git('rev-list', '--count', 'upstream/main..' + branch)) == 1, branch
    assert not git('diff', tested, final), branch
    rows.append({'branch': branch, 'tested_commit': tested, 'published_commit': final, 'tree': tree,
                 'tree_identical': True, 'commits_ahead_of_upstream': 1})
(root / 'submission-history-equivalence.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
