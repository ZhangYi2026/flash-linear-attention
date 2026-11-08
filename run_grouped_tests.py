"""Run the unchanged grouped-head tests skipped by the generic backend matrix."""

import json
from pathlib import Path
import subprocess
import sys

nodes = json.loads(Path(__file__).with_name('grouped-heads-nodes.json').read_text())
result = subprocess.run([sys.executable, '-m', 'pytest', *nodes, '-q', '--tb=short', '-rs'])
raise SystemExit(result.returncode)
