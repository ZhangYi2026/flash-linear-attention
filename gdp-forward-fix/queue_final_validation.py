"""Advance only after every preceding exact-node correctness gate succeeds."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
while True:
    path = root / 'oja-fix-runs.json'
    if path.exists():
        runs = json.loads(path.read_text())
        assert all(r['exitcode'] == 0 for r in runs), 'Standalone Oja validation failed'
        if len(runs) == 2:
            break
    time.sleep(10)
for run in runs:
    subprocess.run([sys.executable, str(root / 'gate_xml.py'), str(root / (run['name'] + '.xml'))], check=True)
assert sum(len(r['nodes']) for r in runs) == 57
subprocess.run([sys.executable, str(root / 'run_current_shared.py')], check=True)
subprocess.run(['bash', str(root / 'run_latest_integration_oja.sh')], check=True)
subprocess.run([sys.executable, str(root / 'gate_xml.py'), str(root / 'latest-integration-oja-20261113.xml')], check=True)
subprocess.run(['bash', str(root / 'run_solver_benchmarks_latest.sh')], check=True)
subprocess.run([sys.executable, str(root / 'summarize_solver_benchmark_latest.py')], check=True)
subprocess.run(['bash', str(root / 'run_oja_benchmarks.sh')], check=True)
print('All queued correctness and measurement steps completed.', flush=True)
