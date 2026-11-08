from pathlib import Path
import json
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
cmdline = Path('/proc/677/cmdline')
while cmdline.exists() and b'run_shared_recovery.py' in cmdline.read_bytes():
    time.sleep(10)
subprocess.run([sys.executable, str(root / 'summarize_results.py')], check=True)
summary = json.loads((root / 'test-summary.json').read_text())
assert summary['solver-fix-shared']['completed_successfully'], 'Original shared recovery must complete before starting another GPU process'
subprocess.run([sys.executable, str(root / 'run_latest_controls.py')], check=True)
