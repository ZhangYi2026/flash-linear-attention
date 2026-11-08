from pathlib import Path
import os,signal
p=Path('/proc/653')
assert str((p/'cwd').resolve())=='/home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix'
status=(p/'status').read_text();print('\n'.join(l for l in status.splitlines() if l.startswith(('Name:','SigIgn:','SigCgt:'))))
os.kill(653,signal.SIGTERM)
print('Terminated only the uncached pytest process; full same-target rerun follows')
