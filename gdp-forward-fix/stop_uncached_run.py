from pathlib import Path
import os,signal
script='/mnt/d/mycode/hw/gdn2/output/fla_gdp_forward_fix_20261108/finish_validation.sh'
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
 except (OSError,UnicodeDecodeError):continue
 if args==['bash',script]:matches.append(int(p.name))
assert len(matches)==1,matches
os.kill(matches[0],signal.SIGTERM);print('Stopped only continuation coordinator',matches[0])
p=Path('/proc/653')
args=(p/'cmdline').read_bytes().decode().split('\0')
assert str((p/'cwd').resolve())=='/home/zhang_e/.cache/fla-rewy-pr-20261108/solver-fix'
assert args[1:3]==['-m','pytest'],args[:3]
os.kill(653,signal.SIGINT);print('Requested graceful pytest interruption for uncached partial run')
