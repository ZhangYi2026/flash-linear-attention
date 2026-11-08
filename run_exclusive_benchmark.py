"""Monitor GPU process occupancy while a benchmark command runs; never stop other tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--gpu',required=True)
parser.add_argument('--audit',type=Path,required=True)
parser.add_argument('command',nargs=argparse.REMAINDER)
a=parser.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
assert command and command[0]=='docker'
process=subprocess.Popen(command)
started=time.monotonic()
samples=[]
while process.poll() is None:
    observed=[]
    try:
        result=subprocess.run(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        for line in result.stdout.splitlines():
            pid=int(line.strip())
            try:
                argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                ours=b'/workspace/fla_rewy_validation_20261108/benchmark_rewy.py' in argv
            except FileNotFoundError:
                ours=None  # Exited between the GPU query and /proc inspection.
            observed.append({'pid':pid,'is_this_benchmark':ours})
        samples.append({'elapsed_seconds':round(time.monotonic()-started,3),'processes':observed})
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        samples.append({'elapsed_seconds':round(time.monotonic()-started,3),'audit_error':type(error).__name__})
    time.sleep(2)
code=process.wait()
interference=any(p['is_this_benchmark'] is False for s in samples for p in s.get('processes',[]))
audit_errors=any('audit_error' in s for s in samples)
a.audit.write_text(json.dumps({'gpu':a.gpu,'command_exit_code':code,'other_process_observed':interference,'audit_errors':audit_errors,'samples':samples},indent=2)+'\n')
raise SystemExit(code if code else (79 if interference or audit_errors else 0))
