from pathlib import Path
import hashlib,json
p=Path('/home/zhang_e/.venvs/kda-residual-wy-20261023/lib/python3.12/site-packages/triton/runtime/autotuner.py')
s=p.read_text()
lines=s.splitlines()
for i,line in enumerate(lines):
    if 'def check_disk_cache' in line:
        print('\n'.join(f'{j+1}: {lines[j]}' for j in range(max(i-5,0),min(i+85,len(lines)))))
print('Source sha256:',hashlib.sha256(p.read_bytes()).hexdigest())
