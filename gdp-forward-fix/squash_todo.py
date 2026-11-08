import sys
from pathlib import Path
p=Path(sys.argv[1])
lines=p.read_text().splitlines()
seen=False
for i,line in enumerate(lines):
 if line.startswith('pick '):
  if seen:lines[i]='fixup '+line[5:]
  seen=True
p.write_text('\n'.join(lines)+'\n')
