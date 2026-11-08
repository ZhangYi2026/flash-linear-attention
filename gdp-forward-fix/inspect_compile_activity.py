from pathlib import Path
import json,time
root=Path.home()/'.triton/cache'
rows=[]
for p in root.glob('*/*.json'):
 try:
  t=p.stat().st_mtime
  if time.time()-t<600:
   x=json.loads(p.read_text());rows.append((t,p.name,x.get('name'),x.get('num_warps')))
 except (ValueError,OSError):pass
for t,name,kernel,warps in sorted(rows)[-12:]:print(round(time.time()-t,1),name,kernel,warps)
