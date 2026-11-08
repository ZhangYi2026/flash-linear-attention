from pathlib import Path
import json
rows=[]
for p in (Path.home()/'.triton/cache').glob('*/merge_16x16_to_64x64_inverse_kernel.json'):
 try:
  x=json.loads(p.read_text())
  if x.get('num_warps')==2:
   rows.append((p.stat().st_mtime,{k:v for k,v in x.items() if k in ['name','num_warps','num_stages','shared','global_scratch_size','global_scratch_align','maxnreg']}))
 except (ValueError,OSError):pass
for _,row in sorted(rows)[-4:]:print(json.dumps(row))
