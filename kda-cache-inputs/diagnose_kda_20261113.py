import sys,os,json,importlib
from pathlib import Path
repo=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/baseline');sys.path.insert(0,str(repo));os.chdir(repo)
import torch,pytest
import torch.nn.functional as F
import tests.cp.test_intracard as test
actual=test.chunk_kda
inputs={}
class Captured(Exception):pass
def capture(**kw):inputs.update(kw);raise Captured()
mp=pytest.MonkeyPatch();mp.setattr(test,'chunk_kda',capture)
try:test.test_chunk_kda_cache(mp)
except Captured:pass
finally:mp.undo()
root=Path(__file__).resolve().parent
report={'rows':[]}
def stats(x):
 if not isinstance(x,torch.Tensor):return None
 finite=torch.isfinite(x);flat=finite.flatten()
 return {'shape':list(x.shape),'dtype':str(x.dtype),'nan':torch.isnan(x).sum().item(),'inf':torch.isinf(x).sum().item(),'max_finite':x[finite].abs().max().item() if finite.any() else None,'first_nonfinite_flat':int((~flat).nonzero()[0]) if not flat.all() else None}
fwd=importlib.import_module('fla.ops.kda.chunk_fwd')
for name in ['kda_gate_chunk_cumsum','chunk_kda_fwd_intra','chunk_gated_delta_rule_fwd_h','chunk_gla_fwd_o_gk']:
 fn=getattr(fwd,name)
 def wrapped(*args,_fn=fn,_name=name,**kwargs):
  out=_fn(*args,**kwargs)
  report['rows'].append({'phase':phase,'stage':_name,'outputs':[stats(x) for x in out] if isinstance(out,tuple) else stats(out)})
  return out
 setattr(fwd,name,wrapped)
for phase in ['intracard0','intracard1']:
 os.environ['FLA_INTRACARD_CP']='0' if phase=='intracard0' else '1'
 from fla.ops.common.intracard_cp import _intracard_cache
 _intracard_cache.clear()
 kw=dict(inputs)
 if phase=='safe_gate':kw['safe_gate']=True
 with torch.inference_mode():out=actual(**kw)
 report[phase]=stats(out[0])
q,k,v,g,beta,A,bias=[inputs[n].detach().double().cpu() for n in ['q','k','v','g','beta','A_log','dt_bias']]
g=-A.exp()*F.softplus(g+bias.reshape(1,1,1,-1))
report['activated_gate_range']=[g.min().item(),g.max().item()]
s=torch.zeros(32,32,dtype=torch.float64);peak=0;rows=[]
for i in range(32768):
 s*=g[0,i,0].exp()[:,None]
 ki=k[0,i,0];vi=v[0,i,0]
 s+= beta[0,i,0]*ki[:,None]*(vi-ki@s)[None,:]
 o=q[0,i,0]@s/32**0.5
 peak=max(peak,s.abs().max().item())
 if i in [0,63,127,255,511,1023,2047,4095,5759,8191,16383,32767]:rows.append({'token':i,'state_max':s.abs().max().item(),'output_max':o.abs().max().item()})
report['cpu_fp64_full_sequence']=rows
(root/'kda-diagnosis-20261113.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
