"""Compare solver launch configurations using the captured identical FP32 matrices."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
a=parser.parse_args();sys.path.insert(0,str(a.repo.resolve()));os.chdir(a.repo)
import torch
m=importlib.import_module('fla.ops.utils.solve_tril')
from fla.ops.utils import prepare_chunk_indices
root=Path(__file__).resolve().parent
r=torch.load(root/'operator-stage-tensors.pt',weights_only=True)
A=r['chunk_scaled_dot_kkt_fwd:0']['dense'][0].cuda();B,T,H,BT=A.shape
packed=A.reshape(1,B*T,H,BT)
cu=torch.arange(0,B*T+1,T,device='cuda',dtype=torch.int32)
indices=prepare_chunk_indices(cu,BT)
wrapper=m.merge_16x16_to_64x64_inverse_kernel
raw=wrapper
while not hasattr(raw,'src'):raw=raw.fn
rows=[]
for dtype in (torch.float32,torch.bfloat16):
    public=[]
    for x,seq in ((A,None),(packed,cu)):
        y=m.solve_tril(x,cu_seqlens=seq,output_dtype=dtype)
        public.append(y.detach().cpu())
        tuner=wrapper
        while not hasattr(tuner,'best_config'):tuner=tuner.fn
        print('[DEBUG-gdp-solver] public config',dtype,seq is not None,str(tuner.best_config),flush=True)
    pairs=[('autotune',public)]
    for warps, stages in ((2,2),(2,3),(2,4),(2,5),(4,2),(4,3),(4,4),(4,5)):
        values=[]
        for x,seq in ((A,None),(packed,cu)):
            y=torch.zeros_like(x,dtype=dtype)
            nt=T//BT if seq is None else B*T//BT
            raw[(nt,x.shape[0]*H)](A=x,Ai=y,cu_seqlens=seq,chunk_indices=None if seq is None else indices,T=x.shape[1],H=H,BT=BT,USE_TMA=False,IS_VARLEN=seq is not None,DOT_PRECISION='ieee',num_warps=warps,num_stages=stages)
            values.append(y.detach().cpu())
        pairs.append((f'w{warps}s{stages}',values))
    for config,values in pairs:
        x,y=[v.flatten() for v in values];d=(x.float()-y.float()).abs();bad=x!=y
        row={'dtype':str(dtype),'config':config,'mismatch_count':bad.sum().item(),'max_abs':d.max().item(),'first_mismatches':bad.nonzero().flatten()[:8].tolist()}
        rows.append(row);print('[DEBUG-gdp-solver]',json.dumps(row),flush=True)
        torch.save(values,root/f'solver-{str(dtype).split(".")[-1]}-{config}.pt')
(root/'solver-config-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
