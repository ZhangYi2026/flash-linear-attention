"""Replay the captured equal GDP inputs and locate the first differing substage."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--inputs',default='layer1-operator-inputs.pt')
args=parser.parse_args();sys.path.insert(0,str(args.repo.resolve()));os.chdir(args.repo)
import torch
chunk=importlib.import_module('fla.ops.gated_delta_product.chunk')
root=Path(__file__).resolve().parent
snaps=torch.load(root/args.inputs,weights_only=True)
records={}; originals={}; phase=None

def capture(name):
    original=getattr(chunk,name);originals[name]=original
    def wrapper(*values,**kwargs):
        result=original(*values,**kwargs)
        items=result if isinstance(result,tuple) else (result,)
        for i,v in enumerate(items):
            if isinstance(v,torch.Tensor):records.setdefault(name+f':{i}',{}).setdefault(phase,[]).append(v.detach().cpu())
        return result
    setattr(chunk,name,wrapper)
for n in ('l2norm_fwd','chunk_scaled_dot_kkt_fwd','solve_tril','dn_recompute_w_u_fwd','chunk_gated_delta_product_fwd_h','chunk_gated_delta_product_fwd_o'):capture(n)
outs=[]
for phase in ('dense','packed'):
    kwargs={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in snaps[phase]['kwargs'].items()}
    outs.append(chunk.chunk_gated_delta_product(**kwargs)[0].detach().cpu())
rows=[]
for name,modes in records.items():
    for index,(a,b) in enumerate(zip(modes['dense'],modes['packed'])):
        shape=list(a.shape);a=a.flatten().float();b=b.flatten().float();assert a.shape==b.shape,(name,a.shape,b.shape)
        d=a-b;different=a!=b
        row={'name':name,'call':index,'dense_shape':shape,'equal':torch.equal(a,b),'max_abs':d.abs().max().item(),'relative_rms':(d.square().mean().sqrt()/(a.square().mean().sqrt()+1e-8)).item(),'mismatch_count':different.sum().item()}
        if different.any():row['first_flat_index']=different.nonzero()[0].item()
        rows.append(row);print('[DEBUG-gdp-stage]',json.dumps(row),flush=True)
(root/'operator-stage-trace.json').write_text(json.dumps(rows,indent=2)+'\n')
torch.save(records,root/'operator-stage-tensors.pt')
a,b=outs;assert torch.equal(a.flatten(),b.flatten()),'Captured equal inputs produce different dense/packed outputs'
