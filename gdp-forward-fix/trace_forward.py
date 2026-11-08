"""Temporary boundary tracing on the unchanged GDP forward regression."""
import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--length',type=int,default=1024)
parser.add_argument('--layer',type=int,default=0)
args=parser.parse_args()
sys.path.insert(0,str(args.repo.resolve()));os.chdir(args.repo)
logging.basicConfig(level=logging.INFO)
import torch
from fla.models import GatedDeltaProductConfig
import tests.models.utils as u
layer=importlib.import_module('fla.layers.gated_deltaproduct')
root=Path(__file__).resolve().parent
records={}; snaps={}; originals=[]

def tensor(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,(tuple,list)):
        return next((tensor(v) for v in value if isinstance(v,torch.Tensor)),None)
    return None

def hook(name):
    def capture(module,inputs,output):
        value=tensor(output)
        if value is not None:records.setdefault(name,[]).append(value)
    return capture

create=u.create_model_and_config

def instrument(*values,**kwargs):
    model,config=create(*values,**kwargs)
    for name,module in model.named_modules():
        if f'layers.{args.layer}.' in name or name in ('model.layers.0','model.layers.1','model.layers.2','model.norm'):
            originals.append(module.register_forward_hook(hook(name)))
    return model,config
u.create_model_and_config=instrument
op=layer.chunk_gated_delta_product
count=0

def tap(**kwargs):
    global count
    n=count;count+=1
    output=op(**kwargs)
    if n%3==args.layer:
        key='dense' if n<3 else 'packed'
        snaps[key]={'kwargs':{k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items()},'output':output[0].detach().cpu()}
        for k,v in kwargs.items():
            if isinstance(v,torch.Tensor) and k!='cu_seqlens':records.setdefault('GDP input '+k,[]).append(v.detach().cpu())
        records.setdefault('GDP output',[]).append(output[0].detach().cpu())
    return output
layer.chunk_gated_delta_product=tap
failed=False
try:
    u.check_forward(3,2,args.length,4,64,GatedDeltaProductConfig,use_l2warp=False,attnres_block_size=None,dtype=torch.bfloat16,varlen=True)
except AssertionError as error:
    failed=True; print('[DEBUG-gdp-boundary] Original assertion:',error,flush=True)
finally:
    for h in originals:h.remove()
rows=[]
for name,vs in records.items():
    if len(vs)!=2:continue
    a,b=(v.flatten().float() for v in vs)
    assert a.shape==b.shape,(name,a.shape,b.shape)
    d=a-b
    row={'name':name,'equal':torch.equal(a,b),'max_abs':d.abs().max().item(),'relative_rms':(d.square().mean().sqrt()/(a.square().mean().sqrt()+1e-8)).item(),'mismatch_count':(a!=b).sum().item(),'elements':a.numel()}
    rows.append(row);print('[DEBUG-gdp-boundary]',json.dumps(row),flush=True)
(root/f'boundary-trace-layer{args.layer}.json').write_text(json.dumps({'original_assertion_failed':failed,'rows':rows},indent=2)+'\n')
torch.save(snaps,root/f'layer{args.layer}-operator-inputs.pt')
