"""Standalone SM120 rounding reproducer using two synthetic FP32 KKT blocks."""
import argparse,importlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo));os.chdir(a.repo)
import torch
m=importlib.import_module('fla.ops.utils.solve_tril')
root=Path(__file__).resolve().parent
fixture=torch.load(root/'solver-rounding-fixture.pt',weights_only=True)
A=fixture['A'].unsqueeze(2).cuda();packed=A.reshape(1,128,1,64)
cu=torch.tensor([0,64,128],device='cuda',dtype=torch.int32)
from fla.ops.utils import prepare_chunk_indices
indices=prepare_chunk_indices(cu,64)
raw=m.merge_16x16_to_64x64_inverse_kernel
while not hasattr(raw,'src'):raw=raw.fn
values={}
for warps in [2,4]:
    outputs=[]
    for x,seq in [(A,None),(packed,cu)]:
        y=torch.zeros_like(x,dtype=torch.bfloat16)
        raw[(1 if seq is None else 2,x.shape[0])](A=x,Ai=y,cu_seqlens=seq,chunk_indices=None if seq is None else indices,T=x.shape[1],H=1,BT=64,USE_TMA=False,IS_VARLEN=seq is not None,DOT_PRECISION='ieee',num_warps=warps,num_stages=2)
        outputs.append(y.cpu().reshape(2,64,64))
    assert torch.equal(*outputs), 'Same warp count must remove the layout mismatch in this reproducer'
    values[warps]=outputs[0]
bad=values[2]!=values[4]
reference=torch.linalg.solve_triangular(torch.eye(64,dtype=torch.float64)+fixture['A'].double(),torch.eye(64,dtype=torch.float64).expand(2,64,64),upper=False,unitriangular=True)
rows=[]
for n,(i,j) in enumerate(fixture['entries'].tolist()):
    rows.append({'matrix':n,'row':i,'column':j,'w2':values[2][n,i,j].float().item(),'w4':values[4][n,i,j].float().item(),'fp64':reference[n,i,j].item(),'fp64_rounded_bf16':reference[n,i,j].bfloat16().float().item()})
result={'device':torch.cuda.get_device_name(),'mismatches_between_warps':bad.sum().item(),'same_warp_dense_packed_equal':True,'entries':rows}
(root/'minimal-solver-reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert bad.any(), 'The original warp-dependent rounding no longer reproduces on this stack'
