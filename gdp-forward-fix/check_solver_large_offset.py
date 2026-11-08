"""Public dense/packed solves with real offsets above 2**31 BF16 elements."""
import argparse,importlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--tag',default='solver-large-offset');a=p.parse_args()
sys.path.insert(0,str(a.repo));os.chdir(a.repo)
import torch
m=importlib.import_module('fla.ops.utils.solve_tril')
root=Path(__file__).resolve().parent
assert not (root/(a.tag+'.json')).exists(), 'Preserve existing result; use a new tag'
assert torch.cuda.get_device_capability()==(12,0)
B,T,H,BT=129,4096,64,64
matrix=(torch.randn(BT,BT,generator=torch.Generator().manual_seed(42))*0.01).to(torch.bfloat16).tril(-1).cuda()
# Warm the same autotune keys with small tensors before allocating 8 GiB.
small=torch.zeros((1,BT,H,BT),device='cuda',dtype=torch.bfloat16)
small[0,:,0].copy_(matrix)
reference=m.solve_tril(small,output_dtype=torch.bfloat16)[0,:,0].clone()
small_cu=torch.tensor([0,BT],device='cuda',dtype=torch.int64)
m.solve_tril(small,cu_seqlens=small_cu,output_dtype=torch.bfloat16)
del small
A=torch.zeros((B,T,H,BT),device='cuda',dtype=torch.bfloat16)
# Both positions lie beyond the signed 32-bit element-address boundary.
A[-1,:BT,1].copy_(matrix)
A[-1,-BT:,-1].copy_(matrix)
eye=torch.eye(BT,device='cuda',dtype=torch.bfloat16)
rows=[]
for layout in ['dense','packed']:
    if layout=='dense':x=A;cu=None
    else:
        x=A.reshape(1,B*T,H,BT)
        cu=torch.arange(0,B*T+1,T,device='cuda',dtype=torch.int64)
    result=m.solve_tril(x,cu_seqlens=cu,output_dtype=torch.bfloat16).reshape(B,T,H,BT)
    assert torch.equal(result[-1,:BT,1],reference)
    assert torch.equal(result[-1,-BT:,-1],reference)
    for batch,chunk,head in [(0,0,0),(B//2,16,H//2),(B-1,0,0)]:
        assert torch.equal(result[batch,chunk*BT:(chunk+1)*BT,head],eye)
    rows.append({'layout':layout,'both_late_blocks_equal_to_small_solve':True,'three_identity_controls_equal':True})
    del result,x,cu
    torch.cuda.empty_cache()
report={'device':torch.cuda.get_device_name(),'shape':[B,T,H,BT],'dtype':'bfloat16','first_checked_element_offset':(B-1)*T*H*BT+BT,'last_checked_element_offset':A.numel()-BT,'input_output_bytes':A.numel()*A.element_size()*2,'rows':rows,'scope':'complete public solves; compare two late nonzero blocks and three identity controls, original full huge-batch test remains unrun'}
report['tma_enabled']=m.IS_TMA_SUPPORTED
report['repo']=str(a.repo)
(root/(a.tag+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
