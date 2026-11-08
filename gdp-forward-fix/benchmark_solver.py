"""Same-device public solve_tril latency; supplementary SM120 cost measurements."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--tma',action='store_true')
a=p.parse_args();sys.path.insert(0,str(a.repo));os.chdir(a.repo)
import torch
import torch.nn.functional as F
import triton
import triton.testing
m=importlib.import_module('fla.ops.utils.solve_tril')
from fla.utils import assert_close
root=Path(__file__).resolve().parent
assert not (root/f'benchmark-{a.tag}.json').exists(), 'Preserve existing result; use a new tag'
assert not torch.backends.cuda.matmul.allow_tf32
assert os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']=='0'
assert os.environ['TRITON_F32_DEFAULT']=='ieee'
assert m.IS_TMA_SUPPORTED == a.tma
assert m.FLA_TRIL_PRECISION=='ieee'
# CPU-generated Gram inputs and a float64 triangular reference, independent of Triton.
gen=torch.Generator().manual_seed(20261108)
rows=[]
for B,T,H in [(2,1024,4),(1,4096,8)]:
    k=F.normalize(torch.randn(B,T//64,H,64,128,generator=gen),dim=-1)
    mat=(k@k.transpose(-1,-2)).tril(-1)*0.5
    eye=torch.eye(64,dtype=torch.float64)
    ref=torch.linalg.solve_triangular(eye+mat.double(),eye.expand_as(mat),upper=False,unitriangular=True)
    cpu=mat.permute(0,1,3,2,4).reshape(B,T,H,64).contiguous()
    reference=ref.permute(0,1,3,2,4).reshape(B,T,H,64).float().cuda()
    original=cpu.cuda();input_hash=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
    for layout in ['dense','packed']:
        x=original if layout=='dense' else original.reshape(1,B*T,H,64)
        cu=None if layout=='dense' else torch.arange(0,B*T+1,T,dtype=torch.int32,device='cuda')
        for dtype in [torch.float32,torch.bfloat16]:
            out=m.solve_tril(x,cu_seqlens=cu,output_dtype=dtype)
            # FP32 preserves the repository solver reference threshold. BF16 validation
            # uses the original model/ops gates recorded separately, not a new tolerance.
            if dtype==torch.float32:assert_close('solver benchmark reference',reference.reshape_as(out),out,0.0001)
            assert torch.isfinite(out).all()
            tuner=m.merge_16x16_to_64x64_inverse_kernel
            while not hasattr(tuner,'best_config'):tuner=tuner.fn
            samples=[triton.testing.do_bench_cudagraph(lambda:m.solve_tril(x,cu_seqlens=cu,output_dtype=dtype),rep=100)*1000 for _ in range(5)]
            row={'shape':[B,T,H,64],'layout':layout,'output_dtype':str(dtype),'input_sha256':input_hash,'config':str(tuner.best_config),'samples_us':samples,'median_us':statistics.median(samples)}
            rows.append(row);print(json.dumps(row),flush=True)
result={'tag':a.tag,'repo':str(a.repo),'solver_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'triton':triton.__version__,'flags':{k:os.environ.get(k) for k in ['FLA_CI_ENV','FLA_REWY','FLA_FLASH_QLA','FLA_TILELANG','FLA_CACHE_RESULTS','TRITON_F32_DEFAULT','FLA_TRIL_PRECISION','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']},'tma':m.IS_TMA_SUPPORTED,'rows':rows}
(root/f'benchmark-{a.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
