from pathlib import Path
import json,torch
root=Path(__file__).resolve().parent
s=torch.load(root/'operator-stage-tensors.pt',weights_only=True)
A=s['chunk_scaled_dot_kkt_fwd:0']['dense'][0]
x,y=torch.load(root/'solver-bfloat16-autotune.pt',weights_only=True)
x=x.flatten();y=y.flatten(); B,T,H,BT=A.shape
rows=[]
for flat in (x!=y).nonzero().flatten().tolist():
 b=flat//(T*H*BT); rem=flat%(T*H*BT);t=rem//(H*BT);h=rem//BT%H;c=rem%BT
 a=A[b,t//BT*BT:(t//BT+1)*BT,h,:].double();eye=torch.eye(BT,dtype=torch.double)
 ref=torch.linalg.solve_triangular(eye+a,eye,upper=False,unitriangular=True)[t%BT,c]
 entries={}
 for name in ['autotune','w2s2','w4s2']:
  z=torch.load(root/f'solver-float32-{name}.pt',weights_only=True)
  entries[name]=[v.flatten()[flat].item() for v in z]
 row={'flat_index':flat,'matrix_index':[b,t//BT,h],'entry':[t%BT,c],'fp64_reference':ref.item(),'fp64_rounded_bf16':ref.bfloat16().float().item(),'bf16_dense':x[flat].float().item(),'bf16_packed':y[flat].float().item(),'fp32':entries}
 rows.append(row)
print(json.dumps(rows,indent=2));(root/'fp64-rounding-oracle.json').write_text(json.dumps(rows,indent=2)+'\n')
