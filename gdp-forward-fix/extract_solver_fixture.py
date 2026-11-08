from pathlib import Path
import torch
root=Path(__file__).resolve().parent
r=torch.load(root/'operator-stage-tensors.pt',weights_only=True)
a=r['chunk_scaled_dot_kkt_fwd:0']['dense'][0]
matrices=torch.stack([a[0,12*64:13*64,2,:],a[1,:64,3,:]])
torch.save({'A':matrices,'entries':torch.tensor([[28,8],[15,0]])},root/'solver-rounding-fixture.pt')
print('Extracted two synthetic KKT blocks:',matrices.shape)
