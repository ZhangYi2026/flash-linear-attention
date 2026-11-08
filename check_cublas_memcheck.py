"""Standalone control for the cuBLAS initialization errors reported by memcheck."""

import torch

torch.manual_seed(42)
a = torch.randn(2, 32, 32, device='cuda')
b = torch.randn_like(a)
c = torch.bmm(a, b)
torch.cuda.synchronize()
assert torch.isfinite(c).all()
print('Standalone torch.bmm completed with finite output.')
