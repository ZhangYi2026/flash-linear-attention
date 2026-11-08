"""Check the ReWY input/state/output addresses past 2**31 elements on a large GPU.

This is an addressing invariant check, not a substitute for the numerical oracle tests.
Only the final chunk has finite inputs. Its results must equal a small, identical chunk.
Both state layouts run through the unchanged production kernel and full dense grid.
"""

import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))

import torch
import triton

from fla.ops.gated_delta_rule.backends.rewy import chunk_gated_delta_rule_bwd_kernel_rewy

free, total = torch.cuda.mem_get_info()
if free < 60 * 1024**3:
    raise SystemExit('Requires at least 60 GiB free GPU memory; no tensors were allocated.')

H, K, V, BT, tail = 2, 64, 64, 64, 63
start = 2**31 // (H * K)
T = start + tail
NT = triton.cdiv(T, BT)
torch.manual_seed(42)
small = {}
large = {}
for name in ('q', 'k', 'v', 'A', 'dr', 'r', 'do', 'dq', 'dk', 'dv', 'g', 'beta', 'db', 'dg', 'h', 'dh'):
    if name in ('h', 'dh'):
        small_shape, large_shape = (1, 1, H, K, V), (1, NT, H, K, V)
    elif name in ('g', 'beta', 'db', 'dg'):
        small_shape, large_shape = (1, tail, H), (1, T, H)
    else:
        width = BT if name == 'A' else K
        small_shape, large_shape = (1, tail, H, width), (1, T, H, width)
    dtype = torch.float32 if name in ('g', 'beta', 'db', 'dg') else torch.float16
    small[name] = torch.randn(small_shape, device='cuda', dtype=dtype) * 0.1
    large[name] = torch.full(large_shape, float('nan'), device='cuda', dtype=dtype)

small['g'] = -torch.rand_like(small['g']).cumsum(dim=1) / 10
small['beta'] = torch.rand_like(small['beta'])
small['A'] = torch.eye(BT, device='cuda', dtype=torch.float16)[:tail][None, :, None, :].expand(1, tail, H, BT).contiguous()
outputs = ('dq', 'dk', 'dv', 'db', 'dg')
for name in small:
    if name not in (*outputs, 'h', 'dh'):
        large[name][:, start:].copy_(small[name])
original_h, original_dh = small['h'], small['dh']

def run(tensors, length, state_v_first):
    chunk_gated_delta_rule_bwd_kernel_rewy[(triton.cdiv(length, BT), H)](
        **tensors,
        cu_seqlens=None,
        chunk_indices=None,
        T=length,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=32,
        BV=64,
        scale=K**-0.5,
        IS_VARLEN=False,
        STATE_V_FIRST=state_v_first,
        num_warps=4,
        num_stages=1,
    )

results = []
for state_v_first in (False, True):
    small['h'] = original_h.transpose(-1, -2).contiguous() if state_v_first else original_h
    small['dh'] = original_dh.transpose(-1, -2).contiguous() if state_v_first else original_dh
    large['h'][:, -1:].copy_(small['h'])
    large['dh'][:, -1:].copy_(small['dh'])
    run(small, tail, state_v_first)
    run(large, T, state_v_first)
    torch.cuda.synchronize()
    for name in outputs:
        actual = large[name][:, start:]
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, small[name], rtol=0, atol=0)
    results.append({'state_v_first': state_v_first, 'passed': True})
    print(json.dumps(results[-1]), flush=True)

args.output.write_text(json.dumps({
    'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'triton': triton.__version__,
    'B': 1, 'T': T, 'H': H, 'K': K, 'V': V, 'chunk_size': BT,
    'last_chunk_input_element_offset': start * H * K,
    'last_chunk_state_element_offset': (NT - 1) * H * K * V,
    'allocated_bytes': torch.cuda.memory_allocated(), 'results': results,
}, indent=2) + '\n')
