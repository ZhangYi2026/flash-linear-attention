"""Supplementary SM120 cost of public Oja2 backward on previously runnable inputs."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--tag', required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
os.chdir(args.repo)
import torch
import triton
import triton.testing
from tests.ops.test_gated_oja_rule2 import _rand_inputs, _ref
from fla.ops.gated_oja_rule2 import chunk_gated_oja_rule2
from fla.utils import assert_close

root = Path(__file__).resolve().parent
module = importlib.import_module('fla.ops.gated_oja_rule.chunk_h')
assert torch.cuda.get_device_capability() == (12, 0)
assert not torch.backends.cuda.matmul.allow_tf32
assert os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == '0'
rows = []
for dtype in [torch.float16, torch.bfloat16]:
    for layout in ['dense', 'packed']:
        bounds = [0, 1, 65, 256] if layout == 'packed' else [0, 256]
        count = len(bounds) - 1
        inputs = _rand_inputs(1, 256, 2, 64, 64, dtype, N=count)
        digest = hashlib.sha256(b''.join(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() for x in inputs)).hexdigest()
        leaves = [x.detach().clone().requires_grad_(True) for x in inputs]
        cu = torch.tensor(bounds, dtype=torch.int64, device='cuda') if layout == 'packed' else None
        q, k, v, gv, b, c, h0 = leaves
        out, final = chunk_gated_oja_rule2(q=q, k=k, v=v, gv=gv, b=b, c=c, initial_state=h0, output_final_state=True, use_q_l2norm=True, use_k_l2norm=True, cu_seqlens=cu)
        torch.manual_seed(43)
        do = torch.randn(out.shape, device='cuda', dtype=torch.float32)
        dht = torch.randn_like(final)

        def backward():
            return torch.autograd.grad((out, final), leaves, (do, dht), retain_graph=True)

        gradients = backward()
        reference_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        ref_outs, ref_states = [], []
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            y, h = _ref(*(x[:, start:end] for x in reference_inputs[:6]), reference_inputs[6][i:i + 1])
            ref_outs.append(y)
            ref_states.append(h)
        ref_out, ref_state = torch.cat(ref_outs, 1), torch.cat(ref_states, 0)
        ref_gradients = torch.autograd.grad((ref_out, ref_state), reference_inputs, (do, dht))
        assert_close('o', ref_out, out, 0.005)
        assert_close('ht', ref_state, final, 0.005)
        for name, ratio, ref, actual in zip(['dq', 'dk', 'dv', 'dgv', 'db', 'dc', 'dh0'], [.01, .01, .01, .02, .02, .02, .01], ref_gradients, gradients):
            assert torch.isfinite(actual).all()
            assert_close(name, ref, actual, ratio)
        tuner = module.chunk_oja_bwd_kernel_dhu_blockdim64
        while not hasattr(tuner, 'best_config'):
            tuner = tuner.fn
        config = str(tuner.best_config)
        samples = [triton.testing.do_bench_cudagraph(backward, rep=100) * 1000 for _ in range(5)]
        row = {'shape': [1, 256, 2, 64, 64], 'qk_dtype': str(dtype), 'v_dtype': str(v.dtype), 'layout': layout, 'bounds': bounds, 'input_sha256': digest, 'bwd_h_config': config, 'samples_us': samples, 'median_us': statistics.median(samples), 'original_reference_thresholds_passed': True}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del out, final, leaves, inputs, reference_inputs, ref_outs, ref_states, gradients, ref_gradients, ref_out, ref_state
        torch.cuda.empty_cache()
result = {'tag': args.tag, 'repo': str(args.repo), 'source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), 'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__, 'boundary': 'public autograd backward including default recomputation, state and normalization gradients; CUDA Graph, excludes forward and Python dispatch', 'rows': rows}
(root / ('oja-benchmark-' + args.tag + '.json')).write_text(json.dumps(result, indent=2) + '\n')
