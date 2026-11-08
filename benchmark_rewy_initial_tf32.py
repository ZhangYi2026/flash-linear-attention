"""Supplementary GDN B/FB measurements; run only after the full correctness gate passes."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--revision', required=True)
parser.add_argument('--rewy', choices=('0', '1'), required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ.update(
    FLA_REWY=args.rewy, FLA_DISABLE_BACKEND_DISPATCH='0', FLA_FLASH_QLA='0',
    FLA_TILELANG='0', FLA_CI_ENV='0', FLA_CACHE_RESULTS='0',
)
sys.path.insert(0, str(args.repo.resolve()))

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

import fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import assert_close

assert Path(fla.__file__).resolve().is_relative_to(args.repo.resolve())
report = {
    'revision': args.revision,
    'gpu': torch.cuda.get_device_name(0),
    'device_selector': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'capability': torch.cuda.get_device_capability(0),
    'torch': torch.__version__,
    'triton': triton.__version__,
    'cuda_runtime': torch.version.cuda,
    'flags': {k: os.environ[k] for k in (
        'FLA_REWY', 'FLA_DISABLE_BACKEND_DISPATCH', 'FLA_FLASH_QLA', 'FLA_TILELANG', 'FLA_CI_ENV', 'FLA_CACHE_RESULTS',
    )},
    'saving_policy': 'upstream default recompute; output_final_state=True; initial-state gradient included',
    'timing': 'CUDA graph; 10 samples of 100 ms per mode; median; Python dispatch excluded',
    'memory': 'eager allocated peak minus live inputs (B also excludes the forward graph); returned gradients included',
    'rows': [],
}
cases = [
    ('dense-fp16', 1024, 64, torch.float16, None),
    ('dense-bf16', 4096, 128, torch.bfloat16, None),
    ('packed-fp16', 1024, 64, torch.float16, [73, 951]),
    ('packed-bf16', 4149, 128, torch.bfloat16, [1, 64, 511, 1024, 2549]),
]
for name, T, D, dtype, lengths in cases:
    torch.manual_seed(42)
    B, H = 1, 4
    q = F.normalize(torch.randn(B, T, H, D, device='cuda'), dim=-1).to(dtype)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn(B, T, H, device='cuda')) / 10
    beta = torch.rand_like(g)
    h0 = torch.randn(B if lengths is None else len(lengths), H, D, D, device='cuda')
    inputs = tuple(x.requires_grad_(True) for x in (q, k, v, g, beta, h0))
    do, dht = torch.randn_like(v), torch.randn_like(h0)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True)
    if lengths is not None:
        cu_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
        kwargs.update(cu_seqlens=cu_cpu.cuda(), cu_seqlens_cpu=cu_cpu)

    def forward_backward():
        o, ht = chunk_gated_delta_rule(**kwargs)
        gradients = torch.autograd.grad((o, ht), inputs, (do, dht))
        return o, ht, *gradients

    os.environ['FLA_REWY'] = '0'
    reference = forward_backward()
    os.environ['FLA_REWY'] = args.rewy
    backend_calls = []
    if args.rewy == '1':
        from fla.ops.gated_delta_rule.backends.rewy import ReWYBackend

        implementation = ReWYBackend.chunk_gated_delta_rule_bwd_dqkvbg

        def spy(self, *values, **options):
            backend_calls.append(True)
            return implementation(self, *values, **options)

        ReWYBackend.chunk_gated_delta_rule_bwd_dqkvbg = spy
        try:
            actual = forward_backward()
        finally:
            ReWYBackend.chunk_gated_delta_rule_bwd_dqkvbg = implementation
        assert backend_calls, (name, 'ReWY was not selected')
    else:
        actual = forward_backward()
    for label, ref, out, tolerance in zip(
        ('o', 'ht', 'dq', 'dk', 'dv', 'dg', 'db', 'dh0'),
        reference,
        actual,
        (0.005, 0.005, 0.008, 0.008, 0.008, 0.02, 0.02, 0.008),
    ):
        assert torch.isfinite(out).all(), (name, label)
        assert_close(label, ref, out, tolerance)
    del reference, actual

    # Autograd follows its forward stream. A non-default forward stream allows
    # the backward to join the side-stream CUDA graph used by Triton's timer.
    forward_stream = torch.cuda.Stream()
    forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(forward_stream):
        o, ht = chunk_gated_delta_rule(**kwargs)
    torch.cuda.current_stream().wait_stream(forward_stream)

    def backward():
        return torch.autograd.grad((o, ht), inputs, (do, dht), retain_graph=True)

    for mode, fn in [('B', backward), ('FB', forward_backward)]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        samples = [float(do_bench_cudagraph(fn, rep=100, return_mode='median')) * 1000 for _ in range(10)]
        gc.collect()
        torch.cuda.empty_cache()
        live = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        outputs = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - live
        del outputs
        row = {
            'case': name, 'B': B, 'T': T, 'H': H, 'K': D, 'V': D,
            'dtype': str(dtype), 'lengths': lengths, 'mode': mode,
            'median_us': statistics.median(samples), 'samples_us': samples,
            'additional_peak_bytes': peak,
            'rewy_selection_verified': bool(backend_calls),
        }
        report['rows'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'samples_us'}), flush=True)
        if mode == 'B':
            del o, ht
    del kwargs, inputs, q, k, v, g, beta, h0, do, dht
    gc.collect()
    torch.cuda.empty_cache()

args.output.write_text(json.dumps(report, indent=2) + '\n')
