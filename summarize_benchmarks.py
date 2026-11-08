"""Summarize every paired workload without selecting the fastest process."""

import json
import math
from pathlib import Path
import statistics

directory = Path(__file__).parent / 'a800' / 'strict-fp32'
runs = {name: json.loads((directory / f'{name}-benchmark.json').read_text()) for name in (
    'baseline-1', 'candidate-1', 'candidate-2', 'baseline-2',
)}
assert len({r['device_selector'] for r in runs.values()}) == 1
for name, report in runs.items():
    audit = json.loads((directory / f'{name}-occupancy.json').read_text())
    assert audit['command_exit_code'] == 0
    assert audit['other_process_observed'] is False and audit['audit_errors'] is False
    assert any(p['is_this_benchmark'] is True for sample in audit['samples'] for p in sample['processes'])
    assert report['torch_matmul_allow_tf32'] is False
    assert report['flags']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == '0'
    assert report['flags']['FLA_TRIL_PRECISION'] == 'ieee'
    assert report['flags']['TRITON_F32_DEFAULT'] == 'ieee'
    expected_revision = ('7608606ab33ee4fc570dd3999053a34851761636' if name.startswith('candidate')
                         else '98193113ecb7835143252a786e06933e371b67cc')
    assert report['revision'] == expected_revision
    assert len(report['rows']) == 8
    assert report['flags']['FLA_CI_ENV'] == '0'
    assert report['flags']['FLA_CACHE_RESULTS'] == '0'
    for row in report['rows']:
        assert row['rewy_selection_verified'] == name.startswith('candidate')
        assert len(row['samples_us']) == 10

rows = []
for index, reference in enumerate(runs['baseline-1']['rows']):
    row = {key: reference[key] for key in ('case', 'mode', 'B', 'T', 'H', 'K', 'V', 'dtype', 'lengths')}
    for name in ('baseline', 'candidate'):
        inputs = [runs[f'{name}-{i}']['rows'][index] for i in (1, 2)]
        assert all(all(x[k] == v for k, v in row.items() if k in reference) for x in inputs)
        samples = [x for record in inputs for x in record['samples_us']]
        assert all(math.isfinite(x) and x > 0 for x in samples)
        row[name + '_us'] = statistics.median(samples)
        row[name + '_run_medians_us'] = [x['median_us'] for x in inputs]
        row[name + '_sample_range_us'] = [min(samples), max(samples)]
        peaks = [x['additional_peak_bytes'] for x in inputs]
        assert peaks[0] == peaks[1], peaks
        row[name + '_additional_peak_bytes'] = peaks[0]
    row['speedup'] = row['baseline_us'] / row['candidate_us']
    row['memory_reduction_fraction'] = 1 - row['candidate_additional_peak_bytes'] / row['baseline_additional_peak_bytes']
    rows.append(row)

summary = {
    'baseline': runs['baseline-1']['revision'], 'candidate': runs['candidate-1']['revision'],
    'gpu': runs['baseline-1']['gpu'], 'device_selector': runs['baseline-1']['device_selector'],
    'torch_matmul_allow_tf32': False,
    'flags_by_variant': {n: runs[n + '-1']['flags'] for n in ('baseline', 'candidate')},
    'sampling': 'ABBA processes; median of all 20 per-invocation medians per mode and revision',
    'status': 'supplementary A800 measurements; datacenter SM90+ evidence remains missing',
    'geometric_mean_speedup': {mode: statistics.geometric_mean(x['speedup'] for x in rows if x['mode'] == mode) for mode in ('B', 'FB')},
    'regressions': [x['case'] + '/' + x['mode'] for x in rows if x['speedup'] < 1],
    'rows': rows,
}
(directory / 'benchmark-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['| Workload | Mode | Baseline (us) | ReWY (us) | Ratio | Additional peak MiB, baseline -> ReWY |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
for r in rows:
    lines.append(f"| {r['case']} | {r['mode']} | {r['baseline_us']:.2f} | {r['candidate_us']:.2f} | {r['speedup']:.3f}x | "
                 f"{r['baseline_additional_peak_bytes'] / 2**20:.2f} -> {r['candidate_additional_peak_bytes'] / 2**20:.2f} |")
(directory / 'benchmark-table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(json.dumps({k: summary[k] for k in ('geometric_mean_speedup', 'regressions')}))
