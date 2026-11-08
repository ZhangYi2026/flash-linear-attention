"""Summarize all stored ABBA samples without selecting favorable runs."""
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent


def summarize(prefix, tags, keys, baseline, candidate, boundary, out):
    runs = {t: json.loads((root / (prefix + t + '.json')).read_text()) for t in tags}
    rows = []
    for i in range(len(runs[tags[0]]['rows'])):
        values = {tag: r['rows'][i] for tag, r in runs.items()}
        initial = values[tags[0]]
        assert all(all(v[k] == initial[k] for k in keys) for v in values.values())
        row = {k: initial[k] for k in keys}
        for label, names in [('baseline', [tags[0], tags[3]]), ('candidate', [tags[1], tags[2]])]:
            row[label + '_median_us'] = statistics.median(sum([values[t]['samples_us'] for t in names], []))
            medians = [values[t]['median_us'] for t in names]
            row[label + '_process_spread_pct'] = (max(medians) - min(medians)) / statistics.mean(medians) * 100
        row['baseline_over_candidate'] = row['baseline_median_us'] / row['candidate_median_us']
        row['latency_change_pct'] = (1 / row['baseline_over_candidate'] - 1) * 100
        rows.append(row)
    result = {'baseline': baseline, 'candidate': candidate, 'boundary': boundary, 'device': runs[tags[0]]['device'],
              'process_order': tags, 'gm_baseline_over_candidate': math.exp(statistics.mean(math.log(r['baseline_over_candidate']) for r in rows)),
              'worst_latency_change_pct': max(r['latency_change_pct'] for r in rows),
              'max_process_spread_pct': max(r[k] for r in rows for k in ['baseline_process_spread_pct', 'candidate_process_spread_pct']),
              'rows': rows}
    (root / out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


summarize('oja-benchmark-', ['stream-A1', 'stream-B1', 'stream-B2', 'stream-A2'], ['shape', 'qk_dtype', 'v_dtype', 'layout', 'bounds', 'input_sha256'],
          '0a649e4c7201f0edcba734832773fe8ab371367a', 'b6172a05f2405a73c96ec190ee41f4083faa31a8',
          'public autograd backward including default recomputation, state and normalization gradients; CUDA Graph, excludes forward and Python dispatch',
          'oja-benchmark-summary.json')
summarize('benchmark-', ['TMA-A1', 'TMA-B1', 'TMA-B2', 'TMA-A2'], ['shape', 'layout', 'output_dtype', 'input_sha256'],
          '0a649e4c7201f0edcba734832773fe8ab371367a', '28e74f8ba0c8ebb0a0edd11c42dc7ef628b27cae',
          'public solve_tril forward, TMA disabled, includes output zero-initialization; CUDA Graph, no Python dispatch or backward',
          'tma-default-benchmark-summary.json')
