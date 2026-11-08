from pathlib import Path
import json,math,statistics
root=Path(__file__).resolve().parent
runs={name:json.loads((root/f'benchmark-{name}.json').read_text()) for name in ['A1','B1','B2','A2']}
rows=[]
for n in range(len(runs['A1']['rows'])):
    values={tag:r['rows'][n] for tag,r in runs.items()}
    keys=['shape','layout','output_dtype','input_sha256']
    assert all(all(v[k]==values['A1'][k] for k in keys) for v in values.values())
    row={k:values['A1'][k] for k in keys}
    for label,tags in [('baseline',['A1','A2']),('candidate',['B1','B2'])]:
        all_samples=sum([values[tag]['samples_us'] for tag in tags],[])
        row[label+'_median_us']=statistics.median(all_samples)
        medians=[values[tag]['median_us'] for tag in tags]
        row[label+'_process_spread_pct']=(max(medians)-min(medians))/statistics.mean(medians)*100
    row['baseline_over_candidate']=row['baseline_median_us']/row['candidate_median_us']
    row['latency_change_pct']=(1/row['baseline_over_candidate']-1)*100
    rows.append(row)
summary={'baseline':'98193113ecb7835143252a786e06933e371b67cc','candidate':'034c3d41a18f1c6a941918c39396ef004ec757ca','device':runs['A1']['device'],'boundary':'public solve_tril forward including zero-initialized output, CUDA Graph; no Python dispatch or backward','autotune_policy':'fresh per-process tuning with FLA_CACHE_RESULTS=0; compiled-kernel and in-process caches retained; tuning excluded from timing','process_order':['A1','B1','B2','A2'],'samples_per_process_and_case':5,'gm_baseline_over_candidate':math.exp(statistics.mean(math.log(r['baseline_over_candidate']) for r in rows)),'worst_latency_change_pct':max(r['latency_change_pct'] for r in rows),'max_process_spread_pct':max(r[k] for r in rows for k in ['baseline_process_spread_pct','candidate_process_spread_pct']),'rows':rows}
(root/'benchmark-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| B,T,H | Layout | Output | Baseline us | Fix us | Latency change |','| --- | --- | --- | ---: | ---: | ---: |']
for r in rows:
    lines.append('| '+','.join(map(str,r['shape'][:3]))+' | '+r['layout']+' | '+r['output_dtype'].split('.')[-1]+' | '+f"{r['baseline_median_us']:.3f} | {r['candidate_median_us']:.3f} | {r['latency_change_pct']:+.2f}% |")
(root/'BENCHMARK_TABLE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
