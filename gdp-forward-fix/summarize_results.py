from pathlib import Path
import json
import xml.etree.ElementTree as ET
root=Path(__file__).resolve().parent
reports={}
def read_report(name):
    p=root/(name+'.xml')
    t=ET.parse(p).getroot()
    cases=list(t.iter('testcase'))
    failed=[c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]
    skipped=[{'node':c.attrib,'reason':c.find('skipped').get('message')} for c in cases if c.find('skipped') is not None]
    return {'total':len(cases),'passed':len(cases)-len(failed)-len(skipped),'failed':failed,'skipped':skipped}
for name in ['submission-tma-default','submission-tma-enabled','submission-combined-tma','submission-integration','baseline-tma-20261113','fixed-warps-models','solver-fix-rewy','solver-fix-rewy-kda-20261113','latest-integration-oja-20261113','kda-fixture-fix-20261113','latest-baseline-gdp','latest-baseline-kda','latest-fixed-kda','oja-fix-test_oja','oja-fix-test_gated_oja_rule2','solver-fix-tma','baseline-oja-fresh','solver-fix-oja-fresh','baseline-oja2-resource-20261113']:
    if (root/(name+'.xml')).exists():reports[name]=read_report(name)
if (root/'solver-fix-shared.xml').exists():
    reports['solver-fix-shared-failed-batch']=read_report('solver-fix-shared')
    reports['solver-fix-shared-failed-batch']['completed_successfully']=False
if (root/'shared-recovery-runs.json').exists():
    runs=json.loads((root/'shared-recovery-runs.json').read_text())
    original=json.loads((root/'solver-fix-shared.nodes.json').read_text())
    actual=[];combined={'total':0,'passed':0,'failed':[],'skipped':[],'reports':[]}
    complete=True
    for run in runs:
        name=Path(run['report']).stem
        record=read_report(name)
        reports[name]=record
        collected=json.loads((root/(name+'.nodes.json')).read_text())
        status=json.loads((root/(name+'.exit.json')).read_text())['exitstatus']
        assert collected==run['nodes']
        actual.extend(collected)
        complete &= status==0 and run['exitcode']==0 and record['total']==len(collected) and not record['failed']
        for key in ['total','passed']:combined[key]+=record[key]
        for key in ['failed','skipped']:combined[key].extend(record[key])
        combined['reports'].append(run['report'])
    complete &= set(actual)==set(original) and len(actual)==len(set(actual))==214
    combined['device_counts']={device:sum(len(r['nodes']) for r in runs if r.get('device')==device) for device in set(r.get('device') for r in runs)}
    combined['completed_successfully']=bool(complete)
    combined['collected']=len(actual)
    reports['solver-fix-shared']=combined
if (root/'current-shared-runs.json').exists():
    runs=json.loads((root/'current-shared-runs.json').read_text())
    original=json.loads((root/'solver-fix-shared.nodes.json').read_text())
    actual=[];combined={'total':0,'passed':0,'failed':[],'skipped':[],'reports':[]}
    complete=True
    for run in runs:
        name=Path(run['report']).stem
        record=read_report(name)
        reports[name]=record
        collected=json.loads((root/(name+'.nodes.json')).read_text())
        status=json.loads((root/(name+'.exit.json')).read_text())['exitstatus']
        assert collected==run['nodes']
        actual.extend(collected)
        complete &= status==0 and run['exitcode']==0 and record['total']==len(collected) and not record['failed']
        for key in ['total','passed']:combined[key]+=record[key]
        for key in ['failed','skipped']:combined[key].extend(record[key])
        combined['reports'].append(run['report'])
    complete &= set(actual)==set(original) and len(actual)==len(set(actual))==214
    combined['device_counts']={device:sum(len(r['nodes']) for r in runs if r.get('device')==device) for device in set(r.get('device') for r in runs)}
    combined['completed_successfully']=bool(complete)
    combined['collected']=len(actual)
    reports['current-shared']=combined
(root/'test-summary.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps({n:{'total':r['total'],'passed':r['passed'],'failed':len(r['failed']),'skipped':len(r['skipped']),'complete':r.get('completed_successfully')} for n,r in reports.items()},indent=2))
