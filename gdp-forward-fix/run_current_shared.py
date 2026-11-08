import subprocess,sys,os,json,time,datetime
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108/latest-integration-oja-20261113')
nodes=json.loads((root/'solver-fix-shared.nodes.json').read_text())
groups={}
remote=json.loads((root/'oja-reference-nodes.json').read_text())
assert len(remote)==4 and all(n.startswith('tests/ops/test_oja.py::test_naive_chunk_oja[') for n in remote)
for node in nodes:
    if node not in remote:groups.setdefault(node.split('::')[0],[]).append(node)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',FLA_CI_ENV='0',FLA_REWY='0',FLA_FLASH_QLA='0',FLA_TILELANG='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',TRITON_F32_DEFAULT='ieee',FLA_CACHE_RESULTS='1',PYTHONPATH=str(root)+':'+str(repo))
results=[]
for i,node in enumerate(remote):
    xml=root/f'a800-naive-oja-{i}.xml'
    assert json.loads(xml.with_suffix('.nodes.json').read_text())==[node]
    subprocess.run([sys.executable,str(root/'gate_xml.py'),str(xml)],check=True)
    results.append({'file':'tests/ops/test_oja.py','nodes':[node],'report':xml.name,'exitcode':0,'device':'A800 SM80','scope':'unchanged pure-PyTorch reference; does not call modified solver'})
for i,(file,selected) in enumerate(groups.items()):
    while True:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if not apps and memory<1800:break
        (root/'recovery-resource-wait.json').write_text(json.dumps({'time':datetime.datetime.now().isoformat(),'apps':apps,'memory_mib':memory,'waiting_for':file},indent=2)+'\n')
        time.sleep(30)
    stem=f'current-shared-{i:02d}-{Path(file).stem}'
    xml=root/(stem+'.xml')
    assert not xml.exists(),f'Preserve existing results: {xml}'
    print('Running',file,len(selected),'unchanged nodes',flush=True)
    with (root/(stem+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,'-m','pytest','-p','audit_collection',*selected,'-q','--tb=short','-rs','--maxfail=1','--junitxml='+str(xml)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('\n'.join((root/(stem+'.log')).read_text().splitlines()[-15:]),flush=True)
    results.append({'file':file,'nodes':selected,'report':xml.name,'exitcode':p.returncode,'device':'RTX 5070 Ti Laptop SM120'})
    (root/'current-shared-runs.json').write_text(json.dumps(results,indent=2)+'\n')
    if p.returncode:raise SystemExit(p.returncode)
    assert json.loads(xml.with_suffix('.nodes.json').read_text())==selected
    subprocess.run([sys.executable,str(root/'gate_xml.py'),str(xml)],check=True)
    config=json.loads(xml.with_suffix('.solver-configs.json').read_text())
    assert config['solver_sha256']=='fba06478219c4086b9b461e508aa35ebe8656cc74237baab78dafeb71906a77d'
    assert config['is_sm120'] and config['active_warp_candidates']==[2]
    assert all(c['num_warps']==2 for c in config['used_configs'])
assert len([node for r in results for node in r['nodes']])==214
assert set(node for r in results for node in r['nodes'])==set(nodes)
print('All 214 original nodes completed: 210 on SM120 plus four unchanged pure-PyTorch references on A800',flush=True)
