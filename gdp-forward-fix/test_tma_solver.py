"""Run existing dense/packed solver references on the affected optional TMA path."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--tag',default='solver-fix-tma');a=p.parse_args()
sys.path.insert(0,str(a.repo));os.chdir(a.repo)
from fla.utils import IS_TMA_SUPPORTED
assert IS_TMA_SUPPORTED, 'TMA must actually be active'
root=Path(__file__).resolve().parent
assert not (root/(a.tag+'.xml')).exists(), 'Preserve existing result; use a new tag'
nodes=json.loads((root/'solver-fix-shared.nodes.json').read_text())
selected=[n for n in nodes if n.startswith('tests/ops/test_solve_tril.py::') and ('B2-T1000-H5-chunk_size64' in n or 'H4-D100-chunk_size64' in n)]
assert len(selected)==2,selected
print('TMA active; unchanged original nodes:',selected,flush=True)
import pytest
sys.exit(pytest.main([*selected,'-q','-p','audit_collection','--tb=short','-rs','--junitxml='+str(root/(a.tag+'.xml'))]))
