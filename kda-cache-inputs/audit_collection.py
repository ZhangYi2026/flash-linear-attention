"""Record selection and exit status; never modify items, tests, skips or results."""
import json
import pytest
from pathlib import Path

def pytest_collection_finish(session):
    path=Path(session.config.option.xmlpath)
    path.with_suffix('.nodes.json').write_text(json.dumps([item.nodeid for item in session.items],indent=2)+'\n')

def pytest_sessionfinish(session,exitstatus):
    path=Path(session.config.option.xmlpath)
    path.with_suffix('.exit.json').write_text(json.dumps({'exitstatus':int(exitstatus)})+'\n')
    import sys,hashlib
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        try:
            stats=torch.cuda.memory_stats(0)
            memory={key:stats.get(key) for key in ['allocated_bytes.all.peak','reserved_bytes.all.peak','num_alloc_retries','num_ooms']}
        except Exception as error:
            memory={'unavailable':str(error)}
        path.with_suffix('.cuda-memory.json').write_text(json.dumps(memory,indent=2)+'\n')
    module=sys.modules.get('fla.ops.utils.solve_tril')
    if module is not None:
        tuner=module.merge_16x16_to_64x64_inverse_kernel
        while not hasattr(tuner,'configs'):
            tuner=tuner.fn
        record={
            'solver_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            'is_sm120':getattr(sys.modules.get('fla.utils'),'IS_NVIDIA_SM120',None),
            'active_warp_candidates':sorted(set(c.num_warps for c in tuner.configs)),
            'used_configs':[{'key':repr(k),'num_warps':v.num_warps,'num_stages':v.num_stages,'kwargs':v.kwargs} for k,v in tuner.cache.items()],
        }
        path.with_suffix('.solver-configs.json').write_text(json.dumps(record,indent=2)+'\n')

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.path.name != 'test_modeling_gated_deltaproduct.py' or item.originalname != 'test_modeling':
        yield
        return
    import torch
    import tests.models.utils as model_utils
    original=model_utils.assert_close
    def observe(prefix,ref,tri,*args,**kwargs):
        try:
            return original(prefix,ref,tri,*args,**kwargs)
        finally:
            if prefix=='output':
                path=Path(item.config.option.xmlpath).with_suffix('.model-comparisons.json')
                records=json.loads(path.read_text()) if path.exists() else {}
                records[item.nodeid]={'bitwise_equal':torch.equal(ref,tri),'max_abs_fp32':(ref.float()-tri.float()).abs().max().item(),'threshold':args[0] if args else kwargs.get('ratio')}
                path.write_text(json.dumps(records,indent=2)+'\n')
    model_utils.assert_close=observe
    try:
        yield
    finally:
        model_utils.assert_close=original
