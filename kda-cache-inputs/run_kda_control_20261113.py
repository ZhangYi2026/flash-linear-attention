import subprocess,sys,os,json
from pathlib import Path
root=Path(__file__).resolve().parent
cache=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
for variant,flag in [('baseline','0'),('solver-fix-rewy','1')]:
 repo=cache/variant; stem='kda-control-20261113-'+variant
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',FLA_CI_ENV='0',FLA_REWY=flag,FLA_FLASH_QLA='0',FLA_TILELANG='0',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',TRITON_F32_DEFAULT='ieee',FLA_CACHE_RESULTS='1',PYTHONPATH=str(root)+':'+str(repo))
 assert not (root/(stem+'.xml')).exists()
 with (root/(stem+'.log')).open('w') as f:
  p=subprocess.run([sys.executable,'-m','pytest','-p','audit_collection','tests/cp/test_intracard.py::test_chunk_kda_cache','-q','--tb=short','-rs','--junitxml='+str(root/(stem+'.xml'))],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(variant,p.returncode,flush=True)
 print((root/(stem+'.log')).read_text()[-7000:],flush=True)
