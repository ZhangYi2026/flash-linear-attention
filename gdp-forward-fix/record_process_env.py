from pathlib import Path
import json
root=Path(__file__).resolve().parent
keys=['CUDA_VISIBLE_DEVICES','FLA_CI_ENV','FLA_REWY','FLA_FLASH_QLA','FLA_TILELANG','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','TRITON_F32_DEFAULT','FLA_TRIL_PRECISION','FLA_CACHE_RESULTS','FLA_USE_TMA']
values=dict(item.split('=',1) for item in Path('/proc/674/environ').read_text().split('\0') if '=' in item)
selected={k:values.get(k) for k in keys}
result={'process':674,'cwd':str(Path('/proc/674/cwd').resolve()),'flags':selected}
(root/'shared-process-environment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
