"""Run the unchanged GDP forward comparison with controlled diagnostic dimensions."""
import argparse
import logging
import json
import time
import os
from pathlib import Path
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--layers',type=int,default=4)
parser.add_argument('--batch',type=int,default=4)
parser.add_argument('--length',type=int,default=1024)
args=parser.parse_args()
logging.basicConfig(level=logging.INFO)
sys.path.insert(0,str(args.repo.resolve()))
os.chdir(args.repo)
import torch
from fla.models import GatedDeltaProductConfig
import tests.models.utils as model_utils
from fla.utils.testing import get_abs_err, get_err_ratio
check_forward = model_utils.check_forward
original_assert = model_utils.assert_close
observations=[]
def observe(prefix, ref, actual, ratio, *values, **options):
    observations.append({'prefix':prefix,'max_abs':get_abs_err(ref,actual),'relative_rms':get_err_ratio(ref,actual),'tolerance':ratio})
    return original_assert(prefix, ref, actual, ratio, *values, **options)
model_utils.assert_close=observe
assert torch.backends.cuda.matmul.allow_tf32 is False
started=time.monotonic()
passed=False
try:
    check_forward(args.layers,args.batch,args.length,4,64,GatedDeltaProductConfig,use_l2warp=False,attnres_block_size=None,dtype=torch.bfloat16,varlen=True)
    passed=True
    print('Unchanged strict GDP forward comparison passed.',flush=True)
finally:
    report={'layers':args.layers,'batch':args.batch,'length':args.length,'passed':passed,'observations':observations,'seconds':time.monotonic()-started}
    target=Path(__file__).with_name(f'minimize-L{args.layers}-B{args.batch}-T{args.length}.json')
    target.write_text(json.dumps(report,indent=2)+'\n')
