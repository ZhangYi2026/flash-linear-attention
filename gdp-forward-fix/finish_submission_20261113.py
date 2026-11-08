"""Sequential, non-overwriting validation of the final source snapshots."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
cache = Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', FLA_CI_ENV='0', FLA_REWY='0', FLA_FLASH_QLA='0',
           FLA_TILELANG='0', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', TRITON_F32_DEFAULT='ieee',
           FLA_TRIL_PRECISION='ieee', FLA_CACHE_RESULTS='1', FLA_USE_TMA='0')
status = []


def run(name, args, repo, **flags):
    log = root / (name + '.log')
    assert not log.exists(), log
    active_env = dict(env, PYTHONPATH=str(root) + ':' + str(cache / repo), **flags)
    print('Running ' + name, flush=True)
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, *args], cwd=cache / repo, env=active_env,
                                stdout=stream, stderr=subprocess.STDOUT)
    status.append({'name': name, 'repo': repo, 'flags': flags, 'exitcode': result.returncode, 'args': args})
    (root / 'submission-run-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print('\n'.join(log.read_text().splitlines()[-5:]), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def test(name, repo, targets, **flags):
    xml = root / (name + '.xml')
    assert not xml.exists(), xml
    run(name, ['-m', 'pytest', '-p', 'audit_collection', *targets, '-q', '--tb=short', '-rs',
               '--junitxml=' + str(xml)], repo, **flags)
    subprocess.run([sys.executable, str(root / 'gate_xml.py'), str(xml)], check=True)


test('submission-tma-default', 'submission-tma-20261113', ['tests/ops/test_solve_tril.py', '-k', 'not large_batch_offsets'])
test('submission-tma-enabled', 'submission-tma-20261113', ['tests/ops/test_solve_tril.py', '-k', 'not large_batch_offsets'], FLA_USE_TMA='1')
test('submission-combined-tma', 'submission-integration-20261113', ['tests/ops/test_solve_tril.py', '-k', 'not large_batch_offsets'], FLA_USE_TMA='1')
for mode in ['0', '1']:
    name = 'submission-large-offset-tma' + mode
    run(name, [str(root / 'check_solver_large_offset.py'), '--repo', str(cache / 'submission-integration-20261113'),
               '--tag', name], 'submission-integration-20261113', FLA_USE_TMA=mode)
test('submission-integration', 'submission-integration-20261113', [
    'tests/ops/test_gdn.py', 'tests/ops/test_gdn_kernels.py', 'tests/ops/test_gdp.py',
    'tests/cp/test_gdn.py', 'tests/cp/test_gdp.py', 'tests/cp/test_intracard.py',
    'tests/layers/test_gated_deltanet.py', 'tests/layers/test_mom_shared_no_grad.py',
    'tests/models/test_modeling_gated_deltanet.py', 'tests/models/test_modeling_gated_deltaproduct.py',
    'tests/models/test_modeling_mom.py'], FLA_REWY='1')
for repo, tag in [('latest-baseline-20261113', 'A1'), ('latest-solver-20261113', 'B1'),
                  ('latest-solver-20261113', 'B2'), ('latest-baseline-20261113', 'A2')]:
    run('benchmark-' + tag, [str(root / 'benchmark_solver.py'), '--repo', str(cache / repo), '--tag', tag],
        repo, FLA_CACHE_RESULTS='0')
subprocess.run([sys.executable, str(root / 'summarize_solver_benchmark_latest.py')], check=True)
for repo, tag in [('latest-baseline-20261113', 'A1'), ('latest-oja-20261113', 'B1'),
                  ('latest-oja-20261113', 'B2'), ('latest-baseline-20261113', 'A2')]:
    run('oja-benchmark-' + tag, [str(root / 'benchmark_oja.py'), '--repo', str(cache / repo), '--tag', tag],
        repo, FLA_CACHE_RESULTS='0')
for repo, tag in [('latest-baseline-20261113', 'TMA-A1'), ('submission-tma-20261113', 'TMA-B1'),
                  ('submission-tma-20261113', 'TMA-B2'), ('latest-baseline-20261113', 'TMA-A2')]:
    run('benchmark-' + tag, [str(root / 'benchmark_solver.py'), '--repo', str(cache / repo), '--tag', tag],
        repo, FLA_CACHE_RESULTS='0')
run('benchmark-TMA-enabled', [str(root / 'benchmark_solver.py'), '--repo', str(cache / 'submission-tma-20261113'),
                            '--tag', 'TMA-enabled', '--tma'], 'submission-tma-20261113', FLA_USE_TMA='1', FLA_CACHE_RESULTS='0')
print('Final correctness and scheduling measurements completed.', flush=True)
