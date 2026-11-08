"""Validate completed evidence, preserving and distinguishing the earlier failures."""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / 'summarize_results.py')], check=True, stdout=subprocess.DEVNULL)
summary = json.loads((root / 'test-summary.json').read_text())
assert summary['current-shared']['completed_successfully']
assert (summary['current-shared']['collected'], summary['current-shared']['passed'], len(summary['current-shared']['skipped'])) == (214, 212, 2)
expected = [('submission-integration', 285), ('latest-integration-oja-20261113', 285), ('latest-fixed-kda', 8),
            ('oja-fix-test_oja', 24), ('oja-fix-test_gated_oja_rule2', 33),
            ('submission-tma-default', 10), ('submission-tma-enabled', 10), ('submission-combined-tma', 10)]
for name, total in expected:
    subprocess.run([sys.executable, str(root / 'gate_xml.py'), str(root / (name + '.xml'))], check=True)
    assert summary[name]['total'] == total
assert summary['submission-integration']['passed'] == 235
assert len(summary['submission-integration']['skipped']) == 50
models = json.loads((root / 'submission-integration.model-comparisons.json').read_text())
assert len(models) == 4 and all(r['bitwise_equal'] and r['max_abs_fp32'] == 0 and r['threshold'] == .001 for r in models.values())
manifest = json.loads((root / 'submission-integration-20261113-verified.json').read_text())
for name in ['submission-integration', 'submission-combined-tma']:
    source = json.loads((root / (name + '.solver-configs.json')).read_text())
    assert source['solver_sha256'] == manifest['files']['fla/ops/utils/solve_tril.py']
    assert source['active_warp_candidates'] == [2]
    assert all(c['num_warps'] == 2 for c in source['used_configs'])
for mode in ['0', '1']:
    offsets = json.loads((root / ('submission-large-offset-tma' + mode + '.json')).read_text())
    assert offsets['first_checked_element_offset'] > 2 ** 31
    assert offsets['tma_enabled'] == (mode == '1')
    assert len(offsets['rows']) == 2 and all(r['both_late_blocks_equal_to_small_solve'] and r['three_identity_controls_equal'] for r in offsets['rows'])
assert len(summary['baseline-tma-20261113']['failed']) == len(summary['solver-fix-tma']['failed']) == 2
assert json.loads((root / 'submission-source-review.json').read_text())['non_tma_specialization_ast_equal']
runs = json.loads((root / 'submission-final-runs.json').read_text())
assert len(runs) == 19 and all(r['exitcode'] == 0 for r in runs)
for name, count in [('benchmark-summary.json', 8), ('oja-benchmark-summary.json', 4), ('tma-default-benchmark-summary.json', 8), ('benchmark-TMA-enabled.json', 8)]:
    assert len(json.loads((root / name).read_text())['rows']) == count
(root / 'FINAL_GATE.json').write_text(json.dumps({'passed': True, 'integration_commit': manifest['commit'],
    'integration': {'passed': 235, 'skipped': 50}, 'shared': {'passed': 212, 'skipped': 2},
    'tma_original_failures_preserved': True, 'large_offset_modes_passed': [False, True],
    'unrun': ['original huge-batch test exceeds available SM120 memory', 'datacenter SM90+ ReWY performance'],
    'performance_scope': 'supplementary affected-device scheduling cost'}, indent=2) + '\n')
print('Final evidence gate passed; unavailable checks remain explicitly unrun.')
