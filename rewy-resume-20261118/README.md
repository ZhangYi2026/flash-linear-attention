# ReWY validation after rebasing onto November 18 main

This report covers only [ReWY #1703](https://github.com/fla-org/flash-linear-attention/pull/1703). The six-file patch remains one commit; no solver, Oja, TMA or KDA fixture fix is included or required as a prerequisite.

- Baseline: `cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b`.
- Candidate: `e9fc25f2714a653995120a751e5f9356fd8bab19`.
- Source trees and archive SHA256 digests: [sources.json](sources.json).
- The ReWY patch is unchanged apart from upstream context: [range-diff.txt](range-diff.txt).
- Device/software: [environment.json](environment.json); RTX 5070 Ti Laptop SM120, driver 595.97, WSL Ubuntu 24.04, Python 3.12.3, PyTorch 2.9.1+cu128, Triton 3.5.1, Transformers 5.20.2.

## Paired results

| Selection | Baseline | ReWY enabled |
| --- | --- | --- |
| Full GDN file | 99 passed, 1 failed, 25 skipped | 114 passed, 0 failed, 25 skipped |
| Dependent GDN/GDP selection | 55 passed, 4 failed, 25 skipped | 55 passed, 4 failed, 25 skipped |

All 14 new ReWY cases passed, including actual dispatch, independent recurrence comparisons of outputs/states/six gradients, packed tails, both layouts/dtypes, zero beta and rejected/disabled-call fallback. No extra candidate failure was observed. This is not a fully passing dependent suite: both revisions failed the same four GDP variable-length forward comparisons before backward. The exact assertions and node IDs are preserved below and in [validation.json](validation.json).

The full baseline GDN run also failed one FP16 reference-gradient assertion (`dq: NaN detected in ref`). The corresponding candidate case passed. This single paired observation does not establish its cause or that ReWY fixes the reference; no rerun was used to replace that failed result.

The dependency helper's broad import-based list is in [dependent-discovery.txt](dependent-discovery.txt). The executed operator/layer/model/CP selection is explicit in [validate_rebased.py](validate_rebased.py). It covers the shared GDN/GDP backward and their direct layer/model consumers; it does not claim the entire repository or every transitive import match passed. Preconditioned GDN and GDN2 use separate backward implementations; the general model API/cache/hybrid files and unrelated operators were not run in this selection. Default wide-value model configurations mainly exercise integration/fallback. Multi-device CP remains unvalidated.

The 25 full-file GDN skips comprise eight existing grouped-head/backend guards, sixteen unavailable FlashQLA cases and one unavailable TileLang case. The 25 dependent skips comprise twelve multi-device CP cases, eleven upstream-disabled MoM cases and two Hopper-only model cases. No new skip or changed tolerance was introduced. Earlier standalone grouped-head/A800/addressing/sanitizer results remain historical evidence at the root of this branch, not new execution of this rebased head.

Current main consolidated `test_mom_shared_no_grad.py` into `test_mom.py`. The first dependent command used the removed filename and exited during collection without executing tests; its log/XML and initial harness/result snapshots remain in this directory. The corrected paired selection uses the existing consolidated file, increasing historical dependent passes from 51 to 55. No production/test source was modified for this target correction.

## Reproduction

Set `FLA_CI_ENV=0 FLA_FLASH_QLA=0 FLA_TILELANG=0 FLA_DISABLE_BACKEND_DISPATCH=0 FLA_CACHE_RESULTS=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee` before imports, with `FLA_REWY=0` on the baseline and `FLA_REWY=1` on the candidate. The harness records complete commands and environment for each selection. It extracts the two exact Git archives into separate directories, verifies archive hashes and preserves completed outputs. Archives themselves are omitted from this text evidence; recreate them with `git archive --format=tar --output=baseline.tar cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b` and the equivalent candidate command. Adapt the harness cache path to a fresh local directory.

See [H100_VALIDATION.md](H100_VALIDATION.md) for checkout-independent test commands, upstream workflow parameters and the B/FB benchmark method. The account only has read permission on upstream: a maintainer must dispatch its H100 workflow. Dispatch must be enabled and eligible existing tests also need `FLA_REWY=1`; the old dispatch-disabled smoke run did not exercise ReWY.

## Static checks and remaining gates

Changed-file repository pre-commit hooks, full repository copyright-header check and `git diff --check` passed for the candidate; see [static-checks.json](static-checks.json). No new performance experiment was run. Forward, saved tensors, original precisions and default recomputation policy are unchanged. Historical A800 public B/FB measurements are supplementary and use the old pinned revisions; they do not measure this rebased head or replace datacenter SM90+ evidence. NCU, training convergence and full multi-device correctness are not claimed.

The PR remains Draft pending maintainer assessment of baseline failures and actual datacenter SM90+ correctness/performance evidence. The checklist records these outstanding gates honestly. The withdrawn four independent submissions and their combined 235-pass experiment are not evidence that this standalone ReWY selection passes.

## Exact failed assertions

### baseline-gdn

- `tests.ops.test_gdn::test_chunk[B4-T1024-H4-HV4-D128-scale0.1-gate_logit_normalizer1-mask_p1.0-use_qk_l2norm_in_kernelFalse-torch.float16]`: `AssertionError: dq: NaN detected in ref`

### candidate-gdn

No failed cases.

### baseline-dependent-current

- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2True-bsNone-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.078125 ratio: 0.006866`
- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2False-bsNone-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.078125 ratio: 0.006866`
- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2False-bs1-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.066406 ratio: 0.003204`
- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2False-bs4-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.078125 ratio: 0.003738`

### candidate-dependent-current

- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2True-bsNone-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.078125 ratio: 0.006866`
- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2False-bsNone-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.078125 ratio: 0.006866`
- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2False-bs1-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.066406 ratio: 0.003204`
- `tests.models.test_modeling_gated_deltaproduct::test_modeling[L4-B4-T1024-H4-D64-l2False-bs4-torch.bfloat16-varlen]`: `AssertionError:           output diff: 0.078125 ratio: 0.003738`
