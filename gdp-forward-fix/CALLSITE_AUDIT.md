# Shared caller audit

Fix: `034c3d41a18f1c6a941918c39396ef004ec757ca`, based on `98193113ecb7835143252a786e06933e371b67cc`.
The same policy was subsequently tested on current upstream `0a649e4c` as `44db92fd` and published as the tree-identical single commit `8ee08891`. Final shared coverage is 212 passed / 2 original skips, including four unchanged pure-PyTorch reference nodes on A800; changed kernels ran on SM120. See README.md and the history-equivalence record for the independent Oja, KDA and TMA fixes included in the combined validation.
Only the SM120 BT=64 autotune warp candidates change, from [2,4,8] to [2]. Stages, precision choices, tensor storage, indexing, interfaces, and derivatives are unchanged. The predicate is the existing exact `(12,0)` hardware helper; SM100 and SM90 are unchanged.

| Caller | Reachability / validation |
| --- | --- |
| GDP forward | BT64 shared solver, both gated and ungated; original four failing dense/packed model checks, operator and generation tests |
| DeltaNet WY | Shared solver at supported BT16/32/64; BT64 changes, both forward and backward recomputation |
| Comba | Shared BT64 solver; operator reference and gradients |
| Gated Oja / Gated Oja 2 | Shared solver at supported chunk sizes; reference output/state/gradient cases |
| Preconditioned GDN | Shared BT64 solver on asymmetric KKT; operator reference/state/gradient cases |
| PATH attention | Chooses BS32 on SM120 because of its shared-memory requirement; unaffected BT32 path, checked as a shared caller |
| GDN native | NVIDIA BT64 uses its independent fused KKT+solve. Unfused BT16/32 remain unchanged. Full GDN and existing ReWY dependency selection validate integration |
| GDN Ascend backend | SM120 predicate false; no change |

The dependency script also lists transitive imports (GDN2, model families, optional backends). Import reachability is not evidence that their computation uses this changed kernel. Multi-GPU CP and unavailable optional backends retain their existing limitations; no test annotations or tolerance changes are made.

Existing GDP model tests are the regression coverage: original upstream tests fail before the fix and pass afterward. No new Cartesian test matrix is needed for this scheduling-only change.

`test_solve_tril_large_batch_offsets` allocates a BF16 [256,16384,12,64] tensor (6 GiB), its result, chunk results and a concatenated reference. It cannot fit the available 12 GiB SM120. The ten ordinary dense/varlen solver cases are selected explicitly; the large-offset test is an unrun hardware gap, not a passing or skipped regression. No address expression is changed by this patch.

The independent TMA coordinate fix preserves int64 global pointer arithmetic and casts only descriptor-relative coordinates. Its ordinary standalone default/TMA tests pass ten cases each. Combined real dense/packed probes above 2^31 elements pass with TMA off/on; these check selected nonzero and identity blocks and do not replace the original huge-batch test. Specializing the final TMA patch with `USE_TMA=False` produces an AST identical to the previous integration, so the shared default-path coverage remains applicable.
