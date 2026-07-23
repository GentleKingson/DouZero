# V3 P2 pilot decision report

## Decision

- Advance directly to P4: **no**.
- Run the P3/H7.1 runtime decision: **yes**.
- Release candidate: **NONE**.
- Release status: **NOT READY**.
- Playing strength: **NOT MEASURED**.

The cooperation and full-hybrid single-process paths are not fast enough for
the frozen formal matrix. The result is a runtime stop signal, not a playing
strength conclusion. P3 must measure the segmented cost and decide whether to
implement the H7.1 async stack before any P4 budget is committed.

## Provenance

- Source SHA: `7c94577159c434b285d66bf06863669fe52e1f44`
- Source tree: `032dac0e3637ecc60e6c30757808869fac7394b2`
- Base SHA: `51ced4e64079deba254f8c3b856e819e08cae347`
- Docker image: `douzero-p2:7c94577`
- Attested image ID: `sha256:eb7e99d4de352653f17a0ca58d5432fcea250a50dbc811a6e23c3c970b20aad8`
- GPU/driver: NVIDIA GeForce RTX 5070 / `595.71.05`
- PyTorch/CUDA: `2.12.1+cu132` / `13.2`
- Topology/ruleset/seed: single process / legacy / `101`
- Protocol: approximately 897 seconds, real SIGTERM at an episode boundary,
  strict checkpoint load in a fresh container, then another approximately
  897 seconds ending with SIGTERM after a post-resume optimizer update
- Seed derivation: `sha256(root_seed,stream_name,worker_id,episode_id)-v1`
- Raw evidence: `/tmp/douzero-p2-evidence/final-7c94577` on `LocalServer`
- Raw evidence manifest: `SHA256SUMS` in that directory

The repository summary is a compact derivative of the validated raw evidence.
The evidence source SHA is the implementation commit immediately before the
final evidence-only report commit; that commit does not alter executable behavior.

## Results

| Variant | Total wall s | Samples | Steps | Resume samples/s | Resume steps/s | Skipped long cooperation episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3_role | 1797.47 | 30,860 | 1,205 | 17.407 | 0.677 | 0 |
| v3_admc | 1793.93 | 31,719 | 1,196 | 18.280 | 0.693 | 0 |
| v3_oracle | 1796.15 | 18,365 | 765 | 10.547 | 0.439 | 0 |
| v3_belief | 1797.71 | 27,884 | 1,029 | 16.140 | 0.593 | 0 |
| v3_farmer_cooperation | 1794.12 | 2,997 | 112 | 1.742 | 0.066 | 3,626 |
| v3_full_hybrid | 1791.65 | 246 | 9 | 0.081 | 0.003 | 1,456 |

All six variants saved a checkpoint after SIGTERM, strict-loaded it in a new
container, advanced the optimizer and policy counters, and published a new
checkpoint. Losses and gradients remained finite in the captured terminal
steps. The evidence summarizer rejected any run pair lacking the SIGTERM or
post-resume update contract.

The dominant failure signal is the episode-atomic cooperation path: episodes
longer than the configured transition batch are skipped, leaving very few
eligible updates. This makes the current full-hybrid single-process path
unsuitable for the frozen P4 matrix even though the component paths train.

## Correctness fixes exposed by the pilot

The pilot added regression coverage for failures found against real
environment trajectories:

1. Exact duplicate legal-action feature rows are normalized without changing
   the first-seen rule-engine order before Oracle action alignment.
2. Oracle warmup receives only the data allowed by its active phase; public
   strategy targets no longer cross the warmup gate.
3. H6 public auxiliary updates contribute to the effective policy version even
   when farmer cooperation is disabled, preserving ADMC `q_old` provenance.
4. Effective pilot ceilings are recorded and overrides above the frozen P1
   budgets fail before training starts.
5. Non-cooperation episodes are trained and checkpointed atomically, so a
   SIGTERM cannot persist a completed episode ID with only a trained prefix.
6. Docker image identity is bound through the container's PID namespace, not
   its configurable hostname.
7. A deal whose collection crosses the wall-clock deadline is discarded before
   training and is not marked complete in resume state.
8. Direct script execution prepends the attested repository root before any
   project import, preventing an installed or `PYTHONPATH` package from
   shadowing the commit-bound checkout.
9. The runner checks the monotonic deadline before every batch piece and
   fails closed rather than starting late work inside an atomic episode.
10. The summarizer also binds its imports to the checked-out repository before
    loading the validation implementation.

Resume throughput is computed from counter deltas rather than cumulative
counters, and the summary validator independently checks that arithmetic.

## Not executed

- The frozen 2,000-5,000 paired-deal pilot evaluations were not executed.
  Current V3 public checkpoints do not yet have a complete paired-evaluator
  backend, and the runtime stop condition was already met. No small-sample
  result is substituted for playing-strength evidence.
- Standard full-game pilot was not executed because the legacy card-play
  cooperation/full-hybrid paths failed the throughput stop condition.
- Search-on evaluation was not executed. Search remains an evaluation wrapper
  around the same training checkpoint.
- P4 multi-seed development and promotion evaluation were not started.

## Next gate

P3 must run the frozen three-repeat comparison and segmented profiling for
base V3+ADMC single-process, async 4x4, async 8x4, and full-hybrid
single-process. It should then either record that single-process is adequate
or implement the required H7.1 capabilities as separate PRs. P4 remains
blocked until that decision is complete.
