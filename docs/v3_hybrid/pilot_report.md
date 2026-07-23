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

- Source SHA: `24fac749b01ddfc551b0afdd3f797e980ad567e0`
- Source tree: `4eacd6c8ee57430cfe14c4eb4d3c5e4796f1c08a`
- Base SHA: `51ced4e64079deba254f8c3b856e819e08cae347`
- Docker image: `douzero-p2:24fac74`
- Attested image ID: `sha256:0c32f97a46a03aa05efadbb40da8feeb53379b6971eba11a57d05733159b3f46`
- GPU/driver: NVIDIA GeForce RTX 5070 / `595.71.05`
- PyTorch/CUDA: `2.12.1+cu132` / `13.2`
- Topology/ruleset/seed: single process / legacy / `101`
- Protocol: approximately 924 seconds, real SIGTERM at an episode boundary,
  strict checkpoint load in a fresh container, then 900 seconds with a
  post-resume optimizer update
- Seed derivation: `sha256(root_seed,stream_name,worker_id,episode_id)-v1`
- Raw evidence: `/tmp/douzero-p2-evidence/final-24fac74` on `LocalServer`
- Raw evidence manifest: `SHA256SUMS` in that directory

The repository summary is a compact derivative of the validated raw evidence.
The evidence source SHA is the implementation commit immediately before the
final evidence-only report commit; that commit does not alter executable behavior.

## Results

| Variant | Total wall s | Samples | Steps | Resume samples/s | Resume steps/s | Skipped long cooperation episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3_role | 1825.80 | 28,628 | 1,122 | 16.634 | 0.649 | 0 |
| v3_admc | 1823.95 | 29,696 | 1,120 | 17.374 | 0.661 | 0 |
| v3_oracle | 1823.04 | 17,427 | 724 | 10.099 | 0.417 | 0 |
| v3_belief | 1825.13 | 26,344 | 974 | 15.444 | 0.570 | 0 |
| v3_farmer_cooperation | 1824.88 | 2,997 | 112 | 1.721 | 0.065 | 3,623 |
| v3_full_hybrid | 1823.08 | 246 | 9 | 0.080 | 0.003 | 1,501 |

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
