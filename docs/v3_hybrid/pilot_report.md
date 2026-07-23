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

- Source SHA: `4b240d8753dadfd84c1759e63d797a473b933c23`
- Source tree: `2d72a59bcb15afd6d2b67f82a010e678224eafc4`
- Base SHA: `51ced4e64079deba254f8c3b856e819e08cae347`
- Docker image: `douzero-p2:4b240d8`
- Attested image ID: `sha256:da776cbf2f74be359c11be52b600fad5a13b52e2c5529b6f40b291183fae94e7`
- GPU/driver: NVIDIA GeForce RTX 5070 / `595.71.05`
- PyTorch/CUDA: `2.12.1+cu132` / `13.2`
- Topology/ruleset/seed: single process / legacy / `101`
- Protocol: 900 seconds, real SIGTERM, strict checkpoint load in a fresh
  container, then another 900 seconds with a post-resume optimizer update
- Seed derivation: `sha256(root_seed,stream_name,worker_id,episode_id)-v1`
- Raw evidence: `/tmp/douzero-p2-evidence/final-4b240d8` on `LocalServer`
- Raw evidence manifest: `SHA256SUMS` in that directory

The repository summary is a compact derivative of the validated raw evidence.
The evidence source SHA is the implementation commit immediately before the
final evidence-only report commit; that commit does not alter executable behavior.

## Results

| Variant | Total wall s | Samples | Steps | Resume samples/s | Resume steps/s | Skipped long cooperation episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3_role | 1802.81 | 28,913 | 1,132 | 17.117 | 0.669 | 0 |
| v3_admc | 1802.68 | 29,866 | 1,103 | 17.991 | 0.657 | 0 |
| v3_oracle | 1800.23 | 16,904 | 703 | 10.013 | 0.414 | 0 |
| v3_belief | 1802.90 | 25,581 | 957 | 15.250 | 0.576 | 0 |
| v3_farmer_cooperation | 1797.24 | 2,997 | 112 | 1.659 | 0.063 | 3,555 |
| v3_full_hybrid | 1799.52 | 246 | 9 | 0.079 | 0.003 | 1,496 |

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

The pilot added regression coverage for three failures found against real
environment trajectories:

1. Exact duplicate legal-action feature rows are normalized without changing
   the first-seen rule-engine order before Oracle action alignment.
2. Oracle warmup receives only the data allowed by its active phase; public
   strategy targets no longer cross the warmup gate.
3. H6 public auxiliary updates contribute to the effective policy version even
   when farmer cooperation is disabled, preserving ADMC `q_old` provenance.

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
