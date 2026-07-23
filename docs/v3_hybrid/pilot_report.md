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

- Source SHA: `57e854c5cecd6198cb75e855392ef6ea20df229e`
- Base SHA: `51ced4e64079deba254f8c3b856e819e08cae347`
- Docker image: `douzero-p2:57e854c`
- Image digest: `sha256:6bd867ed39ad774195f89365b26f39e6b6d044ddf9378888ec65df2173005a55`
- GPU: NVIDIA GeForce RTX 5070
- PyTorch/CUDA: `2.12.1+cu132` / `13.2`
- Topology/ruleset/seed: single process / legacy / `101`
- Protocol: 900 seconds, real SIGTERM, strict checkpoint load in a fresh
  container, then another 900 seconds with a post-resume optimizer update
- Raw evidence: `/tmp/douzero-p2-evidence/final-57e854c` on `LocalServer`
- Raw evidence manifest: `SHA256SUMS` in that directory

The repository summary is a compact derivative of the validated raw evidence.
The evidence source SHA is the implementation commit immediately before the
final evidence-only report commit; that commit does not alter executable behavior.

## Results

| Variant | Total wall s | Samples | Steps | Resume samples/s | Resume steps/s | Skipped long cooperation episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3_role | 1799.83 | 99,284 | 3,941 | 55.202 | 2.228 | 0 |
| v3_admc | 1803.03 | 81,753 | 3,319 | 43.041 | 1.788 | 0 |
| v3_oracle | 1801.02 | 63,462 | 2,621 | 35.230 | 1.459 | 0 |
| v3_belief | 1800.28 | 86,578 | 3,466 | 48.617 | 1.989 | 0 |
| v3_farmer_cooperation | 1799.67 | 3,973 | 152 | 2.859 | 0.110 | 5,518 |
| v3_full_hybrid | 1800.10 | 396 | 15 | 0.231 | 0.009 | 1,971 |

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
