# H7.1 full-hybrid runtime decision

## Frozen gate

This record freezes the P3 decision before the first matched measurement.

- Matrix: base V3+ADMC single process, base V3+ADMC async 4x4, base
  V3+ADMC async 8x4, and full-hybrid single process.
- Repetitions: three, using frozen training seeds 101, 202, and 303.
- Warmup: 30 seconds per repetition.
- Measurement: 300 seconds per repetition.
- Shared constraints: one committed source SHA and source tree, one Docker image
  digest, one RTX 5070, batch size 32, checkpoint cadence 1,000 eligible
  updates, and the same public backbone scale.
- Primary gate: the median full-hybrid single-process learner samples/s must be
  at least 70% of the median base V3+ADMC single-process value.
- Stability gate: checkpoint save and strict reload must succeed, policy lag
  must stay at or below 128, shutdown must quiesce active/in-flight/pending
  work, and metrics must remain finite.

The 70% threshold is frozen before measurement. It will not be adjusted after
results are observed. If throughput is below the threshold, the formal matrix
would exceed the available GPU budget, or any stability gate fails, P3 will
select H7.1. Capabilities must then be implemented in the prescribed H7.1a-e
sequence rather than bundled into this decision PR.

Segment timings are diagnostic and may overlap when a semantic component is
nested inside public inference or learner work. The promotion decision uses
raw counter deltas divided by the end-to-end checkpoint-enabled measurement
window, never the sum of segment timers.

## Current status

- Measurement: complete on committed executable source
  `7e2186b6dfa9c29db4f0bc0cf057e0f64adc74e8`.
- H7.1 decision: required.
- Release candidate: NONE.
- Release status: NOT READY.
- Playing strength: NOT MEASURED.

This PR does not modify the runtime algorithm and does not perform playing
strength evaluation.

## Matched results

The matrix ran in Docker on one NVIDIA GeForce RTX 5070 with driver
595.71.05, PyTorch 2.12.1+cu132, and CUDA 13.2. The immutable protocol hash is
`f46a30deab514e1649696eac7f4b184e7f625967d1905c38f0ba3a6d78e8cfae`;
the image digest is
`sha256:0ecbf17be9ea51961867516907e74327c8ab2a3ac9cb10d654d0ea6d8e08422c`.
Every run used a 30-second warmup, a checkpoint-enabled 300-second measured
window, and seeds 101, 202, and 303.

| Topology | Learner samples/s (three runs) | Median | Median optimizer steps/s |
| --- | --- | ---: | ---: |
| Base single process | 53.678, 46.293, 50.231 | 50.231 | 1.570 |
| Base async 4x4 | 17.797, 28.718, 21.167 | 21.167 | 0.661 |
| Base async 8x4 | 14.654, 14.644, 14.824 | 14.654 | 0.458 |
| Full hybrid single process | 0.319, 0.316, 0.057 | 0.316 | 0.010 |

The full-hybrid/base-single median learner-throughput ratio is 0.00629, far
below the frozen 0.70 gate. All twelve checkpoints saved and strictly
reloaded, all processes quiesced, and all runs observed an update across the
complete learner parameter graph. The full-hybrid runs produced only 3, 4,
and 1 optimizer updates in their measured windows. Their cumulative counters,
including warmup, recorded 382, 391, and 389 skipped oversized cooperation
episodes because enabled cooperation requires episode-atomic batches and the
frozen batch size is 32. The runtime stability checks pass, but the efficiency
gate fails decisively.

Diagnostic medians reinforce the end-to-end result: base single process
reached 6.279 games/s and 373.7 decisions/s, while full hybrid reached 1.178
games/s and 75.0 decisions/s. Full-hybrid collection occupied essentially the
entire measurement window; nested public inference accounted for about
68 seconds, exact belief DP for about 16 seconds, cooperation trajectory
assembly for about 3 seconds, and Oracle learner work for about 0.8 seconds
per run. These overlapping segment timers are not summed to derive
throughput.

## Decision

P3 selects H7.1. The complete hybrid reaches only 0.629% of base
single-process learner throughput under the frozen matched protocol, so it
cannot finish the formal training matrix within the available GPU budget.
Work proceeds as separate PRs from the then-current `main`, starting with
H7.1a belief async, followed by Oracle, farmer cooperation, public
strategy/style auxiliaries, and standard bidding as required. This decision
PR changes benchmark correctness and evidence only, not the production
training runtime.

Machine-readable protocol, raw run records, and the validator-derived summary
are committed under `artifacts/v3-p3/`. Full logs and checkpoint files remain
external test evidence rather than Git payload.
