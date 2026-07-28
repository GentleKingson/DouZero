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

- Measurement: complete on committed source `1308e302aacfb5df2e2fec1c01c744d643347c1a`.
- H7.1 decision: required.
- Release candidate: NONE.
- Release status: NOT READY.
- Playing strength: NOT MEASURED.

This PR does not modify the runtime algorithm and does not perform playing
strength evaluation.

## Matched results

The matrix ran in Docker on one NVIDIA GeForce RTX 5070 with driver
595.71.05, PyTorch 2.12.1+cu132, and CUDA 13.2. The immutable protocol hash is
`63afd2d9b982ac9e8c50dd2cc0eeecfae07872d6bb0f95f9ed0e7302ab4ae5ea`;
the image digest is
`sha256:6da0fed88d3de60814ad3cabd3c6dd93bc9b650aa145060b8a3557d109a48cb6`.
Every run used a 30-second warmup, a checkpoint-enabled 300-second measured
window, and seeds 101, 202, and 303.

| Topology | Learner samples/s (three runs) | Median | Median optimizer steps/s |
| --- | --- | ---: | ---: |
| Base single process | 53.860, 47.487, 51.282 | 51.282 | 1.603 |
| Base async 4x4 | 62.796, 50.865, 57.665 | 57.665 | 1.802 |
| Base async 8x4 | 71.985, 47.680, 62.754 | 62.754 | 1.961 |
| Full hybrid single process | 0.320, 0.316, 0.057 | 0.316 | 0.010 |

The full-hybrid/base-single median learner-throughput ratio is 0.00617, far
below the frozen 0.70 gate. All twelve checkpoints saved and strictly
reloaded, all processes quiesced, and base runs observed parameter updates.
The full-hybrid runs did not observe a parameter change despite recording
eligible optimizer steps. They skipped 383, 389, and 391 oversized cooperation
episodes respectively, so the stability gate also fails. This is not treated
as successful long-running training.

Diagnostic medians reinforce the end-to-end result: base single process
reached 6.410 games/s and 375.9 decisions/s, while full hybrid reached 1.185
games/s and 75.2 decisions/s. Full-hybrid collection occupied essentially the
entire measurement window; nested public inference accounted for about
68 seconds, exact belief DP for about 16 seconds, and cooperation trajectory
assembly for about 3 seconds per run. These overlapping segment timers are not
summed to derive throughput.

## Decision

P3 selects H7.1. The complete hybrid cannot finish the frozen formal matrix
within the available GPU budget and does not meet the parameter-update
stability gate. Work proceeds as separate PRs from the then-current `main`,
starting with H7.1a belief async, followed by Oracle, farmer cooperation,
public strategy/style auxiliaries, and standard bidding as required. This
decision PR intentionally contains no runtime algorithm change.

Machine-readable protocol, raw run records, and the validator-derived summary
are committed under `artifacts/v3-p3/`. Full logs and checkpoint files remain
external test evidence rather than Git payload.
