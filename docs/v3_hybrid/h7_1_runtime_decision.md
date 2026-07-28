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

- Measurement: pending.
- H7.1 decision: pending.
- Release candidate: NONE.
- Release status: NOT READY.
- Playing strength: NOT MEASURED.

This PR does not modify the runtime algorithm and does not perform playing
strength evaluation.
