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
  `1848cc37541f5bc17b570728ae3741491b5a241b`.
- H7.1 decision: required.
- Release candidate: NONE.
- Release status: NOT READY.
- Playing strength: NOT MEASURED.

This PR does not modify the runtime algorithm and does not perform playing
strength evaluation.

## Matched results

The matrix ran in Docker on one NVIDIA GeForce RTX 5070 with driver
595.71.05, PyTorch 2.12.1+cu132, and CUDA 13.2. The immutable protocol hash is
`998e467f497d1c9390701bf65e1ce1bfbcab70b7ee6261776b79382043403023`;
the image digest is
`sha256:2e6179b7077fafd9eec7d485d0fefe2b129452d6cf07df3fa1a866647ca4ab57`.
Every run used a 30-second warmup, a checkpoint-enabled 300-second measured
window, and seeds 101, 202, and 303.

| Topology | Learner samples/s (three runs) | Median | Median optimizer steps/s | Median CPU RSS |
| --- | --- | ---: | ---: | ---: |
| Base single process | 51.588, 46.312, 50.283 | 50.283 | 1.571 | 5.01 GiB |
| Base async 4x4 | 22.204, 28.380, 16.369 | 22.204 | 0.694 | 8.72 GiB |
| Base async 8x4 | 19.395, 17.907, 18.240 | 18.240 | 0.570 | 10.88 GiB |
| Full hybrid single process | 4.435, 4.384, 4.805 | 4.435 | 0.156 | 2.40 GiB |

Async CPU RSS is the aggregate of the parent process and every live Actor
worker sampled before shutdown, not just the parent. The full-hybrid/base-
single median learner-throughput ratio is 0.08820, far below the frozen 0.70
gate. All twelve checkpoints saved and strictly reloaded, all processes
quiesced, and all runs observed an update across the complete learner
parameter graph. The full-hybrid measured windows produced 47, 47, and 51
optimizer updates. Their cumulative counters, including warmup, recorded 363,
369, and 361 skipped oversized cooperation episodes because enabled
cooperation requires episode-atomic batches and the frozen batch size is 32.
Seed 202 also produced one episode where a farmer had no non-forced decision;
the runner explicitly counted and skipped that incomplete cooperation pair
without adding forced-action inference or replay. The runtime stability checks
pass, but the efficiency gate fails decisively.

Diagnostic medians reinforce the end-to-end result: base single process
reached 6.285 games/s and 356.6 decisions/s, while full hybrid reached 1.255
games/s and 80.4 decisions/s. Full-hybrid collection occupied essentially the
entire measurement window; median synchronized segment totals were 52.6
seconds for nested public inference, 13.4 seconds for exact belief DP, 2.94
seconds for cooperation trajectory assembly, 8.72 seconds for Oracle learner
work, and 274.6 seconds for collate and episode preparation. SegmentProfiler
synchronizes CUDA at every segment boundary. These semantic components may
still be nested, so their timings are not summed to derive throughput.

## Decision

P3 selects H7.1. The complete hybrid reaches only 8.82% of base
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
