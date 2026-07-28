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
  updates, the same public backbone scale, and the P1 formal per-episode deal
  seed derivation.
- Learner cadence: one base learner update per four collected games in every
  base topology. Async 4x4 and 8x4 therefore execute four and eight learner
  updates respectively after each 16- and 32-game collection cycle.
- Full-hybrid phase: counters are checkpoint-consistently pre-advanced to the
  first `guided` update (10,000) before warmup. The timed workload therefore
  includes the public student, Adaptive DMC, Oracle guidance, belief,
  cooperation, strategy, and style paths rather than Oracle warmup alone.
- Primary gate: the median full-hybrid single-process learner samples/s must be
  at least 70% of the median base V3+ADMC single-process value.
- Stability gate: checkpoint save and strict reload must succeed, policy lag
  must stay at or below 128, shutdown must quiesce active/in-flight/pending
  work, the restored runtime must complete another collection and optimizer
  update, and metrics must remain finite.

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
  `22df6a299a7eddfe6f5562067ac00bd8fd861828`.
- H7.1 decision: required.
- Release candidate: NONE.
- Release status: NOT READY.
- Playing strength: NOT MEASURED.

This PR does not implement an H7.1 runtime topology and does not perform
playing-strength evaluation. It adds only the identity-bound formal deal-seed
mode needed to compare existing runtime paths on the same deals.

## Matched results

The matrix ran in Docker on one NVIDIA GeForce RTX 5070 with driver
595.71.05, PyTorch 2.12.1+cu132, and CUDA 13.2. The immutable protocol hash is
`7ed28fb9e3df98b31207d93eaad4551ba787a2ba090f1d61dd2a7c03c195bf73`;
the image digest is
`sha256:82baec25a5b85f24d6588c306c6b4a0c9c1edf5da04cbf8b989e424b2622f844`.
Every run used a 30-second warmup, a checkpoint-enabled 300-second measured
window, and seeds 101, 202, and 303.

| Topology | Learner samples/s (three runs) | Median | Median optimizer steps/s | Median CPU RSS |
| --- | --- | ---: | ---: | ---: |
| Base single process | 55.140, 49.100, 45.823 | 49.100 | 1.534 | 4.97 GiB |
| Base async 4x4 | 78.818, 47.766, 70.495 | 70.495 | 2.203 | 7.98 GiB |
| Base async 8x4 | 69.498, 63.603, 75.420 | 69.498 | 2.172 | 10.22 GiB |
| Full hybrid single process | 6.876, 13.608, 19.645 | 13.608 | 0.599 | 2.39 GiB |

Async CPU RSS is the aggregate of the parent process and every live Actor
worker sampled before shutdown, not just the parent. The full-hybrid/base-
single median learner-throughput ratio is 0.27716, far below the frozen 0.70
gate. All twelve checkpoints saved and strictly reloaded, all processes
quiesced, and every restored runtime completed another collection and
optimizer update. All runs observed an update across the complete learner
parameter graph. The full-hybrid measured windows produced 75, 180, and 279
optimizer updates entirely in the guided phase. Their cumulative counters,
including warmup, recorded 221, 101, and 25 skipped oversized cooperation
episodes because enabled
cooperation requires episode-atomic batches and the frozen batch size is 32.
Seed 303 also produced one episode where a farmer had no non-forced decision;
the runner explicitly counted and skipped that incomplete cooperation pair
without adding forced-action inference or replay. The runtime stability checks
pass, but the efficiency gate fails decisively.

Diagnostic medians reinforce the end-to-end result: base single process
reached 6.137 games/s and 391.8 decisions/s, while full hybrid reached 0.900
games/s and 47.3 decisions/s. Full-hybrid collection occupied much of the
measurement window; median synchronized segment totals were 26.9 seconds for
nested public inference, 13.0 seconds for exact belief DP, 1.87 seconds for
cooperation trajectory assembly, 34.0 seconds for Oracle/public learner work,
5.75 seconds for strategy features, and 167.5 seconds for collate and episode
preparation. SegmentProfiler
synchronizes CUDA at every segment boundary. These semantic components may
still be nested, so their timings are not summed to derive throughput.

## Decision

P3 selects H7.1. The complete hybrid reaches only 27.72% of base
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
