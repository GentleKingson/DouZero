# H7.1 full-hybrid runtime decision

## Frozen gate

This record freezes the P3 decision before the first matched measurement.

- Matrix: base V3+ADMC single process, base V3+ADMC async 4x4, base
  V3+ADMC async 8x4, and full-hybrid single process.
- Repetitions: three, using frozen training seeds 101, 202, and 303.
- Warmup: 30 seconds per repetition.
- Measurement: 300 seconds per repetition.
- Measurement seed window: warmup keeps the learned parameters, optimizer, and
  schedule state, but its runtime and episode counters are discarded. Every
  measured topology restarts from formal episode ID 0, so a given seed maps the
  overlapping measured prefix to the same deals. The validator requires every
  raw `counters_before` field to be exactly zero.
- Runtime replacement: the warmup runtime is quiesced, shut down, dereferenced,
  and collected before the measured runtime is constructed. Periodic
  checkpoint cadence continues from the learner's global eligible-update
  counter even though the measured runtime statistics restart at zero.
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
  update, and metrics must remain finite. The full-hybrid checkpoint envelope
  binds the protocol, seed, measurement window, and next formal episode cursor;
  strict reload obtains the cursor only from that checkpoint.

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
  `66bb4d3325f40a5d51a1c3cb1f6631e7a3ec4d3d`.
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
`5feac521a1bee6f303d9aacd32f8c9ae9e2766c1173f102325f3784ca1dc4e70`;
the image digest is
`sha256:15401dcf1a0a55504bf82d9da01d4449b8ed5e2098a8f3ae96d5603b6cdf3266`.
Every run used a 30-second warmup, a checkpoint-enabled 300-second measured
window, and seeds 101, 202, and 303. All twelve measured windows started with
games, decisions, transitions, learner samples, and optimizer steps at zero;
the first measured deal in every topology was formal episode 0.

| Topology | Learner samples/s (three runs) | Median | Median optimizer steps/s | Median CPU RSS |
| --- | --- | ---: | ---: | ---: |
| Base single process | 48.608, 49.942, 52.984 | 49.942 | 1.561 | 4.74 GiB |
| Base async 4x4 | 69.681, 68.431, 66.877 | 68.431 | 2.138 | 7.49 GiB |
| Base async 8x4 | 67.627, 66.630, 70.505 | 67.627 | 2.113 | 10.10 GiB |
| Full hybrid single process | 6.326, 12.805, 18.634 | 12.805 | 0.560 | 2.38 GiB |

Async CPU RSS is the aggregate of the parent process and every live Actor
worker sampled before shutdown, not just the parent. The full-hybrid/base-
single median learner-throughput ratio is 0.25639, far below the frozen 0.70
gate. All twelve checkpoints saved and strictly reloaded, all processes
quiesced, and every restored runtime completed another collection and
optimizer update. All runs observed an update across the complete learner
parameter graph. The three full-hybrid checkpoints stored formal episode
cursors 292, 271, and 261 in the strict
`douzero-v3-p3-full-runtime-checkpoint-v1` envelope and restored those cursors
without process-local state. The full-hybrid measured windows produced 69,
168, and 252 optimizer updates entirely in the guided phase, starting at
learner updates
10,003, 10,002, and 10,011 respectively. The same windows recorded 223, 103,
and 8 skipped oversized cooperation episodes because enabled cooperation
requires episode-atomic batches and the frozen batch size is 32.
Seed 303 also produced one episode where a farmer had no non-forced decision;
the runner explicitly counted and skipped that incomplete cooperation pair
without adding forced-action inference or replay. The runtime stability checks
pass, but the efficiency gate fails decisively.

Diagnostic medians reinforce the end-to-end result: base single process
reached 6.243 games/s and 399.8 decisions/s, while full hybrid reached 0.903
games/s and 48.5 decisions/s. Full-hybrid collection occupied much of the
measurement window; median synchronized segment totals were 31.7 seconds for
nested public inference, 13.2 seconds for exact belief DP, 2.22 seconds for
cooperation trajectory assembly, 32.3 seconds for Oracle/public learner work,
7.11 seconds for strategy features, and 183.4 seconds for collate and episode
preparation. `SegmentProfiler`
synchronizes CUDA at every segment boundary. These semantic components may
still be nested, so their timings are not summed to derive throughput.

## Decision

P3 selects H7.1. The complete hybrid reaches only 25.64% of base
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

## Reproducing the source checkout

The generated evidence commit necessarily follows the executable commit, so
the protocol binds the runner source rather than claiming that the artifact
commit itself executed. The executable remains an ancestor of this PR's
read-only GitHub pull ref. A clean checkout does not depend on the feature
branch surviving:

```bash
git fetch origin refs/pull/43/head
git merge-base --is-ancestor \
  66bb4d3325f40a5d51a1c3cb1f6631e7a3ec4d3d FETCH_HEAD
git worktree add --detach /tmp/douzero-p3-evidence-source \
  66bb4d3325f40a5d51a1c3cb1f6631e7a3ec4d3d
```

From that worktree, build with
`DOUZERO_GIT_SHA=66bb4d3325f40a5d51a1c3cb1f6631e7a3ec4d3d`.
The resulting source tree must equal
`1de97c3df839a53718f0e7283ca251420f840f9e`, and the formal runner fails
closed unless the Git SHA, source tree, image digest, hardware, config, model,
trainer, replay, protocol, and measurement-window identities all match.
