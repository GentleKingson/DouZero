# P2 low-cost pilot protocol

P2 runs the six frozen legacy card-play V3 configurations from
`configs/v3_formal/` without changing their model, feature, loss, seed, or
budget identities. The executable adapter feeds the existing H3-H6 learner;
it is not a second trainer.

## Frozen execution

- Variants: `v3_role`, `v3_admc`, `v3_oracle`, `v3_belief`,
  `v3_farmer_cooperation`, and `v3_full_hybrid`.
- Training seed: `101`; evaluation seed: `41001`; deal-set seed: `51001`.
- Device/topology: one CUDA device, single process, batch size 32.
- Per-variant ceiling: 3,600 seconds, 1,000,000 samples, or 10,000 eligible
  optimizer steps, whichever occurs first.
- Checkpointing is enabled. Each variant must receive a real SIGTERM, load the
  same strict H6 checkpoint in a fresh container, and perform another update.
- Training decisions come from the public V3 policy and environment legal
  action list. Oracle and belief labels are captured in separate training-only
  sidecars. Farmer cooperation remains episode-atomic.

The first pass is legacy card-play. Standard full-game follows only after the
card-play paths are stable. Search is disabled during training and is evaluated
as a wrapper around the same checkpoint.

## Evidence boundary

Raw run summaries, commands, environment records, checkpoint hashes, and
checksums live under the external P2 evidence directory on the test machine.
The repository stores only the compact cross-run decision summary in
`artifacts/v3-pilot/summary.json` and the human-readable report in
`docs/v3_hybrid/pilot_report.md`.

Pilot comparisons are diagnostic. They are not promotion evidence and cannot
produce a release candidate. Until P4 promotion evaluation is complete:

- Release candidate: NONE
- Release status: NOT READY
- Playing strength: NOT MEASURED
