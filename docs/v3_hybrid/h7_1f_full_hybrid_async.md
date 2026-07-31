# H7.1f Full Hybrid Async

H7.1f composes the previously isolated H7.1a-e transports on the existing
bounded single-GPU coordinator. It does not introduce a second request protocol
or change any model, loss, rules, or environment semantics.

## Supported Bundles

Only two combined bundles are supported:

- legacy: belief, Oracle, farmer cooperation, strategy/style, and public export;
- standard: the legacy bundle plus the separate standard bidding action space.

Individual H7.1a-e transports remain supported. Partial multi-capability
combinations fail before CUDA initialization, worker creation, replay
allocation, or checkpoint I/O.

Both bundles use a belief-coupled game-boundary snapshot. `q_old`, public belief
feedback, strategy/style features, and the selected action therefore identify
the same immutable served snapshot.

## Replay Boundary

Card-play replay remains public. Belief and Oracle labels use separate
privileged queues keyed by actor, episode, and trace index. Farmer cooperation
uses its public decision sidecar and episode-atomic alignment. Strategy targets
remain public. Standard bidding retains a separate action space and replay.

The learner publishes a cooperation episode only after every card-play row has
its belief sample, Oracle sample, and strategy target. Incomplete or oversized
farmer pairs discard the corresponding auxiliary records. Capacity limits,
duplicate keys, identity mismatches, and non-quiescent checkpoint boundaries
fail closed.

Public snapshots contain the student and public belief model only. Oracle
parameters, cooperation state, optimizer state, privileged labels, and replay
are not part of public export.

## Identity

Legacy and standard full-hybrid bundles have distinct request and replay
protocols. The runtime and support-matrix versions are compatibility axes.
Existing H7.1a-e checkpoints cannot load under the H7.1f runtime identity.

## Status

- Release candidate: NONE
- Release status: NOT READY
- Playing strength: NOT MEASURED

CUDA parameter update, SIGTERM/resume, two-hour soak, and three-repeat matched
benchmarks are required before this stage can be recommended for formal
training.
