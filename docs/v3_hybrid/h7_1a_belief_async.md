# H7.1a belief async contract

H7.1a extends the existing H7 shared-slot runtime. It does not introduce a
second request protocol, replay implementation, or public policy type.

## Data boundary

The actor derives `BeliefInput` from `ObservationV2.public` and writes it to a
public shared-memory slot beside the existing policy request. Central
inference evaluates a quiescently published pair of public policy and belief
model snapshots. The exact constrained posterior is detached before it enters
the public policy, matching the H4 training contract.

The actor builds the hidden-hand `BeliefLabel` only after the public request is
formed. That label travels through a separate bounded training-only queue,
keyed by `(actor_id, episode_id, trace_index)`. The public replay row continues
to use `V3ReplayTransition` and cannot serialize the label. The learner binds
the two paths with an actor-created public-state fingerprint and rejects
duplicates,
mismatches, overflow, or an unmatched quiescent boundary.

`q_old` is the selected-action value returned by the coupled public snapshot
that served the decision. When Adaptive DMC is disabled, the runtime retains
that provenance but removes it from the ordinary-DMC learner view so the
disabled loss remains an exact no-op.

## Snapshot and recovery

Public policy and belief weights are copied while the request coordinator is
quiescent, then one policy version is published. H7.1a supports the existing
H4 `auxiliary` phase with detached policy feedback. Other belief phase
semantics fail before CUDA initialization or worker spawn.

The outer H7 checkpoint identity binds the H7.1a request, replay, snapshot,
belief-model, and training semantics. The nested H6/H4 checkpoint restores the
belief optimizer, eligible-step counters, phase, RNG state, and public policy
version. Public replay and privileged label queues are deliberately not
checkpointed; checkpoint publication requires an empty alignment backlog.

## Support

H7.1a currently supports legacy card-play in `single_process` and
`async_single_gpu` topologies. It does not add Oracle, cooperation, strategy,
style, standard bidding, league, curriculum, or DDP support. Unsupported
combinations fail during configuration validation.

## Status

- Release candidate: NONE
- Release status: NOT READY
- Playing strength: NOT MEASURED

Matched topology benchmarks, fault evidence, SIGTERM/resume evidence, and the
two-hour soak are execution evidence for the current PR head. They do not
measure playing strength.
