# H7.1c Farmer Cooperation Async Runtime

## Scope

H7.1c extends the existing H7 shared-slot runtime with the H5 farmer
cooperation training path. It does not add a request protocol, policy graph, or
cooperation algorithm. Actors still request actions from the public V3 policy;
the learner alone owns the farmer team-value heads, trajectory encoder, and
optional public monotonic mixer.

The supported H7.1c cell is legacy card play with farmer cooperation enabled.
Belief, Oracle, strategy/style, bidding, league, curriculum, DDP, privileged
mixer state, and combinations of H7.1 sidecars fail before CUDA or worker
startup.

## Public Boundary

For each observed farmer decision, the actor derives:

- the selected environment legal-action index;
- the public H5 feature row;
- pass status;
- trace index;
- served policy and teammate-policy provenance; and
- a SHA-256 binding to the exact public model input tensors.

The sidecar contains no hidden hand, privileged state, mixer parameter, or
complete trajectory. Public replay remains a `V3ReplayTransition` and does not
serialize the sidecar. Landlord decisions never enter cooperation assembly.

## Episode Assembly

The learner aligns public rows and sidecars by
`(actor_id, episode_id, trace_index)`. Actor completion events declare the
number of recorded decisions for each farmer. The learner waits for the exact
declared counts, orders each farmer independently by trace index, and then
constructs one unequal-length up/down pair.

An episode is eligible only when:

- both farmers have at least one recorded decision;
- every public row has exactly one source-bound sidecar;
- episode/deal, role, action, return, and policy provenance agree;
- no transition is duplicated; and
- the complete farmer pair fits the learner batch limit.

Zero-decision farmer episodes are counted as incomplete and skipped.
Over-limit pairs are counted as oversized and skipped. They are never
truncated, padded into the loss, or split across updates.

## Replay And Resume

Complete farmer episodes are stored as bounded, episode-atomic learner replay.
An optimizer step samples one complete pair and supplies both the public rows
and `V3H5FarmerTrajectory` objects to the existing H6/H5 learner.

Only learner state, optimizer state, schedules, counters, RNG state, runtime
identity, and cumulative cooperation statistics are checkpointed. As with the
existing H7 replay contract, replay is cleared after every long-running cycle.
Before checkpoint publication, all inference slots and public/sidecar
alignment state must be quiescent. Thus no partial farmer trajectory crosses a
resume boundary.

The runtime identity binds:

- `v3-hybrid-h7-1c-runtime-v9`;
- `v3-hybrid-h7-runtime-checkpoint-v6`;
- the public cooperation request/replay protocols;
- sidecar and episode capacities;
- the H5/H6 training identity; and
- the support-matrix version.

Older H7 checkpoints and mismatched cooperation identities fail closed.

## Metrics

H7.1c reports cooperation decision samples, complete episodes, optimizer
steps, incomplete and oversized episode counts, episode replay occupancy, and
training-only cooperation parameter VRAM. Frozen benchmark schema v5 requires
positive cooperation sample, episode, optimizer, and parameter metrics when
the protocol enables cooperation.

## Status

- Release candidate: NONE
- Release status: NOT READY
- Playing strength: NOT MEASURED

Runtime and throughput evidence cannot establish playing strength. Formal
multi-seed paired evaluation remains P4/H8b scope.
