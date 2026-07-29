# H7.1b Oracle async contract

H7.1b extends the existing H7 shared-slot runtime. It does not add a second
inference request protocol or replay implementation.

## Data boundary

Actors run only the public student. After a public decision has selected one
of the rule engine's legal actions, the actor captures the training-only
`PrivilegedObservation` and canonical legal-action order. It does not import,
construct, or forward the privileged Oracle model.

The privileged data travels through a separate bounded queue keyed by
`(actor_id, episode_id, trace_index)`. The public `V3ReplayTransition` remains
unchanged and cannot serialize hidden hands. At the learner, a SHA-256 binding
over the public tensor bundle verifies the source state, role, chosen action,
and action alignment before the terminal return is attached. Duplicate,
mismatched, over-capacity, and unmatched sidecars fail closed.

## Training and publication

The existing H3 learner owns the Oracle model, optimizer, and
warmup/guided/public-only schedule. Schedule progression is based on eligible
learner updates and the nested H6 checkpoint restores it without restarting.
Oracle-only warmup updates do not advance the served policy version. Guided
student updates do advance it.

Snapshot publication copies only the public student. The Oracle never enters
the inference model, actor process, public checkpoint, exporter, or deployment
graph. Runtime evidence reports Oracle samples and optimizer steps per second,
plus the Oracle parameter VRAM footprint separately from total peak VRAM.

## Recovery and support

The outer H7 identity binds the H7.1b request and replay protocols and the
nested H6/H3 training identity. Replay and sidecar queues are intentionally
flushed at checkpoint boundaries; unmatched sidecars prohibit publication.

H7.1b supports legacy card-play in `single_process` and `async_single_gpu`.
Combined belief plus Oracle transport, cooperation, public auxiliaries,
standard bidding, league, curriculum, and DDP remain unsupported here and
fail before CUDA initialization or worker spawn.

## Status

- Release candidate: NONE
- Release status: NOT READY
- Playing strength: NOT MEASURED

Runtime validation does not constitute a playing-strength result.
