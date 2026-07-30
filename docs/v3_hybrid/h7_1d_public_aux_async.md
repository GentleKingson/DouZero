# H7.1d Public Auxiliary Async Contract

H7.1d extends the existing bounded H7 shared-slot protocol for the public
strategy and style graph. It does not add a second request protocol and it
does not enable any privileged actor input.

## Supported Scope

- topology: `single_process` and `async_single_gpu`
- ruleset: legacy card play
- graph: V3 role model, Adaptive DMC, strategy features/heads, and style
- request protocol: `v2-shared-slots-v3-dmc-q-public-strategy-style-v1`
- replay protocol:
  `v3-public-replay-plus-public-strategy-trajectory-labels-v1`
- committed config: `configs/v3_hybrid_h7_1d_public_aux.yaml`

Belief, Oracle, cooperation, bidding, league, curriculum, BC, and standard
rules remain fail-closed in this isolated transport stage. H7.1e owns
standard bidding. A later integration stage must define the combined runtime
identity before any H7.1 transports can be enabled together.

## Public Data Path

The actor derives strategy and style tensors from `PublicObservation`. The
same `ModelInputBundle` instance is written to the inference slot, retained
with the selected action, and written to public replay after terminal
labelling. This binds the feature cache to the immutable policy snapshot that
produced `q_old`.

Strategy labels are derived from the public action trajectory and terminal
result. They travel in the existing terminal-labelled replay slab. Style uses
only other-player public action history. Neither path reads or serializes true
hidden hands.

The learner samples strategy labels with the same replay indices as their
public rows. Replay eviction uses equal-capacity deques and fails closed if
their lengths ever diverge.

## Identity And Recovery

The runtime version, checkpoint format, request protocol, replay protocol,
support matrix, model feature layout hashes, strategy solver budgets, style
layout, loss weight, and runtime flag all participate in stable identities.
Checkpoints store cumulative strategy label and optimizer counters. A base,
H7.1a, H7.1b, or H7.1c checkpoint is rejected rather than partially loaded.

## Performance Evidence

Benchmark schema v6 records strategy sample/update rates and public auxiliary
parameter memory in addition to existing microbatch, queue, transfer,
forward, learner, lag, and shutdown metrics. The protocol freezer and runner
accept the committed H7.1d config and still require three matched repeats for
each frozen topology.

Release candidate: NONE

Release status: NOT READY

Playing strength: NOT MEASURED
