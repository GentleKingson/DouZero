# DouZero V3 Hybrid Design

Status: H0 contract frozen; H1-H5 components and the H6 single-process Hybrid
integration contract are implemented. H7 adds the bounded base V3+ADMC async
runtime and public-only selective-search contract; formal topology benchmarks,
long-run evidence, and H8 evaluation remain promotion gates. Playing strength
not measured.

The repository audit and PR #28/#29/#30 file dispositions live in
[`docs/v3_hybrid_contract.md`](../v3_hybrid_contract.md). This document is the
implementation-facing module and interface contract.

## Capability matrix

| Capability | Current state | H1 use |
|---|---|---|
| Observation V2 and public/privileged containers | Complete with leakage tests | Reused unchanged; public only |
| Model V2 encoders and variable-action batching | Complete | Encoder and batching primitives reused |
| Belief exact constrained posterior | Component complete; joint/alternating paths exist; long-run and strength evidence missing | Disabled, no parameters |
| Offline privileged teacher/distillation | Component complete | P10 action alignment and cache identity reused by H3 |
| Human BC, strategy, style | V2 components adapted to the H6 public model/loss graph | Single-process only |
| League and curriculum | Components integrated with V2 | V3 runtime integration deferred to H7 and fails closed |
| Learned bidding | Existing separate public bid space adapted in H6 | Standard rules, single-process only |
| Async/long-running single-GPU trainer | V2 implementation exists; topology-specific gaps remain | Deferred to H7 |
| Paired evaluation and clustered bootstrap | Complete infrastructure | Reused at H8 |
| Public belief search | Budgeted component exists | Deferred to H7 |
| Release package | Strict V2 package exists | H1 supplies a strict V3 public sidecar, not a formal release package |
| Role residual V3 policy | H1 implemented | Integrated by H6 |
| Adaptive DMC, online Oracle, cooperation mixer | H2 Adaptive DMC, H3 Oracle, and H5 training-only sequential farmer cooperation implemented | H2, H3, H5 |
| Formal V3 long-run and playing strength | Missing | H8 |

## Data boundary

```text
environment rules engine
  -> ObservationV2
       -> PublicObservation
       -> ordered LegalActionBatch
  -> shared public encoders
  -> role adapter and role heads
  -> ranking index
  -> existing legal_actions[index]
```

Only the environment constructs legal actions. The model never creates,
repairs, or appends an action. `PrivilegedObservation`, hidden allocations,
teacher/Oracle state, and training labels are rejected from the H1 deployment
path and public checkpoint.

## H1 model graph

```text
public state/card fields -> shared StateEncoder ------+
public history + mask -> shared LSTM/Transformer -----+-> shared fusion
ordered legal action rows -> shared ActionEncoder ----+       |
                                                               +-> landlord adapter (2 blocks) -> landlord heads
                                                               +-> landlord_up adapter (4 blocks, optional gate) -> landlord_up heads
                                                               +-> landlord_down adapter (4 blocks, optional gate) -> landlord_down heads
```

State and history are encoded once per decision. Only action encoding, shared
fusion, the selected physical-role adapter, and that role's heads run per
candidate. The default history encoder is a lightweight LSTM. Transformer is
an identity-bound ablation. H1 supports no attention inside role adapters.

All three physical roles own independent adapter and head parameters. This is
stricter than merely sharing one farmer adapter and preserves the positional
difference between `landlord_up` and `landlord_down`. The optional farmer
channel gate is action-local, so permuting action rows permutes outputs without
cross-row reordering effects.

## Output semantics

Each real action row produces:

- `dmc_q`: independent scalar Monte Carlo return from the acting team view;
  target transform is `raw` or `signed_log` and is checkpoint identity.
- `win_logit` and `p_win`: acting-team win probability.
- `score_if_win`: conditional acting-team signed final score on wins.
- `score_if_loss`: conditional acting-team signed final score on losses.
- `score_mean`: derived conditional-score mixture; not an independent head.

H1 adds no loss implementation. The H0 loss ordering remains frozen for H2-H6:

```text
lambda_dmc * L_admc + lambda_win * L_win + lambda_score * L_score
+ lambda_oracle * L_oracle + lambda_belief * L_belief
+ lambda_coop * L_cooperation + lambda_bc * L_bc
+ lambda_strategy * L_strategy + lambda_bidding * L_bidding
```

All schedule, valid-sample normalization, and role-weight accounting work is
deferred to its owning stage rather than represented by an H1 placeholder.

## H2 Adaptive DMC contract

H2 adds a standalone V3-only selected-action learner. It does not route V3
through the Legacy or V2 trainers and does not implement the H7 actor/runtime
topology. Each adaptive replay row binds the public tensor bundle, selected
real legal-action index, acting role, episode/deal identity, raw terminal MC
return, target transform, complete ruleset id/version/hash, and immutable
`PolicyLease` provenance (`q_old`, policy version, slot, owner, and generation).
Ordinary DMC replay has no `q_old` dependency. Schema, ruleset, and protocol
version mismatches fail closed at replay-buffer and learner admission; a
serialized buffer also rejects record counts above its declared capacity and
validates exact schema-derived card, flat, history, mask, and action shapes.

The learner supports three exclusive modes:

- `disabled`: ordinary selected-action `MSE(q_new, transformed_mc_return)`;
- `paper_ratio`: clamp `q_new / q_old` to `[1-gamma, 1+gamma]`, multiply by
  sign-preserving `q_old`, then clamp to the representable target range; exact
  zero `q_old` uses a branch-safe ordinary-prediction fallback with finite
  gradients;
- `safe_hybrid`: use the ratio path when `abs(q_old) >= epsilon`, otherwise
  clamp the additive change `q_new - q_old` to `[-delta, delta]`.

Gamma follows a learner-update linear schedule. Mode, gamma endpoints and
duration, epsilon, delta, target transform/clamp, optimizer settings, role
weights, replay protocol, and reduction semantics are compatibility identity.
Only gathered real actions enter the loss. Role weights are applied once and
the weighted loss is normalized by their effective sum; per-role sample,
weight, and loss metrics make accidental double weighting observable. A batch
whose roles all have configured weight zero is an exact learner no-op so role
ablations can resample without advancing optimizer, policy, or schedule state.

H2 trainer checkpoints strictly persist model, optimizer, learner update,
clip schedule, policy version, cumulative finite statistics, and Python,
NumPy, Torch, and CUDA RNG states. Resume rejects partial envelopes, changed
model/learner/ruleset/schema identity, stale source SHA, replay protocol drift,
or counter/schedule/statistic disagreement. Cumulative statistics additionally
validate role totals, configured role weights, adaptive-mode-only fields, and
event-count bounds before any state is restored. Optimizer parameter layout and
all identity-bound RMSprop hyperparameters must exactly match the configured
optimizer. A public model carrying a checkpoint ruleset binding cannot be
attached to a learner for another ruleset. Replay is explicitly flushed at a
checkpoint boundary; H7 owns persistent actor/replay runtime integration.

The frozen H1 public sidecar remains unchanged. A trained H2 model is released
through that public-only sidecar, which excludes optimizer state, `q_old`,
statistics, and all later-stage training data. The H2 training identity wraps
the H1 public identity with `oadmcdou-ratio-safe-hybrid-v1`; it does not mutate
or relabel existing H1 checkpoints.

## H3 online Oracle contract

H3 keeps privileged code under `douzero.v3_hybrid.training`. The public
package, model, exporter, loader, DeepAgent, and search import graph remain
unchanged and do not import the Oracle or `PrivilegedObservation`. Importing
the disabled H3 learner configuration also has no privileged import side
effect. An enabled learner lazily creates a separate `V3PrivilegedOracle`; it
owns its own V3 public-shaped backbone plus a gated hidden-hand residual branch.
It may copy the student's initial state, but never shares parameters or a
deployment forward API with the student.

Online samples reuse P10 `OfflineDistillationSample`, canonical action keys,
`align_teacher_output`, and `TeacherCacheIdentity`. H3 adds no parallel
privileged serialization or alignment format. Public replay remains the H2
`V3ReplayTransition`; hidden hands and labels are held separately and required
only during Oracle warmup and guided updates. Each Oracle sample must exactly
match replay public tensors, selected real-action index, acting role,
action-key count, and terminal target.

The learner-update schedule has three exact states:

1. `oracle_warmup`: only the Oracle optimizer advances; student parameters and
   public policy version remain unchanged.
2. `guided`: public DMC/Adaptive-DMC and Oracle value loss train together.
   Temperature KL, top-k ranking, and chosen-action value distillation are
   independently weighted. Oracle value weight, guidance, temperature, and the
   privileged gate anneal by learner update.
3. `public_finetune`: guidance, Oracle loss, and privileged gate are exactly
   zero. Privileged samples are rejected and only public DMC/Adaptive-DMC
   continues for exactly `finetune_updates`. The next learner tick enters an
   immutable `complete` phase and rejects further training batches.

A one-update guided phase applies the configured start weights on its sole
update, then crosses directly to the zero-privilege public-finetune boundary.
Guidance removes batch padding before aligning the student's real legal
actions with the Oracle action keys.

An enabled schedule advances on every admitted positive-role-weight learner
batch, including a deliberately zero-loss boundary tick. This prevents a
zero-weight warmup or fully annealed guided tail from trapping resume at one
update forever. Such ticks consume real samples and are checkpointed, but do
not advance the public policy version or either optimizer. Adaptive replay
provenance may remain attached during Oracle-only phases even though `q_old`
is not consumed; it becomes mandatory again when Adaptive-DMC is active.

Role weights apply once and all losses normalize over real decisions, never
padded action rows. H3 checkpoints bind the public H2 identity, Oracle graph,
loss weights, phase lengths, annealing values, optimizer configurations,
replay semantics, and normalization. They persist student and Oracle states
separately, both optimizers, learner update, policy version, phase state,
cumulative metrics, and Python/NumPy/Torch/CUDA RNG. H3 trainer checkpoints
are privileged training artifacts and the public loader rejects them. Public
export continues to use the H1 sidecar and serializes only the student.
Resume also requires public policy version to equal its configured initial
version plus the persisted number of public optimizer updates.

H3 reports action agreement, chosen-action value error, KL, ranking/value
losses, optimizer gradient norms, phase, temperature, gate, and role weights.
Throughput, VRAM, and playing strength require real CUDA runs and paired
evaluation of the exported public checkpoint. Playing strength not measured.

## H4 conservative belief contract

H4 extends the existing `douzero.belief.BeliefModel`; it does not introduce a
second belief representation. Raw `[B, 15, 5]` rank-count logits receive
privileged true-hand targets only inside the training namespace. The existing
exact constrained dynamic program remains authoritative for marginals, MAP,
and sampling. Every posterior satisfies the unseen-pool per-rank caps, joker
caps, opponent-A total, and opponent-B subtraction constraints.

Policy feedback uses the existing 48-field constrained-posterior layout. The
posterior is converted through the exact evaluation DP and detached before it
enters the V3 role adapters; policy loss therefore cannot backpropagate through
the DP or into the belief model. `belief_feedback` is identity-bound as
`none`, `farmers`, or `all_roles`; `none` creates no projection parameters.
The default feedback target is the two farmer roles. Landlord feedback is an
explicit ablation.

Belief supervision supports two optimizer schedules:

1. `auxiliary`: policy and supervised belief updates occur on each eligible
   batch. Optional shared state/history encoders receive only the supervised
   belief gradient, never a policy gradient through the posterior.
2. `alternating`: identity-bound counts select `policy`, `belief`, and optional
   `joint_shared_encoder` phases using the eligible-update counter. The counter,
   phase, both optimizers, public policy version, cumulative metrics, and RNG
   are restored exactly.

H4 reuses H2 public replay. `V3H4BeliefSample` is a separate training-side
binding of the same public tensors to an optional privileged label. A stable
source-state fingerprint covers both the policy bundle and belief input, so a
same-role sample swap fails closed before either feedback or supervision.
Labels and the fingerprint are never serialized into replay or public
checkpoints. Belief loss and role weights normalize over real decisions
exactly once. A delegated H3 no-op does not advance the H4 phase, sample, or
statistics clocks. Coupled public belief policies remain valid immutable
Adaptive-DMC snapshots, so `q_old` is captured from the exact public policy
that selected the action. H4 checkpoints validate belief optimizer parameter
groups and cross-check nested H3 updates, decisions, policy version, and
optimizer schedule against H4 phase history before restoring. Metrics include
masked CE, MAP rank/exact accuracy,
constrained-posterior calibration error, exact conservation, DP latency, and
separate belief/shared gradient norms.

The coupled H4 public checkpoint contains only the V3 student, public belief
model, strict configs, ruleset/schema identities, and public feedback contract.
The H1 loader rejects it, and the H4 loader rejects training checkpoints or
partial model pairs. Public imports lazily exclude belief labels, privileged
observations, Oracle code, and H4 trainer code. The standalone H4 identity
continues to reject Oracle plus belief feedback; the H6 wrapper admits that
combination under its distinct integration identity and atomic rollback.

| H4 combination | Status |
|---|---|
| Belief auxiliary only | Supported |
| Detached belief feedback to farmers | Supported, preferred |
| Detached belief feedback to all roles | Supported ablation |
| Independent belief encoder updates | Supported |
| Belief updates to shared V3 state/history encoders | Supported by explicit phase |
| Policy gradient through exact DP | Rejected |
| H3 Oracle plus H4 belief in one learner | Supported only through H6 |
| Bidding/BC/strategy/cooperation integration | Supported combinations are owned by H6 |

H4 playing strength not measured. Feedback is not a default until paired
farmer/landlord evaluation demonstrates benefit without role regression.

## Batching and masks

The scalar API accepts any positive action count. The batched API pads to the
largest action bucket and carries one authoritative boolean mask. Selection
and chosen-action gathering reject padded rows. Padded values do not enter H1
losses because H1 has no learner; future loss builders must gather a real
chosen index before reduction.

## Identity and checkpoint

`V3HybridModelConfig.stable_hash()` binds hidden width, history backend and
depth, history heads/dropout, shared fusion depth, all role depths, channel
gate type/reduction, adapter dropout, attention type, score/Q clamps, output
target transform, numerical guard semantics, and the H4 belief-feedback graph.

The public sidecar additionally binds the complete H0 compatibility identity:

- exact ruleset id/version/hash;
- frozen Observation V2 schema hash;
- feature flags;
- model graph and output semantics;
- explicit H1 `not_integrated` loss/optimizer/trainer sections;
- disabled belief and cooperation layouts;
- model-only H1 topology.

The loader requires the caller's expected schema, ruleset, and typed model
config. Unknown/missing envelope fields, wrong access class, V2/Legacy files,
config drift, ruleset drift, identity drift, extra/missing state keys, shape
drift, and forbidden teacher/Oracle names fail closed. Loading never trusts a
checkpoint to choose its own runtime configuration.

## Support matrix

| Combination | H1 status |
|---|---|
| Public Observation V2 card play | Supported |
| LSTM history | Supported, default |
| Transformer history | Supported ablation |
| Landlord/farmer role adapters | Supported |
| Farmer channel gate | Supported ablation |
| Raw/signed-log `dmc_q` identity | Supported; H1 does not train either |
| Padded centralized inference | Supported |
| Scalar arbitrary action count | Supported |
| Bidding, belief, Oracle, BC, strategy, style, cooperation | Rejected/not integrated |
| V2 or Legacy checkpoint migration | Rejected; distillation/conversion must create a new manifest later |

## Single-GPU topology candidates

H1 only makes the model batchable. H7 will select and validate one topology:

1. existing async actors plus centralized GPU inference;
2. colocated learner/inference with bounded microbatch queues;
3. learner-owned snapshot service with versioned actor requests.

Selection requires end-to-end games/decisions/transitions/learner-samples per
second, queue and transfer timing, memory, policy lag, restart/resume, and clean
shutdown evidence. The H1 model benchmark cannot establish a training-speed
claim.

## Ablation and promotion order

H1 model-only order: Model V2, V3 shared-only, V3 role adapters, then farmer
channel gate. Report parameter count, action buckets, forward/backward rate,
decision rate, and VRAM with at least three repetitions.

Cross-stage order remains H1 role model, H2 Adaptive DMC, H3 Oracle, H4 belief,
H5 cooperation, H6 auxiliaries, H7 runtime/search, H8 formal evaluation. A
failed correctness or performance gate stops later stages. Playing-strength
promotion requires paired evaluation and role-specific WP/ADP confidence
intervals; no H1 result is a strength claim.

### H5 sequential farmer cooperation contract

H5 does not apply synchronous QMIX to alternating Dou Dizhu actions. Replay
rows remain the H2 public rows and are grouped by exact episode/deal identity
into an ordered `landlord_up`/`landlord_down` pair. Every trajectory decision
atomically binds one trace index, replay transition, public-feature row, and
pass flag before the sequence is sorted by trace index. A trajectory may have
a different number of decisions and keeps its league policy and
teammate-policy identities. False padded rows are zeroed and excluded from
every loss and statistic.

The public V3 policy retains its three independent role-specific local DMC Q
heads. A training-only sidecar consumes the selected public role-adapted action
embedding, detached conservative-belief teammate summary, and the existing
public strategy quantities for pressure, initiative, teammate feed, and bomb
opportunity cost. It contains role-specific farmer team-value heads and a
shared GRU trajectory encoder. The per-decision team-value loss and the two
trajectory terminal-return/consistency losses start from the raw farmer-team
terminal return, then apply the public DMC head's identity-bound target
transform and clamp. Landlord samples never enter these losses.

The optional mixer first reduces each unequal trajectory to its masked mean
selected local Q. Two non-negative `softplus` weights and a bias then predict
the common farmer return. Weights may be conditioned on public trajectory
state, or on an explicit privileged training-only state for ablation. The
privileged state is a caller-owned loss side channel: it is not replayed and is
never passed to the public model. Mixer mode, dimensions, alignment, reward,
padding, optimizer, schedule, and loss weights are all compatibility identity
axes. H6 admits the previously rejected H3 Oracle/H4 belief/H5 cooperation
combination under a conditional identity and an outer atomic rollback
boundary; standalone H3-H5 identities remain unchanged for their old graphs.

When H5 is disabled, no team head, trajectory encoder, mixer, optimizer, or
data dependency is created and the learner delegates exactly to H4. Public H1
and H4 checkpoint writers serialize only the V3 model (and public belief model
where applicable), so the H5 sidecar and mixer cannot enter a deployment
artifact. H5 training checkpoints are separately marked `training_only` and
strictly bind the nested H4 state, sidecar graph, optimizer, counters, schedule,
statistics, and nested H3 RNG continuity. The actor-visible policy version is
the nested H3 version plus every H5 public optimizer step, so Adaptive DMC
provenance cannot lag behind parameters changed by the cooperation loss.

H5 currently provides the single-process reference topology and correctness
contract. Long-running async workers, SIGTERM manifests, bounded policy lag,
and selective search remain H7. No formal paired evaluation or multi-seed
wall-clock ablation has been run: playing strength not measured.

### H6 Hybrid integration contract

`douzero.v3_hybrid.support_matrix` is the stable machine-readable authority for
capability, ruleset, topology, checkpoint, export, deployment, and search
support. The dedicated `configs/v3_hybrid.yaml` is loaded only by
`douzero.v3_hybrid.integration_config`; Legacy A1 and V2 allowlists remain
unchanged. Validation runs before model construction, CUDA initialization,
checkpoint I/O, replay allocation, or worker creation. H6 supports the
single-process topology. V3 async, DDP, league, curriculum, and selective
search fail closed and remain H7 work. Human BC is restricted to the existing
legacy-rules validated dataset; learned bidding is restricted to the standard
ruleset and its separate public bid action space.

The H6 public graph conditionally reuses the existing V2 listwise prior,
strategy auxiliary, style encoder, and bidding heads. Disabled flags preserve
the H1-H5 parameter graph, config hash, serialized config, and checkpoint load
contract. Enabled public features are serialized by the versioned H6 public
replay format. Belief labels, Oracle samples, human labels, privileged mixer
state, and training trajectories remain separate sidecars and cannot enter
public replay or a public checkpoint.

`V3HybridLossComposer` owns the canonical nine-term ordering. Each enabled
term supplies an unreduced tensor, authoritative valid mask, physical role,
and unique sample identity. It applies role weights once, divides by real
effective weight rather than padding, records per-role counts, and advances a
term schedule only after the owning update succeeds. H3 DMC/Oracle, H4 belief,
and H5 cooperation remain component-owned optimizer phases and are reported as
externally applied terms; win, conditional score, BC, strategy, and bidding
share the H6 public auxiliary phase. A strict pre-update snapshot rolls the
whole nested learner, optimizers, counters, schedules, and policy version back
if a later phase fails or becomes non-finite. Checkpoints persist the nested
H5 artifact, composer state, H6 counters, resolved-config identity,
support-matrix hash, and the sum of H3, H5, and H6 public policy updates.
Externally applied DMC, Oracle, belief, and cooperation terms reject a second
H6 weight schedule because their owning H2-H5 component controls the
executable cadence. Pure BC and pure bidding batches bypass the card-play
optimizer phase rather than performing an empty optimizer step.

The public sidecar contains only public policy and optional public belief,
style, prior, strategy, and bidding modules. Oracle and cooperation/mixer
parameters, optimizer state, replay, and privileged labels remain
training-only. H6 has no formal wall-clock, multi-seed, or paired-strength
evidence: playing strength not measured; release candidate remains NONE and
release status NOT READY.

## Implementation order

| Stage | Packages and interfaces |
|---|---|
| H1 | `douzero.v3_hybrid.{config,layers,model,output,checkpoint,export}`; public Observation V2 and V2 tensor batching |
| H2 | new V3 replay transition and learner loss/metrics; checkpoint resume state; no V2 loss mutation |
| H3 | extend existing `douzero.distillation` alignment/cache safety into a training-only V3 Oracle service and schedules |
| H4 | extend `douzero.belief` joint/alternating interfaces and V3 public feature injection; preserve exact DP |
| H5 | `douzero.v3_hybrid.training.{cooperation,h5_learner}`; public action-embedding training API; sequential farmer pair alignment, sidecar checkpoint, masked losses |
| H6 | implemented in `support_matrix`, `integration_config`, `loss_composer`, `integration_replay`, optional public graph adapters, and `training.h6_learner`; runtime-only league/curriculum remain explicitly unsupported |
| H7 | extend existing async/long-running runtime, policy snapshots, checkpoint manifests, and search ranker |
| H8 | extend paired evaluation, ablation reports, release manifest/package, and rollback evidence |

### H7 runtime and selective-search contract

H7 reuses `douzero.training.async_single_gpu` rather than defining a second
request state machine. The existing shared observation slab now has an explicit
five-channel V2 or six-channel V3 output layout. The sixth V3 channel is the
selected-action `dmc_q`; the CPU actor records that value with the immutable
served snapshot version before terminal labels exist. Shared replay produces
public `V3ReplayTransition` rows and binds actor, episode, snapshot generation,
ruleset, target transform, and protocol identity. Legacy/V2 callers retain the
five-channel default.

`V3AsyncSingleGPUTrainer` currently supports only legacy-rules card play with
the public role model, Adaptive DMC, and public export. Oracle, joint belief,
cooperation, BC, strategy/style, league/curriculum, bidding, standard full-game,
DDP, and search-in-training fail before worker creation. The runtime publishes
at a quiescent game boundary, enforces a configured policy-lag bound, uses the
existing coarse action buckets and pinned staging, and exposes queue, collate,
H2D, forward, D2H, publish, replay-drain, learner, microbatch, lag, and
quiescence metrics. Its training checkpoint includes exact runtime/protocol
identity, learner state, optimizer/schedules, RNG, counters, snapshot version,
and long-running state; replay remains ephemeral at checkpoint boundaries.

`V3SelectiveSearch` wraps the existing belief sampler and endgame solver. Its
composite gate uses only `PublicObservation`, public model values, and a
conserved public belief posterior. Remaining cards, own cards, top-two Q gap,
bomb/rocket risk, spring risk, belief entropy, and card-control state can
trigger search. Disabled search, an incompatible package, global stop,
non-conserved belief, exhausted budgets, action-alignment drift, legal-action
drift, and search exceptions return the exact base action with a fresh
structured reason. Search never creates or filters legal actions. Metrics cover
trigger/fallback reasons, samples, nodes, rollouts, action changes, and
p50/p95/p99 added latency.

The production recommendation remains unchanged until three repeated,
checkpoint-enabled end-to-end measurements compare single-process, 4x4 async,
and 8x4 async under the same SHA/image/config/hardware identity. Centralized
inference is not promoted from a forward microbenchmark. Release candidate:
NONE. Release status: NOT READY. Playing strength not measured.
