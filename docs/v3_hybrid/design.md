# DouZero V3 Hybrid Design

Status: H0 contract frozen; H1 public card-play model and H2 Adaptive DMC
learner components implemented. H3-H8 are not implemented. Playing strength
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
| Offline privileged teacher/distillation | Component complete | Disabled, no import or parameters |
| Human BC, strategy, style, league, curriculum | Components integrated with V2 | Deferred to H6 |
| Learned bidding | V2 component complete with distributed limitations | H1 fails closed |
| Async/long-running single-GPU trainer | V2 implementation exists; topology-specific gaps remain | Deferred to H7 |
| Paired evaluation and clustered bootstrap | Complete infrastructure | Reused at H8 |
| Public belief search | Budgeted component exists | Deferred to H7 |
| Release package | Strict V2 package exists | H1 supplies a strict V3 public sidecar, not a formal release package |
| Role residual V3 policy | H1 implemented | Current stage |
| Adaptive DMC, online Oracle, cooperation mixer | H2 Adaptive DMC implemented; Oracle/cooperation absent | H2, H3, H5 |
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
serialized buffer also rejects record counts above its declared capacity.

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
weight, and loss metrics make accidental double weighting observable.

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
target transform, and numerical guard semantics.

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

## Implementation order

| Stage | Packages and interfaces |
|---|---|
| H1 | `douzero.v3_hybrid.{config,layers,model,output,checkpoint,export}`; public Observation V2 and V2 tensor batching |
| H2 | new V3 replay transition and learner loss/metrics; checkpoint resume state; no V2 loss mutation |
| H3 | extend existing `douzero.distillation` alignment/cache safety into a training-only V3 Oracle service and schedules |
| H4 | extend `douzero.belief` joint/alternating interfaces and V3 public feature injection; preserve exact DP |
| H5 | V3 farmer team mixer, counterfactual credit labels, and masked cooperation loss |
| H6 | adapters for existing BC, strategy, style, league, curriculum, and bidding contracts into one V3 trainer |
| H7 | extend existing async/long-running runtime, policy snapshots, checkpoint manifests, and search ranker |
| H8 | extend paired evaluation, ablation reports, release manifest/package, and rollback evidence |
