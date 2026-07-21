# DouZero V3 Hybrid H0 Contract

Status: H0 contract only. No V3 model, trainer, checkpoint, or deployment path
is enabled by this change. Legacy A1 and V2 defaults remain unchanged.

Playing strength not measured.

## Baseline and repository audit

- Audit date: 2026-07-21.
- Repository: `GentleKingson/DouZero`.
- Base branch: latest `main` after `git pull --ff-only origin main`.
- Exact base SHA: `388870f9a4c40a093fd86fe0d8de2da821e903f6`.
- H0 branch: `codex/v3-h0-contract`, created directly from that SHA.
- Open draft PRs at audit time: #27, #28, #29, and #30.
- No open PR branch is a development base. No PR #30 commit or file is
  cherry-picked into H0.

The audit found that the merged repository already has substantial reusable
V2 infrastructure. V3 extends it rather than creating another parallel stack.

| Area | Existing main capability | V3 decision |
|---|---|---|
| Observation | Versioned `ObservationV2`, immutable `PublicObservation`, separate `PrivilegedObservation`, hidden-reallocation leakage tests | Reuse Observation V2 unchanged; deployment accepts only public input |
| Model | Shared card/state/history/action encoders, role embedding, win and conditional-score heads, optional belief/prior/strategy/style/bidding paths | Extend the V2 concepts with landlord/farmer residual adapters and an independent scalar DMC Q head |
| Belief | Exact card-conserving rank-count DP, public-only inference, frozen/joint/alternating training, strict coupled checkpoints | Reuse the conservative layout; add optional public-posterior policy feedback behind an identity-bound flag |
| Teacher | Offline privileged teacher and public-student distillation | Add online training-only Oracle guiding; do not place teacher modules in deployment packages |
| Auxiliaries | Human BC, strategy auxiliary, style, league, curriculum, learned bidding | Integrate through existing contracts in H6; each remains independently disableable |
| Runtime | Single-process and async single-GPU V2, long-running cycles, atomic latest manifest, strict resume and policy snapshots | Extend the existing runtime in H7; bind topology and bounded policy-lag semantics in the V3 identity |
| Evaluation | Paired deals, role metrics, deal-clustered bootstrap, ablation framework, bidding and search metrics | Reuse and raise development/formal sample gates to 20,000/100,000 paired deals |
| Search | Public belief sampling, environment-owned legal generation, node/rollout/time budgets | Reuse only as an optional H7 ranker over existing legal actions |
| Release | Strict public-policy package, checksums, model construction, public/belief identity, rollback tooling | Extend format under a V3 ABI; public package must exclude Oracle, labels, and true hands |

Known merged limitations remain real: standard learned-bidding DDP,
joint/alternating belief DDP, and distributed trainer checkpoint/resume fail
closed; formal playing-strength and target-GPU evidence do not exist.

## Draft PR audit

### PR #27

PR #27 is an umbrella evidence branch explicitly marked do-not-merge. It mixes
Legacy C0, Legacy A1, and the isolated `gpu_v3` prototype. It is retained only
as historical evidence and is not a V3 base.

### PR #28

PR #28 contains Legacy A1 profiling/runtime experiments. Its own evidence
reports a median 14,242.937 frames/s against 15,475.522 frames/s on `main`, a
7.97% regression. It remains Draft and does not clear its promotion gate. H0
does not import its adaptive split-dense or profiling changes. A later runtime
phase may compare individual techniques only after re-basing and revalidation.

### PR #29

PR #29 contains the opt-in Legacy C0 centralized-inference closeout. Its
checkpoint/resume and bounded-lag evidence is useful, but its sustained Legacy
throughput remains about half of A1. It is not the V3 runtime topology and no
code is imported. H7 may reuse test ideas for correlated request shutdown and
failure propagation after a new review against then-current `main`.

### PR #30, file-by-file disposition

PR #30 is a component prototype, not a runnable actor/learner path. Every file
was inspected from the fetched `origin/pr-30` ref.

| File | H0 disposition |
|---|---|
| `benchmarks/bench_gpu_v3.py` | Do not copy. Synthetic model-forward timing is not end-to-end training evidence; reconsider CUDA-event mechanics in H1/H7 |
| `benchmarks/configs/gpu_v3_distill_legacy_teacher.yaml` | Reject. It is outside the typed config schema and targets Legacy factorized inputs |
| `benchmarks/configs/gpu_v3_independent_role_dual_tower.yaml` | Reject architecture; V3 requires shared encoders with residual role-family adapters |
| `benchmarks/configs/gpu_v3_shared_trunk_role_heads.yaml` | Reimplement conceptually in H1 under the frozen V3 identity, not as a second model name |
| `configs/gpu_v3.yaml` | Reject. It advertises a disabled-checkpoint non-runnable config and broadens version parsing before integration |
| `docs/benchmarks/gpu_v3.md` | Evidence remains prototype-only; parameter/forward results make no training or strength claim |
| `douzero/config/loader.py` | Reject the one-line global `gpu_v3` allowlist expansion; H0 stays fail closed |
| `douzero/dmc/arguments.py` | Reject the Legacy CLI version choice; V3 will use its own validated path when runnable |
| `douzero/gpu_v3/__init__.py` | Reject overlapping package/model name |
| `douzero/gpu_v3/checkpoint.py` | Reimplement later. It binds a small architecture hash but omits schedules, RNG, policy version, adaptive statistics, topology, belief layout, and mixer identity |
| `douzero/gpu_v3/config.py` | Reject dual architecture ambiguity and incomplete compatibility axes |
| `douzero/gpu_v3/distillation.py` | Do not copy. Action-count-stable KL is worth retesting in H3, but the input is a padded Legacy tensor rather than Observation V2 |
| `douzero/gpu_v3/identity.py` | Superseded by the sole canonical name `v3_hybrid` |
| `douzero/gpu_v3/models.py` | Do not copy. It has only scalar values, lacks V2 heads/encoders, and uses whole role heads rather than residual landlord/farmer adapters |
| `tests/test_gpu_v3_benchmark.py` | Replace with end-to-end and repeated throughput gates in the responsible stage |
| `tests/test_gpu_v3_contract.py` | Reuse strict-load test intent, but require the complete V3 identity and exact resume state |
| `tests/test_gpu_v3_distillation.py` | Reuse only the teacher-frozen and KL-scaling test ideas in H3 |
| `tests/test_gpu_v3_models.py` | Rebuild in H1 around Observation V2, V2 multi-head preservation, adapter isolation, and legal-action alignment |

## Canonical identity

The one canonical model/version name is `v3_hybrid`. `gpu_v3` is not an alias.
Hardware placement is a training-topology property, not a model semantic
version. Supporting both names would make checkpoint and release meaning
ambiguous.

The fixed top-level identity is:

| Field | Value |
|---|---|
| contract | `v3-hybrid-h0-contract-v1` |
| model version | `v3_hybrid` |
| feature version | `v2` |
| checkpoint kind | `public_policy` |
| deployment observation | `ObservationV2` containing `PublicObservation` |

H0 deliberately does not add `v3_hybrid` to `TrainingConfig` or Legacy CLI
allowlists. Until H1 supplies a real public policy and strict constructor, a
V3 config must fail closed.

## Public and privileged boundary

Only the environment rules engine creates legal actions. Policy, belief,
Oracle, priors, and search receive the existing legal-action batch and may
only rank its rows. They cannot synthesize, repair, or append moves.

The deployable policy accepts `ObservationV2` only. Public belief posterior
features are allowed. `PrivilegedObservation`, true allocations, hidden-hand
labels, teacher/Oracle weights, and training labels are forbidden from the
public checkpoint, package payload, and deployment import graph. A disabled
privileged feature must remove its parameter/data dependency, not merely
multiply a computed loss by zero.

## Model and output contract

V3 uses the existing Observation V2 card, state, bounded history, public
context, bidding-token, and legal-action encodings. A shared encoder processes
those inputs. Two residual adapter families, landlord and farmer, specialize
the representation while retaining the shared base. The two farmer seats use
the farmer adapter and remain distinguishable through public seat/role context.

The action-conditioned scalar DMC Q head is independent of the preserved V2
win and conditional-score heads. It receives the fused state/action
representation and emits one scalar per existing legal action. V2-compatible
human-prior, strategy, style, belief, and bidding additions remain optional.

## Adaptive DMC contract

H2 implements `per_role_popart_huber_monte_carlo_v1`: terminal team-perspective
Monte Carlo returns, checkpointed float64 running moments per canonical role,
PopArt affine preservation, and Huber loss in normalized space. Statistics are
updated from real selected samples only. Role weights are applied exactly once
and their raw/effective counts and loss contributions are observable.

The running moments, update count, epsilon, decay/update rule, Huber delta,
role weights, and schedule are checkpoint identity and resume state. Disabling
Adaptive DMC removes its head/state/loss unless a later stage explicitly keeps
the Q head for another enabled objective.

## Loss contract

The only canonical combined loss ordering is:

```text
lambda_dmc      * L_admc
+ lambda_win    * L_win
+ lambda_score  * L_score
+ lambda_oracle * L_oracle
+ lambda_belief * L_belief
+ lambda_coop   * L_cooperation
+ lambda_bc     * L_bc
+ lambda_strategy * L_strategy
+ lambda_bidding  * L_bidding
```

Every term has one weight source, one identity-bound schedule, and an
independent disable path. Reductions sum over real valid samples and divide by
the corresponding real valid count, never padded action rows. Sample and role
weights are not folded into a second sampler weight. Metrics report numerator,
denominator, raw count, effective weight sum, and per-role contribution.

## Compatibility and resume

`V3HybridCompatibilityIdentity` hashes the frozen semantic contract plus the
exact ruleset and these required non-empty sections:

- feature flags;
- model graph;
- output semantics;
- optimizer configuration;
- loss configuration;
- loss schedules;
- belief layout;
- cooperation mixer;
- complete trainer configuration;
- training topology.

Unknown or missing sections, non-finite JSON, a payload/hash mismatch, or any
expected/actual identity mismatch fail closed. Later stages may add fields by
bumping the contract version; they may not silently omit an axis.

Exact resume additionally restores model and optimizer state, every loss
schedule, Adaptive DMC statistics, policy version, trainer counters, and
Python/NumPy/Torch RNG state. Permissive partial loading is forbidden.

## Stage boundaries

Each stage starts from then-latest `main` on its own branch and has one Draft
PR. A failed correctness or performance gate stops the sequence.

| Stage | In scope | Explicitly out of scope |
|---|---|---|
| H0 | Audit, naming, semantic and identity contract | Runnable V3 model/training claims |
| H1 | Shared public encoder, role-family residual adapters, scalar Q head, preserved V2 outputs | Adaptive loss, Oracle, belief feedback, mixer |
| H2 | Adaptive DMC state/loss/metrics and exact resume | Privileged teacher |
| H3 | Online training-only Oracle guiding | Belief feedback and farmer mixer |
| H4 | Conservative belief auxiliary and optional public policy feedback | Cooperation mixer |
| H5 | Farmer team value and credit allocation | Full auxiliary/runtime integration |
| H6 | BC/strategy/style/league/curriculum/bidding integration | New runtime/search topology |
| H7 | Single-GPU long-running runtime, bounded policy lag, selective search | Formal strength claim |
| H8 | Three-seed training, ablations, 20k/100k paired evaluation, release | Automatic GitHub GPU workflow |

H8 may claim promotion only if each role reports WP and ADP differences with
deal-clustered bootstrap 95% confidence intervals and no statistically
significant role regression. Until real weights and formal evaluation exist,
the required statement is: playing strength not measured.
