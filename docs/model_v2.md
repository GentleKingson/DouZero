# Model V2 — shared state-action model (P05)

This document describes the **Model V2** architecture introduced in P05: a
unified, role-conditioned, multi-head value model that replaces the three
role-specific legacy MLPs with one shared backbone. It is selected by
`model_version=v2` and consumes the [Observation V2][obs-v2] (public inputs
only).

P05 is a **feature-flagged** addition: the legacy and factorized (P04) model
families are untouched, the default `model_version` stays `legacy`, and no
training semantics change (the `train()` gate still rejects non-`legacy`
training until P06 wires the multi-objective loss and actor/learner loop).

[obs-v2]: observation_v2.md

## What changed in P05

- **New package** `douzero/models_v2/` — the shared model and its sub-modules.
- **New deployment agent** `DeepAgentV2` — a public-only agent with a canonical
  type guard that rejects `PrivilegedObservation` at the boundary.
- **Config widening** — `ModelConfig` gains V2 architecture knobs; the allowed
  `model_version` set and the `--model_version` CLI choices now include `v2`.
- **Checkpoint helpers** — `save_v2_checkpoint` / `load_v2_checkpoint` /
  `save_v2_position_weights` stamp `model_version="v2"` and the feature schema
  hash, and load strictly (no permissive partial load).
- **Benchmark** — `benchmarks/bench_model_v2.py` reports parameter counts and
  CPU forward/act latency.

## What did NOT change

- The legacy models (`douzero/dmc/models.py`), the factorized models
  (`douzero/dmc/models_factorized.py`), and `DeepAgent` are unchanged.
- The default `model_version` is still `legacy`; the default
  `feature_version` is still `legacy`.
- `train()` still rejects `model_version != "legacy"` (training integration
  arrives in P06).
- Legacy checkpoints load exactly as before.

## Architecture

```
state block (once) ──┐
public context ──────┼── StateEncoder ──► state_trunk ──────────────┐
                     │                                                ├──►
history tokens+mask ── HistoryEncoder ──► history_summary ───────────┤    StateActionFusion
                                                                    ├──► (per action) ──► ValueHeads ──► ModelOutput
action feature rows ── ActionEncoder ──► action_embeddings (N) ──────┤      (+ role embed)
                                                                    │
acting role ───────────────────────────────────────────────────────────┘
```

### Modules

| Module | File | Responsibility |
|---|---|---|
| `CardSetEncoder` / `MultiCardSetEncoder` | `card_encoder.py` | Project a 54-wide card-count vector into the hidden space. Shared across all card-set inputs (my hand, played piles, last move, bottom cards). |
| `TransformerHistoryEncoder` / `LSTMHistoryEncoder` | `history_encoder.py` | Summarise the bounded public action-history token sequence into one vector, respecting the padding mask. Transformer is the default; LSTM is a lighter fallback (`history_encoder: lstm`). |
| `ActionEncoder` | `action_encoder.py` | Embed each legal action's 74-wide feature row (cards + move-type + rank + length + pass/bomb flags). |
| `StateEncoder` | `state_encoder.py` | Encode the per-decision state block + public context block into one role-agnostic trunk vector. Runs **once per decision**. |
| `StateActionFusion` | `fusion.py` | Combine the shared state trunk + history summary + per-action embedding + role embedding, via pre-norm residual MLP blocks. |
| `ValueHeads` | `heads.py` | Multi-head output: `win_logit`, `score_if_win`, `score_if_loss`, derived `p_win` and `score_mean`. Score heads are clamped for numerical stability. |
| `ModelV2` | `model.py` | The top-level model: wires the encoders → fusion → heads, and exposes `forward()` + `parameter_count()`. |
| `ModelOutput` | `output.py` | Typed return value: the head tensors + the action mask + `argmax_win()` selection helper. |
| `observation_to_model_inputs` | `batch.py` | Bridge from `ObservationV2` to the model's tensor contract (splits the state/context blocks into card-vector and flat-field portions). |

### Key invariants (tested in `tests/test_model_v2.py`)

- **State/history encoded once per decision.** Only the action path and the
  final fusion run per legal action (the P04 factorized property, generalized).
- **Action embeddings are consumed per-row.** The fusion concatenates each
  action's own embedding with the shared trunk, so two different actions
  produce different logits. Tested by action-sensitivity, permutation
  equivariance, and action-encoder nonzero-gradient tests.
- **State field identity is preserved.** The state encoder concatenates
  per-field embeddings in a fixed schema order (it does NOT sum them).
  Swapping two card fields (e.g. `my_hand` ↔ `other_hand`) changes the trunk.
  Card fields are identified by their canonical schema name, not by a
  width=54 guess.
- **Variable legal-action counts.** The model takes `(N, action_width)` and
  broadcasts the shared trunk; no fixed maximum action count is assumed.
- **Zero legal actions rejected.** `forward()`, `act_v2()`,
  `observation_to_model_inputs()`, and `ModelOutput` all raise on zero action
  rows (a decision with no legal actions is undefined).
- **Padding masks are respected.** Padded history tokens never affect the
  output (tested by corrupting only padded slots and asserting the output is
  unchanged).
- **No BatchNorm.** LayerNorm + residual MLPs throughout (actor inference
  batches are size-1; BatchNorm running stats would be unstable).
- **Finite outputs.** Score heads are clamped to `[-score_clamp, score_clamp]`;
  a runtime NaN/Inf guard (`nan_guard`, default on) checks the fused
  representation AND every head output (`win_logit`, `score_if_win`,
  `score_if_loss`, `p_win`, `score_mean`), raising `NumericalError` on any
  non-finite value — catching both bad inputs and bad weights. The clamp
  alone is insufficient: `torch.clamp(nan)` is `nan`, so a NaN weight in a
  score head produces a NaN score that the clamp cannot remove.
- **Deterministic under `eval()`.** Same input → identical output.
- **Imperfect-information boundary.** The model package imports only the
  public observation modules. Corrupting `infoset.all_handcards` (the true
  hidden hands) does not change the model output.

### Batch scope (one decision per forward)

P05 supports **one decision per forward pass** with a **variable number of
legal actions** (the action path is `(N, action_width)` with no fixed max).
It does NOT yet support a multi-decision training minibatch (padding +
masking across decisions with different action counts). That padded
decision/action batch representation belongs with the P06 multi-objective
training loop, which is where the learner batches across actors. Constructing
a `ModelV2` and calling `forward()` on one decision at a time is the complete,
tested P05 contract; do not assume a `(B, N, ...)` batched forward is available.

### Output dictionary and sign convention

`ModelOutput` carries, per legal action (shape `(N, 1)`):

| Field | Meaning |
|---|---|
| `win_logit` | Raw win logit. `p_win = sigmoid(win_logit)`. |
| `p_win` | Win probability from the **acting player's team** perspective. A farmer win is positive for both farmer roles. |
| `score_if_win` | Conditional final signed score given a win (acting-team perspective). Supervised only on won-episode samples (P06). |
| `score_if_loss` | Conditional final signed score given a loss. Supervised only on lost-episode samples (P06). |
| `score_mean` | Derived: `p_win * score_if_win + (1-p_win) * score_if_loss`. A readout for the decision policy, NOT an independent loss target. |
| `action_mask` | `(N,)` bool, `True` for a valid action. |

All scores are **acting-team perspective, positive = good for the acting team**
(AGENTS.md "Rewards, targets, and action selection"). The loss module (P06) is
responsible for converting terminal labels into this perspective; the heads are
perspective-agnostic.

## Configuration

The V2 architecture knobs live on `ModelV2Config`
(`douzero/models_v2/config.py`). `ModelConfig` (in
`douzero/config/schemas.py`) carries the architecture fields
(`hidden_size`, `history_encoder`, `history_layers`, `history_heads`,
`role_embedding_dim`, `belief_enabled`, `human_prior_enabled`) and the
`version` selector; `ModelV2Config.from_model_config` bridges the two so a
future YAML `model:` block can drive construction.

**P05 scope:** the config loader currently reads only the `model_version`
string (via `--model_version v2` / the YAML top-level `model_version` key) and
validates it against the allowed set. The full YAML `model:` block wiring
(arriving with P06, which introduces V2 training and needs the architecture
knobs at the learner) is not yet connected — today, construct `ModelV2` with a
`ModelV2Config` directly (the defaults match a CPU-friendly smoke-test size):

```python
from douzero.models_v2 import ModelV2, ModelV2Config
from douzero.observation import build_v2_schema

model = ModelV2(build_v2_schema(), ModelV2Config())
```

The intended `model:` block (for when P06 wires it) is:

```yaml
model:
  version: v2
  hidden_size: 256
  history_encoder: transformer   # or lstm
  history_layers: 4
  history_heads: 8
  role_embedding_dim: 32
  belief_enabled: false           # P07 attaches belief heads
  human_prior_enabled: false      # P08/P09 attach a prior head
```

### Divisibility constraint

For the Transformer history encoder, `hidden_size` must be divisible by
`history_heads` (the Q/K/V split). This is the only divisibility coupling and
is validated at construction.

## Deployment: `DeepAgentV2`

`DeepAgentV2` (`douzero/evaluation/deep_agent.py`) is the public-only
deployment agent for Model V2.

### Imperfect-information boundary (the most important property)

`act_v2(obs)` accepts **only** an `ObservationV2`. Passing a
`PrivilegedObservation` raises `TypeError` **before any model call** — this is
the canonical type guard required by the P03/P05 acceptance criteria. The
privileged module is imported **locally** inside `act_v2` (for the `isinstance`
rejection only), so the production import graph never depends on it.

The model itself only ever sees the tensor blocks of the observation; it has
no field for hidden hands.

### Two entry points

- `act_v2(obs: ObservationV2)` — the canonical type-guarded public entry point.
- `act(infoset)` — legacy-compatible, for `douzero.evaluation.simulation`. It
  builds an `ObservationV2` internally via `get_obs_v2` (which never reads the
  true hidden hands) and delegates to `act_v2`. The selected action is mapped
  back onto the infoset's own canonical action object so downstream code keeps
  working.

### Decision modes

- `"win"` (default): argmax `p_win` over valid actions.
- `"score"`: argmax expected score (`score_mean`) over valid actions.

P06 adds lexicographic modes (`win_then_score`, `score_then_win`); P05 keeps
these two simple, fully-tested modes.

### Loading weights

`load_v2_model(model_path, schema, config, ruleset_id=...)` loads a
**manifest-bearing** V2 sidecar (written by `save_v2_position_weights`). The
sidecar carries a minimal manifest (model_version, schema hash, ruleset
identity, checkpoint_kind=`public_policy`); every identity field is validated
against RUNTIME expectations (the `schema` and `ruleset_id` the caller passes),
never against the checkpoint's self-reported values. The state_dict is loaded
with `strict=True` and `weights_only=True` (the safe default).

A bare state_dict sidecar, a legacy/factorized `.ckpt`, a same-shape
different-schema sidecar, or a wrong-ruleset sidecar is rejected with a
precise `CheckpointCompatibilityError`. There is no permissive partial load.

`DeepAgentV2` additionally binds to the model's feature schema hash at
construction and validates every observation's schema hash in `act_v2()`, so
a model trained under schema A cannot silently consume an observation encoded
under schema B. It ALSO requires an explicit `RuleSet`, validates the agent's
RuleSet against the model's checkpoint ruleset identity, and validates every
observation's ruleset identity (id + version + hash) — closing the loophole
where a standard-policy model could be served under a legacy observation
context (the ruleset family/bid/multiplier are observation data values that do
not necessarily change the schema layout).

## Checkpoints

P05 adds V2-aware checkpoint helpers in `douzero/checkpoint/v2.py`. Every load
validates FIVE identity axes against RUNTIME-SUPPLIED expectations (never the
checkpoint's self-reported values):

1. `model_version == "v2"` — rejects a legacy / factorized bundle.
2. `feature_schema_hash` — must equal the runtime schema's `stable_hash()`.
   Catches a same-shape-different-schema drift.
3. `model_config_hash` — must equal `ModelV2Config.stable_hash()`. Catches a
   same-shape-different-semantics config drift (e.g. `history_heads` 8→4 keeps
   projection shapes but changes the Transformer split; `score_clamp` /
   `nan_guard` change runtime behavior) that strict state_dict loading cannot
   detect.
4. `ruleset_id` / `ruleset_version` / `ruleset_hash` — validated against the
   caller's full `RuleSet`, supporting custom rule families and rejecting an
   unknown id (no silent downgrade to legacy).
5. `checkpoint_kind` — `training_checkpoint` vs `public_policy`.

Helpers (all accept the full `RuleSet` object, not an id string; both save
helpers derive the schema/config identity from the model and reject a caller
override that disagrees):

- `save_v2_checkpoint(path, model, *, ruleset, schema_hash=None,
  model_config=None, frames=...)` — writes the full `model_v2.tar` bundle.
- `load_v2_checkpoint(path, expected_schema_hash=...,
  expected_model_config_hash=..., expected_ruleset=...,
  expected_checkpoint_kind=...)` — reads + validates the full bundle. All
  expected values are required runtime arguments.
- `save_v2_position_weights(path, model, *, ruleset, schema_hash=None,
  model_config=None)` — writes the **manifest-bearing** deployment sidecar
  (`.ckpt`). NOT a bare state_dict; same model-derived identity closure as the
  full bundle.
- `load_v2_position_weights(path, expected_schema_hash=...,
  expected_model_config_hash=..., expected_ruleset=...)` — reads + validates
  the sidecar.

`load_v2_model(model_path, schema, ruleset, config)` returns a model with the
verified ruleset identity attached (`model.expected_ruleset_identity`), which
`DeepAgentV2` enforces against its own RuleSet at construction and against
every observation's ruleset identity in `act_v2()`.

The existing `load_checkpoint` (legacy/factorized `model.tar`) already rejects
a `model_version` mismatch via the manifest validator, so a V2 bundle cannot
be silently loaded as legacy.

## Benchmark

`benchmarks/bench_model_v2.py` reports (CPU-only, mirroring `bench_factorized.py`):

- per-submodule + total parameter counts;
- model-forward-only latency at action-count buckets (1, 10, 50, full set);
- full `DeepAgentV2.act` latency (encode + forward + select).

Output: JSON + Markdown under `artifacts/benchmark/`.

Example (default config, CPU):

```
Total parameters: 5,047,651
  (the fusion now consumes the per-action embedding width, ~525K more than a
  prior version that silently ignored action embeddings)
Forward-only medians: ~12-55 ms per decision (varies with host load and the
Transformer warm-up; the per-action path is cheap, the shared Transformer trunk
dominates).
```

These are **measurements on the build host, not strength or speedup claims**.
GPU timing, AMP, and DDP comparison are deferred to P14.

## Tests

`tests/test_model_v2.py` (49 tests) covers:

- construction + parameter-count reporting (default config, LSTM backend,
  invalid-backend rejection, divisibility validation);
- forward shape + finiteness for all three roles;
- `p_win = sigmoid(win_logit)` and `score_mean` decomposition;
- score-head clamping under adversarial weights;
- variable action counts (N=1, N=many, same model);
- variable history (empty first-move, non-empty, padding-mask invariance for
  both Transformer and LSTM backends);
- determinism under `eval()` + backward pass (gradients populate, all roles);
- action selection (argmax respects mask, rejects all-masked);
- save/load equivalence (state_dict round-trip, V2 checkpoint bundle,
  schema-mismatch rejection, legacy-ckpt rejection, strict V2 load);
- `DeepAgentV2` (PrivilegedObservation type guard, wrong-type rejection,
  legal-action selection, single-action short-circuit, decision-mode
  validation, ModelV2-instance requirement);
- imperfect-information boundary (corrupting `all_handcards` does not change
  the output; swapping hidden cards between farmers does not change the
  output);
- `ModelOutput` dataclass validation.

All tests are CPU-only and deterministic. GPU numerical/latency parity is
deferred to P14.

## Migration and rollback

- **To use Model V2 for inference**: construct `ModelV2(schema, config)`, load
  weights via `load_v2_model` or `load_v2_checkpoint`, and wrap in
  `DeepAgentV2`. The existing `evaluate.py` / `simulation.py` harness works
  unchanged via the `act(infoset)` entry point.
- **Rollback**: stop selecting `model_version=v2`. Legacy and factorized paths
  are unchanged; no migration is required to return to them.
- **Training** is NOT yet supported for V2 (the `train()` gate rejects it until
  P06 wires multi-objective losses).

## Out of scope for P05

- Training integration (P06): multi-objective losses, decision policies,
  calibration metrics, and the actor/learner loop for V2.
- Belief model integration (P07): `belief_enabled` now attaches a belief
  feature projection (see `docs/belief_model.md`). When `belief_enabled=True`
  the model gains a `belief_proj` layer that fuses a frozen belief posterior
  into the state trunk; the architecture delta is captured by the existing
  `belief_enabled` identity axis, so belief-disabled checkpoints are
  unaffected.
- Human prior / auxiliary heads (P08/P09): `human_prior_enabled` is carried but
  no prior head is attached yet.
- AMP / DDP / `torch.compile` (P14).
