# Joint Hidden-Hand Belief Model (P07)

> **Status:** implemented behind `belief_enabled` (default off). `frozen`
> remains the compatibility default, while P17 adds explicit `joint` and
> `alternating` optimization through a differentiable, constrained PyTorch DP.
> Legacy and belief-disabled V2 paths remain unchanged. Real playing strength
> is not established by these implementation and CPU correctness tests.

This phase adds the imperfect-information-safe replacement for "reading the
true hidden hands". At deployment the model predicts a **posterior** over how
the unknown cards are split between the two opponents; at training time the
labels come from `PrivilegedObservation` only.

It implements the AGENTS.md "Belief-model rules":

- per-rank allocations cannot exceed unseen counts,
- joker counts are at most one each,
- each opponent's total equals the public remaining-card count,
- the two opponent hands sum exactly to the unknown pool,
- known public bottom cards are assigned correctly,
- decoding and sampling never rely on unbounded rejection loops.

## Representation

From the acting player's view, every unknown card is held by exactly one of two
opponents. A **canonical opponent A** (the NEXT seat, clockwise) is the
prediction target; opponent B (PREVIOUS seat) is fully determined by
subtraction:

```
count_B[rank] = unseen_count[rank] - count_A[rank]
```

with `sum_r count_A[rank] == opponent_A_cards_left` enforced **exactly** by a
bounded dynamic program. There are 15 rank categories (the 13 numeric ranks
`3..14,17` + small/big jokers `20,30`, matching
`douzero.observation.cards`). The model emits logits of shape `[B, 15, 5]`
(count `0..4` per rank), masked to `[0, unseen_count[rank]]` and to `[0, 1]`
for jokers.

The model produces **two** probability views:

- `factor_probs` — the independent per-rank masked softmax (the model's
  per-rank *factor* distribution; NOT conditioned on the total).
- `constrained_probs` — the per-rank **marginals of the constrained posterior**
  `P(c_r = k | sum = total)`, computed by a forward-backward (log-sum-exp)
  dynamic program. These are mutually consistent with the total-count
  constraint: `sum_r E[c_A_r] == opponent_a_total` **exactly**.

The value-fusion features (`belief_features_from_probs`) consume the
**constrained** marginals (and assert the expected total matches the target);
passing the independent `factor_probs` is rejected. The MAP decoder and the
sampler (which enforce the constraint exactly) operate on the same logits.

### Public bottom cards

The unplayed public bottom cards are *known* landlord property, so:

- they are **excluded** from the farmer's belief unknown pool
  (`compute_belief_unknown_pool`),
- when opponent A is the landlord, the DP total is the landlord's **hidden**
  card count (`num_cards_left[landlord] - len(bottom_unplayed)`), and the
  training label is the landlord's true hand **minus** the unplayed bottom
  cards (the model predicts only the hidden portion). The farmer-side
  leakage/conservation tests verify this.

## Modules (`douzero/belief/`)

| Module | Responsibility |
|---|---|
| `constraints.py` | 15-rank category set, canonical opponent A/B, `[15,5]` legal mask, per-rank count helpers, `opponent_unknown_total` (bottom-adjusted). |
| `dynamic_programming.py` | Exact MAP `decode_map` + forward-filter/backward-sample `sample_allocation`. Bounded `O(15 · total · 5)`; no rejection loop. Raises `BeliefDPError` on an infeasible (inconsistent) observation. |
| `torch_dynamic_programming.py` | Differentiable constrained marginals for joint training. It implements the same exact-total posterior in a forced-float32 Torch island so gradients reach belief logits under FP16/BF16 autocast. |
| `features.py` | `build_belief_input(public_obs)` → a fixed-width public feature vector + the constraint totals. Enforces `pool.sum() == A_hidden + B_hidden`. |
| `model.py` | `BeliefModel`: MLP encoder → `[B,15,5]` head + legal mask; `decode_map`, `sample`, and `belief_features_from_probs` (the value-fusion feature vector). `BeliefConfig` with its own `stable_hash()`. |
| `labels.py` | `build_belief_label(...)` — the ONLY place the true hidden allocation is read; produces the `(15,)` target and `(15,5)` one-hot. |
| `losses.py` | Masked cross-entropy + optional count/entropy regularizers + `belief_metrics` (rank accuracy, exact-match, count-MAE). |
| `checkpoint.py` | Manifest-bearing save/load with architecture-hash + ruleset identity validation. |
| `joint_checkpoint.py` | Atomic joint/alternating checkpoint helpers that bind both model states, both config hashes, mode, public-input contract, optimizer state, and ruleset/schema identity. |
| `data.py` | Random self-play collector carrying `(BeliefInput, BeliefLabel)` pairs (privileged label never on the public observation). |

## Imperfect-information boundary

- `BeliefModel.forward` accepts ONLY `BeliefInput` (built from a
  `PublicObservation`). It never imports `douzero.observation.privileged`.
- The belief input is **invariant under hidden re-allocation**: two states
  with identical public footprint but different true farmer splits produce
  byte-identical feature vectors (the leakage test asserts this).
- Belief labels live in `BeliefLabel` and `PrivilegedObservation.hidden_hand_labels`
  only; they never reach `BeliefModel.forward`, `ModelV2.forward`, or
  `DeepAgentV2.act_v2`.

## Model V2 fusion (`belief_enabled`)

When `ModelV2Config.belief_enabled=True`, Model V2 gains a `belief_proj`
linear layer that maps the fixed-dim belief feature vector
(`BELIEF_FEATURE_DIM = 48`: per-rank expected count / entropy / max-prob for
opponent A, opponent-A & B expected totals, total entropy) into the trunk and
**adds** it to the state representation before fusion.

- The architecture delta is **exactly** `belief_enabled` (already an identity
  axis in `ModelV2Config.compatibility_dict`), so **no checkpoint
  identity-version bump** is needed and belief-disabled checkpoints load
  unchanged.
- `belief_stop_gradient=True` detaches belief features for the default
  "pretrain then freeze" path; `belief_proj` remains a trainable value-model
  parameter. P17 joint training calls the public belief encoder with
  `differentiable=True`, computes the same exact-total constrained marginals in
  Torch, converts them to differentiable belief features, and forwards with
  `belief_stop_gradient=False`, allowing value loss to update BeliefModel.
- The belief model itself is **not owned** by Model V2; the caller computes
  the public posterior and passes it in. This keeps the value checkpoint
  decoupled from the belief architecture.
- **Fail-closed**: when `belief_enabled=True` but no `belief_features` are
  supplied, the model raises by default (a belief-trained checkpoint must not
  silently degrade to a zero-feature baseline at deployment). Pass
  `allow_missing_belief_features=True` only for explicit ablations.

## Training the value model with belief fusion

A belief-enabled value model is trained via `train_v2.py` with a strictly
validated pretrained belief model as its initialization. Frozen mode uses it
as a feature source; joint and alternating modes opt into further updates:

```bash
python train_v2.py --config /path/to/belief_v2.yaml \
  --belief_checkpoint /path/to/belief.pt
```

`V2Trainer(model, ..., belief_model=belief)` freezes the BeliefModel
(`requires_grad=False`, `eval()`), computes the constrained posterior features
from each `obs.public` at both the self-play collection and the optimizer
forward, and fuses them into `ModelV2`. Only `belief_proj` (a value-model
parameter) is optimized; the belief posterior is a frozen feature source. The
checkpoint is validated by `load_belief_checkpoint` (ruleset + feature version +
architecture hash). A belief-enabled value model without `--belief_checkpoint`
fails fast at trainer construction.

P17 exposes three explicit modes:

| Mode | Behavior |
|---|---|
| `frozen` (default) | Uses the established NumPy constrained posterior as a detached public feature source. Belief weights never enter the optimizer. |
| `joint` | Uses differentiable Torch constrained marginals. Value loss reaches both the value model and BeliefModel; an optional supervised belief term may be added. |
| `alternating` | Alternates value-only and supervised-belief-only optimizer blocks at `belief_alternating_interval`; labelled belief samples and a positive supervised weight are required. |

For example, with a config whose `model.belief_enabled` is true:

```bash
python train_v2.py --config /path/to/belief_v2.yaml \
  --belief_checkpoint /path/to/belief.pt \
  --belief_training_mode joint \
  --checkpoint_path /path/to/joint-trainer.pt
```

`--resume_checkpoint` strictly restores mode, both model states, optimizer,
counters, and RNG state. The standalone `save_joint_checkpoint` /
`load_joint_checkpoint` API provides the same coupled identity boundary for
callers outside `V2Trainer`. `--belief_supervised_episodes` is a bounded
synthetic labelled-data smoke path, not a substitute for an approved training
dataset. Joint and alternating modes currently reject DDP because belief
gradients are not synchronized, and reject `compile_model` because this graph
has not been validated through Torch compilation.

## Deployment: `DeepAgentV2`

A belief-enabled value model deploys via `DeepAgentV2(position, model, ruleset,
belief_model=...)`. The agent holds the eval-mode `BeliefModel`, builds a
`BeliefInput` from the **public** observation (`obs.public` — never a hidden
hand), runs the constrained posterior, projects it into the 48-dim feature
vector, and passes it into `ModelV2.forward` (cast to the value model's
device/dtype inside the forward).

- A belief-enabled value model **without** a `belief_model` is rejected at
  construction with a precise error (no silent crash at inference).
- A `belief_model` supplied to a belief-**disabled** value model is also
  rejected (configuration mismatch).
- Both the canonical `act_v2(obs)` and the legacy `act(infoset)` paths fuse
  belief features. The imperfect-information boundary is preserved: the belief
  model reads only `obs.public`.
- A format-2 belief-enabled deployment package carries the strictly validated,
  checksummed belief checkpoint and attaches the reconstructed BeliefModel to
  the loaded value model. `DeepAgentV2` consumes this attachment by default,
  avoiding an untracked external belief artifact at serving time. Direct
  programmatic construction may still pass `belief_model=` explicitly.

## Training and evaluation CLI

```bash
# CPU smoke: collect random self-play, train, save a manifest checkpoint.
python train_belief.py --save_dir /tmp/belief_smoke \
    --num_episodes 20 --epochs 3 --batch_size 32 --seed 0

# Evaluate: rank accuracy, exact-match, count-MAE, argmax-total conservation.
python evaluate_belief.py --checkpoint /tmp/belief_smoke/belief.pt \
    --num_episodes 20 --seed 42
```

`train_belief.py` uses the masked cross-entropy loss, clips gradients, guards
against non-finite loss, and writes a checkpoint with `model_version =
"belief_v1"`, the `BeliefConfig.stable_hash()`, the ruleset identity, git sha,
torch/python version, and frame count. `load_belief_checkpoint` loads with
`weights_only=True` by default (safe unpickling — untrusted checkpoints cannot
trigger arbitrary code execution) and validates the full manifest identity
(schema version, model version, checkpoint kind, feature version, architecture
hash, ruleset identity). `expected_ruleset` is **required**.

The standalone P07 `train_belief.py` collector still runs on the **legacy**
card-play env only (`Env("adp")`), so checkpoints created by that command are
stamped `legacy`, never mislabeled as `standard`. This is a limitation of that
pretraining command, not of the P17 standard V2 game loop: standard full-game
training and learned bidding are available through `train_v2.py`, but require
a belief checkpoint whose ruleset identity matches that run.

`evaluate_belief.py` reports metrics for **both** decoders:

- `factor_argmax_*` — the independent per-rank argmax (informational; does not
  respect the total-count constraint),
- `constrained_map_*` — the DP MAP decoder used at deployment, and
- `constrained_map_conservation` — the fraction of DP decodes that satisfy the
  exact total constraint (**must be 1.0** by construction).

## Checkpoint impact

- **Belief checkpoints** (`belief_v1`) are a separate kind with their own
  manifest (`douzero/belief/checkpoint.py`); they are not loadable by the V2
  value loader and vice versa.
- **Coupled checkpoints** for `joint` and `alternating` atomically bind both
  state dictionaries and the exact optimization mode. Cross-mode, config,
  ruleset, feature, optimizer, or public-input-contract mismatches fail closed.
- **Value checkpoints**: `belief_enabled` is already in
  `ModelV2Config.compatibility_dict`, so a belief-enabled value model has a
  different config hash from a belief-disabled one (enforced as an identity
  axis). Existing P05/P06 checkpoints (`belief_enabled=False`) are unchanged.

## Test coverage

`tests/test_p07_belief.py`, `tests/test_p07_belief_pipeline.py`,
`tests/test_p07_model_fusion.py`:

- **1000-random-state conservation sweep**: every MAP and sampled allocation
  satisfies the per-rank cap, `sum == opponent_A_hidden_total`, opponent B =
  pool − A is non-negative, and A + B reconstructs the pool exactly; the
  constrained-marginal expected total equals the target exactly.
- joker counts never exceed one in any sample.
- DP MAP correctness on handcrafted examples; sampler distribution matches
  normalized weights within tolerance; constrained marginals' expected total
  matches the target for totals 0/1/2, and different totals yield different
  posteriors.
- Leakage: identical public footprint + swapped farmer cards → identical belief
  input; the label changes.
- Masked CE loss: finite, gradient-bearing, λ=0 regularizers disable cleanly.
- Checkpoint security: `weights_only=True` default rejects a crafted pickle;
  schema/kind/feature-version/config-hash/ruleset mismatches are rejected;
  `expected_ruleset` is required.
- Model V2 fusion: belief on/off gate, output changes with features,
  stop-gradient on/off, wrong-dim rejection, disabled-model rejection,
  fail-closed on missing features, parameter_count includes belief_proj.
- P17 differentiable DP: Torch/NumPy marginal parity, exact-total conservation,
  finite float32 behavior under CPU BF16 autocast, and gradients through the
  constrained posterior into belief logits.
- Frozen/joint/alternating trainer behavior: frozen belief weights stay fixed,
  value-only joint loss updates belief weights, alternating phases isolate
  optimizer ownership, and coupled save/resume restores both models.
- Device/dtype: `BeliefModel.double()` produces float64 logits.
- CLI smoke (`train_belief` / `evaluate_belief` end-to-end on CPU).

## What is **not** measured / out of scope

- No GPU runs; no real playing-strength evaluation. The CPU smoke verifies the
  loss decreases (e.g. `0.56 → 0.37` cross-entropy over 3 epochs on random
  play) and that **every** decoded/sampled allocation is card-conservative —
  it does **not** claim a stronger belief or a stronger value model. That
  requires the P15 paired-evaluation framework and model-guided self-play
  collection (P14), both out of scope here.
- Joint and alternating training have not been validated under DDP,
  `torch.compile`, or on CUDA in this closure. Those requests fail closed where
  applicable; CPU tests are not evidence of multi-GPU correctness or speed.
- `DeepAgentV2` **does** wire belief features end-to-end (see "Deployment"
  above) on CPU; the optional CUDA path is symmetric but not exercised in CI
  (the test image is CPU-only).

## Migration and rollback

This is purely additive and default-off. To roll back, leave `belief_enabled`
at its `False` default and do not run the belief CLIs — no legacy or V2
behavior changes. To enable belief fusion for a value model, set
`belief_enabled: true` under the `model:` block of a V2 config and supply a
trained belief checkpoint at deployment.
