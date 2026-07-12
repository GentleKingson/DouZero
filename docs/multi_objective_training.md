# Multi-Objective Training, Decision Policy, and Calibration (P06)

P06 introduces the multi-objective training layer on top of the P05
:doc:`model_v2` architecture. The single-scalar MSE on one value head is
replaced by a team-perspective combination of:

- a BCE-with-logits **win-probability** loss on `win_logit`,
- masked **Huber conditional-score** losses on `score_if_win` and
  `score_if_loss`,
- an optional **log-score** auxiliary loss,
- an optional **uncertainty-NLL** auxiliary loss (default off).

The deployment path gains a configurable **decision policy**
(`pure_win`, `pure_score`, `win_then_score`, `score_then_win`,
`risk_aware`) and the evaluation harness gains **calibration metrics**
(Brier score, NLL, expected calibration error, reliability bins).

> Status: implemented and CPU-tested. Training throughput is intentionally
> minimal (single-process trainer); high-throughput multiprocessing is
> deferred to P14. Strength is **not measured** here — it will be reported
> by P15's unified evaluation framework.

## Sign convention (centralized)

AGENTS.md mandates that every value be expressed from the **current acting
player's team** perspective: a farmer win is positive for both farmer
roles. P06 removes the scattered farmer negation that existed in the legacy
actor loop (`douzero/dmc/utils.py` negated `episode_return` for farmer
positions) by centralizing the conversion in
`douzero/training/labels.py`:

```python
from douzero.training.labels import team_targets

labels = team_targets(game_result, position)
# {'target_win': 1.0/0.0, 'target_score': ±s, 'target_log_score': sign(s)*log1p(|s|)}
```

The helpers accept either a `GameResult` instance or a plain dict. They
read only the public terminal result (winner team + scores); they never
access hidden hands.

For the **landlord**, `team_score = game_result.landlord_score` (the
landlord plays for two). For either **farmer**, `team_score =
game_result.farmer_score` (per farmer, same sign and magnitude by
construction). Score conservation `landlord_score + 2*farmer_score == 0`
holds.

## Team-perspective terminal labels

`Env.step` now populates two additive keys in the terminal `info` dict
(both legacy and standard modes):

- `info['team_targets']` — `{position: {target_win, target_score,
  target_log_score}}` for each of the three positions.
- `info['terminal_result']` — the public `GameResult`-compatible dict the
  labels were derived from.

The legacy `reward` field is **unchanged**. The new keys are purely
additive, so existing consumers of `Env.step` are unaffected unless they
opt in to the new labels.

## Loss module

`douzero/training/losses.py` exposes `MultiObjectiveLoss`, an `nn.Module`
combining the four loss terms with configurable weights:

```python
from douzero.training import LossConfig, MultiObjectiveLoss

loss_fn = MultiObjectiveLoss(LossConfig(
    lambda_win=1.0,         # BCEWithLogitsLoss on win_logit
    lambda_score=0.5,       # masked Huber on score_if_win / score_if_loss
    lambda_log=0.0,         # optional log-score auxiliary
    lambda_uncertainty=0.0, # optional NLL (default off)
    score_delta=1.0,        # Huber delta for the score loss
))
components = loss_fn.forward_gathered(win_logit, score_if_win, score_if_loss, batch_labels)
components.total.backward()
```

### Conditional masking (the critical correctness property)

`score_if_win` is supervised **only** on samples whose team won
(`target_win == 1`); `score_if_loss` only where the team lost
(`target_win == 0`). When a minibatch is all-win or all-loss, the
un-supervised term is exactly zero and produces no NaN/Inf. The unit tests
cover both empty-mask cases (`test_p06_losses.py`).

### Tail stability

The score heads are clamped at `±score_clamp` (default 32.0) inside the
model. The Huber delta further bounds the gradient contribution of large
bomb/rocket residuals, so a 32× multiplier game does not dominate the
gradient.

## Decision policy

`douzero/training/decision_policy.py` exposes `select_action(output,
config)`:

| Mode | Behaviour |
| --- | --- |
| `pure_win` (alias `win`) | argmax `p_win` among valid actions. Default. |
| `pure_score` (alias `score`) | argmax `score_mean` among valid actions. |
| `win_then_score` | keep actions within tolerance of the best `p_win`, then argmax `score_mean`. |
| `score_then_win` | keep actions within tolerance of the best `score_mean`, then argmax `p_win`. |
| `risk_aware` | `score_mean - risk_penalty * uncertainty_proxy`. Default off (`risk_penalty = 0`). |

### Tolerance semantics (sign-safe)

The lexicographic tolerance band is **additive**:

```
value >= best - abs_tol - rel_tol * max(1, |best|)
```

This is deliberately NOT a multiplicative threshold. A multiplicative
threshold `|x - best| <= rel * |best|` would behave inconsistently across
the negative/positive score range (it would shrink toward zero as `best`
approaches 0 and widen for large magnitudes). The `max(1, |best|)` factor
scales the relative tolerance smoothly through zero. The
`test_p06_decision_policy.py::test_tolerance_band_negative_safe` test pins
this.

### Uncertainty proxy (no new head)

P06 does **not** add an uncertainty head to Model V2. The `risk_aware`
mode uses a derived proxy:

```
win_uncertainty   = p_win * (1 - p_win)              # peaks at p_win=0.5
score_uncertainty = |score_if_win - score_if_loss|   # head spread
penalty           = win_uncertainty + 0.5 * (score_uncertainty / spread_max)
```

This is an experimental auxiliary regularizer intended for ablation; it is
disabled by default (`risk_penalty = 0`).

## Calibration metrics

`douzero/training/calibration.py` provides the standard scalar calibration
diagnostics:

- `brier_score(p_win, target_win)` — mean squared error vs the 0/1 outcome.
- `nll(p_win, target_win)` — Bernoulli negative log-likelihood.
- `expected_calibration_error(p_win, target_win, n_bins=15)` — count-weighted
  mean absolute gap between bin confidence and bin accuracy.
- `reliability_bins(...)` — per-bin `(accuracy, confidence, count)`.

`douzero/evaluation/metrics.py` adds `RoleMetrics` (per-role WP / mean
score / game count) and `CalibrationAggregator` (running per-role
calibration over a stream of V2 decisions). P15's unified evaluation
framework will consume these.

## V2 trainer and CLI

`douzero/training/v2_trainer.py` (`V2Trainer`) is a single-process
self-play trainer:

1. Runs N episodes with an epsilon-greedy policy over the current
   `ModelV2` outputs.
2. Records one `ObservationV2` per decision plus the chosen action index.
3. At terminal, reads `info['team_targets']` and stamps team-perspective
   Monte-Carlo labels on every transition of the matching position.
4. Samples a minibatch, forwards each decision, gathers the chosen
   action's head values, and calls `MultiObjectiveLoss.forward_gathered`.

The CLI entry is `train_v2.py`:

```bash
python train_v2.py --config configs/enhanced.yaml \
    --episodes 8 --optimizer_steps 1 --batch_size 16 --seed 0
```

CPU smoke (no GPU required):

```bash
python train_v2.py --episodes 4 --optimizer_steps 1 --seed 0
```

The legacy `train.py` multiprocessing path is **untouched**. The training
gate in `douzero/dmc/dmc.py` that rejects `model_version != "legacy"` is
preserved on purpose — V2 training goes through `train_v2.py`.

## Configuration

`LossConfig` and `DecisionPolicyConfig` are nested sub-configs of
`TrainingConfig`:

```yaml
loss:
  lambda_win: 1.0
  lambda_score: 0.5
  lambda_log: 0.0
  lambda_uncertainty: 0.0
  score_delta: 1.0
  log_score_delta: 1.0

decision_policy:
  mode: pure_win
  abs_tol: 0.0
  rel_tol: 0.0
  risk_penalty: 0.0
```

`configs/legacy.yaml` carries all-zero `loss` weights so the legacy
single-target MSE path is byte-for-byte unchanged. `configs/enhanced.yaml`
carries the V2 multi-objective defaults (`lambda_win: 1.0`,
`lambda_score: 0.5`).

## Checkpoint impact

The loss and decision-policy configuration is recorded in the checkpoint
manifest's `effective_config` block (audit-only). P06 does **not** add
them as identity axes, so existing V2 checkpoints load unchanged; the
checkpoint's loss config is informational only.

## What is **not** measured

In line with AGENTS.md "do not fake results":

- Playing strength (win-rate vs baselines) — deferred to P15.
- Training throughput (frames/sec) — deferred to P14.
- Multi-GPU DDP — deferred to P14.
- Real-data calibration (Brier/NLL/ECE on a held-out cohort) — deferred to
  P15; only synthetic-data sanity is tested here.

## Test coverage

`tests/test_p06_*.py`:

- `test_p06_labels.py` — team-perspective sign convention, score
  conservation, log-score transform.
- `test_p06_losses.py` — BCE, empty-mask, large-multiplier tail,
  λ=0 disables, `forward_gathered`.
- `test_p06_decision_policy.py` — each mode, negative-safe tolerance,
  aliases, risk-aware.
- `test_p06_calibration.py` — Brier, NLL, ECE, reliability bins.
- `test_p06_env_team_targets.py` — terminal info populated in both legacy
  and standard modes; no hidden-hand leakage.
- `test_p06_trainer_smoke.py` — V2 trainer runs ≥1 optimizer step;
  parameters change; buffer capacity eviction.
- `test_p06_eval_metrics.py` — per-role metrics, calibration aggregator,
  DeepAgentV2 accepts every supported decision mode.
- `test_p06_legacy_unchanged.py` — legacy `compute_loss` is plain MSE;
  legacy config defaults are zero; `douzero.dmc.dmc` does not import
  `douzero.training`; the v2 training gate still raises.
