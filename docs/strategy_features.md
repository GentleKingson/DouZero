# P09 Strategy Features and Auxiliary Tasks

> **Status:** implemented as an opt-in Model V2 extension. Every feature,
> auxiliary loss, and deployment prior is disabled at its controlling boundary
> by default. Legacy, factorized, and P08 V2 behaviour is unchanged.

## Safety and compatibility

The strategy layer only receives `PublicObservation`: the acting hand, legal
actions, public history, played cards, and public remaining-card counts. It
does not import or accept `PrivilegedObservation`, `all_handcards`, or a true
opponent allocation. Strategy code never adds or removes actions; the rule
engine remains the sole legality source.

Enabling `model.strategy_features_enabled` appends a fixed, versioned
`strategy_v1` feature row to every legal action before `ActionEncoder`.
Enabling `model.strategy_aux_enabled` adds five learned heads. Both architecture
choices and all feature-group toggles enter `ModelV2Config.stable_hash()`.
Checkpoint identity version 3 records these P09 fields. Version-2 P06-P08
checkpoints remain loadable only into a strategy-disabled runtime.

## Feature layout

`douzero.strategy.features.STRATEGY_FEATURE_NAMES` is the canonical 28-column
layout. It contains five independently ablatable groups:

- **Hand decomposition:** bounded minimum turns before/after the action,
  delta, and exactness. `hand_decomposition` uses memoized DP for hands up to
  20 cards. A node budget is the deterministic primary bound; an optional time
  budget is cooperatively checked at DP and candidate-enumeration boundaries.
  Either timeout returns a fixed O(n), legal rank-group upper bound rather than
  re-entering the combinatorial generator. Non-zero time-budget outcomes are
  never stored in the process-wide LRU cache.
- **Structure:** single/pair/triple/straight/serial-pair/airplane/bomb deltas,
  bomb break, rocket split, high-control-card use, and total structure cost.
- **Control:** initiative proxy, move control strength, and blocking features
  when the threatening opponent has one, two, or three cards.
- **Farmer cooperation:** teammate/landlord cards left, teammate suppression,
  a one-card teammate feed signal, and explicit `landlord_up` versus
  `landlord_down` columns.
- **Risk:** spring-risk and bomb opportunity-cost proxies. Spring risk uses the
  public per-role non-pass action counts, never the number of cards played.

Disabled groups keep their columns as zeros so an ablation does not silently
change the tensor width. `strategy_v1`, the ordered names, normalization
divisors, and formula-semantics revision produce a stable layout hash that is
part of the Model V2 checkpoint identity. These are learned inputs, not
hard-coded policy rules.

## Auxiliary labels and losses

The optional heads predict `min_turns_after`, `regain_initiative`,
`teammate_finish`, `spring_probability`, and `structure_cost` for every legal
action. Label provenance is explicit:

| Target | Source |
|---|---|
| `min_turns_after` | Exact bounded decomposition of the acting hand after the selected action; fallback upper bounds are masked out |
| `structure_cost` | Direct deterministic structure calculation |
| `regain_initiative` | Future public trajectory: the acting team later leads a non-pass trick |
| `teammate_finish` | Terminal winner position; masked for landlord samples |
| `spring_probability` | Terminal spring or anti-spring flag, or a replay-derived legacy equivalent |

`Episode.label_strategy_auxiliary` creates these training-only labels after a
trajectory ends. They are never model inputs. Each `loss.lambda_*` weight is an
independent ablation switch; all default to zero. Active terms are logged as
`aux_*` diagnostics by `V2Trainer`. `target_min_turns_exact_mask` prevents a
budget fallback from being treated as an exact regression label.

## Uncertainty-gated prior

`decision_policy.mode: uncertainty_gated_prior` implements:

```text
final_score = score_mean
            + prior_alpha * (4 * p_win * (1 - p_win)) * normalized_prior
```

The prior is normalized over the current masked legal-action list. The gate is
largest at uncertain `p_win=0.5` and vanishes at confident endpoints.
`prior_alpha` defaults to `0`, which is exactly `pure_score`; the mode requires
the P08 public human-prior head and never manufactures an action.

## Configuration

See `configs/enhanced.yaml`. A minimal opt-in experiment is:

```yaml
model:
  human_prior_enabled: true
  strategy_features_enabled: true
  strategy_aux_enabled: true
  strategy_node_budget: 500

loss:
  lambda_min_turns: 0.1
  lambda_regain_initiative: 0.1
  lambda_teammate_finish: 0.1
  lambda_spring: 0.1
  lambda_structure: 0.1

decision_policy:
  mode: uncertainty_gated_prior
  prior_alpha: 0.0
```

## Measurement status

Synthetic/unit coverage validates determinism, budgets, known tactical hands,
all roles, gradients, masks, and checkpoint compatibility. Real human-data
effects, playing strength, GPU throughput, and latency percentiles are **not
measured** in P09. The default 500-node bound limits DP state expansion; callers
that require a wall-clock constraint must also set `strategy_time_budget_ms`.
Neither default is a claim that it is an optimal strength/latency setting.

Rollback requires only disabling the P09 model/loss/decision fields or loading
a P06-P08 checkpoint under a strategy-disabled `ModelV2Config`. No rule,
observation, human-data, or evaluation-data migration is required.
