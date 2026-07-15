# P15 Evaluation Protocol

P15 evaluates a candidate against one fixed baseline with protocol identity
`p15_paired_v1`. It is separate from training and never tunes a model,
threshold, search budget, or decision mode on the evaluation set.

## Scenarios

`cardplay_only` replays each legacy deal twice. The candidate first controls
the landlord against baseline farmers, then the baseline controls the landlord
against candidate farmers. Both legs retain the same deal ID, hands, and bottom
cards.

`full_game` replays each standard deck three times. The candidate rotates
through neutral seats 0, 1, and 2 while the deck, first bidder, and clockwise
bidding order stay fixed. Bidding determines the landlord before seat agents
are mapped to landlord roles. A bundle declares its bidding policy explicitly:
`rule`, `random`, `pass`, or `max`. The current learned card-play checkpoints do
not expose a learned bidding head; `rule` is a fixed public hand-strength policy
and reports must not describe it as model bidding.

`cardplay_only` uses common random streams derived from scenario seed, deal ID,
and role. `full_game` derives them from deal ID and physical seat. Inference
timing is measured and is therefore the only field expected to vary slightly
between identical runs.

The protocol identity is closed over its seat schedule. `cardplay_only` must
contain exactly the candidate-landlord and candidate-two-farmers legs once
each. `full_game` must contain exactly one candidate in each neutral seat once.
Missing, duplicated, reordered, or additional permutations are rejected. In
In `full_game`, stochastic policies use common random streams by physical seat so
identical candidate and baseline policies replay the same game under rotation.

## Statistical Unit

Confidence intervals resample complete deals. Mirrored card-play legs and the
three full-game seat rotations are averaged within a deal before percentile
bootstrap resampling. Decisions and seats are never treated as independent
samples. Reports include the paired deal count, bootstrap sample count, and
confidence level.

For `cardplay_only`, the paired estimate is candidate win rate minus 0.5. For
`full_game`, raw win percentage is descriptive because a landlord win gives
one winning seat while a farmer win gives two. Its paired estimate is instead
the standard zero-sum seat score: landlord and both farmer scores sum to zero
for every game. Identical policies therefore have an estimate of zero.

Only `cardplay_only` may produce the `PromotionEvaluation` consumed by the P11
promotion gate. Promotion additionally requires confidence level 0.95, at
least 1000 bootstrap samples, the official permutation hash, and the
`cardplay_win_rate_delta` estimator. `PromotionGate` validates all of these
fields again; `full_game` reports cannot be promoted.

## Metrics And Outputs

Every run writes JSON, per-game CSV, and Markdown. The report includes:

- overall, team, landlord, landlord-up, and landlord-down win percentage;
- mean raw score and signed `log1p(abs(score))` score;
- bid and landlord-acquisition rate in full-game mode;
- bomb, rocket, spring, anti-spring, and game-length rates;
- selected-action `p_win` Brier, NLL, and 15-bin ECE when a V2 agent exposes
  predictions;
- candidate inference p50/p95/p99 and inference-only actor FPS;
- deal/game/decision sample counts and paired 95% confidence intervals.

Unavailable metrics are JSON `null` and Markdown `n/a`, not invented zeros.
The raw game rows remain in JSON/CSV so headline results are auditable.

## Model Matrix And Ablations

Built-in `random` and `rule` bundles require no weights. A JSON model matrix can
register arbitrary bundle names such as `legacy-wp`, `legacy-adp`, `bc-v1`,
`v2-p13`, or historical policy IDs. Supported backends are `legacy`,
`legacy_factorized`, and manifest-validated `v2`, plus the built-ins. `bc` is
an explicit alias for the V2 loader so behavior-cloned bundles remain visibly
distinct. Historical policies use their real backend plus a `historical` tag.
Weighted bundles provide all three role checkpoint paths.

```json
{
  "bundles": {
    "legacy-wp": {
      "backend": "legacy",
      "checkpoints": {
        "landlord": "/models/wp/landlord.ckpt",
        "landlord_up": "/models/wp/landlord_up.ckpt",
        "landlord_down": "/models/wp/landlord_down.ckpt"
      }
    },
    "legacy-no-bidding": {
      "backend": "legacy_factorized",
      "checkpoints": {
        "landlord": "/models/no-bidding/landlord.ckpt",
        "landlord_up": "/models/no-bidding/landlord_up.ckpt",
        "landlord_down": "/models/no-bidding/landlord_down.ckpt"
      }
    },
    "v2-no-search": {
      "backend": "v2",
      "checkpoints": {
        "landlord": "/models/v2-no-search/landlord.ckpt",
        "landlord_up": "/models/v2-no-search/landlord_up.ckpt",
        "landlord_down": "/models/v2-no-search/landlord_down.ckpt"
      }
    }
  },
  "ablations": {
    "no_search": "v2-no-search",
    "no_bidding": {
      "candidate": "legacy-no-bidding",
      "baseline": "legacy-wp"
    }
  }
}
```

The recognized ablations are `no_bidding`, `single_head`, `no_belief`,
`no_human_bc`, `no_auxiliary`, `no_distillation`, `no_population`, and
`no_search`. Each points to an explicit compatible bundle. The runner does not
flip architecture flags under an existing checkpoint because that violates
checkpoint identity or produces a cosmetic rather than trained ablation.
When a `full_game` matrix runs `no_bidding`, the first bidder is fixed as
landlord and the exact deck/bottom cards are converted to `cardplay_only`.
Candidate and baseline bundles for that row must therefore carry legacy-
ruleset-compatible checkpoints; the optional object form above supplies a
different compatible baseline when needed.

## Reproducible CPU Smoke

```bash
python evaluate_paired.py \
  --mode cardplay_only \
  --candidate rule \
  --baseline random \
  --num-deals 8 \
  --seed 7 \
  --output artifacts/evaluation/p15-smoke
```

Use `--mode full_game` for bidding and seat rotation. Use `--eval-data` for a
fixed trusted pickle generated by `generate_eval_data.py`. Pickle files must
come from a trusted source because loading pickle can execute code.

Public eval sets may be versioned outside the code package and identified by
their content hash. Private holdouts require `--dataset-scope private_holdout`
and an explicit `--eval-data`; their path/name is redacted from reports. Private
deal contents are never written to the repository by the evaluator.

Regression gates should be configured outside the holdout run and cover legacy
behavior, rules, latency, calibration, and a minimum for each role. Failed
gates block promotion; they must not trigger automatic parameter search on the
test or holdout set.

Pass a predeclared JSON file with `--gates`. Supported thresholds are
`max_p95_latency_ms`, `max_brier`, `max_ece`,
`min_overall_win_percentage`, and role keys under
`min_role_win_percentage`. External regression jobs feed their already-known
boolean results through `required_checks`, for example
`{"legacy_behavior": true, "environment_rules": true}`. A failed gate is
written into the report and makes the command exit with status 2.
`required_checks` accepts only string keys and JSON boolean values; strings
such as `"false"` are rejected rather than coerced.
