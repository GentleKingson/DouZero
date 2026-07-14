# Policy league and public opponent style (P11)

P11 adds an opt-in population self-play layer without changing legacy training,
rules, observations, checkpoint loading, or evaluation defaults. The feature is
disabled in `configs/enhanced.yaml`.

## League manifest

`douzero.league.LeagueManifest` stores every policy as a `PolicyEntry` with:

- `policy_id`
- `checkpoint_paths_by_role`
- `model_version` and `ruleset_hash`
- objective, creation step, rating, tags, and training eligibility

The JSON manifest is written through a temporary file, fsynced, and atomically
renamed. Unknown fields, duplicate IDs, unsupported schema versions, missing
checkpoints, and incompatible model/rule identities fail closed or are skipped
with a visible warning. Missing optional baselines therefore do not prevent a
run from starting.

Tags describe policy sources without adding another compatibility axis. Common
tags are `historical`, `legacy-wp`, `legacy-adp`, `bc`, `current`, `random`,
`rule`, `milestone`, and `pinned`. A `pinned` or `user` policy is never removed
by retention.

## Population self-play

`PolicyPool.sample_bundle(game_index)` deterministically selects a complete
three-seat bundle from the run seed. A bundle is immutable and hash-checked at
every turn. It is selected at the game boundary and cannot change mid-game.

`PopulationEpisodeRunner` supports two modes:

- `single`: current policy controls and trains from all three seats, matching
  the prior V2 self-play behavior.
- `population`: learner seats rotate deterministically; opponents are sampled
  from compatible eligible policies and built-in agents.

Only decisions made from `learner_controlled_seats` are appended to the V2
episode. Opponent and teammate actions remain in the public action trace for
correct trajectory labels but never become learner transitions. Farmer
transitions record `teammate_policy_id`. `V2Trainer(policy_pool=...)` wires this
runner into the bounded V2 learner.

Historical weights must be loaded with `build_frozen_policy_model`, which
strict-loads a detached clone, switches it to evaluation mode, and disables
gradients. It never calls `load_state_dict` on the learner.

Every completed game may be written as JSONL with seat assignments, learner
seats, teammate IDs, ruleset hash, result, score, and immutable bundle hash.

The current runner intentionally remains legacy card-play-only. Standard-rule
bidding needs a policy interface for the bidding phase before it can be enabled;
passing a standard `RuleSet` raises instead of silently choosing bids.

## Snapshots and promotion

`SnapshotManager` publishes all role files atomically before registering the
policy. Retention keeps the newest policies, periodic milestones, highest-rated
policies, the primary policy, and all user/pinned policies.

`PromotionGate` accepts only `p15_paired_v1` results. It records the paired
sample count, estimate, confidence interval, configured thresholds, decision,
and deal-set identifier. P15 supplies the evaluator; P11 deliberately does not
substitute unpaired self-play returns for that protocol.

## Public style encoder

`build_style_features(PublicObservation)` uses only `acting_role` and the public
card-play action sequence. For each opponent it computes an observed flag,
turn count, pass rate, high-card consumption, bomb rate, repeated-rank split
tendency, mean action size, and mean rank. It does not accept hidden hands or a
persistent player identity.

`StyleEncoder` has learned cold-start embeddings for opponents with no observed
turns. `style_enabled` optionally fuses the encoding into both `ModelV2` and
`BeliefModel`. With style disabled:

- no style module is constructed;
- no style tensor is required;
- the P09/P10 model-config hashes remain unchanged;
- existing checkpoint keys and forward behavior are preserved.

Enabling style changes the model compatibility hash by binding the embedding
width, feature version, and layout hash. A style-enabled checkpoint therefore
cannot be loaded as a style-disabled model or under a different layout.
The bounded belief pretrainer exposes `--style_enabled` and
`--style_embedding_dim`; its minibatches carry the same public statistics and
never add privileged labels to the model input.

## Configuration

The `model` block controls value-model conditioning:

```yaml
model:
  style_enabled: false
  style_embedding_dim: 64
```

The `league` block controls sampling, paths, retention, and promotion:

```yaml
league:
  enabled: false
  mode: single
  manifest_path: ""
  match_log_path: ""
  seed: 0
  learner_seats_per_game: 1
  include_random_agent: true
  include_rule_agent: false
  snapshot_interval_steps: 0
  keep_recent: 5
  milestone_interval: 0
  keep_top_rated: 3
  promotion_min_pairs: 1000
  promotion_min_ci_lower_bound: 0.0
```

No data conversion or legacy checkpoint migration is required while P11 is
disabled. Enabling style requires a newly trained style-enabled checkpoint.
