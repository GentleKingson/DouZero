# Policy league and public other-player style (P11)

P11 adds an opt-in population self-play layer without changing legacy training,
rules, observations, checkpoint loading, or evaluation defaults. The feature is
disabled in `configs/enhanced.yaml`.

## League manifest

`douzero.league.LeagueManifest` stores every policy as a `PolicyEntry` with:

- `policy_id`
- `checkpoint_paths_by_role`
- `model_version`, `ruleset_hash`, and `checkpoint_kind`
- `feature_schema_hash`, `model_config_hash`, and
  `model_config_identity_version`
- optional style, strategy, and belief layout/config hashes
- objective, creation step, rating, tags, and training eligibility

The JSON manifest is written through a temporary file, fsynced, and atomically
renamed. Unknown fields, duplicate IDs, unsupported schema versions, missing
checkpoints, and incompatible model/rule identities fail closed or are skipped
with a visible warning. A `PolicyLoaderContract` binds each model family to one
exact identity. V2, legacy, factorized, and BC policies require separate,
explicitly registered loaders; a broad model-version allowlist is not used.
The loaded selector is bound back to that contract before an episode can use
it. `PolicyLoaderContract.for_v2_runtime()` derives the V2 contract from the
live schema and model config, and `V2Trainer` verifies that contract again
before collection. Missing optional baselines therefore do not prevent a run
from starting.

The manifest schema is `2`. It replaces the unreleased P11 draft schema `1`;
draft manifests must be regenerated from trusted checkpoint metadata so the
new identity fields are not guessed.

Tags describe policy sources without adding another compatibility axis. Common
tags are `historical`, `legacy-wp`, `legacy-adp`, `bc`, `current`, `random`,
`rule`, `milestone`, and `pinned`. A `pinned` or `user` policy is never removed
by retention.

## Population self-play

`PolicyPool.sample_bundle(game_index)` deterministically selects a complete
three-seat bundle from the run seed. A bundle is immutable and its digest is
recomputed from the current assignment at every turn. It is selected at the
game boundary and cannot change mid-game.

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

Historical V2 weights may be loaded with `build_frozen_policy_model` only after
the concrete loader contract validates every manifest identity field and
matches the learner clone's runtime contract. The helper strict-loads a
detached clone, switches it to evaluation mode, and disables gradients. Legacy,
factorized, and BC checkpoints use their own loaders rather than cloning a V2
learner.

Every completed game may be written as JSONL with seat assignments, learner
seats, teammate IDs, ruleset hash, result, score, and immutable bundle hash.
The logger is a serialized, single-process writer and flushes and fsyncs each
record; multi-process actor deployments must route records through one writer.

The current runner intentionally remains legacy card-play-only. Standard-rule
bidding needs a policy interface for the bidding phase before it can be enabled;
passing a standard `RuleSet` raises instead of silently choosing bids.

## Snapshots and promotion

`SnapshotManager` requires an explicit `snapshot_root`. It owns only immutable
three-role bundles at `snapshot_root/policies/<policy_id>/<role>.ckpt`; path
traversal, outside-root paths, symlinks, directories, and policy-ID/path
mismatches are rejected. All roles are staged and fsynced before one atomic
directory rename, so a failed writer cannot partially overwrite a registered
bundle.

Retention first atomically removes policies from the active manifest and adds
`pending_deletes` tombstones. It then deletes only manager-owned regular files
and clears the tombstones. A crash or deletion error therefore leaves a
replayable cleanup record rather than an active policy pointing at missing
files; the next `SnapshotManager.load()` resumes cleanup. Retention keeps the
newest policies, periodic milestones, highest-rated policies, the primary
policy, and all user/pinned policies.

`PromotionGate` accepts only `p15_paired_v1` results. It records the paired
sample count, estimate, confidence interval, configured thresholds, decision,
and deal-set identifier. P15 supplies the evaluator; P11 deliberately does not
substitute unpaired self-play returns for that protocol.

## Public style encoder

`build_style_features(PublicObservation)` uses only `acting_role` and the public
card-play action sequence. For each other player it computes an observed flag,
turn count, pass rate, high-card consumption, bomb rate, repeated-rank split
tendency, mean action size, and mean rank. It does not accept hidden hands or a
persistent player identity. For a farmer, the two rows are the landlord and the
farmer teammate in canonical seat order; they are not both described as
opponents.

`StyleEncoder` has learned cold-start embeddings for players with no observed
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
  snapshot_root: ""
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
