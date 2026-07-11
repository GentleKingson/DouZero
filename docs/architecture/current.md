# DouZero — Current Architecture (Legacy Baseline)

> Status: **P00 baseline freeze**. This document describes the codebase as it
> exists at commit `718a5c9` on `main`, verified by direct source reads plus
> the P00 test/baseline artifacts. It is the reference every later phase
> (P01–P16) compares against.

The goal of this page is to make the **interfaces** that later phases must
preserve or version explicit. It is not a user guide.

## 1. Repository layout (as of P00)

```
douzero/
  dmc/            distributed training: learner loop, actor loop, models, buffers
    arguments.py  argparse CLI (legacy)
    dmc.py        learner: loss, optimizer step, checkpoint save, actor sync
    utils.py      actor loop, shared buffers, batching, optimizers
    env_utils.py  Environment adapter (env <-> model tensor plumbing)
    models.py     LandlordLstmModel, FarmerLstmModel, Model wrapper, model_dict
    file_writer.py structured run logging (logs.csv / meta.json)
  env/            game rules + RL env
    env.py        Env (gym-style wrapper), get_obs, _cards2array, encoders
    game.py       GameEnv (rules, turn order, legal actions, terminal utility)
    move_generator.py  MovesGener (candidate move enumeration, all 14 types)
    move_detector.py   get_move_type (classify a play into TYPE_0..TYPE_15)
    move_selector.py   filter_type_* (which candidates beat the rival move)
    utils.py      constants: TYPE_*, MIN_SINGLE_CARDS=5, MIN_PAIRS=3, MIN_TRIPLES=2
  evaluation/     deployment / opponents
    deep_agent.py     DeepAgent.act(infoset) -> deterministic legal action
    random_agent.py   RandomAgent (random.choice over legal actions)
    rlcard_agent.py   RLCardAgent (wraps rlcard's rule-based bot)
    simulation.py     multiprocessing evaluate(...)
train.py            training entry (legacy CLI)
evaluate.py         evaluation entry
generate_eval_data.py  pickled list of fixed deals
```

Added by P00 (no production code touched): `tests/`, `tools/`,
`benchmarks/`, `docs/`, `.docker/`, `pyproject.toml` (pytest config only),
`requirements-dev.txt`, `.github/workflows/ci.yml`, `douzero/_version.py`.

## 2. Cards and roles

- 54-card deck: four copies of ranks `3..14` and `17` (the "2"), plus small
  joker `20` and big joker `30`. **Rank `16` is intentionally unused.**
  Canonical deck: `douzero/env/env.py:15-19`.
- The "landlord" is whoever holds the first 20 cards after a shuffle; there is
  **no bidding** in the legacy mode. `Env.reset` shuffles and deals
  `landlord=_deck[:20]`, `landlord_up=_deck[20:37]`,
  `landlord_down=_deck[37:54]`, `three_landlord_cards=_deck[17:20]`.
- Turn order: `landlord → landlord_down → landlord_up → landlord` (note
  **down before up**). See `game.py:154-168`.
- The two farmers share team utility but are distinct positions.

## 3. Legal actions (rule source of truth)

Pipeline in `GameEnv.get_legal_card_play_actions` (`game.py:177-263`):

1. `MovesGener(hand)` eagerly enumerates singles/pairs/triples/bombs/rocket.
2. Determine `rival_move` (last non-pass action; or empty on the opening lead).
3. `get_move_type(rival_move)` classifies it into `TYPE_0..TYPE_15`.
4. For `TYPE_0_PASS` (lead): enumerate **all** move types (`gen_moves()`).
   For each other type: generate candidates, then `filter_type_*` keeps those
   that strictly beat the rival (rank comparison; serials compare low card).
5. **Bombs + rocket are always appended** unless the rival is itself a
   bomb/rocket/pass (`game.py:253-255`).
6. **Pass `[[]]` appended** only when there is a rival move to pass on.
7. Every move is sorted in place: `m.sort()` (`game.py:260-261`).

> Rule of the house: a model, prior, or search may **rank** legal actions but
> must never manufacture an illegal one. Pass legality is rule-controlled.

## 4. Observation encoding (`get_obs`)

`get_obs(infoset)` (`env.py:188`) dispatches by `infoset.player_position` to one
of three role-specific encoders. The returned `obs` dict is identical in
**keys** across roles:

| key | dtype | shape | meaning |
|---|---|---|---|
| `position` | str | — | acting role |
| `x_batch` | float32 | `(N, D_x)` | per-action features (only the trailing 54-dim block varies per action) |
| `z_batch` | float32 | `(N, 5, 162)` | history features, identical across the N rows |
| `legal_actions` | list[list[int]] | len N | candidate actions (each sorted) |
| `x_no_action` | int8 | `(D_x − 54,)` | per-state features, no batch dim |
| `z` | int8 | `(5, 162)` | one row of `z_batch` |

where `N = len(legal_actions)` (no fixed maximum).

### Dimension ledger (no more magic numbers)

- `_cards2array` → `(54,)` int8. Layout = Fortran-flatten of a `4×13` rank
  matrix (column `r` holds the multiplicity of rank `r` as `NumOnes2Array`)
  **plus** 2 trailing joker slots (`[small, big]`). So slot offsets are
  `Card2Column[rank]*4` for ranks and `[52]=small, [53]=big`.
- History: `_action_seq_list2array` pads/truncates to the last **15** moves and
  reshapes to `(5, 162)` (5 rounds × 3 moves × 54). This is the LSTM input.
- `landlord` `x_no_action = 319` = 54×5 (my/other/last/up-played/down-played) +
  17 (up cards-left one-hot) + 17 (down cards-left one-hot) + 15 (bomb one-hot).
  `x_batch = 319 + 54 = 373`.
- farmer `x_no_action = 430` = 54×7 (my/other/landlord-played/teammate-played/
  last/last-landlord/last-teammate) + 20 (landlord cards-left one-hot) +
  17 (teammate cards-left one-hot) + 15 (bomb one-hot). `x_batch = 430 + 54 = 484`.

> These dimensions are asserted by `tests/test_model_shapes.py` and encoded as
> `EXPECTED_X_DIM` there. P03 must derive them from a schema, not hard-code.

### Privileged vs public

`InfoSet` (`game.py:337`) is **perfect information**: it includes
`all_handcards` and `other_hand_cards` (opponents' true hands).
`get_obs` is the *public* projection. The leakage guard
(`tests/test_leakage_guard.py`) asserts `get_obs`'s output keys are exactly the
six public keys above and that swapping cards between the two farmers leaves
the **landlord's** public observation byte-identical (the legacy encoder pools
opponents into `other_hand_cards`, which is swap-invariant).

## 5. Models

`douzero/dmc/models.py`:

- `LandlordLstmModel`: `LSTM(162 → 128)` over `z`, then `concat([lstm(128), x(373)])`
  → `Linear(501→512)` ×5 (ReLU) → `Linear(512→1)`. Returns one scalar value per
  legal action. Output shape `(N, 1)`.
- `FarmerLstmModel`: identical, except `dense1 = Linear(484+128, 512)`.
- No dropout, no BatchNorm → fully deterministic under `model.eval()`.
- `Model(device)` wraps all three roles for training; `model_dict` maps each
  role to its class and is used **only** in evaluation (`deep_agent._load_model`).
- `forward(z, x, return_value=False, flags=None)`: with `return_value=True`
  returns `{'values': (N,1)}`; otherwise epsilon-greedy action selection
  (`flags.exp_epsilon`). `DeepAgent.act` always uses `return_value=True`, so it
  is fully deterministic.

## 6. Reward and terminal

- `GameEnv.game_done` (`game.py:67`): terminal when any player's hand is empty.
- `compute_player_utility` (`game.py:78`): landlord-empty → `{landlord:+2,
  farmer:-1}`; else `{landlord:-2, farmer:+1}`.
- `Env._get_reward` (`env.py:97`) is from the **landlord's perspective**:
  - `adp`: `±2 ** bomb_num`
  - `logadp`: `±(bomb_num + 1)`
  - `wp`: `±1.0`
- `bomb_num` increments on any of the 13 four-of-a-kinds **or** the rocket
  `[20,30]` (`game.py:13-16, 111-112`).
- The training actor negates the return for farmer positions
  (`utils.py:152`): `episode_return = env_output['episode_return'] if p ==
  'landlord' else -env_output['episode_return']`. There is **no** discounting;
  the Monte-Carlo return is broadcast across the episode's timesteps.

> Sign-convention landmine for P06: reward sign is flipped in two places
> (`Env._get_reward` and the actor loop). P06 must centralise perspective
> conversion.

## 7. Training loop (actor/learner)

```
train(flags)
 ├─ Model(device) per actor device  (shared memory)
 ├─ learner_model = Model(training_device)
 ├─ create_buffers  (shared-memory tensors per position/device)
 ├─ actor processes (spawn ctx): act(i,device,free_q,full_q,model,buffers,flags)
 └─ batch_and_learn threads: get_batch -> learn(position, ...)
```

- `learn` (`dmc.py:23`): MSE loss between model values and the Monte-Carlo
  target; `RMSprop`; grad-clip `max_grad_norm=40`.
- **Actor weight sync happens on every gradient step** (`dmc.py:57-58`):
  `actor_model.load_state_dict(learner_model.state_dict())`. This is the race
  P14 must eliminate (publish versioned snapshots, switch at game boundaries).
- Buffer layout (`utils.py:78-108`): `done`, `episode_return`, `target`,
  `obs_x_no_action (T, 319|430) int8`, `obs_action (T, 54) int8`,
  `obs_z (T, 5, 162) int8`. `T = unroll_length`.

## 8. Checkpoint formats (two, coexisting)

1. **Eval weights** `{savedir}/{xpid}/{position}_weights_{frames}.ckpt` — a bare
   `state_dict`. `DeepAgent._load_model` loads it with permissive key filtering:
   `pretrained = {k:v for k,v in pretrained.items() if k in model_state_dict}`.
   > P16 will replace this permissive filter with a strict manifest check.
2. **Training bundle** `{savedir}/{xpid}/model.tar` — dict with keys
   `model_state_dict` (per-role), `optimizer_state_dict` (per-role), `stats`,
   `flags` (`vars(flags)`), `frames`, `position_frames`. Used to resume.

Both formats are round-tripped by `tests/test_checkpoint_loader.py` using
synthetic (init-only) weights.

## 9. Evaluation

- `generate_eval_data.py`: pickled Python `list[dict]`; each dict has keys
  `landlord` (20 cards), `landlord_up` (17), `landlord_down` (17),
  `three_landlord_cards` (3). **Only deals, no bidding.**
- `evaluate.py` + `simulation.py`: multiprocessing; each worker replays fixed
  deals with injected agents. Reports **WP** (win percentage) and **ADP**
  (average doubled score) for landlord vs farmer team.
- Agents share the `.act(infoset) -> list[int]` interface:
  `DeepAgent`, `RandomAgent`, `RLCardAgent`, and `DummyAgent`.

## 10. What is MISSING (scope for later phases)

The legacy baseline deliberately omits several things AGENTS.md requires:

- ~~**No bidding / scoring state machine.**~~ **Resolved in P02:** a
  configurable `RuleSet` and a `DEAL→BIDDING→REVEAL_BOTTOM→PLAYING→TERMINAL`
  state machine are added to `GameEnv` (opt-in via `--ruleset standard`).
  Legacy mode is unchanged. See `docs/rules_and_scoring.md`. Training does
  not yet support standard mode (P05/P06).
- ~~**No spring / anti-spring** affecting reward.~~ **Resolved in P02:**
  spring/anti-spring detection and multipliers are in `douzero/env/scoring.py`
  (standard mode only; legacy has `spring_multiplier=0`).
- **TYPE_15 anomaly (generator/detector inconsistency).**
  `gen_type_11_serial_3_1` builds serial-3+1 wings by selecting single cards
  from `[c for c in hand if c not in serial_3_set]`. When the hand contains a
  four-of-a-kind, those four equal-rank singles become four "wings", yielding
  a 16-card move such as `[3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,7]` (four triples +
  a quad used as four single wings). `get_move_type` then classifies it as
  `TYPE_15_WRONG` because its serial-3 branch rejects any card whose
  multiplicity is 4 (`move_detector.py:92`).

  **This affects actual play, not just classification.** `gen_moves()` (the
  opening-lead enumerator) returns these actions, a model may play one, and
  the next player then sees it as the rival move. In
  `GameEnv.get_legal_card_play_actions` no `elif` matches `TYPE_15_WRONG`, so
  the response set collapses to `bombs + rocket + pass` — the responder
  cannot answer with a normal higher serial/triple. With a bomb-less hand the
  only legal response is pass.

  For the fixed P00 deal (landlord `[3,4,5,6,7]` ×4) the exact exception set is
  two moves (the quad can be either the wing rank or a serial-triple rank):
  `[3,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7]` and
  `[3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,7]`. Pinned exactly by
  `test_legal_actions_snapshot.test_landlord_opening_actions_classify_valid_with_known_exceptions`
  and the response-collapse test
  `test_quad_wing_wrong_move_collapses_response_to_bomb_or_pass`.
  **P02 decision: not fixed.** The anomaly is preserved in both legacy and
  standard modes to keep the P00 frozen snapshots unchanged. Reconciling the
  generator and detector is deferred to a later phase.
- ~~**No observation versioning.** Feature widths (319/373/430/484) are encoded
  implicitly in `dense1` shapes and buffer specs.~~ **Resolved in P03:** the
  `douzero/observation/` package introduces a versioned `FeatureSchemaManifest`
  whose every field width is derived from named constants in
  `cards.py` / `seats.py` / `schema.py` — no magic 319/373/430/484. Opt-in via
  `feature_version="v2"`; the legacy encoder is unchanged. See
  `docs/observation_v2.md`.
- ~~**Privileged data is not type-separated.** `InfoSet` mixes public and private
  fields; only `get_obs`'s projection keeps them out of deployment.~~
  **Resolved in P03:** `PublicObservation` (public only) and
  `PrivilegedObservation` (true hidden hands, training-only, in a `privileged`
  module/type) are separate, explicitly named types. The public encoder
  `get_obs_v2` recomputes the unseen pool from public information and ignores
  `all_handcards`; a leakage test asserts public obs is identical under hidden
  reallocation. Training integration → P05/P06.
- **No belief model.** Opponents' cards are pooled into one `other_hand_cards`
  vector, not modelled as a joint posterior. (→ P07)
- **Permissive checkpoint loader** silently drops unknown keys. (→ P16)
- **No calibration, no paired CI, no ablation harness.** Evaluation reports
  only raw WP/ADP. (→ P15)
- **No internal seeding anywhere.** The sole RNG is `np.random.shuffle` in
  `Env.reset` (`env.py:60`); reproducibility is the caller's responsibility.
  (→ P01 seeds; P00 adds seeding in tests/tools.)
- ~~**`file_writer.py` imports `git` (GitPython) at module load.**~~ **Resolved
  in P01:** the `import git` is now lazy (inside `gather_metadata`), GitPython
  is declared in `install_requires`, and `import douzero.dmc` / `train.py
  --help` work without GitPython. Actual training still needs the `git` binary
  for run-metadata stamping (the P00 Docker image still installs it).

## 11. Interface boundaries later phases must respect

| Boundary | Where | Later phase |
|---|---|---|
| Rule legality | `move_generator` / `move_detector` / `move_selector` | P02 `RuleSet` added (rules.py); move_generator/detector/selector NOT modified (TYPE_15 anomaly deferred) |
| Bidding/scoring | `GameEnv` (ruleset=None=legacy) / `Env` / `scoring.py` | P02 resolved (standard mode); training integration → P05/P06 |
| Observation schema | `get_obs` (role encoders) | P03 resolved: `douzero/observation/` adds versioned `PublicObservation`/`PrivilegedObservation` + schema + legacy adapter; `get_obs` unchanged (default `feature_version=legacy`) |
| Model input/widths | `Landlord/FarmerLstmModel.forward(z,x)` | P04 factorized forward (parity); P05 Model V2 |
| Deployment selection | `DeepAgent.act(infoset)` | P05/P16 `DeepAgentV2` (public-only) |
| Reward sign | `Env._get_reward` + actor negation | P06 centralised perspective |
| Checkpoint manifest | `{pos}_weights_*.ckpt`, `model.tar` | P16 strict `ModelManifest` |
| Eval deal format | pickled `list[dict]` | P02 resolved: v1 (legacy) + v2 (standard with deck/bidding); legacy adapter auto-detects |

## 12. Reproducing this baseline

See `docs/reproducibility.md` for the exact commands (seed, Docker, pytest,
`tools/capture_baseline.py`, `benchmarks/bench_legacy.py`).
