# Observation V2 — Public, Privileged, and History Schema (P03)

> Status: **P03 complete.** This page documents the versioned observation V2
> schema introduced in P03. The legacy encoder
> (`douzero/env/env.py:get_obs`) is unchanged and remains the default; V2 is
> opt-in via `feature_version="v2"`. Training integration arrives in P05/P06.

P03 separates **public** information (everything a deployment model may see)
from **privileged** information (true hidden hands, training-only) and gives
both a stable, versioned schema derived from named constants — no magic widths.

## 1. Why

The legacy `InfoSet` mixes public and private fields. The only thing keeping
hidden hands out of deployment is the `get_obs` projection. P03 makes that
boundary explicit and type-enforced:

- `PublicObservation` — the ONLY input a deployment model accepts.
- `PrivilegedObservation` — true hidden hands + training labels; lives in a
  module and type whose names contain `privileged`, and is never returned by
  the public encoder.

A leakage test asserts that two states with identical public information but
different hidden allocations produce **identical** public observations.

## 2. Package layout

```
douzero/observation/
  __init__.py        public re-exports
  cards.py           versioned 54-dim card encoding (source of truth)
  seats.py           canonical relative-seat mapping
  schema.py          FeatureSchemaManifest (every width derived from constants)
  history.py         HistoryTokenBatch + configurable max_history_len + mask
  public.py          PublicObservation + public unseen-pool helpers
  privileged.py      PrivilegedObservation (training-only)
  encode_v2.py       get_obs_v2 -> ObservationV2 (state once + action batch)
  legacy_adapter.py  reconstruct legacy x_batch/z_batch from V2 (parity bridge)
```

## 3. Public observation

`PublicObservation` contains only public information:

| field | meaning |
|---|---|
| `acting_role`, `seat_context` | acting role + relative-seat map (SELF/NEXT/PREVIOUS/LANDLORD/TEAMMATE/OPPONENT) |
| `phase`, `ruleset_id`, `bid_value`, `bidding_history` | phase + public bidding state |
| `bomb_count`, `rocket_count`, `total_multiplier` | public multiplier state |
| `my_handcards` | the acting player's hand |
| `other_handcards` | the public unseen pool (swap-invariant; equals the legacy `other_hand_cards` union) |
| `played_cards` | cumulative played cards per role |
| `last_move`, `last_move_dict` | the move to beat + per-role last move |
| `bottom_cards` | revealed bottom cards, their unplayed subset, owner (`landlord`) |
| `num_cards_left` | per-role remaining-card count |
| `legal_actions` | the candidate moves |

The unseen pool is recomputed from public information by
`compute_unseen_pool`:

```
parity pool = deck − my hand − all played cards
```

This reproduces the legacy `other_hand_cards` union exactly (the landlord sees
the 34 farmer cards; a farmer sees the 37 cards held by the landlord +
teammate). It is invariant under any re-allocation of hidden cards among
opponents — the property the leakage test enforces.

### Public bottom cards

- The bottom cards are revealed once the landlord is determined and are owned
  by the landlord (`bottom_cards.owner == "landlord"`).
- `bottom_cards.unplayed` is the subset not yet played by the landlord, tracked
  on a first-match removal basis (mirrors `GameEnv.step`).
- The bottom cards are **not** subtracted from `other_handcards` (the parity
  pool), because they are already part of the landlord's hand (landlord view)
  or part of the opponent pool (farmer view). The legacy encoder pools
  opponents into one vector and does not distinguish them, so V2 does the same
  at the feature level.
- A separate helper, `compute_belief_unknown_pool`, produces the pool a
  **belief model** (P07) must distribute among opponents. For a farmer this
  pool **excludes** the 3 unplayed public bottom cards (P03 spec point 6:
  "unplayed public bottom cards must not enter the unknown-card pool"). For the
  landlord the two pools are equal.

## 4. Privileged observation

`PrivilegedObservation` (module `douzero.observation.privileged`) holds:

- `all_handcards` — the true per-role hands (perfect information).
- `hidden_hand_labels` — optional belief-training labels.
- `terminal_target_win`, `terminal_target_score` — optional Monte-Carlo
  training labels.

It carries `kind="privileged"` so a type guard can reject it without
introspection. `is_privileged(obj)` returns True for the type or any dict
carrying that kind marker.

The public encoder (`get_obs_v2`) never constructs or returns a
`PrivilegedObservation`. The leakage test additionally asserts that corrupting
`infoset.all_handcards` does not change the encoded public observation — the
encoder recomputes the unseen pool from public info and ignores the privileged
field.

## 5. Schema (no magic widths)

`FeatureSchemaManifest` (`build_v2_schema`) records every field's name, shape,
and dtype. Widths are derived from named constants in `cards.py` / `seats.py` /
`schema.py`:

- `CARD_VECTOR_DIM = 54` (13 ranks × 4 + 2 jokers).
- `SEAT_ONEHOT_WIDTH = 6` (SELF/NEXT/PREVIOUS/LANDLORD/TEAMMATE/OPPONENT).
- `MOVE_TYPE_ONEHOT_WIDTH = 16` (TYPE_0..TYPE_15).
- `BOMB_ONEHOT_WIDTH = 15`, `MAX_CARDS_LEFT = 20`.

The legacy widths (319/373/430/484) are reproduced from the same constants by
`legacy_landlord_state_width()` / `legacy_farmer_state_width()` for
documentation and adapter parity — they are NOT used by the V2 encoder.

The state block is encoded **once per decision** (no legal-action batch dim);
the legal actions are encoded into a `LegalActionBatch` (one row per action);
the history is a padded `HistoryTokenBatch`.

## 6. History tokens

Each history token carries the required P03 fields: `actor_role`, `is_pass`,
`move_type`, `main_rank`, `length`, `card_count`, `cards_encoding`,
`cards_left_after`, `bomb_flag`, `phase`, and a `valid` padding mask.

`max_history_len` is configurable (default 100, which comfortably covers a full
game). Real tokens are placed at the start of the sequence; the rest is
zero-padding with a zero mask. Older moves beyond the cap are dropped.

## 7. Legacy adapter

`legacy_observation_from_v2(obs, card_play_action_seq=...)` rebuilds the legacy
`x_batch` / `z_batch` / `x_no_action` / `z` tensors from a V2 observation. This
is a transition bridge so a legacy model can consume a V2 observation without
any model-side change. Parity is asserted per role in
`tests/test_observation_legacy_adapter.py`.

## 8. Enabling V2

V2 is opt-in and does not change the default:

- CLI: `--feature_version v2` (accepted since P03; default is still `legacy`).
- YAML/dict config: `feature_version: v2`.
- Programmatic: `get_obs_v2(infoset)`.

The legacy `get_obs` path, the legacy checkpoints, and the legacy training loop
are byte-for-byte unchanged. Training does not yet consume V2 observations;
that wiring arrives in P05 (Model V2) and P06 (multi-objective training).

## 9. Tests

- `tests/test_observation_v2.py` — cards parity, relative seats, schema shapes,
  serialisation, the leakage invariant (public obs identical under hidden
  reallocation; privileged changes; encoder ignores `all_handcards`), card
  conservation, encode-once, standard-mode bottom cards.
- `tests/test_observation_legacy_adapter.py` — V2 → legacy tensor parity for
  all three roles, and legacy widths derived from constants.
