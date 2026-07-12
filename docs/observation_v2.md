# Observation V2 — Public, Privileged, and History Schema (P03)

> Status: **P03 complete.** This page documents the versioned observation V2
> schema introduced in P03. The legacy encoder
> (`douzero/env/env.py:get_obs`) is unchanged and remains the default; V2 is
> opt-in via `feature_version="v2"`. **V2 is NOT yet wired into training** —
> `train()` rejects `feature_version="v2"` until P05/P06 integrate the V2 model
> and buffers. V2 observations can be built and round-tripped today (evaluation
> adapters, tests); they just are not consumed by the actor/learner.

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
  cards.py           versioned 54-dim card encoding — the V2 source of truth
                     (the legacy _cards2array in env.py is a frozen compatibility
                     copy; new code should use cards.cards_to_vector)
  seats.py           canonical relative-seat mapping
  schema.py          FeatureSchemaManifest (every width derived from constants)
                     + stable_hash / compatibility_dict (identity stamps)
  history.py         HistoryTokenBatch: bounded history + left-truncation + mask
  public.py          PublicObservation + public unseen-pool helpers +
                     BiddingTokenBatch
  privileged.py      PrivilegedObservation (training-only)
  encode_v2.py       get_obs_v2 -> ObservationV2 (state once + action batch)
  legacy_adapter.py  reconstruct legacy x_batch/z_batch from V2 ALONE (no extra
                     args; parity bridge)
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

In addition to the state/action/history groups, the schema describes two
further model-consumable tensor blocks (item 3, so P05 never bypasses the
schema to read `obs.public` ad hoc):

- a **public context block** (`context_fields`): bottom-card revealed/unplayed
  card vectors, bid-value one-hot, phase one-hot, rocket count, total
  multiplier (int32 — unbounded in the standard ruleset), and a ruleset-family
  id one-hot. Encoded once into `PublicContextBlock`.

  The context block encodes the **ruleset family id** (legacy/standard), not
  the full `ruleset_version`/`ruleset_hash`. The complete rule identity remains
  compatibility metadata on `PublicObservation` and must be checked by the
  checkpoint/runtime boundary (a model is bound to one ruleset hash via the
  manifest, not via a feature). If a future model must distinguish multiple
  custom rulesets, it should encode the actual rule parameters as features,
  not the SHA-256 hash bytes.
- a **bidding-token block** (`bidding_token_fields`): `[bid_seat(3),
  bid_value(4), is_pass(1)]` per bid. Encoded into `SchemaBiddingTokenBatch`.

Both groups are covered by `stable_hash()`, so adding/removing/reordering any
field changes the schema identity.

## 6. History tokens (bounded history, left-truncation)

The V2 history is a **bounded** history: it keeps at most `max_history_len`
tokens (configurable; default 100, which comfortably covers a full game). When
the public action history exceeds the cap, the **oldest** moves are dropped
(left-truncation) and the real tokens are left-aligned. This bounded/truncated
contract is recorded by the `TRUNCATION_SEMANTICS_VERSION` stamp.

Each history token carries the required P03 fields: `actor_role`, `is_pass`,
`move_type`, `main_rank`, `length`, `card_count`, `cards_encoding`,
`cards_left_after`, `bomb_flag`, `phase`, and a `valid` padding mask.

Mask contract (explicit, item 6):

- `valid_mask`: int8, `1` for a real token, `0` for padding.
- `key_padding_mask`: bool, `True` for **padding** (the PyTorch Transformer
  convention where True means "ignore this position"). It is the exact boolean
  negation of `valid_mask`.
- `original_length`: the full move count before truncation.
- `was_truncated`: `True` iff some oldest moves were dropped.
- `truncation_side`: always `"left"` for this schema.

Padding slots are all-zero. Editing padding content cannot affect the mask
contract: the mask is what defines validity, not the token content.

## 7. Schema identity (stable hash)

`FeatureSchemaManifest.stable_hash()` is a description-stable SHA-256 of
`compatibility_dict()`, which includes every field's name/shape/dtype (per
group), `max_history_len`, and the semantic version stamps (card encoding,
move-type encoding, seat mapping, history encoding, mask semantics, truncation
semantics). `description` text is excluded, so documentation churn does not
change a model's identity contract.

`ObservationV2` carries `feature_schema_version` and `feature_schema_hash` (the
full 64-char hash) so a checkpoint/model can reject an incompatible schema
precisely. The hash changes on any field name/shape/dtype/order change,
`max_history_len` change, or stamp change.

## 8. Deep immutability

`ObservationV2`, `StateBlock`, `LegalActionBatch`, `BiddingTokenBatch`, and
`HistoryTokenBatch` are `frozen` dataclasses; `PrivilegedObservation` is
`frozen` + `slots`. Every numpy array they hold is read-only (`write=False`).
Caller-supplied lists/dicts are copied at construction, so mutating the source
infoset after building an observation does not retroactively alter it, and the
public and privileged containers share no ndarray.

## 9. Legacy adapter (depends only on ObservationV2)

`legacy_observation_from_v2(obs)` rebuilds the legacy `x_batch` / `z_batch` /
`x_no_action` / `z` tensors from a V2 observation **alone** — it takes no extra
infoset or action-sequence argument. The raw public action sequence it needs to
rebuild `z` is stored on `ObservationV2.card_play_action_seq` (an immutable
tuple of tuples) at encode time. This is a transition bridge so a legacy model
can consume a V2 observation without any model-side change. Parity is asserted
per role, for short and long (>15-move) histories, and across varying
legal-action counts in `tests/test_observation_legacy_adapter.py`.

## 9b. Rule identity (never empty, never inconsistent)

`get_obs_v2` resolves the ruleset identity so the stamped hash is never empty
and never self-contradictory (review round 3, blocker 3):

- **Preferred:** pass `ruleset=RuleSet.legacy()` / `RuleSet.standard()`. The
  encoder derives `ruleset_id` / `ruleset_version` / `ruleset_hash` from
  `RuleSet.identity()`.
- **Validated fallback:** pass `ruleset_id` (+ optional version/hash). Legacy
  auto-fills the canonical legacy hash when omitted; standard requires an
  explicit version AND hash and rejects `standard` + `legacy-v1`; a hash that
  disagrees with the canonical one is rejected; passing both `ruleset=` and
  `ruleset_id=` is rejected.

The legacy default (`get_obs_v2(infoset)` with no rule arguments) now stamps
the full canonical legacy hash, not an empty string.

## 10. Enabling V2

V2 is opt-in and does not change the default:

- CLI: `--feature_version v2` (accepted since P03; default is still `legacy`).
- YAML/dict config: `feature_version: v2`.
- Programmatic: `get_obs_v2(infoset)`.

The legacy `get_obs` path, the legacy checkpoints, and the legacy training loop
are byte-for-byte unchanged. **Training does not yet consume V2 observations**:
`train()` rejects `feature_version="v2"` up front (before any CUDA/checkpoint/
actor initialisation) until P05 (Model V2) and P06 (multi-objective training)
wire the V2 schema into the actor/learner and buffers.

## 11. Deployment boundary note

The imperfect-information boundary for the V2 path is enforced at multiple
layers:

- the public encoder recomputing the unseen pool from public info and ignoring
  `infoset.all_handcards` (the leakage test replaces `all_handcards` with an
  access-throws sentinel and asserts `get_obs_v2` still succeeds);
- the public encoder not importing the `privileged` module;
- `PublicObservation` serialization containing no hidden-hand field;
- **P05**: `DeepAgentV2` (`douzero/evaluation/deep_agent.py`) provides the
  canonical type guard — `act_v2(obs)` rejects a `PrivilegedObservation` by
  type **before any model call**. See `docs/model_v2.md`.

## 12. Tests

- `tests/test_observation_v2.py` — cards parity, relative seats, schema shapes,
  serialisation, the leakage invariant (public obs identical under hidden
  reallocation; privileged changes; encoder ignores `all_handcards`), card
  conservation, encode-once, standard-mode bottom cards.
- `tests/test_observation_v2_hardening.py` — schema identity (description-stable
  hash, hash changes), public bottom-card semantics (revealed vs unplayed),
  deep immutability (frozen/readonly arrays/source-isolation), the history
  contract (valid_mask/key_padding_mask/truncation), leakage hardening
  (access-throws sentinel, no privileged import, no hidden field), and
  model-consumable public-input presence.
- `tests/test_observation_legacy_adapter.py` — V2 → legacy tensor parity for
  all three roles (short and >15-move histories, varying legal-action counts),
  the no-arg adapter signature, and legacy widths derived from constants.
