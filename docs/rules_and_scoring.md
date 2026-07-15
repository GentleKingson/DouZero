# Rules, Bidding, and Scoring

> Status: the P02 rule engine is complete, and P17 adds opt-in standard V2
> training with a learned public bidding head. The legacy mode (no bidding,
> no spring, `bomb_num` doubling) remains the default and is unchanged.

## Overview

P02 upgrades the "randomly assigned landlord" environment into a configurable
DouDizhu environment with a full bidding and scoring state machine. Two
rulesets are supported:

| Feature | `legacy` (default) | `standard` (opt-in) |
|---|---|---|
| Bidding | None — landlord gets 20 cards | 0/1/2/3 score bidding |
| All-pass redeal | N/A | Yes |
| Landlord selection | Fixed (first 20 cards) | Highest bidder |
| Bottom cards | Dealt into landlord hand at start | Revealed after bidding |
| Spring / anti-spring | No | Yes (×2 each) |
| Bomb multiplier | `2 ** bomb_num` (rocket counted as bomb) | `2 ** bomb_count` |
| Rocket multiplier | (included in bomb_num) | ×2 (separate) |
| Base score | 2 (landlord wins +2, farmer loses −1) | 1 × bid |
| Training | Supported | V2 single-process, opt-in (`configs/standard_v2.yaml`) |
| Evaluation | Cardplay-only | End-to-end bidding+cardplay |

## RuleSet dataclass

All rule parameters are centralised in `douzero/env/rules.py` as a frozen
`RuleSet` dataclass. No magic numbers for multipliers or base scores should
appear elsewhere.

```python
from douzero.env.rules import RuleSet

# Legacy (default): reproduces the original environment exactly.
rs = RuleSet.legacy()

# Standard: 0/1/2/3 bidding, spring, full scoring.
rs = RuleSet.standard()

# Custom (from dict/YAML):
rs = RuleSet.from_dict({"ruleset_id": "standard", "spring_multiplier": 3})
```

### Fields

| Field | Type | Legacy | Standard | Description |
|---|---|---|---|---|
| `ruleset_id` | str | `"legacy"` | `"standard"` | Recorded in checkpoint manifest |
| `bidding_mode` | str | `"none"` | `"score_0_1_2_3"` | Bidding phase type |
| `bid_values` | tuple[int] | `()` | `(0,1,2,3)` | Allowed bid values |
| `allow_rob` | bool | False | False | Reserved (抢地主) |
| `all_pass_redeal` | bool | False | True | Redeal if all pass |
| `bid_multiplier` | bool | False | True | Bid multiplies base score |
| `bomb_multiplier` | int | 2 | 2 | Per-bomb multiplier (exponent base) |
| `rocket_multiplier` | int | 2 | 2 | Rocket multiplier |
| `spring_multiplier` | int | 0 | 2 | Spring multiplier (0 = disabled) |
| `anti_spring_multiplier` | int | 0 | 2 | Anti-spring multiplier (0 = disabled) |
| `allow_double` | bool | False | False | Reserved (加倍) |
| `base_score` | int | 2 | 1 | Base score before multipliers |
| `max_multiplier` | int\|None | None | None | Optional cap |

## State machine

In standard mode, `GameEnv` transitions through five phases:

```
DEAL → BIDDING → REVEAL_BOTTOM → PLAYING → TERMINAL
```

Each phase only accepts its corresponding actions:

- **DEAL**: cards are dealt (17+17+17+3 bottom).
- **BIDDING**: each bidder submits a bid value (0/1/2/3). After all three
  bid, the highest bidder becomes the landlord. Ties are broken by bidding
  order (first bidder wins). If all pass and `all_pass_redeal=True`, the game
  signals a redeal.
- **REVEAL_BOTTOM**: the 3 bottom cards are added to the landlord's hand
  (now 20 cards). The bottom card identities are preserved for tracking.
- **PLAYING**: standard DouDizhu cardplay (identical to legacy — the same
  `move_generator`/`move_detector`/`move_selector` pipeline).
- **TERMINAL**: a `GameResult` is produced with the full scoring breakdown.

Calling an action from the wrong phase raises `IllegalPhaseError`. Submitting
an invalid bid raises `IllegalActionError`.

## Bidding observations

The environment's raw bidding-phase observation contains only public
information. Before a landlord exists, seats are neutral identifiers
(`"0"`, `"1"`, and `"2"`), not landlord/farmer roles:

```python
{
    'phase': 'bidding',
    'position': '0',                  # current neutral seat
    'my_handcards': [3, 3, 4, ...],   # bidder's 17 cards
    'bidding_history': [('0', 2), ('1', 0)],
    'bidding_order': ['0', '1', '2'],
    'bid_values': [0, 1, 2, 3],
    'num_cards_left': {'0': 17, '1': 17, '2': 17},
}
```

It does **not** contain other players' hands or the bottom cards. P17's
`get_bidding_obs_v2()` converts this mapping to a separate, versioned
`BiddingObservationV2`. The encoder reads an explicit public allow-list,
requires exactly 17 private cards for the current bidder, preserves neutral
seats, and binds the fixed `0/1/2/3` action mask, ruleset identity, redeal
count, auction history, phase, and finite public-style features into the
bidding schema hash. It ignores unknown keys rather than admitting hidden
state. This contract is intentionally separate from card-play
`ObservationV2`, which uses roles only after the auction is resolved.

## Scoring

### Legacy scoring (unchanged)

```
bomb_num = bomb_count + rocket_count  (rocket counted as a bomb)
multiplier = 2 ** bomb_num
landlord_score = ±2 * multiplier
farmer_score = ∓1 * multiplier
```

Score conservation: `landlord_score + 2 * farmer_score == 0`.

### Standard scoring

```
base = base_score * bid_value              (bid_value ≥ 1)
multiplier = bomb_multiplier**bomb_count
           * rocket_multiplier**rocket_count
           * (spring ? spring_multiplier : 1)
           * (anti_spring ? anti_spring_multiplier : 1)
total = base * multiplier                  (capped by max_multiplier if set)
landlord_score = ±2 * total
farmer_score = ∓total
```

Score conservation: `landlord_score + 2 * farmer_score == 0` always holds.

### Spring / anti-spring

- **Spring** (地主春天): the landlord wins and neither farmer ever played a
  valid (non-pass) action. The farmers only passed throughout the game.
- **Anti-spring** (农民反春): the farmers win and the landlord played exactly
  one valid action (the opening lead), then never played again.

Pass (`[]`) does not count as a valid play for spring detection.

## GameResult

`douzero/env/scoring.py` defines a `GameResult` dataclass with:

- `winner_team`: `"landlord"` or `"farmer"`
- `winner_position`: the specific position that emptied its hand
- `bid_value`: 0 (legacy) or 1/2/3 (standard)
- `bomb_count`, `rocket_count`: separate counts
- `spring`, `anti_spring`: booleans
- `multiplier_breakdown`: dict of each multiplier component
- `total_multiplier`: the final multiplier
- `landlord_score`, `farmer_score`: signed scores

`Env.step()` returns `GameResult.to_dict()` in the terminal `info` dict for
standard mode. Legacy mode still returns an empty `info` dict.

## Evaluation data formats

Two formats coexist, auto-detected by `load_eval_data()`:

### Legacy format (v1)

A pickled `list[dict]`; each dict has:
- `landlord` (20 sorted cards)
- `landlord_up` (17 sorted cards)
- `landlord_down` (17 sorted cards)
- `three_landlord_cards` (3 sorted cards)

### Standard format (v2)

A pickled `list[dict]`; each dict has:
- `format_version`: `2`
- `ruleset_id`: `"standard"`
- `deck`: full 54-card order (not pre-sliced)
- `first_bidder`: one of `"0"`, `"1"`, or `"2"`
- `bidding_order`: a rotation/permutation of `["0", "1", "2"]`
- `bidding_script`: `None` (reserved for fixed bidding scripts)

The adapter raises a precise error if the format does not match the requested
ruleset.

## Usage

### Generating standard eval data

```bash
python generate_eval_data.py --output eval_standard --num_games 100 --ruleset standard
```

### Running standard evaluation

```bash
python evaluate.py \
  --landlord random \
  --landlord_up random \
  --landlord_down random \
  --eval_data eval_standard.pkl \
  --num_workers 1 \
  --ruleset standard
```

This historical `evaluate.py` path still uses deterministic-seeded random
bidding. Learned-bidding evaluation uses `evaluate_paired.py` and a bundle
whose manifest-bearing V2 checkpoint has `bidding_enabled=true`; the evaluator
strictly validates the bidding schema and invokes the separate bidding head.

### Using the environment directly

```python
from douzero.env.env import Env
from douzero.env.rules import RuleSet

# Legacy (default):
env = Env("adp")
obs = env.reset()  # returns get_obs dict

# Standard:
env = Env("adp", ruleset=RuleSet.standard())
obs = env.reset()  # returns bidding observation
obs, reward, done, info = env.step(None, bid_value=1)  # bidding
obs, reward, done, info = env.step([3], bid_value=None)  # cardplay
```

## Training and current limitations

- **Entry point**: legacy `train.py` still rejects standard rules by design.
  Use `train_v2.py --config configs/standard_v2.yaml` for full
  bidding/redeal/reveal/card-play training. The V2 trainer records separate
  card-play and bidding transitions, discards auction transitions from
  abandoned all-pass deals, labels terminal auctions from the landlord
  perspective, and can atomically save/resume the combined state.
- **Bidding head**: `model.bidding_enabled` defaults to `false`, so existing
  V2 model graphs and checkpoint identities remain unchanged. Standard V2
  training requires it to be `true`. Its masked policy CE, landlord-win BCE,
  landlord-score Huber, and optional regret terms are configured independently.
  `random`, `rule`, `max`, and `pass` are explicit warm-start policies;
  `learned` mixes the learned head with a configured fallback while retaining
  action-source provenance.
- **Distributed and compile**: bidding-enabled standard training currently
  fails closed under DDP. It also rejects `compile_model`; the auction uses a
  separate `forward_bidding` contract that has not been validated through
  `torch.compile`. Run this path single-process and eager until those two
  implementation gates are closed.
- **Evaluation scope**: `evaluate.py` remains the random-bidding compatibility
  path. Use the P15/P17 paired evaluator for learned bidding; no playing-strength
  claim follows merely from the learned head being wired end to end.
- **Observation**: the legacy compatibility encoder remains unchanged.
  Standard V2 training uses versioned `ObservationV2`, whose public context
  includes the terminal bid and bounded bidding-history encoding; this is
  distinct from the neutral-seat pre-landlord bidding observation above.
- **TYPE_15 anomaly**: the known generator/detector inconsistency
  (`gen_type_11_serial_3_1` producing 16-card quad-wing moves classified as
  `TYPE_15_WRONG`) is **not fixed** in P02. It affects both legacy and
  standard modes identically. See `docs/architecture/current.md` §10.
- **Reward sign**: standard-mode environment reward is derived from
  `GameResult.landlord_score`. The V2 path converts the terminal result once
  into explicit landlord/farmer `team_targets`; card-play replay reads the
  acting role's targets, while neutral bidding replay deliberately uses the
  landlord-side targets.
