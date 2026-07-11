# Rules, Bidding, and Scoring (P02)

> Status: **P02**. This document describes the configurable rule engine,
> bidding state machine, and scoring system introduced in P02. The legacy
> mode (no bidding, no spring, `bomb_num` doubling) remains the default and is
> byte-for-byte unchanged.

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
| Training | Supported | **Not yet** (P05/P06) |
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

## Bidding observation

The bidding-phase observation contains only public information:

```python
{
    'phase': 'bidding',
    'position': 'landlord',           # current bidder
    'my_handcards': [3, 3, 4, ...],   # bidder's 17 cards
    'bidding_history': [('landlord', 2), ('landlord_down', 0)],
    'bidding_order': ['landlord', 'landlord_down', 'landlord_up'],
    'bid_values': [0, 1, 2, 3],
    'num_cards_left': {'landlord': 17, 'landlord_up': 17, 'landlord_down': 17},
}
```

It does **not** contain other players' hands or the bottom cards. The
bidding observation is separate from the cardplay `get_obs` encoding (which
is unchanged in P02; observation versioning arrives in P03).

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
- `first_bidder`: `"landlord"`
- `bidding_order`: `["landlord", "landlord_down", "landlord_up"]`
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

Standard-mode evaluation uses random bidding (the bidding policy is not yet
learned; it arrives with the model integration in P05/P06).

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

## Limitations (P02)

- **Training**: `--ruleset standard` is rejected by `train()`. Standard-mode
  training requires bidding observations in the buffer and a bidding value
  head in the model (P05/P06).
- **Bidding policy**: standard-mode evaluation uses random bidding, not a
  learned policy. Real bidding strength requires P05/P06.
- **Observation**: the cardplay-phase observation in standard mode is the same
  as legacy `get_obs` (it does not include bidding history). Observation
  versioning arrives in P03.
- **TYPE_15 anomaly**: the known generator/detector inconsistency
  (`gen_type_11_serial_3_1` producing 16-card quad-wing moves classified as
  `TYPE_15_WRONG`) is **not fixed** in P02. It affects both legacy and
  standard modes identically. See `docs/architecture/current.md` §10.
- **Reward sign**: standard-mode reward is derived from
  `GameResult.landlord_score`; the actor loop's farmer negation in
  `utils.py` is unchanged. Centralised perspective conversion arrives in P06.
