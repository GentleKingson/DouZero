# Human-Game Data Pipeline and Listwise Behaviour Cloning (P08)

> **Status:** implemented behind `human_prior_enabled` (default off). The data
> pipeline (ingest → validate → split → sample) is production-shaped but is
> exercised only on **synthetic random self-play** data here; real
> playing-strength is **not measured** (no authorized human dataset in this
> phase). Legacy and prior-disabled V2 paths are byte-identical to P07.

This phase adds the AGENTS.md "Human-game data and strategy priors" layer: a
validated human-game dataset pipeline and a **listwise** behaviour-cloning
prior that scores the current legal-action list (never a global action-class
catalogue).

It implements the project's data-safety rules:

- Use only lawfully obtained and authorized data. **No** scraping, account
  automation, anti-detection, or platform-ToS-bypass code lives in this repo.
- Do not store personal identifiers or credentials (forbidden metadata keys are
  dropped at the ingest boundary).
- Canonicalize and validate every recorded game by **replaying it through the
  rule engine**; quarantine invalid games, never silently repair them.
- Split data by complete game to prevent leakage.
- Do not train only on won games (the record keeps the full `final_result` so
  downstream sampling can stratify).

## Imperfect-information boundary

The raw [`HumanGameRecord`](../douzero/human_data/schema.py) carries
`initial_hands` (the true deal) and the recorded human actions — these are
**privileged training-only data**, analogous to
[`PrivilegedObservation`](../douzero/observation/privileged.py). The BC
*student* model only ever receives the public
[`ObservationV2`](../douzero/observation/encode_v2.py) produced by replaying
the record; the recorded human action becomes the `human_action_index` label,
carried in a separately-stamped [`BCSample`](../douzero/human_data/sample.py)
(`kind="bc_sample"`) that the deployment `DeepAgentV2.act` never accepts.

## Canonical data format (JSONL)

A canonical human game is one JSONL line — a self-describing
`HumanGameRecord` with `format_version` / `schema_version` / `kind` stamps.
JSONL is chosen over Parquet because it adds **no new runtime dependency**
(`pyarrow`/`pandas` are not project deps), is cross-language, and avoids
pickle's arbitrary-code-execution risk. Fields:

| Field | Purpose |
|-------|---------|
| `game_id` | Opaque `dzg_<64-hex-digest>` id (split + dedupe key) |
| `ruleset_id` / `ruleset_version` / `ruleset_hash` | Rule identity the game was played under |
| `seats` | Ordered role tuple the `action_history` positions refer to |
| `initial_hands` | The true deal (**privileged**); legacy 4-key card_play_data |
| `bottom_cards` | The 3 revealed bottom cards (entity identity) |
| `bidding_history` | `((seat, bid_value), ...)`; empty for legacy cardplay |
| `action_history` | `((position, sorted_cards), ...)`; empty tuple = pass |
| `final_result` | `winner_team`, `winner_position`, + optional scoring |
| `player_skill_weight` | Per-role non-negative sample weight |
| `source_metadata` | Audit-only provenance (sanitized) |

## Pipeline (four stages)

```
external records ──ingest_human_games──► canonical JSONL
                                              │
                                              ▼
                                   validate_human_games (replay)
                                       │           │
                                    valid      quarantine
                                       │
                                       ▼
                                 split (by game_id)
                                       │
                                       ▼
                              BCSample (public obs + index)
                                       │
                                       ▼
                              pretrain_bc (listwise CE)
```

1. **Ingest** (`ingest_human_games.py`): an
   [`Adapter`](../douzero/human_data/adapters.py) converts an external-format
   payload into a canonical record. No platform format is hard-coded; adapters
   are plugged via a dotted import path. Metadata is sanitized (forbidden
   identifier/credential keys dropped), duplicates removed by `game_id`, output
   sorted for reproducibility. `--synthetic` generates deterministic
   random-self-play records when no `<HUMAN_DATA_PATH>` exists.

2. **Validate** (`validate_human_games.py`): every record is replayed
   action-by-action through [`GameEnv`](../douzero/env/game.py) (the rule
   engine is the single source of truth for legality). A record is valid iff
   the deal partitions the 54-card deck, every action is played by the role
   whose turn it is, every action is legal at its decision, the game
   terminates with exactly one empty hand, and the winner matches
   `final_result`. Invalid records are **quarantined** with a diagnostic
   reason — never silently repaired.

3. **Split** ([`split.py`](../douzero/human_data/split.py)): by `game_id` into
   train/val/test with **no overlap** (asserted). Optional `winner_team`
   stratification guards survivorship bias. Deterministic per
   `(seed, ratios)`.

4. **Sample** ([`sample.py`](../douzero/human_data/sample.py)): at each
   non-trivial decision during replay, snapshot the public `ObservationV2` and
   locate the recorded human action's index in `obs.actions.legal_actions`
   (sorted-tuple canonical match). That index is the listwise-CE target.

## Listwise BC (the model side)

- [`PriorHead`](../douzero/models_v2/heads.py): an `nn.Linear(hidden_size, 1)`
  producing one prior logit per legal action, gated by the existing
  `human_prior_enabled` config flag (already a checkpoint identity axis).
- [`ModelOutput.prior_logit`](../douzero/models_v2/output.py): optional; the
  `pure_prior` decision mode (P08 ablation) argmaxes it.
- [`listwise_bc_loss`](../douzero/training/bc_loss.py): masked cross-entropy
  over the N legal actions. **No global 27472-class catalogue.**
- [`BCTrainer`](../douzero/training/bc_trainer.py): pretrains the prior head
  on BC samples (per-decision forward, RMSprop, fail-closed non-finite guard,
  early stopping, top-1/NLL metrics).
- **RL + BC combination** (task 11): set `loss.lambda_bc > 0` and pass
  `bc_aux_samples=` to [`V2Trainer`](../douzero/training/v2_trainer.py); each
  optimizer step adds `lambda_bc * L_BC` to the multi-objective RL loss in a
  single backward pass. `bc.schedule` (`constant` | `linear_decay`) controls
  how the weight evolves.

## Usage

### CPU smoke (no real data)

```bash
# 1. Generate synthetic canonical records.
python ingest_human_games.py --synthetic --num_synthetic 16 \
    --output /tmp/games.jsonl

# 2. Validate by replay (writes valid.jsonl + valid.quarantine.jsonl).
python validate_human_games.py --input /tmp/games.jsonl --output /tmp/valid

# 3. Pretrain the prior head.
python pretrain_bc.py --data /tmp/valid.jsonl \
    --save_dir /tmp/bc_prior --epochs 5 --batch_size 16
```

### With authorized human data

Write an [`Adapter`](../douzero/human_data/adapters.py) using the strict keyed
contract
`(raw_mapping, *, pseudonymizer) -> AttestedAdapterRecord` and configure the
project key outside the repository:

```bash
export DOUZERO_HUMAN_DATA_HMAC_KEY_FILE=/secret-store/douzero-hmac.key
python ingest_human_games.py --input raw_export.jsonl \
    --adapter mypkg.adapters.PlatformAAdapter \
    --output /tmp/games.jsonl
python validate_human_games.py --input /tmp/games.jsonl --output /tmp/valid
python pretrain_bc.py --data /tmp/valid.jsonl --save_dir /tmp/bc_prior
```

The CLI accepts a function, a callable instance, or a callable class with a
zero-argument constructor. A class that needs configuration must be exposed as
an adapter function that closes over configuration or as a preconfigured
callable instance.

Adapters must use the pseudonymizer supplied by ingest and return its opaque
attestation with the record:

```python
from douzero.human_data import AttestedAdapterRecord

def convert(raw, *, pseudonymizer):
    identity = pseudonymizer.pseudonymize(raw["platform_game_id"])
    record = build_record(raw, game_id=identity.game_id)
    return AttestedAdapterRecord(record=record, identity=identity)
```

External ingest fails closed unless `--hmac-key-file` or
`DOUZERO_HUMAN_DATA_HMAC_KEY_FILE` supplies at least 32 bytes. The adapter never
receives raw key bytes. Ingest verifies the keyed attestation and rejects an
unkeyed SHA-256 or any merely regex-shaped `dzg_...` identifier. Synthetic mode
continues to use internal fixture IDs and requires no secret.

## Configuration (`configs/enhanced.yaml`)

The BC auxiliary loss is enabled **iff** `loss.lambda_bc > 0` (single source of
truth — there is no separate `bc.enabled` or `bc.lambda_bc`). To turn on RL+BC:

```yaml
model:
  human_prior_enabled: true    # build the prior head

loss:
  lambda_bc: 0.3               # > 0 enables the BC auxiliary term (sole switch)

bc:                            # BC-specific settings (no enabled/lambda_bc here)
  data_path: /path/to/validated.jsonl   # validated canonical JSONL
  temperature: 1.0
  label_smoothing: 0.0
  skill_weight_clip: 10.0
  schedule: constant           # constant | linear_decay
  schedule_steps: 0
  schedule_floor: 0.0          # residual prior floor (never forced to 0)
```

## Compatibility

- **Legacy / prior-disabled V2**: byte-identical to P07. The prior head is an
  additive `nn.Linear` behind `human_prior_enabled` (default `False`); no
  existing checkpoint changes.
- **Prior-enabled V2 checkpoints**: a different `stable_hash()` (the flag is
  in `compatibility_dict`) plus extra `prior_head.*` keys; the strict V2 loader
  rejects a cross-load by design.
- **No new runtime dependencies** (stdlib + existing `pyyaml`/`torch`).

## Data authorization, privacy, and audit

- Adapters MUST NOT reach the network or perform scraping/automation. They
  consume already-acquired, authorized files only.
- `source_metadata` is a flat allowlist: `source`, `license`,
  `dataset_version`, `batch_id`, and `collection_method`. Each value has a
  bounded type, length, and character set. Unknown/nested fields are dropped by
  [`audit_source_metadata`](../douzero/human_data/adapters.py) and rejected at
  the canonical record boundary.
- Canonical schema v2 rejects raw external game IDs. Existing draft schema-v1
  JSONL must be re-ingested from the authorized source with keyed HMAC IDs;
  there is deliberately no converter that would copy old raw IDs forward.
- Deletion: removing a record from the canonical JSONL and re-running
  `validate_human_games.py` fully removes it from the pipeline (records are
  never copied elsewhere).
- License provenance is recorded in `source_metadata.license` and audited in
  [third_party_review.md](third_party_review.md).

## What is NOT measured here

- Real playing-strength (no authorized human dataset in this phase). The smoke
  only verifies the pipeline runs and the BC loss decreases on synthetic
  random self-play (a trivial target). Any strength claim requires authorized
  data + paired evaluation (P15) and is recorded as **未测** here.
- The uncertainty-gated prior fusion in deployment (that is P09's
  `final_score = base_utility + alpha * uncertainty_gate * prior`); P08 only
  adds the head, the listwise loss, the `pure_prior` ablation mode, and the
  RL+BC combination hook.
