# P17 Authorized Human-Data Canary

## Result

**Real authorized-data canary: NOT RUN**  
**Reason: no DOUZERO_HUMAN_DATA_PATH supplied**

`DOUZERO_HUMAN_DATA_HMAC_KEY_FILE` was also unset on the audit host. No private
dataset or secret was opened, copied, logged, committed, placed in a
checkpoint, or included in a model package.

No real-data game counts, rule/seat/outcome/bid distributions, duplicate
counts, quarantine rates, BC metrics, paired estimates, or confidence
intervals exist. They must remain `NOT AVAILABLE` until an authorized export
is supplied and the entire canary completes.

## Synthetic Code Smoke

A deterministic synthetic fixture exercised ingest, validation, game-level
split, BC pretraining, RL+BC, and paired-evaluation code paths. This is only a
software smoke, not evidence about human behavior or model strength:

```bash
.venv/bin/python ingest_human_games.py --synthetic \
  --num_synthetic 4 --synthetic_seed 17 \
  --output /tmp/douzero-p17-synthetic-games.jsonl
.venv/bin/python validate_human_games.py \
  --input /tmp/douzero-p17-synthetic-games.jsonl \
  --output /tmp/douzero-p17-synthetic-valid
```

Observed synthetic result: 4 total, 4 valid, 0 quarantined, 0 parse errors.
All four records use the legacy ruleset and contain no bidding phase. Seats are
balanced at four records per role; winners are landlord 2, landlord_down 1,
landlord_up 1 (landlord team 2, farmer team 2). All four IDs are unique and
match the canonical `dzg_<64 hex>` form. There is no raw platform identity in
a synthetic fixture, so that check does not substitute for a real adapter
privacy audit.

The deterministic complete-game split used seed 17 and produced train 2,
validation 1, test 1, with zero `game_id` overlap. A one-epoch BC smoke over
115 decisions completed and wrote a manifest-bearing checkpoint. On the
single-game test split, the descriptive metrics were:

| Role | Decisions | Top-1 | Top-3 | NLL |
| --- | ---: | ---: | ---: | ---: |
| landlord | 10 | 0.400 | 0.700 | 1.7933 |
| landlord_down | 8 | 0.375 | 0.750 | 1.4841 |
| landlord_up | 5 | 0.200 | 0.600 | 1.8394 |

These tiny, synthetic holdout values are pipeline diagnostics only. They are
not a BC quality claim.

```bash
.venv/bin/python pretrain_bc.py \
  --data /tmp/douzero-p17-synthetic-valid.jsonl \
  --save_dir /tmp/douzero-p17-bc --save_name synthetic-bc.pt \
  --epochs 1 --batch_size 4 --val_ratio 0.25 \
  --hidden_size 16 --history_layers 1 --history_heads 1 \
  --history_encoder lstm --seed 17

.venv/bin/python train_v2.py \
  --config /tmp/douzero-p17-rlbc.yaml --episodes 1 \
  --optimizer_steps 1 --batch_size 1 --buffer_capacity 64 \
  --checkpoint_path /tmp/douzero-p17-rlbc.pt --seed 17
```

The RL+BC command collected 19 card-play transitions, completed one optimizer
step, changed parameters, and recorded finite total loss 1.4469 plus BC
cross-entropy 2.6577. The CLI logged only the aggregate 115-sample count and
did not print the configured dataset path.

After converting the training checkpoints to strict public-policy sidecars,
the P15 path compared the synthetic BC prior with its untrained initialization
on four generated deals and 2,000 deal-level bootstrap samples:

```bash
.venv/bin/python evaluate_paired.py --mode cardplay_only \
  --candidate synthetic-bc --baseline synthetic-bc-untrained \
  --model-matrix /tmp/douzero-p17-bc-matrix.json \
  --num-deals 4 --seed 17 --bootstrap-samples 2000 \
  --output /tmp/douzero-p17-bc-before-after
```

Observed smoke estimate: +0.1250 with CI [0.0000, 0.3750]. Four synthetic
deals are far below the 1,000-deal promotion gate; the number is recorded only
to prove the paired path executed and must not be interpreted as improvement.

## Authorized Invocation

The ingest CLI defaults `--input` from `DOUZERO_HUMAN_DATA_PATH` and loads the
project key from `--hmac-key-file` or
`DOUZERO_HUMAN_DATA_HMAC_KEY_FILE`. External ingest fails closed without it.
The CLI supplies adapters a redacted pseudonymizer object; adapters return its
opaque keyed identity attestation with each record. Ingest verifies the
attestation before writing, so a merely regex-shaped canonical ID is rejected.
Neither the key nor its path is logged or copied into canonical data.

```bash
export DOUZERO_HUMAN_DATA_PATH=/outside/repo/authorized-export.jsonl
export DOUZERO_HUMAN_DATA_HMAC_KEY_FILE=/secret-store/douzero-hmac.key

.venv/bin/python ingest_human_games.py \
  --adapter authorized_adapter.convert \
  --output /outside/repo/canary/canonical.jsonl
.venv/bin/python validate_human_games.py \
  --input /outside/repo/canary/canonical.jsonl \
  --output /outside/repo/canary/validated
```

The adapter must consume an already authorized export and must not scrape,
automate accounts, bypass platform controls, or access a network service.

Exercise a game-level deletion by rebuilding to a new file. The command emits
aggregate counts only and never prints the excluded ID or record contents:

```bash
.venv/bin/python tools/rebuild_human_dataset.py \
  --input /outside/repo/canary/validated.jsonl \
  --output /outside/repo/canary/validated-after-deletion.jsonl \
  --exclude-ids-file /secret-store/deletion-request-ids.txt
```

Revalidate and regenerate every downstream split/checkpoint from the rebuilt
file; do not edit an already trained package and call that deletion complete.

## Canary Gates

| Stage | Status |
| --- | --- |
| Authorization and license evidence | Blocked: no dataset |
| HMAC pseudonymization | Implemented; real run not tested |
| Canonical ingest and deduplication | Synthetic smoke only |
| Replay validation and quarantine | Synthetic smoke only |
| Train/validation/test split and leakage audit | Synthetic smoke only: 2/1/1, zero overlap |
| BC pretraining and per-role top-k/NLL | Synthetic smoke only; table above |
| Optional belief labels | Not run |
| RL+BC short canary | Synthetic smoke only: one finite optimizer step |
| P15 paired before/after evaluation | Synthetic smoke only: 4 deals, not a strength result |
| Deletion/rebuild exercise | Covered by code tests; real run not tested |
| Package exclusion of canonical/raw data | Enforced by strict package allowlist |

Before release, the canary report must include complete-game counts, valid and
quarantined counts with reasons, dedupe counts, rules/seats/outcomes/bids,
split sizes, a zero-overlap game-ID audit, per-role BC metrics, and paired
before/after confidence intervals. Logs must remain aggregate-only and never
print a complete raw game.
