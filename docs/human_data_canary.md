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

A deterministic fixture exercised canonical serialization, provenance
verification, replay validation, BC pretraining, RL+BC, and paired-evaluation
code paths on clean commit
`b7db29a3856324d65170b49ef32d17be7d3a6996`. The synthetic CLI does **not**
exercise an external adapter, HMAC pseudonymization, or external deduplication;
those have unit evidence only. This is software evidence, not evidence about
human behavior or model strength:

```bash
.venv/bin/python ingest_human_games.py --synthetic \
  --num_synthetic 4 --synthetic_seed 17 \
  --output /tmp/douzero-p17-human-b7db29a/canonical.jsonl
.venv/bin/python validate_human_games.py \
  --input /tmp/douzero-p17-human-b7db29a/canonical.jsonl \
  --output /tmp/douzero-p17-human-b7db29a/validated
```

Observed synthetic result: 4 total, 4 valid, 0 quarantined, 0 parse errors.
All four records use the legacy ruleset and contain no bidding phase. Both the
canonical and validated JSONL sidecars verified `lineage_verified=true`, four
records, the full source SHA above, the strict legacy ruleset hash, and content
SHA-256
`042dfd75801da7f3220484800f5f76b9aa45a27de0d9b4086cc5ba445f5fd72b`.

Every supported canonical, validated, and rebuilt JSONL has a sibling
`<path>.manifest.json`. It carries the dataset/record schema versions, full
source SHA, configuration-identity hash, ruleset identities, record count,
dataset SHA-256, privileged access class, and lineage flag, but no game IDs,
paths, raw identifiers, or secrets. Default validation, `pretrain_bc --data`,
the `train_v2` BC path, and deletion rebuild reject a missing, tampered, or
unverified manifest. `--allow-unverified-input` exists only to quarantine
malformed legacy data; it writes `lineage_verified=false`, which training and
release readers reject. Rebuild manifests bind the source dataset SHA.

```bash
.venv/bin/python pretrain_bc.py \
  --data /tmp/douzero-p17-human-b7db29a/validated.jsonl \
  --save_dir /tmp/douzero-p17-human-b7db29a/bc \
  --save_name synthetic-bc.pt \
  --epochs 1 --batch_size 4 --val_ratio 0.25 \
  --hidden_size 16 --history_layers 1 --history_heads 1 \
  --history_encoder lstm --seed 17

.venv/bin/python train_v2.py \
  --config /tmp/douzero-p17-human-b7db29a/rlbc.yaml --episodes 1 \
  --optimizer_steps 1 --batch_size 1 --buffer_capacity 64 \
  --checkpoint_path /tmp/douzero-p17-human-b7db29a/rlbc.pt --seed 17
```

The one-epoch BC smoke produced 115 samples (44 landlord, 35 landlord-down,
36 landlord-up), validation loss 1.2304 and validation top-1 0.486. Its
checkpoint configuration binds the input dataset SHA. These tiny synthetic
values are pipeline diagnostics only, not a BC-quality claim. RL+BC collected
19 card-play transitions, completed one optimizer step, changed parameters,
and recorded finite total loss 1.4469 plus BC cross-entropy 2.6577.

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

Observed current-schema smoke estimate: -0.1250 with CI [-0.5000, 0.2500].
The result uses `p15-paired-result-v2` and binds the full source SHA, evaluator
configuration, legacy ruleset, checkpoint identities, and both V2 feature
schemas. Four synthetic
deals are far below the 1,000-deal promotion gate; the number is recorded only
to prove the paired path executed and must not be interpreted as improvement.

## Authorized Invocation

The ingest CLI defaults `--input` from `DOUZERO_HUMAN_DATA_PATH` and loads the
project key from `--hmac-key-file` or
`DOUZERO_HUMAN_DATA_HMAC_KEY_FILE`. External ingest fails closed without it.
The HMAC key must contain at least 32 bytes. The dataset configuration identity
also binds the selected adapter implementation/CLI identity.
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
The rebuilt sidecar is regenerated atomically and binds the source dataset
SHA-256 without disclosing excluded IDs.

## Canary Gates

| Stage | Status |
| --- | --- |
| Authorization and license evidence | Blocked: no dataset |
| HMAC pseudonymization | Implemented; external unit tests pass, real run not tested |
| External adapter and deduplication | Unit-tested only; synthetic CLI bypasses this path |
| Canonical serialization/provenance | Synthetic smoke: 4 records; both manifests verified |
| Replay validation and quarantine | Synthetic smoke: 4 valid, 0 quarantined |
| Train/validation/test split and leakage audit | Unit/synthetic tests only; real run not tested |
| BC pretraining | Synthetic smoke: 115 samples, one epoch |
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
