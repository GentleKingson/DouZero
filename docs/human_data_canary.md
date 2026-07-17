# P17 Authorized Human-Data Canary Procedure

## Result

**Repository release evidence: NOT AVAILABLE**

Whether an authorized run occurred is an external, commit-bound fact. The run
ledger must name the source head, workflow run, input attestation, output
digests, and reviewer; this tracked procedure must not copy a moving run result
or private input identity.

No real-data game counts, rule/seat/outcome/bid distributions, duplicate
counts, quarantine rates, BC metrics, paired estimates, or confidence
intervals exist. They must remain `NOT AVAILABLE` until an authorized export
is supplied and the entire canary completes.

## Synthetic Code Smoke

A deterministic fixture can exercise canonical serialization, provenance
verification, replay validation, BC pretraining, RL+BC, and paired-evaluation
code paths. The synthetic CLI does **not**
exercise an external adapter, HMAC pseudonymization, or external deduplication;
those have unit evidence only. This is software evidence, not evidence about
human behavior or model strength:

```bash
run_root="$(mktemp -d /tmp/douzero-p17-human.XXXXXX)"
.venv/bin/python ingest_human_games.py --synthetic \
  --num_synthetic 4 --synthetic_seed 17 \
  --output "$run_root/canonical.jsonl"
.venv/bin/python validate_human_games.py \
  --input "$run_root/canonical.jsonl" \
  --output "$run_root/validated"
```

The command must fail closed on any invalid record or sidecar. Counts and
digests belong in the commit-bound run artifact, not in this procedure.

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
  --data "$run_root/validated.jsonl" \
  --save_dir "$run_root/bc" \
  --save_name synthetic-bc.pt \
  --epochs 1 --batch_size 4 --val_ratio 0.25 \
  --hidden_size 16 --history_layers 1 --history_heads 1 \
  --history_encoder lstm --seed 17

.venv/bin/python train_v2.py \
  --config "$run_root/rlbc.yaml" --episodes 1 \
  --optimizer_steps 1 --batch_size 1 --buffer_capacity 64 \
  --checkpoint_path "$run_root/rlbc.pt" --seed 17
```

The smoke must record finite diagnostics and bind its checkpoint configuration
to the input dataset SHA. Synthetic values are pipeline diagnostics only, not
a BC-quality claim.

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

This local comparison is not accepted by formal P17 collation. Four synthetic
deals are far below the 1,000-deal promotion gate and must not be interpreted
as improvement.

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
The rebuild writes the JSONL and manifest into one private immutable version,
fsyncs both files and their directories, then publishes them with one atomic
pointer replacement. Supported readers pin one pointer/version for the whole
read, so a concurrent switch cannot mix data and manifest. A pre-commit
failure leaves the prior complete version active. After pointer replacement,
`RebuildPostCommitError` reports `committed=true` plus explicit `durable` and
`current` state; the CLI returns status 3 and emits that state as JSON rather
than implying rollback. An uncertain switch returns status 4 and retains the
staged version for recovery. Publication locks are advisory, so the dataset
parent must be an owner-controlled trusted directory; versioned publication is
POSIX-only and fails closed when `flock` is unavailable. The new manifest binds
the same pinned source snapshot SHA-256 without disclosing excluded IDs.
After the first publication, the active output is the authoritative base for
later deletion requests; `--input` is bootstrap-only, so removed games cannot
be resurrected by retrying with an older export.

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
