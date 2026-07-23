# P1 formal experiment freeze

P1 freezes the comparison inputs used by later pilot, development, and
promotion work. It does not run long training or provide playing-strength
evidence.

## Matrix

The runnable files in `configs/v3_formal/` match the H8a support contract:

| Variant | Legacy | Standard |
|---|---:|---:|
| `legacy_a1` | yes | no |
| `model_v2` | yes | yes |
| `v3_role` | yes | yes |
| `v3_admc` | yes | yes |
| `v3_oracle` | yes | yes |
| `v3_belief` | yes | yes |
| `v3_farmer_cooperation` | yes | yes |
| `v3_full_hybrid` | yes | yes |

There is no BC row because no authorized dataset identity is frozen. Search is
an evaluation/deployment wrapper over the same full-hybrid training checkpoint,
not a separate training variant. Standard full-game configs enable the separate
bidding head; legacy configs do not.

## Initialization

No official immutable checkpoint is shipped in this repository. P1 therefore
freezes deterministic fresh initialization rather than naming a mutable or
unavailable artifact:

* Legacy A1 uses `legacy-a1-seeded-fresh-v1`.
* Model V2 uses `model-v2-seeded-fresh-v1`.
* Every V3 ablation uses `v3-role-seeded-fresh-v1`, seed `101`, so the public
  backbone starts from the same deterministic rule rather than a selectively
  favorable student checkpoint.

Changing to checkpoint initialization requires a new config identity. The
freeze preflight verifies the file SHA-256 and an exact sidecar containing the
checkpoint kind, model family/hash, and ruleset hash. Cross-family, cross-rule,
partial, or missing identities fail closed.

## Seeds and budgets

Formal training seeds are `101`, `202`, and `303`. Evaluation seed is `41001`
and deal-set seed is `51001`. Environment, exploration, Python, NumPy, Torch,
and CUDA streams use the frozen derivation contract
`sha256(root_seed,stream_name,worker_id,episode_id)-v1`.

| Tier | Seeds | Wall clock/seed | Samples | Optimizer steps | Paired deals |
|---|---:|---:|---:|---:|---:|
| Pilot | 1 | 3,600 s | 1,000,000 | 10,000 | 5,000 |
| Development | 3 | 14,400 s | 5,000,000 | 50,000 | 20,000 |
| Promotion | 3 | 28,800 s | 20,000,000 | 200,000 | 100,000 |

All rows use single-process CUDA, batch size 32, replay capacity 100,000,
checkpoint cadence 1,000 updates, checkpointing enabled, and policy lag limit
zero. Later runtime work may introduce a new topology identity; it may not edit
these frozen rows after seeing results.

Deal-set identities hash the canonical ruleset identity, deal seed, paired-deal
count, and `fixed-deal-seat-swap-v1` generation strategy. Bootstrap unit is the
deal at 95% confidence.

## Tools

Validate a config without CUDA, worker, replay, or checkpoint side effects:

```bash
python tools/validate_v3_formal_config.py configs/v3_formal/v3_role_legacy.yaml
```

Freeze byte-stable canonical JSON:

```bash
python tools/freeze_v3_experiment.py \
  configs/v3_formal/v3_role_legacy.yaml /tmp/v3-role-legacy-freeze
```

The identity contains the resolved-config SHA-256, training-semantics hash,
workload hash, ruleset/model/support-matrix identities, initial checkpoint
hash, seed contract, budgets, and per-tier deal-set hashes. Metadata changes
alter the complete config hash but not training semantics or workload hashes.

## Status

* Release candidate: NONE
* Release status: NOT READY
* Playing strength: NOT MEASURED

Long training, multi-seed ablations, paired-strength evaluation, and release
promotion are explicitly out of scope for P1.
