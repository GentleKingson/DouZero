# DouZero Unreleased Model Card Draft

## Release Status

- Release candidate: **NONE**
- Release status: **NOT READY**
- Model/checkpoint version: `NOT AVAILABLE`
- Source Git SHA for an RC: `NOT AVAILABLE`
- Model ABI and implementation hash: `NOT AVAILABLE`
- Feature schema version/hash: `NOT AVAILABLE`
- Ruleset ID/hash: `NOT AVAILABLE`
- Model configuration hash: `NOT AVAILABLE`
- Training configuration hash: `NOT AVAILABLE`
- Learned bidding schema/version: `NOT AVAILABLE`
- Supported roles: `NOT AVAILABLE`
- Numeric dtype and target hardware: `NOT AVAILABLE`

This is a truthful pre-release draft, not a card for a trained model. No
checkpoint currently satisfies the standard full-game, learned-bidding,
empirical-evaluation, target-hardware, and packaging gates.

## Training

- Training configuration: `NOT AVAILABLE`
- RC training hardware: `NOT AVAILABLE`
- Implementation smoke hardware: `SINGLE-PROCESS CPU`
- Standard full-game training duration: `NOT MEASURED`
- RC checkpoint resume: `NOT MEASURED`
- Implementation checkpoint resume smoke: `PASS ON SINGLE-PROCESS CPU`
- Belief mode: `NOT AVAILABLE`
- Search training/evaluation configuration: `NOT AVAILABLE`

## Data

- Data categories: `NOT AVAILABLE`
- Authorized human-data status: `NOT USED; REAL CANARY NOT RUN`
- Authorization/license evidence: `NOT AVAILABLE`
- Date range and filtering: `NOT AVAILABLE`
- Personal-identifier handling: the canonical pipeline requires HMAC game IDs,
  allowlisted provenance, replay validation, and package exclusion. Supported
  training readers reject missing, tampered, or unverified dataset sidecars;
  this infrastructure change contains no authorized-data release evidence.

## Evaluation

- Card-play paired results by role: `NOT MEASURED`
- Standard full-game learned-bidding results: `NOT MEASURED`
- At least 1,000 paired deals: `NOT MET`
- Eight required ablations: `NOT MEASURED`
- Brier/NLL/ECE calibration: `NOT MEASURED`
- Search timeout/fallback behavior: `NOT MEASURED`
- RC belief quality metrics: `NOT MEASURED`
- Belief conservation correctness: `CPU-TESTED; NOT A QUALITY METRIC`

Paired outputs are accepted for release collation only when their full source
SHA, ruleset, scenario/configuration, checkpoint, and card-play/bidding feature
schema identities validate. Short synthetic/equality smokes remain outside the
RC metrics above.

Random/rule smoke games are excluded from this section because they do not
measure a trained model's playing strength.

## Latency and Hardware

- CPU inference p50/p95/p99 for an RC: `NOT MEASURED`
- GPU inference p50/p95/p99: `NOT MEASURED`
- Peak GPU memory and throughput: `NOT MEASURED`
- FP16/BF16 AMP: `NOT MEASURED`
- NCCL DDP: `NOT MEASURED`
- `torch.compile`: `NOT MEASURED; NOT RECOMMENDED`

## Known Limitations

There is no RC to characterize. Standard full-game learned bidding and joint
belief training have CPU implementation smokes only; their playing strength,
target-GPU behavior, and long-run stability are unmeasured. Authorized
human-data behavior, standard learned-bidding DDP, joint/alternating belief
DDP synchronization, and distributed trainer resume are **not implemented**
and fail closed. Sustained GPU operation, multi-GPU shutdown, full ablations,
calibration, and release-candidate latency are unmeasured. Repository-wide
provenance is not yet universal across every historical coach/distillation/
evaluation-data artifact path, so release remains blocked even though the P17
trainer, belief, canonical-human-data, paired-result, and format-2 package
paths carry strict identities.
Package rollback is covered by a known-good fixed-state inference test, but no
packaged release candidate exists to exercise in production.

## Intended and Prohibited Uses

The repository is for offline research and authorized competition or
deployment evaluation. It must not be used for scraping, account automation,
anti-detection, platform-control bypass, access to undisclosed hidden
information, or any use that violates law, license, authorization, or service
terms.

## Rollback Conditions

No model should be deployed from this draft. A future RC must be rolled back
for checksum or identity failure, non-finite inference, illegal actions,
privileged-information leakage, target-hardware instability, unexplained
paired regression, or failure of any declared release gate.

## License

Repository code is Apache-2.0. Every package must include
`THIRD_PARTY_NOTICES` and must pass the package checksum and identity verifier.
