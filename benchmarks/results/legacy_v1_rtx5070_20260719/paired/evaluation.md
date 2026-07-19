# P15 Evaluation: a2_bf16 vs a1_fp32

- Protocol: `p15_paired_v1`
- Mode: `cardplay_only`
- Ruleset: `legacy`
- Deal set: `3a4b35bb293a246a5dfddad08a10335f20d6a8bab411ab2bb79a4cdf3b551842` (public)
- Seed: `20260719`
- Result schema: `p15-paired-result-v3`
- Source Git SHA: `ce6c4237fd18207b9dd8c3c15b94e9cdaa8fdac5`
- Source Git tree: `0618936f5cb3f699d4d4c03e9a9706606c81bf41`
- Tracked source SHA-256: `c040f2b01ab7a3451a966ad46fc66e271dbd2cd19ca46073cfbe863e8b25bdf2`
- Clean/stable source: `False` / `True`
- Execution provider/run: `local` / `None`
- Evaluator image: `None`
- Hardware identity: `{"cuda_available": false, "cuda_device_count": 0, "cuda_device_names": [], "cuda_runtime_version": "13.2", "machine": "x86_64", "python_implementation": "CPython", "python_version": "3.12.3", "release": "7.0.0-28-generic", "system": "Linux", "torch_version": "2.12.1+cu132"}`
- Complete result SHA-256: `d42914d183092e550aa811d424df0345e00c756583a2ea2868ef71fd4c8f5312`
- Evaluation config SHA-256: `b77cf001a83c5f3fdb43160213e4849ea7809f50715f54a3671e4ee625dea3f0`
- Feature schemas: `{"baseline": {"bidding_feature_schema_hash": null, "feature_schema_hash": null, "feature_version": "legacy"}, "candidate": {"bidding_feature_schema_hash": null, "feature_schema_hash": null, "feature_version": "legacy"}}`
- Deals / games: 1000 / 2000

## Headline Metrics

| Metric | Value |
| --- | ---: |
| Candidate WP | 0.4975 |
| Paired WP delta | -0.0025 [-0.0190, +0.0140] |
| Mean score | +0.0130 |
| Mean log score | +0.0042 |
| Mean game length | 38.45 |

## Per Role

| Role | Games | WP | Mean score |
| --- | ---: | ---: | ---: |
| landlord | 1000 | 0.5220 | +0.0880 |
| landlord_up | 1000 | 0.4730 | -0.0620 |
| landlord_down | 1000 | 0.4730 | -0.0620 |

## Rules And Systems

| Metric | Value |
| --- | ---: |
| Bid rate | n/a |
| Landlord acquisition | n/a |
| Bomb / rocket rate | 0.1390 / 0.3190 |
| Spring / anti-spring | 0.0000 / 0.0000 |
| Inference p50 / p95 / p99 ms | 0.2409 / 0.3902 / 0.7095 |
| Actor FPS (P15 alias; inference calls/s) | 5882.0610 |
| Search timeout / fallback rate | n/a / n/a |
| p_win Brier / NLL / ECE | n/a / n/a / n/a |

Confidence intervals resample complete deals. Mirrored legs and seat rotations from one deal are clustered before bootstrap resampling.
