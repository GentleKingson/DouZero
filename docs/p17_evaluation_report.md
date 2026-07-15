# P17 Evaluation and Ablation Report

## Result

**Formal playing-strength evaluation: NOT COMPLETED**

No release-compatible trained weights were present in the repository or
supplied externally. Consequently, no formal card-play pairing, calibration
study, target-latency study, or independently trained ablation was executed.
Random/rule games and short-lived smoke checkpoints are not model-strength
evidence.

## Learned Full-Game Code Smoke

On clean commit `b7db29a3856324d65170b49ef32d17be7d3a6996`, the
CPU closure trained a four-episode standard V2 smoke checkpoint, converted it
to a strict public-policy sidecar, and ran the same checkpoint as both sides
of a two-deal, three-neutral-seat-rotation full-game scenario. Both bundles
used the manifest-validated learned bidding head and current
`v2-bidding-2` canonical neutral-seat features. With 2,000 deal-level bootstrap
resamples, candidate-equals-baseline produced the expected zero paired estimate
and `[0, 0]` interval.

```bash
.venv/bin/python evaluate_paired.py --mode full_game \
  --candidate v2_full_stack --baseline v2_base \
  --candidate-bidding learned --baseline-bidding learned \
  --model-matrix /tmp/douzero-p17-eval/full-matrix.json \
  --num-deals 2 --seed 17 --bootstrap-samples 2000 \
  --output /tmp/douzero-p17-eval/full-game-equality
```

This validates strict learned-bidding loading, full-game state transitions,
seat rotation, deal-level clustering, and the equality sanity check. The six
game rows made five bidding-inference calls, had no redeal-cap fallback, and
collated into the fixed seven-file P17 layout. Readiness was correctly
`insufficient` because two deals are 998 short of the P17 minimum. Both sides
are identical and the checkpoint is not a release candidate. No
playing-strength or target-latency claim follows.

Every serialized paired JSON result now uses `p15-paired-result-v2` and
requires a full source Git SHA. Its runtime identity binds the protocol,
ablation, complete scenario/evaluation configuration hash, ruleset hash, and
candidate/baseline model feature-schema identities, including the learned
bidding schema. CSV rows and Markdown reports carry the same identity. P17
collation recomputes the expected identities from the scenario and rejects any
protocol, mode, ruleset, schema, configuration, or checkpoint mismatch.

## Model Matrix

| Model | Card-play | Full game |
| --- | --- | --- |
| Legacy WP | Unavailable: weights not supplied | Unavailable |
| Legacy ADP | Unavailable: weights not supplied | Unavailable |
| Legacy factorized | Unavailable: weights not supplied | Unavailable |
| Model V2 base | Unavailable: compatible formal checkpoint not supplied | Unavailable: smoke only |
| V2 multi-objective | Unavailable | Unavailable |
| V2 + belief frozen | Unavailable | Unavailable |
| V2 + belief joint | Unavailable | Unavailable |
| V2 + human BC | Unavailable | Unavailable |
| V2 + strategy auxiliary | Unavailable | Unavailable |
| V2 + distillation | Unavailable | Unavailable |
| V2 + population | Unavailable | Unavailable |
| V2 + coach | Unavailable | Unavailable |
| V2 + search | Unavailable | Unavailable |
| V2 full stack | Unavailable | Unavailable: smoke only |

Full-game availability additionally requires a manifest-validated V2 model
with a learned bidding head. An external `rule`, `random`, `pass`, or `max`
bidding policy cannot satisfy that row.

The evaluator now accepts `bidding_policy: learned` only for V2/BC bundles
with an explicit `bidding_checkpoint`, `model_config.bidding_enabled=true`,
and a strict V2 checkpoint whose ruleset, card-play schema, model config,
bidding head, action schema, and bidding feature schema identities all match
the runtime. Bidding uses only `BiddingObservationV2`; the learned path does
not call the card-play action encoder.

## Ablations

The required `no_bidding`, `single_head`, `no_belief`, `no_human_bc`,
`no_auxiliary`, `no_distillation`, `no_population`, and `no_search` experiments
are all **NOT RUN**. No architecture flag was toggled at evaluation time to
simulate an ablation. Each row requires its own semantically compatible,
independently trained checkpoint.

## Prepared Tooling

`evaluate_paired.py` runs mirrored `cardplay_only` and three-neutral-seat
rotation `full_game` scenarios. Confidence intervals cluster all legs from one
deal before bootstrap resampling. Reports include role/team win percentage,
raw and signed-log score, bid and landlord-acquisition rates, per-bid outcomes,
bombs/rockets, springs, game length, calibration, measured inference latency,
candidate inference calls/s, search timeout/fallback counts, and redeal audit
counts. The deprecated JSON `actor_fps` key is retained as the exact same
inference-only rate for P15 consumer compatibility; it is not actor wall-clock
FPS.

The versioned `p17_empirical_readiness_v1` release policy requires at least
2,000 bootstrap samples and 1,000 paired deals without changing the closed P15
promotion contract. The P17 collation tool refuses missing matrix rows,
missing checkpoint files, smoke-only random/rule model backends, unavailable
ablation rows, and any full-game row without learned bidding.
It also binds each supplied result to path-free SHA-256 identities for every
matrix-validated role/bidding/belief checkpoint, recomputes paired evidence
and confidence intervals from complete game rows, and makes any deal that
exhausted the redeal cap smoke-only and release-ineligible.

```bash
.venv/bin/python tools/prepare_p17_evaluation.py \
  --write-matrix-template /tmp/p17-model-matrix.json
.venv/bin/python tools/prepare_p17_evaluation.py \
  --matrix /tmp/p17-model-matrix.json \
  --output artifacts/evaluation/p17
```

The second command produces the fixed artifact layout with explicit `not_run`
records when no result is supplied:

```text
model_matrix.json
cardplay_results.json
full_game_results.json
ablations.json
calibration.json
latency.json
report.md
```

Formal P17 release readiness remains blocked until compatible weights, learned
bidding, at least 1,000 paired deals per gate, every independent ablation,
calibration, and target-hardware latency measurements are available. Holdout
results must not be used for automatic tuning.
