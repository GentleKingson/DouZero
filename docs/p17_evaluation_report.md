# P17 Evaluation and Ablation Report

## Result

**Formal playing-strength evaluation: NOT COMPLETED**

No release-compatible trained weights were present in the repository or
supplied externally. Consequently, no formal card-play pairing, calibration
study, target-latency study, or independently trained ablation was executed.
Random/rule games and short-lived smoke checkpoints are not model-strength
evidence.

## Historical Learned Full-Game Smoke

The earlier CPU closure ran the same smoke checkpoint on both sides of a
two-deal, three-neutral-seat-rotation full-game scenario. That run predates the
current result-v3 replay and attestation boundary. It remains a descriptive
state-machine/equality smoke only; it is not a formally collatable result for
the current implementation and does not validate the current PR head.

```bash
.venv/bin/python evaluate_paired.py --mode full_game \
  --candidate v2_full_stack --baseline v2_base \
  --candidate-bidding learned --baseline-bidding learned \
  --model-matrix /tmp/douzero-p17-eval/full-matrix.json \
  --num-deals 2 --seed 17 --bootstrap-samples 2000 \
  --output /tmp/douzero-p17-eval/full-game-equality
```

The historical run had only two paired deals, identical policies, and no
release candidate, so no playing-strength or target-latency claim follows.

Current serialized results use `p15-paired-result-v3`. Every game row carries
the complete bidding and card-play action traces, a trace digest, a full
`deal_hash`, and integer nanosecond timing evidence. Formal collation obtains
the approved deal payloads independently and deterministically replays every
trace through `GameEnv`; winner, role mapping, bid value, bombs, rockets,
spring/anti-spring, terminal score, and game length are derived from replay,
not accepted from result summaries.

Formal runtime identity comes from a real Git checkout, never from
`DOUZERO_GIT_SHA`. It records the actual HEAD commit and tree, a SHA-256 over
tracked working-tree bytes, the source ref, tracked-file count, and clean/stable
inspection state. The result integrity envelope hashes the complete canonical
`{protocol, ablation, scenario, metrics, games, runtime_identity}` payload.
Formal collation additionally requires a detached GitHub Actions
OIDC/Sigstore attestation for the exact result bytes and constrains repository,
source commit/ref, signer workflow, and signer workflow digest. Accepted
provenance records the immutable workflow-run URL and whether the runner was
GitHub-hosted or self-hosted.

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
instrumented inference calls/s, search timeout/fallback counts, and redeal audit
counts. Each candidate call records a non-negative integer
`perf_counter_ns` duration; milliseconds, percentiles, and instrumented
throughput are derived from those same nanosecond values. The timing envelope
also records the total evaluation wall time and exact call/count sums.
Instrumented calls/s is calls divided by summed candidate inference time, not
end-to-end actor or evaluation throughput. The deprecated `actor_fps` key is
only an exact compatibility alias.

The versioned `p17_empirical_readiness_v1` release policy requires exactly the
closed 95% `paired_percentile_bootstrap_v1` protocol with `deal` as the
statistical unit, at least 2,000 bootstrap samples, and 1,000 paired deals
without changing the closed P15 promotion contract. The P17 collation tool
refuses missing matrix rows, missing checkpoint files, smoke-only random/rule
model backends, unavailable ablation rows, and any full-game row without
learned bidding. It binds each supplied result to path-free SHA-256 identities
for every matrix-validated role, bidding, and belief checkpoint.

Hashes and terminal summaries are not replay evidence. Formal collation also
requires the separately approved ordered deal payloads. It verifies their
pre-registered deal-set root, maps every indexed full `deal_hash` to that set,
and replays complete bidding/card-play traces. The collator then derives
terminal facts, candidate outcomes/scores, paired intervals, and diagnostics.
Calibration rows contain only `(role, prediction)`; labels are derived from the
replayed winner. Published `calibration.json` and `latency.json` are built from
this recomputed evidence and inherit `insufficient` whenever the result is not
eligible. Missing or inconsistent raw evidence is unavailable, never replaced
with caller summaries. Redeal-cap fallbacks remain smoke-only.

Private holdout payloads and all per-game/per-decision evidence stay inside the
trusted evaluation/collation environment. Replay occurs there before
publication. The public result block is rebuilt from a strict positive
allowlist: it retains only the mode, deal-set hash, deal count, candidate and
baseline names, plus an explicit zero-row redaction marker. Aggregate
recomputed readiness/diagnostics and verified provenance live in their own
fixed report fields. No trace, prediction, role/seat decision, per-call timing,
raw metric, or unknown nested field is copied from the signed source result.
The original signed result digest remains in the provenance summary; the
projection is never presented as the signed source artifact.

Completed results require a real clean/stable evaluator checkout, an approved
full evaluator commit, the approved deal-set root and deal payloads, plus a
detached GitHub OIDC/Sigstore attestation for the exact result file. An
allowlisted SHA claimed by the result is not sufficient. Raw/unverified local
results may be used only by descriptive library tooling and are always
`insufficient`; the formal CLI requires verified attestations. Formal P17 uses
the canonical legacy/standard rulesets; custom-rule P15 runs remain
release-ineligible.

The producer is `.github/workflows/formal-evaluation.yml`, gated by the
protected `formal-evaluation` environment and an isolated self-hosted runner.
Its dispatch surface accepts only the source SHA. A single strict JSON request,
selected and hash-bound by protected `FORMAL_EVALUATION_REQUEST_PATH` and
`FORMAL_EVALUATION_REQUEST_SHA256` environment variables, supplies the mode,
dataset scope, deal path/digest/set ID, evaluator and P17 matrix paths/digests,
separate approved checkpoint roots, candidate, baseline, and bootstrap count.
The request is snapshotted and parsed offline in the immutable image with exact
schema/type/format checks. Its original path is step-scoped, and the raw path
handoff is deleted after the approved files are snapshotted. Publication uses
the validated scope output rather than caller input. It pins the
source/workflow commit, snapshots the strict JSON deal file and both matrices by
those approved SHA-256 identities, verifies and executes an immutable protected
`repo@sha256:<digest>` OCI image, and requires that image's sorted package
manifest to match a second protected digest. Checkpoints must resolve below the
declared roots and are copied by an installed image-owned module running with
isolated Python, no checkout mount, and only root-read/snapshot-write access.
The package digest is checked before either checkpoint root is mounted, and
later containers receive only the files and snapshot directories required by
their stage.
Strict request/private-projection/manifest validators and dependency inventory
also run from the image without mounting the checkout; only evaluation,
result-identity binding, and collation intentionally execute the exact approved
source. Host orchestration performs system-level hashing/copying, Docker, GitHub
attestation, and upload operations but does not execute checkout Python, shell,
or binaries against protected inputs.
Control, audit, signed-result, and P17 output directories are disjoint. The
audit directory is never mounted into approved-source containers; the collator
sees the signed result read-only and can write only the P17 directory. Uploads
recheck the package, result, and manifest digests immediately before transfer.
Evaluation runs offline with read-only source/input/checkpoint mounts. The
workflow records those identities, its run and hardware inventory, then attests
the exact result. Private results are replayed and collated offline in the same
image before cleanup; detached
bundle verification does not expose holdout evidence to the network, and the
raw trace result is never included in the uploaded private artifact.

```bash
.venv/bin/python tools/prepare_p17_evaluation.py \
  --write-matrix-template /tmp/p17-model-matrix.json

# The protected workflow first runs evaluate_paired.py with --formal-release,
# --expected-source-git-sha, and the approved --eval-data, then signs the exact
# result JSON. Collation verifies that detached attestation as follows:
.venv/bin/python tools/prepare_p17_evaluation.py \
  --matrix /tmp/p17-model-matrix.json \
  --full-game-result /trusted/results/full-game.json \
  --full-game-attestation /trusted/results/full-game.attestation.json \
  --expected-evaluator-git-sha "$APPROVED_EVALUATOR_SHA" \
  --expected-full-game-deal-set-id "$APPROVED_FULL_GAME_SET_ID" \
  --approved-full-game-eval-data /trusted/holdout/full-game.json \
  --attestation-repository "$ATTESTATION_REPOSITORY" \
  --attestation-signer-workflow "$ATTESTATION_SIGNER_WORKFLOW" \
  --attestation-signer-digest "$ATTESTATION_SIGNER_DIGEST" \
  --attestation-source-ref "$ATTESTATION_SOURCE_REF" \
  --output artifacts/evaluation/p17
```

Approved deal inputs use `douzero-formal-eval-data-v1` strict JSON. The formal
collator does not load pickle, including for legacy card-play holdouts.

Use the corresponding `cardplay` flags for card-play results. Every
`--ablation-result NAME=PATH` also needs a matching
`--ablation-attestation NAME=PATH`; repeat the evaluator SHA option only for an
explicitly approved cross-version allowlist. With no results, the matrix-only
command still produces the fixed layout with explicit `not_run` records:

```text
model_matrix.json
cardplay_results.json
full_game_results.json
ablations.json
calibration.json
latency.json
report.md
manifest.json
```

`manifest.json` hashes every other P17 artifact. The protected workflow signs
that exact manifest with a second detached GitHub attestation, so a public
strict-allowlist projection is cryptographically bound to the protected
collation output rather than relying on an in-process Python wrapper.

Formal P17 release readiness remains blocked until compatible weights, learned
bidding, at least 1,000 paired deals per gate, every independent ablation,
calibration, and target-hardware latency measurements are available. Holdout
results must not be used for automatic tuning.
