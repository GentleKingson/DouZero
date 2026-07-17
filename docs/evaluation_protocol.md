# P15 Evaluation Protocol

P15 evaluates a candidate against one fixed baseline with protocol identity
`p15_paired_v1`. It is separate from training and never tunes a model,
threshold, search budget, or decision mode on the evaluation set.

## Scenarios

`cardplay_only` replays each legacy deal twice. The candidate first controls
the landlord against baseline farmers, then the baseline controls the landlord
against candidate farmers. Both legs retain the same deal ID, hands, and bottom
cards.

`full_game` replays each standard deck three times. The candidate rotates
through neutral seats 0, 1, and 2 while the deck, first bidder, and clockwise
bidding order stay fixed. Bidding determines the landlord before seat agents
are mapped to landlord roles. A bundle declares its bidding policy explicitly:
`learned`, `rule`, `random`, `pass`, or `max`. `learned` requires a V2/BC
bundle with an explicit bidding checkpoint and `bidding_enabled` model config;
the strict loader validates ruleset, model, card-play schema, bidding head,
action schema, and bidding feature schema identity before inference. `rule` is
a fixed public hand-strength policy and reports must not describe it as learned
model bidding.

`cardplay_only` uses common random streams derived from scenario seed, deal ID,
and role. `full_game` derives them from deal ID and physical seat. Inference
timing is measured with `time.perf_counter_ns` and is therefore the only field
expected to vary slightly between identical runs. Bidding and card-play action
traces remain deterministic for a fixed scenario and policy.

The protocol identity is closed over its seat schedule. `cardplay_only` must
contain exactly the candidate-landlord and candidate-two-farmers legs once
each. `full_game` must contain exactly one candidate in each neutral seat once.
Missing, duplicated, reordered, or additional permutations are rejected. In
`full_game`, stochastic policies use common random streams by physical seat so
identical candidate and baseline policies replay the same game under rotation.

## Statistical Unit

Confidence intervals resample complete deals. Mirrored card-play legs and the
three full-game seat rotations are averaged within a deal before percentile
bootstrap resampling. Decisions and seats are never treated as independent
samples. Reports include the paired deal count, bootstrap sample count, and
confidence level.

For `cardplay_only`, the paired estimate is candidate win rate minus 0.5. For
`full_game`, raw win percentage is descriptive because a landlord win gives
one winning seat while a farmer win gives two. Its paired estimate is instead
the standard zero-sum seat score: landlord and both farmer scores sum to zero
for every game. Identical policies therefore have an estimate of zero.

Only `cardplay_only` may produce the `PromotionEvaluation` consumed by the P11
promotion gate. The closed `p15_paired_v1` contract requires confidence level
0.95, at least 1,000 bootstrap samples, the official permutation hash, and the
`cardplay_win_rate_delta` estimator; its paired-deal minimum remains the
configured `PromotionGate.min_pairs`. `PromotionGate` validates all of these
fields again, and `full_game` reports cannot be promoted.

P17 release collation applies a distinct readiness policy,
`p17_empirical_readiness_v1`, to a valid P15 result. It requires at least 2,000
bootstrap samples and 1,000 paired deals. This stricter release bar does not
change or relabel the underlying P15 protocol result.

Current paired results use `p15-paired-result-v3`. Result-v3 is not merely a
summary schema: each game includes the complete bidding/card-play action trace,
the full deal hash, a canonical trace digest, and per-call nanosecond timing
evidence needed by formal P17 collation.

## Metrics And Outputs

Every run writes JSON, per-game CSV, and Markdown. The report includes:

- overall, team, landlord, landlord-up, and landlord-down win percentage;
- mean raw score and signed `log1p(abs(score))` score;
- bid and landlord-acquisition rate plus per-bid win/score in full-game mode;
- bomb, rocket, spring, anti-spring, and game-length rates;
- selected-action `p_win` Brier, NLL, and 15-bin ECE when a V2 agent exposes
  predictions; raw rows store `(role, prediction)`, never a caller-supplied
  outcome label;
- candidate inference p50/p95/p99 and candidate inference calls/s;
- search timeout/fallback rates and all-pass redeal/max-redeal audit counts;
- deal/game/decision sample counts and paired 95% confidence intervals.

Unavailable metrics are JSON `null` and Markdown `n/a`, not invented zeros.
Every candidate bidding/card-play call stores one non-negative integer
in `candidate_latencies_ns`; the matching compatibility value in
`candidate_latencies_ms` must equal that same integer divided by 1,000,000.
Percentiles are recomputed from nanoseconds. Instrumented calls/s divides the
exact call count by summed inference nanoseconds, while the timing envelope
separately records total evaluation wall nanoseconds and its scope. It is not
end-to-end actor/evaluation FPS. The deprecated P15 `actor_fps` key is an exact
alias only. Missing, malformed, or inconsistent timing makes the diagnostic
unavailable.

Formal calibration labels are derived by the collator from each replayed
winner and the recorded candidate role. A result file cannot improve Brier,
NLL, or ECE by supplying or changing labels. Raw game rows and traces remain in
the signed result so headline results can be independently reconstructed.

## Model Matrix And Ablations

Built-in `random` and `rule` bundles require no weights. A JSON model matrix can
register arbitrary bundle names such as `legacy-wp`, `legacy-adp`, `bc-v1`,
`v2-p13`, or historical policy IDs. Supported backends are `legacy`,
`legacy_factorized`, and manifest-validated `v2`, plus the built-ins. `bc` is
an explicit alias for the V2 loader so behavior-cloned bundles remain visibly
distinct. Historical policies use their real backend plus a `historical` tag.
Weighted bundles provide all three role checkpoint paths.
Full-game learned bidding additionally supplies `bidding_policy: learned`, an
explicit `bidding_checkpoint`, and the exact `model_config` used to save that
manifest-bearing V2 sidecar. The evaluator never guesses a bidding checkpoint
from a role path.

P17 matrix normalization hashes every role, bidding, and belief checkpoint
after strict manifest loading. Result scenarios carry those path-free hashes,
and collation requires an exact match. Formal release readiness is rebuilt by
deterministically replaying every complete result-v3 trace against the
separately supplied approved deal payload at the same ordered index. The
collator derives bidding history, role mapping, winner, bombs/rockets,
spring/anti-spring, game length, score, calibration labels, paired confidence
intervals, latency, and search diagnostics. It rejects an action that is
illegal, out of turn, post-terminal, incomplete, or inconsistent with the
approved deal and ruleset. Deal hashes or internally consistent terminal
summaries alone are never sufficient. A bounded forced all-pass fallback
remains visible only as a smoke row and is excluded from formal statistics.

```json
{
  "bundles": {
    "legacy-wp": {
      "backend": "legacy",
      "checkpoints": {
        "landlord": "/models/wp/landlord.ckpt",
        "landlord_up": "/models/wp/landlord_up.ckpt",
        "landlord_down": "/models/wp/landlord_down.ckpt"
      }
    },
    "legacy-no-bidding": {
      "backend": "legacy_factorized",
      "checkpoints": {
        "landlord": "/models/no-bidding/landlord.ckpt",
        "landlord_up": "/models/no-bidding/landlord_up.ckpt",
        "landlord_down": "/models/no-bidding/landlord_down.ckpt"
      }
    },
    "v2-no-search": {
      "backend": "v2",
      "checkpoints": {
        "landlord": "/models/v2-no-search/landlord.ckpt",
        "landlord_up": "/models/v2-no-search/landlord_up.ckpt",
        "landlord_down": "/models/v2-no-search/landlord_down.ckpt"
      }
    }
  },
  "ablations": {
    "no_search": "v2-no-search",
    "no_bidding": {
      "candidate": "legacy-no-bidding",
      "baseline": "legacy-wp"
    }
  }
}
```

The recognized ablations are `no_bidding`, `single_head`, `no_belief`,
`no_human_bc`, `no_auxiliary`, `no_distillation`, `no_population`, and
`no_search`. Each points to an explicit compatible bundle. The runner does not
flip architecture flags under an existing checkpoint because that violates
checkpoint identity or produces a cosmetic rather than trained ablation.
When a `full_game` matrix runs `no_bidding`, the first bidder is fixed as
landlord and the exact deck/bottom cards are converted to `cardplay_only`.
Candidate and baseline bundles for that row must therefore carry legacy-
ruleset-compatible checkpoints; the optional object form above supplies a
different compatible baseline when needed.

## Formal P17 Trust Boundary

Formal evaluation starts only from a real Git checkout. `--formal-release`
rejects a dirty tree (including untracked files), an index or tracked byte that
does not match HEAD, an unstable checkout that changes between identity
samples, an unsupported submodule, and a HEAD that differs from
`--expected-source-git-sha`. Runtime identity records the real commit/tree,
tracked-tree SHA-256, ref, and file count. `DOUZERO_GIT_SHA` and package
metadata remain available only to local descriptive runs and cannot establish
formal identity.

Formal deal inputs use the safe `douzero-formal-eval-data-v1` JSON schema and
must contain exactly `schema_version`, `mode`, `ruleset`, and `deals`. The
loader rejects duplicate object keys, `NaN`/infinities, extra keys,
non-canonical field types, mismatched rule identity, malformed cards, and
duplicate deals. It never invokes pickle. A `.pkl` path is rejected before its
contents can be deserialized, both by `evaluate_paired.py --formal-release`
and by the P17 collator. Generate an approved input candidate with:

```bash
python generate_eval_data.py \
  --output /trusted/holdout/cardplay \
  --num_games 1000 \
  --ruleset legacy \
  --output-format formal-json
```

Inside its immutable evaluator image, the protected
`.github/workflows/formal-evaluation.yml` job executes the equivalent of:

```bash
python evaluate_paired.py \
  --mode cardplay_only \
  --candidate "$CANDIDATE" --baseline "$BASELINE" \
  --model-matrix "$MODEL_MATRIX" \
  --eval-data "$APPROVED_CARDPLAY_DATA" \
  --dataset-scope private_holdout \
  --bootstrap-samples 2000 \
  --formal-release \
  --expected-source-git-sha "$APPROVED_EVALUATOR_SHA" \
  --output /trusted/results/cardplay
```

This is not a local formal-mode recipe. Formal validation also requires the
GitHub workflow/run identity, a protected self-hosted runner, the immutable OCI
`repo@sha256:<digest>` configured as the protected
`FORMAL_EVALUATOR_IMAGE` environment variable, an exact sorted `pip freeze`
digest configured as protected `FORMAL_PYTHON_PACKAGES_SHA256`, and a recorded
hardware/runtime inventory. The workflow verifies the image's `RepoDigests`
and actually runs evaluation and collation in that image with networking
disabled, a read-only root filesystem, all Linux capabilities dropped,
`no-new-privileges`, and source/input/checkpoint mounts read-only. The output
for each stage is a dedicated persistent writable mount; `/tmp` is an
ephemeral size-bounded tmpfs. `control`, `audit`, `result`, and `p17` are
separate sibling directories. Approved-source containers never receive the
audit or control directory, and the collator receives the signed result only
as a read-only file. Setting lookalike environment variables locally cannot
produce the detached attestation required by collation.

Image pull, GitHub attestation publication/verification, and artifact upload are
host-orchestration steps and therefore use network access. Their pinned action
SHAs, protected-environment configuration, self-hosted runner administration,
and upload allowlists are part of the trust boundary. The workflow file alone
does not prove that required reviewers, runner isolation, protected variables,
or holdout access controls were configured correctly; each formal run must
audit those external controls. For private holdouts, the upload steps exclude
the raw result, deals, traces, and per-decision evidence and publish only the
strict projection, aggregate reports, manifests, and attestation audit data.

Control-plane helpers are separated from approved-source execution. Strict
request, private-projection, and manifest validators use isolated stdlib Python
(`python -I -S -`) without the checkout mounted and without `PYTHONPATH`.
Dependency inventory likewise runs from the pinned image without the checkout
using isolated module execution, and its protected digest is checked before
the image receives any checkpoint root. Only the core evaluator, result-identity
binder, and P17 collator intentionally mount and execute the exact approved
source commit, read-only and offline. This distinction prevents checkout
startup hooks from influencing the validators that decide which protected
inputs or private outputs cross the boundary.

The host side is limited to system-level checkout/hash/install/path, Docker,
GitHub CLI/attestation, permission, and upload orchestration. It does not run a
Python module, shell script, or executable from the checkout against protected
inputs. Approved-source Python starts only inside the isolated evaluator
containers described above.

The dispatch surface accepts only `source_sha`, which must equal the protected
`main` ref, workflow SHA, and checked-out HEAD. All other run selection comes
from one request file selected by the protected environment variables
`FORMAL_EVALUATION_REQUEST_PATH` and
`FORMAL_EVALUATION_REQUEST_SHA256`. The workflow snapshots and hash-checks that
file in a step-scoped environment, then passes only the run-local snapshot to
the immutable evaluator image. The original path is not job-wide. Its strict
`formal-evaluation-request-v2` JSON object contains exactly:

```json
{
  "schema_version": "formal-evaluation-request-v2",
  "mode": "full_game",
  "dataset_scope": "private_holdout",
  "eval_data_path": "/protected/evaluation/deals.json",
  "eval_data_sha256": "<64 lowercase hex characters>",
  "deal_set_id": "<64 lowercase hex characters>",
  "model_matrix_path": "/protected/evaluation/models.json",
  "model_matrix_sha256": "<64 lowercase hex characters>",
  "model_checkpoint_root": "/protected/evaluation/model-checkpoints",
  "p17_matrix_path": "/protected/evaluation/p17-models.json",
  "p17_matrix_sha256": "<64 lowercase hex characters>",
  "p17_checkpoint_root": "/protected/evaluation/p17-checkpoints",
  "candidate": "candidate-v1",
  "baseline": "baseline-v1",
  "bootstrap_samples": 2000
}
```

Missing or extra fields, duplicate keys, non-finite numbers, wrong JSON types,
unsafe/non-canonical paths, embedded newlines, unsupported modes/scopes, and
malformed identities fail before any evidence file is opened. Publication
scope is taken from the validated request step output, so a dispatch caller
cannot route a private holdout through the public artifact path.
Version 1 requests are rejected because they do not declare the two protected
checkpoint roots. The parser's raw asset paths exist only in a private,
strictly generated handoff file for the immediately following snapshot step;
that file is deleted after use. Later steps receive only run-local snapshot
paths and non-secret hashes, names, mode, scope, and count identities.

The two checkpoint roots are canonical absolute non-symlink directories,
disjoint from the checkout and run/output directories. Every matrix checkpoint
path must resolve beneath its corresponding approved root. Snapshotting runs in
the pinned evaluator image through its installed
`douzero.evaluation.snapshot_cli` module with `python -I -B -m`; that container
does not mount the checkout and does not receive `PYTHONPATH`. It sees only the
approved matrix inputs read-only, one approved checkpoint root read-only, and
one snapshot directory writable. The rewritten matrices point only to the
verified snapshots subsequently mounted read-only for evaluation/collation.
This image-owned step prevents an approved matrix from turning arbitrary host
paths into evidence or executing checkout-controlled snapshot helpers.

The protected image digest, protected GitHub environment, isolated self-hosted
runner, and approved source SHA are explicit trust roots. Mount isolation and
fixed-schema projection reduce accidental or opportunistic disclosure, but
cannot prove that a deliberately malicious approved evaluator or collator has
no covert encoding. Independent source approval and protected-run review remain
required.

Because `source_sha` must already be the protected `main` commit containing the
same workflow, a Draft PR branch cannot create formal evidence under this
contract. The implementation may be reviewed and merged as fail-closed
infrastructure, but release evidence is generated only from a subsequently
approved `main` commit. A green PR matrix is not a substitute for that run.

The protected GitHub Actions evaluator signs the exact result JSON with its
OIDC identity/Sigstore-backed GitHub artifact attestation. The canonical
result digest covers exactly `protocol`, `ablation`, `scenario`, `metrics`,
`games`, and `runtime_identity`; changing a trace, summary, timing sample,
configuration, or provenance field changes the digest and signed artifact.
The protected workflow snapshots both evaluator and P17 matrices plus the
strict JSON deal file by SHA-256 identities in that approved request, runs at
least 1,000 deals and 2,000 bootstraps, replays/collates before private cleanup,
and uploads no private per-game or per-decision evidence.

Formal collation requires all trust inputs explicitly:

```bash
python tools/prepare_p17_evaluation.py \
  --matrix "$P17_MATRIX" \
  --cardplay-result /trusted/results/cardplay.json \
  --cardplay-attestation "$CARDPLAY_ATTESTATION_BUNDLE" \
  --expected-evaluator-git-sha "$APPROVED_EVALUATOR_SHA" \
  --expected-cardplay-deal-set-id "$APPROVED_CARDPLAY_SET_ID" \
  --approved-cardplay-eval-data "$APPROVED_CARDPLAY_DATA" \
  --attestation-repository "$ATTESTATION_REPOSITORY" \
  --attestation-signer-workflow "$ATTESTATION_SIGNER_WORKFLOW" \
  --attestation-signer-digest "$ATTESTATION_SIGNER_DIGEST" \
  --attestation-source-ref "$ATTESTATION_SOURCE_REF" \
  --attestation-trusted-root "$ATTESTATION_TRUSTED_ROOT" \
  --attestation-trusted-root-sha256 "$ATTESTATION_TRUSTED_ROOT_SHA256" \
  --output artifacts/evaluation/p17
```

Use the equivalent `full-game` options for that mode. Each ablation requires a
matched `--ablation-result NAME=PATH` and
`--ablation-attestation NAME=PATH`. The verifier pins the exact repository,
source commit/ref, protected workflow path and workflow Git digest, result-file
SHA-256, digest-bound trusted-root snapshot, and canonical result digest. The
trusted root is fetched before entering the offline verifier and passed with
`--custom-trusted-root`; bundle presence alone is not an offline-verification
claim. The verifier also records the immutable GitHub
workflow-run URL and the attested hosted/self-hosted runner class. A
missing/failed attestation, missing approved deal payload, replay failure, or
raw unverified result can never be `eligible`; descriptive library output
remains `insufficient`.

For a private holdout, approved deals and every per-game/per-decision field
stay in the trusted evaluation/collation environment. Replay and attestation
verification happen there. The public P17 result block is reconstructed from
a strict allowlist containing only the mode, deal-set hash, deal count, model
names, and a zero-row redaction marker. It never copies traces, predictions,
roles/seats, per-call latency, raw metrics, or unknown nested fields. Separate
fixed-schema artifacts publish recomputed aggregate diagnostics and the
verified source digest/attestation summary. The projection does not expose the
cards or claim to be the original signed artifact. A canonical
`manifest.json` hashes every projection/report file, and the protected
workflow creates and verifies a second detached attestation for that manifest
before upload.

## Reproducible CPU Smoke

```bash
python evaluate_paired.py \
  --mode cardplay_only \
  --candidate rule \
  --baseline random \
  --num-deals 8 \
  --seed 7 \
  --output artifacts/evaluation/p15-smoke
```

This command is intentionally a local smoke. Without `--formal-release`, an
approved deal file, and detached attestation it cannot produce formal P17
evidence regardless of its sample count.

Use `--mode full_game` for bidding and seat rotation. A local, non-formal run
may still use a trusted historical pickle generated by
`generate_eval_data.py`; pickle can execute code and is retained only for
backward compatibility. Formal evaluation accepts only strict `.json` deal
sets generated with `--output-format formal-json`.

Public eval sets may be versioned outside the code package and identified by
their content hash. Private holdouts require `--dataset-scope private_holdout`
with `--formal-release` and an explicit formal JSON `--eval-data`; their
path/name is redacted from reports. The signed source result contains the
traces needed for trusted replay and must remain in protected storage. Only the
post-verification P17 strict-allowlist projection may leave that environment;
private deal contents and per-decision evidence must never be committed or
exported.

Regression gates should be configured outside the holdout run and cover legacy
behavior, rules, latency, calibration, and a minimum for each role. Failed
gates block promotion; they must not trigger automatic parameter search on the
test or holdout set.

Pass a predeclared JSON file with `--gates`. Supported thresholds are
`max_p95_latency_ms`, `max_brier`, `max_ece`,
`min_overall_win_percentage`, and role keys under
`min_role_win_percentage`. External regression jobs feed their already-known
boolean results through `required_checks`, for example
`{"legacy_behavior": true, "environment_rules": true}`. A failed gate is
written into the report and makes the command exit with status 2.
`required_checks` accepts only string keys and JSON boolean values; strings
such as `"false"` are rejected rather than coerced.
