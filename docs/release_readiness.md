# P17 Readiness Infrastructure

## 1. Baseline Information

- Repository: `https://github.com/GentleKingson/DouZero`
- Base branch/SHA: `main` / `fa9f76de5ca31b4de33d4237e14b36e102c67655`
- Mutable PR identity is deliberately not duplicated in this tracked file.
  The authoritative head SHA, merge-result SHA, workflow run IDs, job results,
  and review state live in [PR #20](https://github.com/GentleKingson/DouZero/pull/20)
  and its immutable GitHub Actions records. A repository commit cannot name
  its own final object ID without creating another commit and immediately
  making that statement stale.
- Commit-bound validation therefore uses an external evidence tuple:
  `{repository, head_sha, merge_sha, workflow_run_ids, image_digest,
  artifact_digest, attestation_identity}`. This document records the required
  gates and commands, not a moving copy of that tuple.
- `.github/workflows/pr-evidence.yml` enforces the stable PR claim contract on
  every open, synchronize, reopen, edit, and ready-for-review event. It rejects
  a closure/release-ready title, literal object IDs, final-head language, fixed
  test counts, and unbound audit claims. Its uploaded JSON binds the PR head,
  tested merge commit, workflow commit, run identity, title hash, and body hash.
  A later push necessarily creates a different head-bound check and artifact.
- PR evidence is engineering evidence only. It is not a protected evaluator
  result, an OIDC/Sigstore release attestation, or model-strength evidence.
- The formal workflow implements request-v2 protected checkpoint roots,
  step-scoped protected paths, image-owned checkpoint snapshots, disjoint
  control/audit/result/P17 outputs, and exact stage-minimal mounts. Whether a
  particular PR head passed those checks must be established from the external
  evidence tuple above.
- Scope: P00-P16 regression audit and P17 full-game training, joint belief,
  empirical-validation tooling, evaluation, packaging, and fail-closed release
  infrastructure. This is not a release-readiness closure or a release-ready
  model delivery.

### Trust-Boundary Status

The result-v3 implementation targets earlier trust-boundary findings with full
action traces, deterministic replay against separately approved deal payloads,
replay-derived calibration labels, nanosecond timing evidence, a canonical
whole-result digest, real clean/stable Git identity, and detached GitHub
OIDC/Sigstore artifact attestation. Unit, Docker, workflow-contract, packaging,
and tamper-test methods are present. Their commit-bound outcomes are recorded
externally rather than frozen here. The current Draft/ready/review disposition
is read from PR #20 rather than asserted here. A protected Environment,
self-hosted evaluator, approved assets, and an actual protected workflow run
must exist externally before this can produce formal evidence. Until such a run
is recorded, the release candidate remains `NONE`; GPU,
authorized-data, formal strength, ablation, latency, and 1,000-deal gates stay
open.

## 2. Baseline Findings

Inspection of `main` at `fa9f76d` confirmed the following rather than relying
on phase documentation:

| Question | Baseline finding | Source |
|---|---|---|
| Did `train_v2.py` accept standard rules? | No. `_assert_v2_identity` raised for any non-legacy ruleset and `_resolve_ruleset` routed standard to a trainer rejection. | `train_v2.py` |
| Did the V2 trainer accept standard rules? | No. `_legacy_only` rejected every non-`None` ruleset. | `douzero/training/v2_trainer.py` |
| Did population self-play accept standard rules? | No. `PopulationEpisodeRunner.__init__` raised `NotImplementedError`. | `douzero/league/self_play.py` |
| Did Model V2 have learned bidding? | No. It exposed only card-play value/prior/auxiliary heads. | `douzero/models_v2/model.py`, `douzero/models_v2/output.py` |
| How did full-game evaluation bid? | External `rule`, `random`, `pass`, or `max` policies only. | `douzero/evaluation/protocol.py`, `douzero/evaluation/simulation.py`, `evaluate_paired.py` |
| Did the standard environment implement the complete state machine? | Yes: deal, neutral-seat bidding, bottom reveal/role remap, playing, terminal scoring, bounded all-pass redeal. | `douzero/env/game.py`, `douzero/env/env.py`, `douzero/env/scoring.py`, `tests/test_bidding_state_machine.py` |
| Was joint belief training usable? | No. constrained marginals crossed a NumPy boundary and `belief_stop_gradient=False` raised. | `douzero/belief/model.py`, `douzero/models_v2/model.py`, `tests/test_p07_model_fusion.py` |

## 3. Phase Status

`Complete` identifies an implemented repository contract. Its validation must
still be rerun for each candidate head and does not imply model-strength, GPU,
production, or private-data validation.

| Phase | Status | Evidence |
|---|---|---|
| P00 baseline freeze | Complete | `tests/test_baseline_invariants.py`, `tools/capture_baseline.py`, `docs/architecture/current.md` |
| P01 config/checkpoint/package metadata | Complete for its historical contract | P17 strict loaders add full-SHA identities; repository-wide legacy artifact provenance is a separate open gate below |
| P02 standard rules/scoring | Complete | `douzero/env/rules.py`, `game.py`, `scoring.py`, bidding/scoring tests |
| P03 public/privileged Observation V2 | Complete | `douzero/observation/`, leakage and observation tests |
| P04 legacy factorization | Complete | `douzero/dmc/models_factorized.py`, parity tests |
| P05 Model V2 | Complete | `douzero/models_v2/`, model/checkpoint tests |
| P06 multi-objective training | Complete | `douzero/training/`, V2 trainer/loss/calibration tests |
| P07 belief | Complete | constrained decoder/sampler/conservation, standard bootstrap/evaluator, and full-SHA checkpoint tests; joint path is counted under P17 |
| P08 human data/BC | Partial | synthetic canonical-manifest/validation/BC/RL+BC/paired smokes and external-ingest privacy unit tests pass; authorized-data canary not run |
| P09 strategy | Complete | `douzero/strategy/`, `tests/test_p09_strategy.py` |
| P10 distillation | Complete | `douzero/distillation/`, `tests/test_p10_distillation.py` |
| P11 league/style | Complete | `douzero/league/`, `douzero/style/`, P11 tests |
| P12 coach curriculum | Complete | `douzero/coach/`, P12 tests |
| P13 search | Complete | `douzero/search/`, P13 tests |
| P14 AMP/DDP runtime | Partial | CPU BF16/Gloo legacy tests pass; CUDA AMP/NCCL are untested and standard learned-bidding DDP is not implemented |
| P15 paired evaluation | Partial | `p15-paired-result-v3` defines full traces, strict result integrity, real-checkout identity, nanosecond timing, and isolated protected-input snapshots. Commit-bound CI is read from the external PR evidence tuple; a protected formal run remains pending. |
| P16 deployment package | Complete (current format) | format-2 strict package/checksum/access/identity, loaded bidding/card-play inference, self-contained belief/search, rollback, and explicit format-1 rejection diagnostics pass; no RC exists |
| P17 readiness infrastructure | Partial | Replay, attestation, snapshot isolation, synthetic engineering smoke, and commit-bound PR evidence contracts are implemented. A protected formal run, standard/joint DDP, and external empirical gates remain open. |

## 4. Release Gates

| Gate | State | Evidence / reason |
|---|---|---|
| Legacy regression | PASS | Full CPU suite and deterministic baseline smoke |
| Standard rules | PASS (CPU) | Rule/state-machine tests plus full-game rollout |
| Learned bidding | PASS (implementation/CPU) | Bidding schema, model head/loss, standard trainer tests |
| Public/privileged boundary | PASS (CPU) | leakage, hidden-reallocation invariance, deployment access tests |
| Belief conservation | PASS (CPU) | exact decoder and differentiable marginal conservation tests |
| Joint belief training | PASS (CPU) | gradient/frozen/joint/alternating tests; GPU remains untested |
| Human-data canary | BLOCKED | No authorized data path was supplied; synthetic is not a real canary. External ingest now requires a keyed, HMAC-attested adapter result. |
| CPU tests | REVALIDATE PER HEAD | The required host, isolated Docker, and Python 3.11-3.13 matrix commands are listed below. Exact counts and outcomes are accepted only from the external evidence tuple for the head under review. |
| Single GPU | BLOCKED | No CUDA/NVIDIA device |
| Multi-GPU NCCL DDP | BLOCKED | No CUDA/NCCL runtime; standard learned bidding, joint/alternating belief, and distributed trainer resume also fail closed as implementation blockers |
| CUDA FP16/BF16 AMP | BLOCKED | No CUDA device; CPU BF16 is not a substitute |
| Checkpoint resume | PASS (focused CPU) | Full TrainerConfig/hash, value/optimizer/belief, GradScaler/AMP fallback, counters, first-bidder schedule, and Python/NumPy/Torch CPU/CUDA RNG are validated through temporary objects before commit. Frozen belief weights are hash-bound; replay uses an explicit flushed checkpoint boundary. Injected optimizer restore failure leaves the active model/optimizer unchanged. |
| Paired evaluation | PARTIAL / BLOCKED | Existing smokes predate result-v3 and are not formally collatable. Current collation requires separately approved deals, complete trace replay, a whole-result digest, real clean/stable Git identity, and detached GitHub OIDC/Sigstore attestation; no protected result has satisfied that boundary. |
| Full-game learned evaluation | PARTIAL | Strict learned-bidding two-deal equality smoke passes; 1,000-deal formal gate not run |
| Eight-ablation matrix | BLOCKED | Semantically trained ablation checkpoints unavailable |
| Calibration | BLOCKED | Labels are now intended to be derived from replayed winners, but no attested release-candidate predictions/results exist |
| Target latency/throughput | BLOCKED | Nanosecond evidence and instrumented-throughput recomputation are implemented; no validated target GPU result exists |
| Model package | PASS (tooling) | Verification now strict-constructs the runtime model and strict-loads finite tensors, so verification implies CPU loadability. Format-2 provenance is inherited and cross-checked from the source public checkpoint; in-memory/legacy migration exports are explicitly `release_eligible=false`. |
| Model card | PASS (documentation) | Honest no-RC draft; missing metrics marked unavailable |
| Artifact provenance | PARTIAL / BLOCKED | Paired-result digest, real Git identity, detached-attestation verification, strict formal JSON inputs, and checkpoint-root isolation are implemented and contract-tested. No protected-workflow result has been accepted. Several historical coach/distillation/eval-data side artifacts also remain below the universal requirement. |
| License review | PASS | `docs/third_party_review.md`, `THIRD_PARTY_NOTICES`; no copied external code |
| Rollback test | PASS (tooling) | Tampered candidate is rejected, then the immutable known-good package reloads and reproduces fixed-state inference |

## 5. Delivery Boundary

This change set delivers readiness infrastructure and fail-closed unsupported
paths. It does not deliver a release candidate or close the model-release
program. Merging the infrastructure must not change any `NONE`, `NOT READY`,
`BLOCKED`, `PARTIAL`, or `unavailable` result into a positive claim.

The public synthetic workflow exercises checkpoint snapshotting, offline
evaluation, deterministic replay, production attestation-aware P17 collation,
detached synthetic attestation, isolation flags, failure-to-upload gating, and
cleanup on a GitHub-hosted runner. It runs automatically for same-repository PR
merge results while recording the PR head as a separate identity; this keeps
the evaluated source digest identical to the GitHub attestation source digest.
Fork-origin code is skipped before an attestation-writing job receives a token.
It uses only generated public inputs, has a distinct signer-workflow identity,
and always emits `release_eligible=false` with an `insufficient` P17 result. It
cannot satisfy the protected producer policy.

The following remain external model-release gates rather than acceptance
claims for this infrastructure delivery: protected formal evaluation, approved
private-holdout replay, authorized human-data canary, target CUDA/NCCL
validation, independently trained ablations, and the 1,000-deal candidate
evaluation. Their absence keeps the model release blocked, even if repository
CI and the synthetic engineering smoke pass.

## 6. Blockers

### Blocker

- Every candidate merge head must have an external evidence tuple that names
  the exact head and merge-result SHAs and shows all three build plus all three
  test jobs passing. Local host/Docker evidence is supporting evidence, not a
  substitute for that commit-bound record. This tracked document intentionally
  cannot assert whether the moving PR head currently satisfies the gate.
- No protected GitHub Actions formal-evaluator workflow is currently recorded
  as having produced the required detached OIDC/Sigstore attestation. Verifier
  code is not an attestation, and an allowlisted SHA inside a result is not
  software-supply-chain proof.
- No private holdout has been replayed against approved deal payloads inside a
  trusted environment under the new boundary. The public trace-redacted
  projection therefore has no formal release evidence yet.
- No release candidate satisfies the minimum model-quality gate. There is no
  standard full-game trained checkpoint with learned bidding and 1,000 paired
  deals.
- Standard learned-bidding DDP is not implemented: the mixed bidding/card-play
  parameter graph fails closed before collection. Missing GPUs are therefore
  not the only reason the NCCL gate is open.
- Joint/alternating belief DDP gradient synchronization and DDP trainer
  checkpoint save/resume are not implemented. Both requests fail closed.
- CUDA single-GPU, CUDA AMP, and two-GPU NCCL DDP validation were not possible
  on this host.
- The real authorized human-data canary was not run because no authorized data
  path or HMAC key file was supplied.
- The full eight-row ablation matrix cannot be claimed without independently
  trained, identity-compatible checkpoints.
- Repository-wide provenance is not universally closed. Historical coach
  label/checkpoint, distillation/intermediate dataset, and quarantine artifacts
  do not all have full producer manifests. Format-2 release packages now inherit
  checkpoint provenance rather than accepting caller re-declaration, but that
  does not retroactively establish lineage for older side artifacts.

### High

- Target-hardware throughput, peak memory, latency percentiles, long-run
  checkpoint recovery, and `torch.compile` benefit remain unmeasured.
- Result-v3 records per-call nanoseconds and total evaluation wall time, but no
  attested target-hardware run has established instrumented or end-to-end
  throughput. The compatibility `actor_fps` field is not actor-loop FPS.
- Result-v3 and the protected workflow now record the immutable GitHub run,
  runner class, evaluator OCI image digest, dependency digest, and detailed
  hardware/runtime inventory. No attested target-hardware run has populated
  that evidence yet, so benchmark claims remain blocked.
- Learned-bidding full-game paired strength, calibration, and regression
  confidence intervals remain unmeasured.
- Standard learned-bidding and joint/alternating `torch.compile` paths are
  unsupported and unvalidated; both fail closed rather than being recommended.

### Medium

- Authorized-data deletion/rebuild and package-data exclusion have unit/synthetic
  evidence only, not a production dataset exercise.
- Canonical dataset sidecars expose only a configuration hash and a
  producer-asserted lineage flag. Supported readers verify checksum, count,
  rulesets, and lineage, but independent configuration reconstruction still
  requires the external run ledger.
- P16 format-1 directories remain tied to their matching P16 runtime. The P17
  verifier rejects them rather than guessing missing format-2 identities; they
  must be rebuilt from the original manifest-bearing public checkpoint and
  reviewed metadata.

### Low

- The fresh-image Docker CPU gate remains required for every candidate head.
  Its image digest, exact source identity, complete log, and outcome belong in
  the external commit-bound evidence rather than as mutable host state in this
  tracked report. `.venv` remains excluded from the Docker context.

## 7. Validation Commands

Run these gates from a clean checkout of the exact candidate head. Store their
outputs and identities with the PR or protected evaluation artifact; do not
copy a moving head SHA back into this file.

```bash
test -z "$(git status --porcelain)"
git rev-parse HEAD
git diff --check
actionlint .github/workflows/*.yml
.venv/bin/python -m compileall -q douzero tools tests
.venv/bin/python -m pytest -q
```

The isolated CPU gate must receive the same SHA and record it in the baseline:

```bash
head_sha="$(git rev-parse HEAD)"
docker run --rm --tmpfs /tmp:rw,exec,nosuid,size=4g \
  -e "DOUZERO_GIT_SHA=$head_sha" \
  -v "$PWD:/workspace" -w /workspace \
  douzero-p17-test:latest bash .docker/run_release_gate.sh
```

| Required evidence | Acceptance rule |
|---|---|
| Focused bidding-credit, deletion-publication, replay, provenance, workflow-isolation, private-projection, and negative-tamper tests | All pass on the candidate tree |
| Host full suite, compileall, `actionlint`, and `git diff --check` | All pass from a clean candidate checkout |
| Isolated Docker CPU release gate | Passes with `DOUZERO_GIT_SHA` equal to the candidate head; image identity and complete log are retained externally |
| Wheel build and installed-module smoke | Wheel digest and isolated import log are retained externally |
| Python 3.11, 3.12, and 3.13 GitHub matrix | All three build and all three test jobs pass for the candidate head/merge result |
| Review disposition | Draft is removed only after blocking code review is cleared; the exact review state is read from GitHub |
| Protected evaluator, approved-deal replay, detached attestation, and formal collation | Required for model release; result and attestation digests are external immutable evidence |

## 8. Conclusion

```text
Release candidate: NONE
Release status: NOT READY
```

```text
Commit-bound code evidence: see PR and workflow records
Formal model-release evidence: pending
```

The tracked report defines stable gates and validation methods. It does not
self-assert a moving PR head or CI result; those commit-bound facts must be read
from the external evidence tuple and the PR evidence artifact. Passing code CI
or the public synthetic workflow is not a model-release claim and does not
substitute for a protected attested replay. This infrastructure status does not
override unresolved repository-wide provenance work.
Standard learned-bidding DDP, joint/alternating belief DDP, distributed trainer
checkpoint resume, and standard/joint compile support remain explicit
implementation blockers, not external-validation euphemisms. The open
blockers prohibit `READY` or `READY WITH CONDITIONS`: model strength, target
GPU behavior, real authorized data, formal ablations, and the evaluation sample
size are release evidence, not optional polish.

Rollback is default-off: use `configs/enhanced.yaml` with `ruleset: legacy`,
`model.bidding_enabled: false`, and frozen belief mode; load the last known P16
public-policy package only with its matching runtime. The executable rollback
test rejects a tampered candidate and then reloads an immutable checksummed
known-good package to reproduce fixed-state inference. For a code rollback,
revert the P17 commits in reverse order rather than partially loading a P17
checkpoint into P16 code.
