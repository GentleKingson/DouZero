# P17 Release Readiness

## 1. Baseline Information

- Repository: `https://github.com/GentleKingson/DouZero`
- Base branch/SHA: `main` / `fa9f76de5ca31b4de33d4237e14b36e102c67655`
- Review branch: `codex/p17-release-readiness-closure`
- PR head observed during the 2026-07-17 evidence reconciliation:
  `bf4f643edece3c192ec237475ea9fafd8564c214`.
- Result-v3 replay/attestation remediation code baseline:
  `bf4f643edece3c192ec237475ea9fafd8564c214`. The source tree that passed the
  host and bind-mounted Docker gates was committed at this SHA, and all six PR
  build/test checks subsequently passed for Python 3.11, 3.12, and 3.13.
- GitHub's PR checks evaluated merge result
  `8ad96b199c91c9d6f7ca95f3d07e7a9f21d3b783`, which merges `bf4f643` into
  `fa9f76d`. This is commit-bound merge-result evidence, not a protected formal
  evaluator run.
- Post-`bf4f643` final-audit amendments add request-v2 protected checkpoint
  roots, step-scoped protected paths, an image-owned checkpoint snapshot
  boundary, disjoint control/audit/result/P17 outputs, and exact stage-minimal
  mounts. The current working-tree bytes passed the host and isolated Docker
  CPU gates, focused contract tests, wheel-install smoke, and two independent
  local boundary audits. They are not yet assigned a final validation SHA in
  this document. The next commit necessarily changes the PR head, and its SHA
  cannot be embedded in its own contents. After it is pushed, read the
  authoritative value from `headRefOid`, copy it into the PR description, and
  wait for all six checks on that new head before calling the PR validation
  final. `bf4f643` remains historical evidence, not the final head for these
  amendments.
- Historical implementation/test/Docker evidence was collected at
  `b7db29a3856324d65170b49ef32d17be7d3a6996`; it does not validate the current
  evaluation trust-boundary changes.
- Review window: 2026-07-15 through 2026-07-17 (Asia/Hong_Kong)
- Host: macOS 26.5.2, arm64, Apple A18 Pro; no NVIDIA device or
  `nvidia-smi`
- Isolated CPU runtime: Python 3.11.15, PyTorch 2.13.0+cpu, Linux arm64;
  CUDA build `None`, CUDA devices 0, NCCL unavailable
- Docker: Docker Desktop 29.6.1; only `runc` runtimes are installed (no
  NVIDIA container runtime)
- Authorized human-data variables: `DOUZERO_HUMAN_DATA_PATH` and
  `DOUZERO_HUMAN_DATA_HMAC_KEY_FILE` were both unset
- Scope: P00-P16 regression audit and P17 full-game training, joint belief,
  empirical-validation tooling, evaluation, packaging, and release gates

The working tree was clean at the start. `origin/main` was fetched and matched
the local base SHA before this branch was created.

### Independent Deep Review (2026-07-16)

An independent adversarial review invalidated the earlier code-merge judgment
at `6f66650`. A second review of published head `852b630` then showed that
hash-bound rows and internally recomputed summaries still did not prove the
declared game was played, that evaluator SHAs were still self-assertable,
calibration labels were caller-controlled, and standalone diagnostics copied
untrusted summaries. Green CI at that head therefore does not close the
formal-evaluation trust boundary. At that review point, PR #20 had to remain
Draft and could not be merged.

Published code baseline `bf4f643` targets those findings with result-v3 full
action traces, deterministic replay against separately approved deal payloads,
replay-derived calibration labels, nanosecond timing evidence, a canonical
whole-result digest, real clean/stable Git identity, and detached GitHub
OIDC/Sigstore artifact attestation. Local CPU, Docker, workflow-static, and
adversarial tamper validation pass, and the six-job PR matrix observed for
`bf4f643` is green. Post-`bf4f643` snapshot-isolation amendments pass current
working-tree host, Docker, workflow-contract, packaging, and independent local
audit validation, but still require a final commit and green PR matrix.
PR #20 remains Draft pending commit-bound review of the eventual head. No
protected formal evaluation has produced an
attested result under this boundary. The release candidate remains `NONE`; GPU,
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

`Complete` means the repository implementation and its CPU evidence were
reviewed in this closure. It does not imply model-strength, GPU, production,
or private-data validation.

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
| P15 paired evaluation | Partial | `bf4f643` defines `p15-paired-result-v3` full traces, strict result integrity, real-checkout identity, and nanosecond timing; its local and PR CI pass. Post-`bf4f643` snapshot-boundary amendments pass current working-tree validation but still need a final SHA, green PR matrix, and protected formal run. |
| P16 deployment package | Complete (current format) | format-2 strict package/checksum/access/identity, loaded bidding/card-play inference, self-contained belief/search, rollback, and explicit format-1 rejection diagnostics pass; no RC exists |
| P17 closure | Partial | Replay/attestation remediation at `bf4f643` has green local/Docker/PR CI evidence. Final snapshot-isolation amendments pass current working-tree validation and two independent local audits, but have no final validated SHA or new PR matrix yet; a protected formal run, standard/joint DDP, and external empirical gates remain open. |

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
| CPU tests | PASS locally; final CI pending | Host Python 3.14 and isolated Docker Python 3.11 each passed all 1,736 tests for the `bf4f643` baseline. The current uncommitted post-`bf4f643` working tree passed all 1,742 tests in both environments. The earlier six-job Python 3.11-3.13 matrix is green, but the eventual new head still requires its own matrix. |
| Single GPU | BLOCKED | No CUDA/NVIDIA device |
| Multi-GPU NCCL DDP | BLOCKED | No CUDA/NCCL runtime; standard learned bidding, joint/alternating belief, and distributed trainer resume also fail closed as implementation blockers |
| CUDA FP16/BF16 AMP | BLOCKED | No CUDA device; CPU BF16 is not a substitute |
| Checkpoint resume | PASS (focused CPU) | Full TrainerConfig/hash, value/optimizer/belief, GradScaler/AMP fallback, counters, first-bidder schedule, and Python/NumPy/Torch CPU/CUDA RNG are validated through temporary objects before commit. Frozen belief weights are hash-bound; replay uses an explicit flushed checkpoint boundary. Injected optimizer restore failure leaves the active model/optimizer unchanged. |
| Paired evaluation | PARTIAL / BLOCKED | Existing smokes predate result-v3 and are not formally collatable. Current collation requires separately approved deals, complete trace replay, a whole-result digest, real clean/stable Git identity, and detached GitHub OIDC/Sigstore attestation; no protected result has satisfied that boundary. |
| Full-game learned evaluation | PARTIAL | Strict learned-bidding two-deal equality smoke passes; 1,000-deal formal gate not run |
| Eight-ablation matrix | BLOCKED | Semantically trained ablation checkpoints unavailable |
| Calibration | BLOCKED | Labels are now intended to be derived from replayed winners, but no attested release-candidate predictions/results exist |
| Target latency/throughput | BLOCKED | Nanosecond evidence and instrumented-throughput recomputation are present at `bf4f643`; no validated target GPU result exists |
| Model package | PASS (tooling) | Verification now strict-constructs the runtime model and strict-loads finite tensors, so verification implies CPU loadability. Format-2 provenance is inherited and cross-checked from the source public checkpoint; in-memory/legacy migration exports are explicitly `release_eligible=false`. |
| Model card | PASS (documentation) | Honest no-RC draft; missing metrics marked unavailable |
| Artifact provenance | PARTIAL / BLOCKED | Paired-result digest, real Git identity, detached-attestation verification, and strict formal JSON inputs were locally/CI validated at `bf4f643`. Post-`bf4f643` checkpoint-root isolation passes current working-tree host/Docker/contract validation and two independent local audits, but still lacks a final SHA, new PR matrix, and accepted protected-workflow result. Several historical coach/distillation/eval-data side artifacts also remain below the universal requirement. |
| License review | PASS | `docs/third_party_review.md`, `THIRD_PARTY_NOTICES`; no copied external code |
| Rollback test | PASS (tooling) | Tampered candidate is rejected, then the immutable known-good package reloads and reproduces fixed-state inference |

## 5. Blockers

### Blocker

- The result-v3 replay/provenance code baseline is committed at `bf4f643`, and
  its six Python 3.11-3.13 build/test checks pass. Post-`bf4f643` checkpoint-root
  and image-owned snapshot changes pass host, Docker, packaging, and workflow
  contract validation on the current working tree. Two independent local
  audits found no remaining P0-P2 issue. They are not yet bound to a final SHA,
  and the eventual PR head must complete all six checks before merge.
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

- A fresh `douzero-p17-test` Docker image build reached final layer extraction
  but the local Docker VM lacked space. No Docker data was deleted. Validation
  used the existing isolated `douzero-p16-test` Python 3.11 CPU image with an
  ephemeral memory-backed `/tmp`. Both the source tree later committed as
  `bf4f643` and the current post-`bf4f643` working-tree bytes passed that gate;
  the latter run passed 1,742 tests plus compile, CLI, baseline, and diff
  checks. This is byte-equivalent bind-mounted working-tree evidence, not a
  fresh clean-checkout or commit-bound Docker run. `.venv` remains excluded
  from the Docker context.

## 6. Validation Commands

The commands below validate the source tree committed as code baseline
`bf4f643`. Host and Docker commands ran immediately before commit on the same
tree bytes; GitHub then validated the corresponding PR merge result. The Docker
run was not performed from a fresh clean checkout of `bf4f643`, so it must be
described as byte-equivalent pre-commit evidence rather than commit-bound
Docker provenance. Earlier 1,575/1,582-test runs, hash-only collation, and
historical Docker gates still must not be cited as current-code proof. The
post-`bf4f643` final-audit bytes now have fresh focused, full-host, and isolated
Docker validation, recorded separately below. They still require a final SHA
and new PR matrix; working-tree validation is not commit-bound evidence.

| Validation for code baseline `bf4f643` | Current status |
|---|---|
| Focused result-v3 replay, provenance, timing, calibration, strict JSON, protected-workflow, private-projection, and negative tamper tests | PASS; 200 focused tests are included in the final full run; the final four high-risk suites also passed independently |
| Host full suite, compile, `actionlint`, and `git diff --check` | PASS; 1,736 tests, one expected `lambda_bc==0` warning, compileall/actionlint/diff clean |
| Wheel build and new-module inventory | PASS; `douzero-1.1.0-py3-none-any.whl` contains replay, provenance, strict formal-data, and checkpoint-input modules |
| Docker release gate | PASS on byte-equivalent pre-commit source bind-mounted into the isolated Python 3.11.15/PyTorch 2.13 CPU image; 1,736 tests plus compile/CLI/baseline/diff gates passed. This was not a fresh clean-checkout run at `bf4f643`. Fresh image extraction separately failed for local Docker disk space. |
| Python 3.11, 3.12, and 3.13 GitHub CI matrix | PASS for all six build/test checks on PR merge result `8ad96b1` derived from head `bf4f643`; the eventual post-audit head requires a new green matrix |
| Protected GitHub evaluator run, detached OIDC/Sigstore attestation, approved-deal replay, and formal collation | NOT RUN |

| Post-`bf4f643` current working-tree validation | Current status |
|---|---|
| Host full suite | PASS; 1,742 tests and one expected `lambda_bc==0` warning |
| Isolated Docker CPU release gate | PASS; Python 3.11.15/PyTorch 2.13 CPU, 1,742 tests plus compile, CLI, baseline, and diff gates. This was a byte-equivalent bind-mounted working-tree run, not a fresh commit checkout. |
| Snapshot/workflow isolation contract | PASS; 32 focused tests, and two independent local audits found no remaining P0-P2 issue |
| `actionlint`, compileall, and `git diff --check` | PASS |
| Wheel build and isolated installed-module smoke | PASS; wheel SHA-256 `950aa5a4fe267888e56fc3502f00929c83a941a4d2d74e8db0fec7614404fa81`, with image-owned `douzero.evaluation.snapshot_cli` import/help verified from a fresh temporary virtual environment outside the checkout |
| Final commit SHA and Python 3.11-3.13 GitHub CI matrix | PENDING |
| Protected GitHub evaluator run and detached attestation | NOT RUN |

### PR Description Reconciliation

The PR #20 body observed on 2026-07-17 still names `6f66650` as the final head,
claims the intervening changes are documentation-only, and cites the historical
1,575-test/Docker baseline. Those statements are stale. After the final
post-audit commit is pushed, update the remote body without changing its
Draft or `NOT READY` status:

- obtain the exact final `headRefOid` from GitHub rather than predicting it in
  this file;
- identify `bf4f643` as the previous result-v3 validation baseline, and identify
  the eventual post-audit head separately rather than calling later changes
  documentation-only;
- cite 1,736 host/Docker tests only as the `bf4f643` baseline; cite the 1,742
  host/Docker result as post-audit working-tree evidence with the Docker
  provenance limitation above, not as a clean final-head checkout;
- cite all six Python 3.11-3.13 checks only after they are green on the final PR
  head; and
- retain `Release candidate: NONE` and `Release status: NOT READY`.

### Archived Pre-v3 Evidence

These entries are retained only as historical engineering evidence tied to
their stated older SHA. The words "current tree" and "post-review" in an
archived command referred to that earlier review point, not the present
working tree.

| Archived command | Historical result |
|---|---|
| `git status`; `git branch --show-current`; `git rev-parse HEAD`; `git log --oneline --decorate -20`; `git fetch origin main` | PASS; clean start, new P17 branch from local/remote `fa9f76d` |
| Host Python/Torch/CUDA, `nvidia-smi`, OS, Docker/runtime probes | PASS as probes; no CUDA device, driver interface, NCCL, or NVIDIA Docker runtime |
| `docker build -f .docker/Dockerfile.test -t douzero-p17-test .` | FAILED: Docker VM ran out of space during `apt` installation; user Docker data was not deleted |
| `.venv/bin/python -m compileall -q douzero tests *.py`; required shell syntax checks | PASS |
| All 12 requested CLI `--help` commands; Docker gate's 17 Python/shell entry points | PASS, all exit 0 |
| `.venv/bin/python -m pytest -q tests/test_p17_bidding.py tests/test_p17_joint_belief.py tests/test_p17_release_tooling.py` | PASS, 49 tests |
| `.venv/bin/python -m pytest --collect-only -qq`; `.venv/bin/python -m pytest -q` on clean `b7db29a` | 1,575 collected; PASS, 1,575 tests and one expected `lambda_bc==0` warning |
| First post-review Docker `run_tests.sh` attempt | FAILED: the source-SHA tamper test replaced the image's fixed all-zero test SHA with the same value; the test now always chooses a different valid SHA |
| Historical bind-mounted checkout `docker run ... bash .docker/run_tests.sh -q` | PASS, 1,575 tests and one expected warning |
| `/usr/bin/time -p docker run ... bash .docker/run_release_gate.sh` on clean `b7db29a` | PASS; 1,575 tests in 93.60 s plus compile/CLI/diff/baseline, 110.31 s total |
| Fresh `train_v2.py --config configs/standard_v2.yaml ... --bidding_policy max` on `b7db29a` | PASS; 2 full games, 114 play and 2 bid transitions, one finite optimizer step, parameters changed, no redeals/caps |
| Fresh strict standard `--resume_checkpoint ...`, one additional game | PASS; cumulative 3 games, 137 play and 3 bid transitions, 2 optimizer steps, parameters changed; full code/rule/model/`v2-bidding-2` identities matched |
| Standard `--bidding_policy learned --bidding_warm_start_policy max --bidding_learned_probability 0.5` | FAILED CLOSED: the untrained head produced only abandoned auctions, so no labelled bid minibatch existed; no optimizer step was claimed |
| Fresh standard learned warm-start probability `0.1`, 4 episodes | PASS; 151 play and 5 bid transitions, one finite optimizer step, parameters changed, no redeals/caps |
| `train_belief.py --ruleset standard ...` then `evaluate_belief.py --ruleset standard ...` | PASS; 32 training labels; 28 evaluation samples; constrained conservation 28/28; result binds full SHA/rule/schema/checkpoint/config identities |
| Joint `train_v2.py ... --belief_training_mode joint` and strict resume | PASS; coupled checkpoint version 2; 1 then 2 optimizer steps, finite losses, parameters changed |
| Final-SHA standard/joint checkpoint focused tests | Initial joint selector name was wrong and collected no such test; corrected command PASS, 2 tests |
| First two Draft-PR test matrices | FAILED on Linux CPU BF16: the random value-only fixture produced a zero belief update across the runner's PyTorch versions, including after the sensitive belief path was moved to float32 |
| Float32 joint-belief path plus deterministic supervised AMP fixture; focused host and Docker tests | PASS; the belief encoder/DP/features opt out of outer autocast, and the AMP test now uses an explicit weighted belief target rather than relying on a random nonzero value gradient |
| Synthetic canonical serialization then manifest-default `validate_human_games.py` | PASS; 4 total/valid, 0 quarantined/parse errors; both full-SHA sidecars verified count/rule/checksum/lineage |
| Strict external-ingest HMAC/attestation tests (`tests/test_p08_ingest_cli.py`) | PASS, 10 tests; missing key, unsalted IDs, leakage, and wrong attestations fail closed; real adapter run not performed |
| Dataset manifest missing/tamper/count/ruleset/unverified-lineage tests | PASS; training/rebuild consumers fail closed; migration output cannot train or release |
| `pretrain_bc.py ... --epochs 1 --batch_size 4 ...` | PASS; 115 samples, val loss 1.2304, val top-1 0.486; checkpoint config binds dataset SHA; synthetic only |
| RL+BC `train_v2.py --config /tmp/douzero-p17-rlbc.yaml ...` | PASS; 19 transitions, one finite optimizer step, parameters changed |
| Fresh synthetic BC before/after `evaluate_paired.py ... --num-deals 4 --bootstrap-samples 2000` | PASS as a pre-v3 code smoke; estimate -0.1250, CI [-0.5000, 0.2500]; its old result identity is not accepted by current formal collation |
| Historical learned full-game equality on `v2-bidding-2`, 2 deals/2,000 bootstrap | PASS as code smoke; 6 seat-rotated games, 5 learned-bid calls, estimate 0, CI [0, 0], no cap fallback |
| Historical `prepare_p17_evaluation.py` hash/set-only command | OBSOLETE for formal use: it did not supply approved deal payloads or a detached attestation and cannot establish current eligibility |
| P17 result-integrity and all-pass-cap tests | PASS; forged evaluator SHA, ordered deal hash/set, terminal outcome/score, confidence/count/CI evidence is rejected; forced cap fallbacks are cleared, audited, excluded, and release-ineligible |
| `tools/package_model.py ...` with the fresh standard learned-bidding sidecar | PASS; format-2 public package binds `b7db29a`, `v2-bidding-2`, rule/model/training identities and summaries |
| First manual package-load command | FAILED: omitted required runtime schema/rules/config and used a nonexistent rollback test selector; no inference was claimed |
| Second manual package-load command | MODEL LOAD FAILED: treated wrapped `model_config.json` as bare config; two independently selected package tests still passed because the shell lacked `set -e` |
| Corrected `set -e` package load plus rollback/self-contained-belief tests | PASS; checksum verification, learned bid legal, loaded card-play action legal, rollback and actual search-enabled belief inference tests passed |
| `.venv/bin/python -m build --wheel ...` | FAILED: repository `build/` package shadowed the absent build frontend; no wheel was claimed from this command |
| `.venv/bin/python -m pip wheel . --no-deps --no-build-isolation ...` | PASS; `douzero-1.1.0-py3-none-any.whl` built |
| Docker temporary venv install/import from `/tmp` | First assertion used nonexistent `douzero.__version__` and failed after successful install; corrected `importlib.metadata.version` check PASS, importing from venv site-packages with Torch 2.13.0+cpu |
| `scripts/validate_gpu_training.sh --probe-only`; full script | Probe PASS; full script expected exit 3 with CUDA unavailable and DDP implementation blockers, no GPU metrics claimed |
| `git diff --check`; clean-tree and SHA readback after code validation | PASS; clean `b7db29a3856324d65170b49ef32d17be7d3a6996` |
| Draft PR build/test matrix on `43b50a3e223e08844724762ea2b49f458564794f` | PASS: build and test on Python 3.11/3.12/3.13; slowest test job 3m59s |
| Independent-review directed regressions (`tests/test_p17_deep_review_fixes.py`) | PASS: canonical deal identity/raw evidence, fixed statistics, strict package loadability, source-checkpoint provenance, epsilon-label masking, frozen-belief identity, atomic restore failure, and whole-game redeal-cap exclusion |
| Post-review focused evaluation/deployment/bidding/belief suites | PASS |
| Post-review host full suite | PASS, 1,582 tests and the existing expected `lambda_bc==0` warning |
| Post-review Docker release gate (`douzero-p16-test:latest`, historical bind-mounted tree) | PASS, Python 3.11.15/Linux arm64; compile, CLI, 1,582 tests, baseline capture, and `git diff --check` |

## 7. Conclusion

```text
Release candidate: NONE
Release status: NOT READY
```

```text
bf4f643 baseline validated; final isolation head and formal evidence pending
```

Code baseline `bf4f643` has local, byte-equivalent Docker, and PR CI evidence.
Post-audit snapshot-isolation amendments have current working-tree host,
byte-equivalent Docker, packaging, contract, and independent-audit evidence,
but still need a final SHA and green PR checks. None of that is a release claim,
and no protected attested replay run has occurred.
It does not override unresolved repository-wide provenance work.
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
