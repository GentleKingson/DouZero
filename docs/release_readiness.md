# P17 Release Readiness

## 1. Baseline Information

- Repository: `https://github.com/GentleKingson/DouZero`
- Base branch/SHA: `main` / `fa9f76de5ca31b4de33d4237e14b36e102c67655`
- Review branch: `codex/p17-release-readiness-closure`
- Reviewed implementation/test/CI SHA: `5661671b6b752ceb9b48d4d7ca520a0daa4379ce`
  (the documentation-only closure commit follows this reviewed code SHA)
- Review date: 2026-07-15 (Asia/Hong_Kong)
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
| P01 config/checkpoint/package metadata | Complete | `douzero/config/`, `douzero/checkpoint/`, config/checkpoint tests |
| P02 standard rules/scoring | Complete | `douzero/env/rules.py`, `game.py`, `scoring.py`, bidding/scoring tests |
| P03 public/privileged Observation V2 | Complete | `douzero/observation/`, leakage and observation tests |
| P04 legacy factorization | Complete | `douzero/dmc/models_factorized.py`, parity tests |
| P05 Model V2 | Complete | `douzero/models_v2/`, model/checkpoint tests |
| P06 multi-objective training | Complete | `douzero/training/`, V2 trainer/loss/calibration tests |
| P07 belief | Complete | constrained decoder/sampler and conservation tests; joint path is counted under P17 |
| P08 human data/BC | Partial | full synthetic ingest/split/BC/RL+BC/paired code smoke and privacy tests pass; authorized-data canary not run |
| P09 strategy | Complete | `douzero/strategy/`, `tests/test_p09_strategy.py` |
| P10 distillation | Complete | `douzero/distillation/`, `tests/test_p10_distillation.py` |
| P11 league/style | Complete | `douzero/league/`, `douzero/style/`, P11 tests |
| P12 coach curriculum | Complete | `douzero/coach/`, P12 tests |
| P13 search | Complete | `douzero/search/`, P13 tests |
| P14 AMP/DDP runtime | Partial | CPU BF16/Gloo legacy tests pass; CUDA AMP/NCCL are untested and standard learned-bidding DDP is not implemented |
| P15 paired evaluation | Partial | statistics, positional/API compatibility, synthetic BC card-play, and identical-model learned full-game smokes pass; formal matrix was not run |
| P16 deployment package | Complete (current format) | format-2 strict package/checksum/access/identity, self-contained belief, rollback, and explicit format-1 rejection diagnostics pass; no RC exists |
| P17 closure | Partial | single-process CPU implementation and validation completed; standard/joint DDP and external empirical gates remain open |

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
| CPU tests | PASS | Isolated Docker test suite and release gate |
| Single GPU | BLOCKED | No CUDA/NVIDIA device |
| Multi-GPU NCCL DDP | BLOCKED | No CUDA/NCCL runtime; standard learned bidding, joint/alternating belief, and distributed trainer resume also fail closed as implementation blockers |
| CUDA FP16/BF16 AMP | BLOCKED | No CUDA device; CPU BF16 is not a substitute |
| Checkpoint resume | PASS (CPU) | Standard trainer save/load followed by optimizer step |
| Paired evaluation | PARTIAL | 4-deal synthetic BC and 2-deal identical-model smokes only; no formal candidate. P17 collation re-hashes matrix checkpoints and recomputes deal-level evidence from game rows. |
| Full-game learned evaluation | PARTIAL | Strict learned-bidding two-deal equality smoke passes; 1,000-deal formal gate not run |
| Eight-ablation matrix | BLOCKED | Semantically trained ablation checkpoints unavailable |
| Calibration | BLOCKED | No release-candidate predictions/results |
| Target latency/throughput | BLOCKED | No target GPU or release candidate |
| Model package | PASS (tooling) | Standard bidding and self-contained belief packages clean-load; tamper/identity/incomplete-state rejection tests pass |
| Model card | PASS (documentation) | Honest no-RC draft; missing metrics marked unavailable |
| License review | PASS | `docs/third_party_review.md`, `THIRD_PARTY_NOTICES`; no copied external code |
| Rollback test | PASS (tooling) | Tampered candidate is rejected, then the immutable known-good package reloads and reproduces fixed-state inference |

## 5. Blockers

### Blocker

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

### High

- Target-hardware throughput, peak memory, latency percentiles, long-run
  checkpoint recovery, and `torch.compile` benefit remain unmeasured.
- Learned-bidding full-game paired strength, calibration, and regression
  confidence intervals remain unmeasured.

### Medium

- Authorized-data deletion/rebuild and package-data exclusion have unit/synthetic
  evidence only, not a production dataset exercise.
- P16 format-1 directories remain tied to their matching P16 runtime. The P17
  verifier rejects them rather than guessing missing format-2 identities; they
  must be rebuilt from the original manifest-bearing public checkpoint and
  reviewed metadata.

### Low

- A fresh Docker image build was attempted but the local Docker VM lacked
  space. Validation used the existing isolated `douzero-p16-test` CPU image
  with the current checkout bind-mounted. `.venv` is now excluded from Docker
  context to prevent the 723 MB local environment from being sent again.

## 6. Validation Commands

The final command log records only commands that validate deliverables, not
routine source-inspection commands. Machine-specific and large outputs stay
under ignored `artifacts/`.

| Command | Result |
|---|---|
| `git status`; `git branch --show-current`; `git rev-parse HEAD`; `git log --oneline --decorate -20`; `git fetch origin main` | PASS; clean start, new P17 branch from local/remote `fa9f76d` |
| Host Python/Torch/CUDA, `nvidia-smi`, OS, Docker/runtime probes | PASS as probes; no CUDA device, driver interface, NCCL, or NVIDIA Docker runtime |
| `docker build -f .docker/Dockerfile.test -t douzero-p17-test .` | FAILED: Docker VM ran out of space during `apt` installation; user Docker data was not deleted |
| `.venv/bin/python -m compileall -q douzero *.py tools benchmarks` | PASS |
| All 12 required CLI `--help` commands, including `tools/package_model.py --help` | PASS, all exit 0 |
| `.venv/bin/python -m pytest -q tests/test_p17_bidding.py tests/test_p17_joint_belief.py tests/test_p17_release_tooling.py` | PASS, 38 tests |
| `.venv/bin/python -m pytest -q` | PASS; 1,563 tests, one expected warning |
| First post-review Docker `run_tests.sh` attempt | FAILED: the source-SHA tamper test replaced the image's fixed all-zero test SHA with the same value; the test now always chooses a different valid SHA |
| `docker run ... douzero-p16-test:latest bash .docker/run_tests.sh` | PASS after the test fix; 1,563 tests, one expected warning, 117.62 s |
| `docker run ... douzero-p16-test:latest bash .docker/run_release_gate.sh` | PASS; 1,563 tests, compile/CLI/diff/baseline gate, 106.90 s |
| Host and Docker `train_v2.py --config configs/standard_v2.yaml ... --bidding_policy max` | PASS; 2 full games, 97 play and 2 bid transitions, one optimizer step, parameters changed, checkpoint saved |
| Host and Docker standard `--resume_checkpoint ...` | PASS; totals reached 3 games, 138 play and 3 bid transitions, 2 optimizer steps, parameters changed after restore |
| Standard `--bidding_policy learned --bidding_warm_start_policy max --bidding_learned_probability 0.5` | FAILED CLOSED: the untrained head produced only abandoned auctions, so no labelled bid minibatch existed; no optimizer step was claimed |
| Standard learned warm-start smoke with probability `0.1`, 4 episodes | PASS; 162 play and 4 bid transitions, one finite optimizer step, parameters changed |
| `train_belief.py --num_episodes 1 --epochs 1 ...` | PASS; 28 labelled samples, 7 finite batches, strict belief checkpoint saved |
| Joint `train_v2.py ... --belief_training_mode joint` and strict resume | PASS; coupled checkpoint version 2; 1 then 2 optimizer steps, finite losses, parameters changed |
| Final-SHA standard/joint checkpoint focused tests | Initial joint selector name was wrong and collected no such test; corrected command PASS, 2 tests |
| First Draft-PR test matrix | FAILED on Linux CPU BF16: a value-only joint-belief update could quantize to zero under the runner's PyTorch/oneDNN combination |
| Float32 joint-belief path fix; focused host and Docker tests | PASS; the belief encoder, constrained DP, and belief feature projection now opt out of outer autocast while the value model remains AMP-enabled |
| `scripts/validate_gpu_training.sh --probe-only` | PASS; sanitized `environment.json` written |
| `scripts/validate_gpu_training.sh` | EXPECTED BLOCKED (exit 3); GPU cases `not_run`, DDP `blocked_implementation` |
| Synthetic `ingest_human_games.py` then `validate_human_games.py` | PASS; 4 total, 4 valid, 0 quarantined, 0 parse errors |
| Strict external-ingest HMAC/attestation tests | PASS; missing key, shape-only/unsalted IDs, adapter leakage, and wrong-run attestations fail closed; synthetic stays keyless |
| Game-level split and per-role BC metric program | PASS; train/val/test 2/1/1, zero ID overlap; descriptive metrics recorded in `human_data_canary.md` |
| `pretrain_bc.py ... --epochs 1 --batch_size 4 ...` | PASS; 115 decisions, val loss 1.2304, val top-1 0.486; synthetic smoke only |
| RL+BC `train_v2.py --config /tmp/douzero-p17-rlbc.yaml ...` | PASS; 19 transitions, one finite optimizer step, parameters changed |
| Synthetic BC before/after `evaluate_paired.py ... --num-deals 4 --bootstrap-samples 2000` | PASS as code smoke; estimate +0.1250, CI [0, 0.375], not strength evidence |
| Learned full-game equality `evaluate_paired.py ... --num-deals 2 --bootstrap-samples 2000` | PASS as code smoke; identical bundles estimate 0, CI [0, 0] |
| `prepare_p17_evaluation.py --write-matrix-template ...` and `--matrix ... --output artifacts/evaluation/p17` | PASS; fixed seven-file artifact set, all formal rows explicitly unavailable/not run |
| P17 result-integrity and all-pass-cap tests | PASS; result checkpoint hashes must match the validated matrix, deal evidence/CI is recomputed from rows, and forced smoke fallbacks are excluded and release-ineligible |
| `tools/package_model.py ...` with a strict standard learned-bidding sidecar | PASS; format-2 public package created with full identity and summaries |
| `.venv/bin/python -m build --wheel ...` | FAILED: repository `build/` package shadowed the absent build frontend; no wheel was claimed from this command |
| `.venv/bin/python -m pip wheel . --no-deps --no-build-isolation ...` | PASS; `douzero-1.1.0-py3-none-any.whl` built |
| Docker temporary venv install/import from `/tmp` | First assertion used nonexistent `douzero.__version__` and failed after successful install; corrected `importlib.metadata.version` check PASS, importing from venv site-packages with Torch 2.13.0+cpu |
| `git diff --check` and shell syntax checks | PASS |

## 7. Conclusion

```text
Release candidate: NONE
Release status: NOT READY
```

```text
Implementation complete, external empirical validation pending
```

That sentence is scoped to the single-process CPU implementation. Standard
learned-bidding DDP, joint/alternating belief DDP, and distributed trainer
checkpoint resume remain explicit implementation blockers, not
external-validation euphemisms. The open blockers prohibit `READY` or `READY
WITH CONDITIONS`: model strength, target GPU behavior, real authorized data,
formal ablations, and the evaluation sample size are release evidence, not
optional polish.

Rollback is default-off: use `configs/enhanced.yaml` with `ruleset: legacy`,
`model.bidding_enabled: false`, and frozen belief mode; load the last known P16
public-policy package only with its matching runtime. The executable rollback
test rejects a tampered candidate and then reloads an immutable checksummed
known-good package to reproduce fixed-state inference. For a code rollback,
revert the P17 commits in reverse order rather than partially loading a P17
checkpoint into P16 code.
