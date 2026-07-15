# P17 Release Readiness

## 1. Baseline Information

- Repository: `https://github.com/GentleKingson/DouZero`
- Base branch/SHA: `main` / `fa9f76de5ca31b4de33d4237e14b36e102c67655`
- Review branch: `codex/p17-release-readiness-closure`
- Reviewed implementation/test/Docker SHA: `b7db29a3856324d65170b49ef32d17be7d3a6996`
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
| P15 paired evaluation | Partial | `p15-paired-result-v2` runtime identities, statistics, synthetic BC card-play, and identical-model learned full-game smokes pass; formal matrix was not run |
| P16 deployment package | Complete (current format) | format-2 strict package/checksum/access/identity, loaded bidding/card-play inference, self-contained belief/search, rollback, and explicit format-1 rejection diagnostics pass; no RC exists |
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
| Artifact provenance | PARTIAL / BLOCKED | P17 trainer/belief checkpoints, canonical human datasets, paired results, and format-2 packages carry strict identities; several historical coach/distillation/eval-data side artifacts do not yet meet the universal requirement |
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
- Repository-wide provenance is not universally closed. Historical coach
  label/checkpoint, distillation/intermediate dataset, bare evaluation-deal,
  and quarantine artifacts do not all have full producer manifests; package
  training metadata is caller-supplied rather than cryptographically derived
  from a source trainer checkpoint. The strict P17 paths do not erase this
  cross-cutting release blocker.

### High

- Target-hardware throughput, peak memory, latency percentiles, long-run
  checkpoint recovery, and `torch.compile` benefit remain unmeasured.
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
| `.venv/bin/python -m compileall -q douzero tests *.py`; required shell syntax checks | PASS |
| All 12 requested CLI `--help` commands; Docker gate's 17 Python/shell entry points | PASS, all exit 0 |
| `.venv/bin/python -m pytest -q tests/test_p17_bidding.py tests/test_p17_joint_belief.py tests/test_p17_release_tooling.py` | PASS, 49 tests |
| `.venv/bin/python -m pytest --collect-only -qq`; `.venv/bin/python -m pytest -q` on clean `b7db29a` | 1,575 collected; PASS, 1,575 tests and one expected `lambda_bc==0` warning |
| First post-review Docker `run_tests.sh` attempt | FAILED: the source-SHA tamper test replaced the image's fixed all-zero test SHA with the same value; the test now always chooses a different valid SHA |
| Current-tree `docker run ... bash .docker/run_tests.sh -q` | PASS, 1,575 tests and one expected warning |
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
| Fresh synthetic BC before/after `evaluate_paired.py ... --num-deals 4 --bootstrap-samples 2000` | PASS as code smoke; estimate -0.1250, CI [-0.5000, 0.2500]; `p15-paired-result-v2` identities verified; not strength evidence |
| Fresh learned full-game equality on current `v2-bidding-2`, 2 deals/2,000 bootstrap | PASS as code smoke; 6 seat-rotated games, 5 learned-bid calls, estimate 0, CI [0, 0], no cap fallback |
| `prepare_p17_evaluation.py --matrix ... --full-game-result ...` | PASS; fixed seven files; result/checkpoint/runtime identities validated; recomputed readiness `insufficient` at 2/1,000 deals |
| P17 result-integrity and all-pass-cap tests | PASS; forged runtime/checkpoint/count/CI evidence is rejected; forced cap fallbacks are cleared, audited, excluded, and release-ineligible |
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

## 7. Conclusion

```text
Release candidate: NONE
Release status: NOT READY
```

```text
Implementation complete, external empirical validation pending
```

That required sentence is scoped to the implemented P17 single-process CPU
surface. It does not override unresolved repository-wide provenance work.
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
