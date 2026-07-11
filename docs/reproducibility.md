# Reproducibility

> P00 establishes how to reproduce the legacy baseline deterministically.
> Later phases add their own seed/version plumbing on top of this.

## Why seeding matters here

The legacy DouZero codebase performs **no internal seeding**. The only source
of randomness during a normal episode is:

```
douzero/env/env.py:60   np.random.shuffle(_deck)   # inside Env.reset
```

`generate_eval_data.py:19` also shuffles without seeding. There is no
`np.random.seed`, `random.seed`, or `torch.manual_seed` anywhere in `douzero/`.

**Consequence:** reproducibility is the **caller's** responsibility. P00 does
not change any production module; instead it centralises seeding in the test
fixtures and tooling:

- `tests/conftest.py` — `set_seed(seed)` seeds `random`, `numpy`, and `torch`
  before every test (autouse fixture).
- `tools/capture_baseline.py` — derives a per-deal seed (`base_seed + i`) and
  reseeds all three RNGs before each deal and each model initialisation.
- `benchmarks/bench_legacy.py` — seeds before each measured scenario.

P01 will lift this into a project-wide seeding utility (covering CUDA and
per-actor derived seeds); P00 only needs offline, single-process CPU
determinism.

## Determinism contract (what is actually pinned)

There are two layers — see "Same-environment diagnostic vs. portable contract"
below the table for why they are separate.

| Quantity | Layer | How it is pinned | Test |
|---|---|---|---|
| Deal order | diagnostic | `np.random.seed` before `Env.reset` | `test_env_rollout.test_reset_reseeds_deck` |
| Legal actions for a fixed deal | **portable** | constructed directly (no RNG); exact count + TYPE_15_WRONG exception set as committed integers/sets | `test_baseline_invariants` + `test_legal_actions_snapshot` |
| Observation shapes/dtypes/widths | **portable** | frozen integer constants per role (373/484, 319/430, 5×162) | `test_baseline_invariants.test_frozen_obs_shapes_and_dtypes` |
| Model forward output (same env) | diagnostic | byte-equal across two runs in one environment | `test_model_shapes.test_fixed_init_output_hash_is_stable_across_runs` |
| Model forward output (tolerance) | **portable** | `assert_allclose(rtol=1e-6, atol=1e-7)` + `array_equal` across two same-seed instantiations | `test_baseline_invariants.test_baseline_float_tolerance_same_seed_two_models` |
| `DeepAgent.act` | **portable** | deterministic argmax under fixed weights; identical action across two agents | `test_deepagent_selection.test_act_is_deterministic_under_fixed_weights` |
| Terminal invariants | **portable** | winner∈{landlord,farmer}; 54-card conservation; reward sign/scale per objective | `test_env_rollout` |

The legacy models contain no dropout and no BatchNorm, so under `model.eval()`
the forward pass is bit-for-bit deterministic on CPU given identical weights
*in one environment*. Across environments, float bytes may differ; the
portable contract therefore pins shapes/sets and uses tolerance for floats.

## Running the suite (Docker, CPU-only)

The test image is `python:3.11-slim` and hides CUDA entirely, so the legacy
models (which have no `device` argument) run on CPU:

```bash
docker build -f .docker/Dockerfile.test -t douzero-p00-test .
docker run --rm douzero-p00-test
```

`run_tests.sh` runs: `compileall`, all three `--help`, then `pytest -q`.

To run a subset inside the container:

```bash
docker run --rm douzero-p00-test python -m pytest -k model
```

## CI Python matrix

P01 raised the minimum supported Python to **3.11** (see `pyproject.toml`
`requires-python`). The Docker image stays pinned to `python:3.11-slim` for a
fast, reproducible local/CPU path; the GitHub Actions matrix covers the full
supported range so cross-version regressions are caught upstream:

- **`Tests` workflow** (`.github/workflows/ci.yml`): mandatory matrix
  **3.11 / 3.12 / 3.13**, CPU-only, `DOUZERO_GIT_SHA` injected, runs
  `compileall`, the three `--help`, `pytest -q`, and a 2-deal baseline smoke.
- **`Building` workflow** (`.github/workflows/python-package.yml`): same matrix,
  `python -m build` + `pip install dist/*.whl` + import smoke from outside the
  repo (proves the wheel is importable, not the source tree).

Python 3.14 is not yet in the mandatory matrix; it will be added (initially as
a non-blocking smoke) once `rlcard` / `GitPython` compatibility on 3.14 is
confirmed. Both workflows pin `actions/checkout` and `actions/setup-python` to
**v6** tag commit SHAs (`df4cb1c0…` and `ece7cb06…` respectively; Node 24,
requires runner ≥ 2.327.1).

## Running locally without Docker

If torch/numpy/rlcard are already installed on the host:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

## P04 factorized forward verification

The factorized forward (deployment-only, P04) is numerically equivalent to
the legacy per-action forward under the same weights. To verify parity and
the LSTM-call-count reduction:

```bash
# Parity + state_dict + LSTM-count + DeepAgent backend parity.
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_factorized_parity.py

# Latency + LSTM call count (writes artifacts/benchmark/bench_factorized.json).
CUDA_VISIBLE_DEVICES="" python benchmarks/bench_factorized.py --rounds 30
```

See `docs/factorized_forward.md` for the equivalence derivation and the
training-integration boundary.

## Capturing the baseline

```bash
CUDA_VISIBLE_DEVICES="" python tools/capture_baseline.py \
    --num_deals 64 \
    --output artifacts/baseline/baseline.json
```

The JSON is **self-describing**: it embeds `environment` (python / torch /
numpy / rlcard versions, git SHA, platform, CUDA availability) and the
`base_seed`. To verify reproducibility, run it twice in the *same* environment
and `diff` the two JSON files — `legal_actions_sha256` and
`model_output_sha256` must be byte-identical within one environment.

### Same-environment diagnostic vs. portable contract

There are two layers to the baseline, and they must not be conflated:

- **Portable contract (committed).** Integer/string invariants that do not
  depend on float reproducibility: observation shapes and dtypes, the
  role→feature-width mapping (373 landlord / 484 farmer), the fixed-deal
  TYPE_15_WRONG exception set, and terminal winner/bomb/card-conservation
  invariants. These live in **`tests/test_baseline_invariants.py`** and run in
  CI on every push. They are the real cross-machine gate.
- **Same-environment diagnostic (git-ignored).** The raw
  `model_output_sha256` bytes in `artifacts/baseline/baseline.json` are a
  determinism diagnostic for *one* environment (same python/torch/numpy, CPU).
  Float model-output bytes can legitimately differ across torch builds or
  platforms while behaviour is still equivalent, so the raw hash is **not** a
  cross-machine hard gate. Float-equivalence across two model instantiations
  in one environment is checked with a tolerance in
  `tests/test_baseline_invariants.test_baseline_float_tolerance_same_seed_two_models`.

Because `artifacts/` is git-ignored, the committed test suite carries the
stable expectations; the JSON artifact is regenerated locally when a
diagnostic is needed.

A 2-deal CI smoke variant runs in `.github/workflows/ci.yml`.

## Benchmarking

```bash
CUDA_VISIBLE_DEVICES="" python benchmarks/bench_legacy.py \
    --rounds 30 \
    --output artifacts/benchmark/bench.json
```

Produces `bench.json` + `bench.md`. **These numbers are host-specific.** They
measure deterministic CPU paths and are *not* strength or optimisation claims.
GPU timing is only collected when CUDA is available (it is not in the P00
image).

## Known RNG locations (checklist for future phases)

- `douzero/env/env.py:60` — deck shuffle (the only env RNG).
- `generate_eval_data.py:19` — deal shuffle (separate copy of the deck).
- `douzero/dmc/models.py:40-43,75-78` — exploration branch
  (`np.random.rand() < exp_epsilon`, `torch.randint`). Reached **only** via the
  training actor (`utils.py:139`) with `return_value=False`; never by
  `DeepAgent.act`.
- `douzero/evaluation/random_agent.py:9` — `random.choice` (Python `random`,
  not numpy).
- `douzero/evaluation/rlcard_agent.py:83,85` — `random.choice` fallback.

## What is explicitly NOT measured in P00

- GPU correctness (no GPU in the test image) — recorded as **not run**.
- Playing strength (no pretrained weights are downloaded; tests use
  init-only synthetic weights and assert **determinism**, not win rate).
- Cross-machine benchmark portability (CPU numbers depend on host and torch
  build; only the *hash* snapshots are portable).
