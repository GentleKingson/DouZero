# P04: Factorized legacy forward (state/action decoupling)

> Scope: a deployment-only, checkpoint-compatible forward that encodes the
> shared history and shared state **once per decision** instead of once per
> legal action. Numerically equivalent to the legacy forward under the same
> weights. Training integration is **not** included (it arrives in P05/P06).

## Why

The legacy models (`douzero/dmc/models.py`: `LandlordLstmModel`,
`FarmerLstmModel`) score `N` legal actions by *tiling* the shared state
`x_no_action` and the shared history `z` across all `N` rows, then running the
LSTM once per row:

```text
z_batch : (N, 5, 162)      # N identical copies of the same history (5, 162)
x_batch : (N, D_state + 54)  # D_state columns identical; last 54 per-action
lstm_out, _ = lstm(z_batch)        # N identical (N, 5, 128)
h = cat([lstm_out[:, -1, :], x_batch], dim=-1)   # (N, 128 + D_state + 54)
values = dense1..6(h)              # (N, 1)
```

Because every row of `z_batch` is the **same** history and every row's state
block is the **same** `x_no_action`, the LSTM output is identical across rows
(the LSTM has no per-row state in `eval()` mode, and the legacy models use no
dropout/BatchNorm). Only the trailing 54-dim `my_action` block varies. The
legacy encoder (`douzero/env/env.py`) builds these tiled batches with
`np.repeat`; `DeepAgent.act` then forwards the whole `(N, ...)` batch.

This makes the LSTM process `N` identical copies of the shared history per
decision — pure waste (the LSTM is called once, but it does `N` rows of
identical work).

## What P04 changes

`douzero/dmc/models_factorized.py` introduces drop-in models that compute the
same result while running the shared history LSTM and the shared state
projection **exactly once** per decision:

```text
z_single : (1, 5, 162)         # the shared history, encoded once
x_state  : (1, D_state)        # the shared state, encoded once
x_action : (N, 54)             # per-action card vectors
lstm_out, _ = lstm(z_single)   # (1, 128) — ONE LSTM call
h = cat([lstm_out[:, -1, :].expand(N, 128),
         x_state.expand(N, D_state),
         x_action], dim=-1)    # (N, 128 + D_state + 54)
values = dense1..6(h)          # (N, 1)
```

`expand` creates a **view** (no copy), so the per-action rows share the same
memory for the state/history block. The MLP then maps each row to a value.

### Numerical equivalence

The factorized forward is a pure rearrangement of the **same arithmetic** on
the **same weights and inputs**: the LSTM runs once on the shared history (its
output is identical to every row's output in the legacy path, because every
legacy row was the same history), and the state block is broadcast rather than
tiled. On CPU float32 the outputs match within `atol=1e-6, rtol=1e-5` (in
practice bit-identical). The parity tests
(`tests/test_factorized_parity.py`) pin this for all three roles, action
counts 1/2/many, and many random deals.

> **CPU only.** Numerical and argmax parity are **tested on CPU**. GPU
> numerical and argmax parity are **not yet measured**: mathematical
> equivalence does not imply bitwise or universal argmax identity across
> CPU/GPU (different kernels, reduction order, and cuDNN RNN
> non-determinism can change results). The factorized backend is safe to use
> on GPU, but "identical selection to legacy on GPU" is an empirical claim
> that must be measured before being asserted.

## Checkpoint compatibility

The factorized models intentionally declare the **same submodule names and
shapes** as the legacy models (`lstm`, `dense1` … `dense6`), so a legacy
per-position `.ckpt` (a bare `state_dict`) loads with **no conversion** via
the existing `load_legacy_position_ckpt` path. `state_dict()` keys and shapes
are byte-for-byte identical; only `forward` differs.

This means:

- existing `baselines/` and `*_weights_*.ckpt` files work with both backends;
- a checkpoint trained with the legacy model can be served with the factorized
  backend, and vice versa;
- no migration tool or weight conversion is required.

## How to use

### DeepAgent (deployment)

```python
from douzero.evaluation.deep_agent import DeepAgent

# Default: legacy per-action forward (unchanged behavior).
agent = DeepAgent("landlord", "path/to/landlord.ckpt")

# Opt-in: factorized forward (same weights, same selection, one LSTM call).
agent = DeepAgent("landlord", "path/to/landlord.ckpt",
                  backend="legacy_factorized")
```

`backend` defaults to `"legacy"`, so all existing callers are unchanged. The
factorized backend accepts the **same** `.ckpt` and produces the **same**
selected action (pinned by `test_deepagent_factorized_matches_legacy_split_path`
and `test_deepagent_factorized_uses_split_observation`).

The factorized backend consumes a **split observation** that never tiles the
shared state or history. `DeepAgent._act_factorized` calls
`get_obs_factorized(infoset)` (which returns `z_single (1,5,162)`,
`x_state_single (1, D_state)`, `x_action (N, 54)` directly — no `np.repeat` on
the shared blocks) and `model.forward_factorized(...)`. This removes the
NumPy tiling allocation, the tiled CPU tensor allocation, and the tiled
CPU→GPU transfer. The legacy backend (`_act_legacy`) is unchanged except
that both backends now run under `torch.inference_mode()` (the original
`act()` built an autograd graph during inference; `inference_mode` reduces
memory without changing outputs under `eval()`).

### Input validation

The factorized model's `forward` (legacy-batched interface) validates shapes
and the shared-row invariant before slicing: non-identical `z_batch` rows or
non-identical state-block rows raise `ValueError` instead of silently using
row 0. `forward_factorized` (split interface) validates the singleton
shapes. Opt out of the shared-row check via `DUZERO_FACTORIZED_STRICT=0` for
hot paths that validate upstream. Tests in
`tests/test_factorized_parity.py` pin the rejection of malformed input.

### Configuration

`--model_version` now accepts `legacy` (default) and `factorized`:

```bash
# Config / CLI carry the version for manifest stamping. Training still
# rejects non-legacy model_version (see below).
python train.py --model_version factorized   # rejected by the training gate
```

`model_version="factorized"` is accepted by the config loader and CLI (so it
can be recorded in a checkpoint manifest), but the **training gate** in
`douzero/dmc/dmc.py:train()` rejects it up front with a precise error, because
the actor/learner loop is not yet wired to the factorized model.

## Training integration boundary

P04 is **deployment-only**. The actor/learner loop
(`douzero/dmc/utils.py`, `douzero/dmc/dmc.py`) continues to use the legacy
`Model` and the legacy per-action forward. The training gate rejects
`model_version != "legacy"` (and `feature_version != "legacy"`,
`ruleset != "legacy"`) before any CUDA/buffer/actor initialization, so a
factorized training run cannot silently produce a checkpoint stamped with the
wrong `model_version`.

Rationale: the training buffer already stores `obs_x_no_action`,
`obs_action`, and `obs_z` **separately** (the decoupling exists at the buffer
layer), but the learner's `learn()` recombines them into `x_batch`/`z_batch`
and calls the legacy `forward`. Wiring the factorized forward into the learner
is a training-semantics change that belongs with Model V2 (P05) and
multi-objective training (P06), not with a parity-only performance step.

## LSTM work reduction (efficiency proof)

`tests/test_factorized_parity.py` instruments `model.lstm.forward` to record
the **batch size** (number of rows) fed to the LSTM per decision. Both the
legacy and factorized forwards call the LSTM once per decision; the
distinction is that the legacy path feeds it `N` identical rows (the tiled
history) while the factorized path feeds it exactly `1` row (the shared
history, encoded once). The waste P04 removes is the N-fold redundant LSTM
computation over identical rows.

| legal actions | legacy LSTM rows | factorized LSTM rows |
|---:|---:|---:|
| 1 | [1] | [1] |
| 10 | [10] | [1] |
| N | [N] | [1] |

The benchmark (`benchmarks/bench_factorized.py`) reports:
- **model-forward-only** latency (legacy vs factorized, split-obs path) at
  action-count buckets 1/10/50/full — isolates the model cost;
- **end-to-end `DeepAgent.act`** latency (encode + tensor + forward + argmax)
  for all three roles — the real deployment number;
- **CPU peak RSS** for the full act path;
- **LSTM rows per decision** (the work-reduction proof above).

The model-forward-only numbers are **not** end-to-end DeepAgent numbers;
both are reported separately and labelled clearly. Run it:

```bash
# CPU (default). Does NOT force-hide CUDA at import; --device cuda works
# when a GPU is present.
python benchmarks/bench_factorized.py --rounds 30
python benchmarks/bench_factorized.py --device cuda --rounds 30
```

## Validation

```bash
# Parity + state_dict + LSTM-count + DeepAgent parity tests.
python -m pytest -q tests/test_factorized_parity.py

# Full suite (includes the config/CLI/training-guard tests widened for P04).
python -m pytest -q

# Benchmark (writes JSON + Markdown to artifacts/benchmark/).
python benchmarks/bench_factorized.py
```

Docker (CPU-only, hermetic):

```bash
docker build -f .docker/Dockerfile.test -t douzero-p04-test .
docker run --rm douzero-p04-test
docker run --rm douzero-p04-test python -m pytest -k factorized -q
docker run --rm douzero-p04-test python benchmarks/bench_factorized.py --rounds 20
```

## What P04 does NOT introduce

- No Transformer, residual network, or new loss (those are P05+).
- No new observation schema (P03's `ObservationV2` is the data-side
  groundwork; P04 is the model-side counterpart).
- No change to the default legacy behavior (`backend="legacy"` and
  `model_version="legacy"` remain the defaults).
- No change to the training path.

## Files

| File | Role |
|---|---|
| `douzero/dmc/models_factorized.py` | Factorized role models + wrapper + `split_legacy_batch` + input validation |
| `douzero/env/env.py` | `get_obs_factorized` — split observation encoder (no tiling of shared state/history) |
| `douzero/evaluation/deep_agent.py` | `backend` selection; `_act_factorized` consumes split obs + `inference_mode` |
| `douzero/dmc/arguments.py` | `--model_version` choices widened to `{legacy, factorized}` |
| `douzero/config/loader.py` | Allowed `model_version` set widened |
| `douzero/config/schemas.py` | Version-field comments updated |
| `douzero/dmc/dmc.py` | Training gate rejects `model_version != legacy` |
| `tests/test_factorized_parity.py` | Numerical parity, state_dict, LSTM-count, split-obs parity, invariant enforcement, DeepAgent parity |
| `tests/test_config.py`, `tests/test_ruleset.py`, `tests/test_training_ruleset_guard.py` | Config/CLI/guard tests widened |
| `benchmarks/bench_factorized.py` | model-forward-only + end-to-end DeepAgent.act latency + CPU peak RSS + LSTM rows |
