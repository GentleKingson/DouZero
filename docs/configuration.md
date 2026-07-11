# Configuration

> P01 introduces a typed configuration system. This page documents how to use
> it and how it stays compatible with the legacy CLI.

## Overview

P01 adds `douzero/config/`, a set of frozen dataclasses whose defaults mirror
the legacy `douzero/dmc/arguments.py` argparse defaults **exactly**. The legacy
CLI remains the default entry point; the typed config is an opt-in layer that
makes the configuration inspectable, serializable, and YAML-loadable.

- `douzero/config/schemas.py` — frozen dataclasses: `RuntimeConfig`,
  `TrainingConfig`, `OptimizerConfig`, `ModelConfig`, `RuleConfig`,
  `CheckpointConfig`, `EvaluationConfig`.
- `douzero/config/legacy.py` — `LegacyConfig`: a frozen snapshot of the current
  legacy defaults, used as a comparison target.
- `douzero/config/loader.py` — `load_config` (YAML), `from_argparse` /
  `to_argparse_namespace` (Namespace ↔ `TrainingConfig`), `merge` (CLI overrides
  YAML), `serialize`.
- `configs/legacy.yaml` — the YAML form of the exact legacy defaults.

## Using the legacy CLI (unchanged default)

`train.py` works exactly as before:

```bash
python train.py --xpid myrun --actor_device_cpu --training_device cpu
```

All 23 original flags keep their names and defaults. `train.py` now calls
`parse_args()` (from `douzero.dmc.arguments`) instead of `parser.parse_args()`
directly, but with no `--config` the behavior is identical.

## Using a YAML config

```bash
python train.py --config configs/legacy.yaml
python train.py --config my_run.yaml --batch_size 64   # CLI overrides YAML
```

Precedence (highest wins): **CLI flags > YAML file > dataclass defaults**.
The YAML file may specify any subset of fields; missing fields fall back to
the defaults. Unknown keys raise (no silent acceptance).

`configs/legacy.yaml` is field-for-field equal to the argparse defaults; you
can copy it and edit the values you want to change.

## The version fields

P01 adds optional flags. `--seed` and `--deterministic` are wired into the
unified seeding utility (see `douzero/runtime/seeding.py`); the three version
identifiers are enforced as `legacy`-only in P01 (later phases widen the set):

| Flag | Default | Purpose |
|---|---|---|
| `--seed` | `0` | Base RNG seed (`0` = legacy unseeded behavior; non-zero activates seeding) |
| `--deterministic` | off | Force deterministic torch algorithms |
| `--feature_version` | `legacy` | Observation feature version (P03 introduces `v2`) |
| `--ruleset` | `legacy` | Rule set identifier (P02 introduces `standard`) |
| `--model_version` | `legacy` | Model version (P05 introduces `v2`) |

In P01 only `legacy` is supported for the version fields; the CLI uses
`choices=['legacy']` and the YAML/dict loader rejects any other value. They are
recorded in the checkpoint manifest (see `docs/checkpoint_compatibility.md`) so
future phases can reject incompatible loads.

## Boolean flags and `--no-<flag>` overrides

The four boolean flags (`--actor_device_cpu`, `--load_model`,
`--disable_checkpoint`, `--deterministic`) use `argparse.BooleanOptionalAction`.
This means:

- `--<flag>` sets it to `True` (the legacy `store_true` behavior, unchanged).
- `--no-<flag>` sets it to `False` (new in P01).

The `--no-<flag>` form lets you override a YAML `true` from the CLI. For
example, if `configs/run.yaml` has `deterministic: true`, you can force it off:

```
python train.py --config configs/run.yaml --no-deterministic
```

The default is `False` for all four flags, matching the pre-P01 `store_true`
defaults exactly.

## Programmatic use

```python
from douzero.config import load_config, serialize, from_argparse
from douzero.dmc.arguments import parser

cfg = load_config("configs/legacy.yaml")          # -> TrainingConfig
ns = parser.parse_args(["--batch_size", "16"])    # legacy argparse
cfg2 = from_argparse(ns)                           # -> TrainingConfig
print(serialize(cfg))                              # JSON/YAML-serializable dict
```

## Guarantees (P01 scope)

- Every dataclass default equals the corresponding argparse default verbatim.
  `tests/test_config.py` pins this with `legacy.yaml == LegacyConfig`.
- No reward, model, observation, or actor semantics changed. The config layer
  is plumbing only.
- The legacy CLI path leaves the P00 baseline invariants (observation widths
  373/484, the `TYPE_15_WRONG` exception set, terminal accounting) unchanged.
