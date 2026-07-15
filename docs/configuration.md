# Configuration

> P01 introduced the typed configuration system. This page documents the
> current legacy and V2 entry points while preserving the original defaults.

## Overview

P01 adds `douzero/config/`, a set of frozen dataclasses whose defaults mirror
the legacy `douzero/dmc/arguments.py` argparse defaults **exactly**. The legacy
CLI remains the default entry point; the typed config is an opt-in layer that
makes the configuration inspectable, serializable, and YAML-loadable.

- `douzero/config/schemas.py` — frozen dataclasses for runtime, training,
  optimizer, model, loss, bidding, rules, checkpoints, evaluation, and the
  other opt-in feature blocks.
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
unified seeding utility (see `douzero/runtime/seeding.py`). The version
identifiers are enforced by the config loader:

| Flag | Default | Allowed values | Purpose |
|---|---|---|---|
| `--seed` | `0` | int | Base RNG seed (`0` = legacy unseeded behavior; non-zero activates seeding) |
| `--deterministic` | off | bool | Force deterministic torch algorithms |
| `--feature_version` | `legacy` | `legacy`, `v2` | Observation feature version (P03 widens to include `v2`) |
| `--ruleset` | `legacy` | `legacy`, `standard` | Rule set identifier (P02 widens to include `standard`) |
| `--model_version` | `legacy` | `legacy`, `factorized`, `v2` | Model version (P04 adds `factorized`; P05 adds `v2`) |

P02 widens `--ruleset` to accept `standard` (in addition to `legacy`).
P03 widens `--feature_version` to accept `v2` (the versioned observation schema
in `douzero/observation/`; the legacy encoder remains the default and is
byte-for-byte unchanged — see `docs/observation_v2.md`). P04 widens
`--model_version` to accept `factorized` (a deployment-only,
checkpoint-compatible forward). P05 widens it further to `v2` (the shared
state-action model with multi-head outputs — see `docs/model_v2.md`). The
default stays `legacy`. All version identifiers are recorded in the checkpoint
manifest (see `docs/checkpoint_compatibility.md`) so incompatible loads are
rejected.

### Standard ruleset

When `--ruleset standard` is set, a `rules:` block in the YAML config
specifies the rule parameters (bidding mode, multipliers, base score, etc.).
See `configs/standard.yaml` and `docs/rules_and_scoring.md` for details.
The two training entry points intentionally differ:

- legacy `train.py` still rejects `standard`, `feature_version=v2`, and
  `model_version=v2` rather than approximating them in the legacy actor loop;
- `train_v2.py --config configs/standard_v2.yaml` runs the complete standard
  auction/redeal/reveal/card-play loop with `model.bidding_enabled=true`.

The standard V2 path is single-process and eager today. It fails closed when
DDP or `compile_model` is requested because the mixed bidding/card-play graph
and separate `forward_bidding` contract have not been validated there.

### Learned bidding block

P17 adds a default-off `bidding:` block, consumed only by `train_v2.py` in
standard mode:

```yaml
model:
  bidding_enabled: true
  bidding_hidden_size: 128
  bidding_uncertainty_enabled: false
bidding:
  enabled: true
  policy: learned
  warm_start_policy: rule
  learned_probability: 0.10
```

`policy` accepts `random`, `rule`, `max`, `pass`, or `learned`. In learned
mode, `learned_probability` controls a gradual handoff from the non-learned
warm-start policy; each collected auction transition records its true source.
The bidding architecture and its versioned public schema enter checkpoint
identity only when `bidding_enabled=true`, preserving the hashes of prior
bidding-disabled V2 checkpoints.

### Belief optimization controls

P17 also adds `belief_training_mode` with `frozen` (default), `joint`, and
`alternating` values plus supervised-loss weight, interval, batch-size, and
bounded synthetic-smoke episode controls. Joint and alternating modes require
`model.belief_enabled=true` and a strictly validated `--belief_checkpoint`.
Alternating mode additionally requires a positive supervised weight and
labelled samples. Joint/alternating modes are single-process and eager: DDP
does not synchronize belief gradients, and `compile_model` rejects these modes.

### Distillation block

P10 adds an optional `distillation:` block for the dedicated privileged
teacher and public student commands. `enabled` defaults to `false`; the legacy
trainer, `train_v2.py`, and deployment agents do not consume the teacher path.
The block carries offline dataset/cache paths, temperature/top-k settings, and
separate weights for KL, ranking, teacher regression, and retained terminal
supervision. See `configs/enhanced.yaml` and `docs/distillation.md`.

### Curriculum block

P12 adds an optional `curriculum:` block consumed only by `train_v2.py`.
`enabled` defaults to `false`. Guided modes load a separately versioned coach
checkpoint, enforce a configurable true-random sample floor in every phase,
and can write policy-versioned outcome labels plus reconstructable sampling
audit JSONL. Evaluation ignores the block and never imports the coach sampler.
Coach loading also requires an exact `policy_version` match and enforces
`max_coach_age_steps` both at startup and before every episode; training fails
closed before coach inference if the checkpoint becomes stale. The
true-random floor constrains configured sampling probability rather than every
finite window's empirical ratio. Per-game labels freeze the effective
optimizer step at episode start. `max_label_age_steps` independently controls
refit data age.
See `configs/enhanced.yaml` and `docs/coach_curriculum.md`.

### Search block

P13 adds an optional deployment-only `search:` block. `enabled` defaults to
`false`; legacy and ordinary V2 selection do not run a belief model or search.
Pass `cfg.search` to `DeepAgentV2(search_config=...)` together with a P07
belief model to enable top-k belief rollouts and the small-endgame solver.
Node, rollout, and millisecond limits are mandatory fallback boundaries; a
zero limit returns the base-policy action. See `configs/enhanced.yaml` and
`docs/search.md`.

### Training system controls

P14 adds `sync_interval_updates`, `policy_snapshot_slots`, `amp_enabled`,
`amp_dtype`, `amp_fallback_on_nonfinite`, `pin_memory`, `ddp_enabled`,
`ddp_backend`, and `compile_model`. Optional accelerators default off. The
versioned actor snapshot mechanism is always active because it fixes a weight
publication race without changing the learning objective. See
`docs/training_system.md` for the torchrun and profiler commands.

DDP support applies to the legacy-ruleset V2 card-play trainer with compatible
optional features. Standard learned bidding, joint/alternating belief,
curriculum/coach output, and RL+BC fail closed under DDP. `torch.compile` is
likewise opt-in for the base V2 forward only; it rejects learned bidding and
joint/alternating belief rather than silently running an unvalidated graph.

## Boolean flags and `--no-<flag>` overrides

Boolean flags such as `--actor_device_cpu`, `--load_model`,
`--disable_checkpoint`, `--deterministic`, and the P14 system toggles use
`argparse.BooleanOptionalAction`.
This means:

- `--<flag>` sets it to `True` (the legacy `store_true` behavior, unchanged).
- `--no-<flag>` sets it to `False` (new in P01).

The `--no-<flag>` form lets you override a YAML `true` from the CLI. For
example, if `configs/run.yaml` has `deterministic: true`, you can force it off:

```
python train.py --config configs/run.yaml --no-deterministic
```

The compatibility-sensitive flags retain their pre-P01 defaults. P14's AMP,
DDP, pinned-memory, and compile toggles default to `False`; anomaly fallback
defaults to `True`.

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
