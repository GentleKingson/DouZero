"""Configuration loading and conversion utilities (P01).

The legacy training entry point (``douzero.dmc.train``) consumes an
``argparse.Namespace`` whose attributes are the 23 legacy flags plus the
optimizer flags. To preserve that contract exactly, this module converts
between ``TrainingConfig`` and ``argparse.Namespace``.

Precedence (highest wins): CLI flags > YAML config file > dataclass defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from douzero.config.schemas import OptimizerConfig, TrainingConfig


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> dict:
    """Load a YAML file into a dict without requiring PyYAML at import time.

    PyYAML (``pyyaml``) is a declared runtime dependency (see ``pyproject.toml``
    ``[project] dependencies``). It is imported lazily here so that plain
    ``--help`` and module imports never require it; only ``--config <yaml>`` does.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "PyYAML (pyyaml) is a declared dependency of douzero and is required "
            "to load YAML configs. It should be installed automatically; if it is "
            "missing, run `pip install pyyaml`. The import is lazy so that plain "
            "`--help` and module imports never require it."
        ) from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(data)}")
    return dict(data)


def load_config(yaml_path: str) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.

    Unknown keys raise; missing keys fall back to the dataclass defaults.
    Nested ``optimizer`` mapping is supported.
    """
    raw = _load_yaml(yaml_path)
    return _build_training_config(raw)


def load_legacy_config() -> TrainingConfig:
    """Load the bundled legacy config (shipped inside the wheel).

    Uses importlib.resources so it works from an installed wheel (no repo
    checkout needed). The bundled file is douzero/config/data/legacy.yaml and
    is identical to the repo's configs/legacy.yaml.
    """
    from importlib.resources import files

    legacy_path = files("douzero.config.data").joinpath("legacy.yaml")
    with legacy_path.open("r", encoding="utf-8") as fh:
        import yaml

        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise RuntimeError("bundled legacy.yaml is not a mapping")
    return _build_training_config(raw)


# --------------------------------------------------------------------------- #
# dict <-> dataclass
# --------------------------------------------------------------------------- #
def _build_training_config(raw: Mapping[str, Any]) -> TrainingConfig:
    """Construct a TrainingConfig from a raw mapping, validating keys."""
    valid_top = {f.name for f in fields(TrainingConfig) if f.name != "optimizer"}
    valid_opt_names = {f.name for f in fields(OptimizerConfig)}

    # 'optimizer' and 'rules' are valid top-level keys (handled separately).
    unknown_top = set(raw.keys()) - valid_top - {"optimizer", "rules"}
    if unknown_top:
        raise ValueError(f"Unknown config keys: {sorted(unknown_top)}")

    optimizer_raw = raw.get("optimizer", {})
    if not isinstance(optimizer_raw, Mapping):
        raise TypeError("'optimizer' must be a mapping")
    unknown_opt = set(optimizer_raw.keys()) - valid_opt_names
    if unknown_opt:
        raise ValueError(f"Unknown optimizer config keys: {sorted(unknown_opt)}")

    # P02: a 'rules' block is accepted but not stored on TrainingConfig (which
    # only carries the version string). We validate it here so a malformed
    # rules block fails loudly, and so the ruleset version string is
    # cross-checked against it.
    rules_raw = raw.get("rules")
    if rules_raw is not None:
        from douzero.env.rules import RuleSet
        RuleSet.from_dict(rules_raw)  # validates types/ranges; result discarded

    kwargs: dict[str, Any] = {}
    for name in valid_top:
        if name in raw:
            kwargs[name] = raw[name]
    if optimizer_raw:
        kwargs["optimizer"] = OptimizerConfig(**dict(optimizer_raw))
    cfg = TrainingConfig(**kwargs)
    _validate_types(cfg)
    _validate_legacy_only_versions(cfg)
    return cfg


# Expected runtime types per field, for validating YAML/dict input. Booleans
# must be real bools (YAML bool, not the string "true"); ints/floats must be
# numbers; strings must be str. This catches wrong-type YAML values that a
# frozen dataclass would otherwise silently accept.
_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "xpid": str, "save_interval": int, "objective": str,
    "actor_device_cpu": bool, "gpu_devices": str, "num_actor_devices": int,
    "num_actors": int, "training_device": str, "load_model": bool,
    "disable_checkpoint": bool, "savedir": str, "total_frames": int,
    "exp_epsilon": float, "batch_size": int, "unroll_length": int,
    "num_buffers": int, "num_threads": int, "max_grad_norm": float,
    "seed": int, "deterministic": bool, "config": str,
    "feature_version": str, "ruleset": str, "model_version": str,
    "learning_rate": float, "alpha": float, "momentum": float, "epsilon": float,
}


def _check_field(name: str, value: Any, source: str) -> None:
    expected = _FIELD_TYPES.get(name)
    if expected is None:
        return
    # bool is a subclass of int; for int fields we must reject bools, and for
    # float fields we accept int (numpy/JSON style) but reject bool.
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Config field {name!r} ({source}) must be int, got "
                f"{type(value).__name__}: {value!r}"
            )
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Config field {name!r} ({source}) must be float, got "
                f"{type(value).__name__}: {value!r}"
            )
    elif expected is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"Config field {name!r} ({source}) must be bool, got "
                f"{type(value).__name__}: {value!r}"
            )
    else:
        if not isinstance(value, expected):
            raise TypeError(
                f"Config field {name!r} ({source}) must be {expected.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )


def _validate_types(cfg: TrainingConfig) -> None:
    for name in _FIELD_TYPES:
        if name in {"learning_rate", "alpha", "momentum", "epsilon"}:
            _check_field(name, getattr(cfg.optimizer, name), "optimizer")
        elif hasattr(cfg, name):
            _check_field(name, getattr(cfg, name), "training")


# P01 only supports the "legacy" feature/rule/model versions. Later phases
# (P02 rules, P03 observations, P05 model) widen these sets. A YAML or dict
# config that sets a non-"legacy" value is rejected here so it fails loudly
# rather than silently producing a run the codebase does not support.
_LEGACY_ONLY_VERSIONS: dict[str, frozenset[str]] = {
    # P03 widens the feature_version allowed set to include "v2". The V2
    # observation schema is opt-in; the legacy encoder remains the default and
    # is byte-for-byte unchanged. Training still uses the legacy observation
    # path until P05/P06 wire the V2 model and multi-objective losses.
    "feature_version": frozenset({"legacy", "v2"}),
    # P02 widens the ruleset allowed set to include "standard".
    "ruleset": frozenset({"legacy", "standard"}),
    "model_version": frozenset({"legacy"}),
}


def _validate_legacy_only_versions(cfg: TrainingConfig) -> None:
    """Reject version identifiers this codebase does not support.

    The argparse flags enforce ``choices`` for CLI input; this validator covers
    the YAML/dict path so a config file cannot smuggle in an unsupported
    version either. The allowed sets are widened per phase: P02 added
    ``ruleset="standard"``, P03 added ``feature_version="v2"``.
    """
    for name, allowed in _LEGACY_ONLY_VERSIONS.items():
        val = getattr(cfg, name)
        if val not in allowed:
            raise ValueError(
                f"Config field {name!r} has unsupported value {val!r}. "
                f"Supported values are {sorted(allowed)}. Later phases widen "
                f"the allowed set."
            )


def serialize(cfg: TrainingConfig) -> dict:
    """Convert a TrainingConfig to a JSON/YAML-serializable dict."""
    return asdict(cfg)


# --------------------------------------------------------------------------- #
# argparse Namespace <-> TrainingConfig
# --------------------------------------------------------------------------- #
# The set of attribute names train(flags) reads off the Namespace. These are
# the EXACT argparse dests from douzero/dmc/arguments.py.
_TRAINING_NAMESPACE_FIELDS: tuple[str, ...] = (
    "xpid", "save_interval", "objective",
    "actor_device_cpu", "gpu_devices", "num_actor_devices", "num_actors",
    "training_device", "load_model", "disable_checkpoint", "savedir",
    "total_frames", "exp_epsilon", "batch_size", "unroll_length",
    "num_buffers", "num_threads", "max_grad_norm",
    "learning_rate", "alpha", "momentum", "epsilon",
    # P01-added argparse dests (optional; default to legacy values if absent).
    "seed", "deterministic", "config",
    # Version identifiers carried through config <-> Namespace (item 4).
    "feature_version", "ruleset", "model_version",
)


def from_argparse(ns: argparse.Namespace) -> TrainingConfig:
    """Build a TrainingConfig from a legacy argparse Namespace.

    Optimizer fields (learning_rate/alpha/momentum/epsilon) live at the
    Namespace top level (that is how arguments.py declares them), so they are
    pulled into the nested OptimizerConfig.
    """
    opt_keys = {"learning_rate", "alpha", "momentum", "epsilon"}
    opt_kwargs = {k: getattr(ns, k) for k in opt_keys if hasattr(ns, k)}
    optimizer = OptimizerConfig(**opt_kwargs) if opt_kwargs else OptimizerConfig()

    training_kwargs: dict[str, Any] = {}
    for name in _TRAINING_NAMESPACE_FIELDS:
        if name in opt_keys:
            continue
        if hasattr(ns, name):
            training_kwargs[name] = getattr(ns, name)
    training_kwargs["optimizer"] = optimizer
    return TrainingConfig(**training_kwargs)


def to_argparse_namespace(cfg: TrainingConfig) -> argparse.Namespace:
    """Convert a TrainingConfig back to a legacy argparse Namespace.

    The returned Namespace has the SAME attributes that ``train(flags)``
    reads (the optimizer fields are flattened to the top level, matching how
    arguments.py declares them). This keeps ``train.py`` unchanged: it can
    call ``train(flags)`` with this Namespace exactly as before.
    """
    d: dict[str, Any] = {}
    for name in _TRAINING_NAMESPACE_FIELDS:
        if name == "config":
            d[name] = getattr(cfg, name, "")
        elif name in {"learning_rate", "alpha", "momentum", "epsilon"}:
            d[name] = getattr(cfg.optimizer, name)
        elif hasattr(cfg, name):
            d[name] = getattr(cfg, name)
    return argparse.Namespace(**d)


# --------------------------------------------------------------------------- #
# Merge (CLI overrides YAML)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Merge (CLI overrides YAML)
# --------------------------------------------------------------------------- #
def merge(base: TrainingConfig, override_ns: argparse.Namespace) -> TrainingConfig:
    """Overlay explicitly-set CLI overrides from a Namespace onto a base config.

    Precedence: dataclass defaults < YAML (base) < explicit CLI flags.

    ``override_ns`` is expected to contain ONLY the flags the user explicitly
    typed (produced by re-parsing with default=SUPPRESS in
    ``arguments.parse_args``). Because absent flags are simply missing from the
    Namespace, every attribute present is a genuine override -- including
    ``store_true`` flags, which appear only when the user actually typed them.
    This avoids the classic "argparse default clobbers YAML" bug for booleans.
    """
    base_d = asdict(base)
    opt_overrides = {}
    for name in _TRAINING_NAMESPACE_FIELDS:
        if not hasattr(override_ns, name):
            continue
        val = getattr(override_ns, name)
        if name in {"learning_rate", "alpha", "momentum", "epsilon"}:
            opt_overrides[name] = val
        else:
            base_d[name] = val
    if opt_overrides:
        base_d["optimizer"] = {**base_d["optimizer"], **opt_overrides}
    return _build_training_config(base_d)
