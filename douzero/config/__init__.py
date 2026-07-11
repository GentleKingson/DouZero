"""Typed configuration for DouZero (P01).

This package introduces versioned, dataclass-based configuration that mirrors
the legacy argparse defaults EXACTLY. The legacy CLI (``douzero.dmc.arguments``)
remains the default entry point; this package adds:

  - ``schemas``: frozen dataclasses (RuntimeConfig, TrainingConfig, ...)
  - ``loader``: load YAML, convert argparse Namespace <-> config, serialize
  - ``legacy``: a frozen snapshot of the current legacy defaults for comparison

Design rules (P01 scope):
  - Do NOT change any training default. Every dataclass default must equal the
    corresponding argparse default verbatim.
  - Do NOT change reward/model/observation semantics. This is configuration
    plumbing only.
  - The legacy argparse surface stays the default; ``--config`` is opt-in.
"""

from douzero.config.legacy import LegacyConfig
from douzero.config.loader import (
    from_argparse,
    load_config,
    load_legacy_config,
    merge,
    serialize,
    to_argparse_namespace,
)
from douzero.config.schemas import (
    CheckpointConfig,
    EvaluationConfig,
    ModelConfig,
    OptimizerConfig,
    RuleConfig,
    RuntimeConfig,
    TrainingConfig,
)

__all__ = [
    "CheckpointConfig",
    "EvaluationConfig",
    "LegacyConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RuleConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "from_argparse",
    "load_config",
    "load_legacy_config",
    "merge",
    "serialize",
    "to_argparse_namespace",
]
