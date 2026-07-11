"""Frozen dataclass schemas for DouZero configuration (P01).

EVERY default here MUST equal the corresponding argparse default in
``douzero/dmc/arguments.py`` (for training) and ``evaluate.py`` (for
evaluation). These schemas are pure configuration plumbing: they do not change
reward, model, observation, or actor semantics.

The field names match the argparse dest names (dashes -> underscores) so that
``from_argparse`` and ``to_argparse_namespace`` are direct mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Runtime (cross-cutting; P00 had no seed plumbing, P01 adds opt-in defaults)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeConfig:
    """Cross-cutting runtime knobs.

    ``seed`` / ``deterministic`` are NEW in P01 and default to values that
    preserve legacy behavior (no forced determinism). They are wired into the
    unified seeding utility in a later slice; for now they are carried so the
    config is complete and serializable.
    """

    seed: int = 0
    deterministic: bool = False
    device: str = "cuda"
    feature_version: str = "legacy"
    ruleset: str = "legacy"
    model_version: str = "legacy"


# --------------------------------------------------------------------------- #
# Optimizer (RMSProp) -- mirrors the "Optimizer settings" argparse group
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 0.0001
    alpha: float = 0.99
    momentum: float = 0
    epsilon: float = 1e-5


# --------------------------------------------------------------------------- #
# Training -- mirrors douzero/dmc/arguments.py exactly (all 23 args)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingConfig:
    # General
    xpid: str = "douzero"
    save_interval: int = 30
    objective: str = "adp"

    # Training settings
    actor_device_cpu: bool = False
    gpu_devices: str = "0"
    num_actor_devices: int = 1
    num_actors: int = 5
    training_device: str = "0"
    load_model: bool = False
    disable_checkpoint: bool = False
    savedir: str = "douzero_checkpoints"

    # Hyperparameters
    total_frames: int = 100000000000
    exp_epsilon: float = 0.01
    batch_size: int = 32
    unroll_length: int = 100
    num_buffers: int = 50
    num_threads: int = 4
    max_grad_norm: float = 40.0

    # New P01 knobs (carried, not yet enforced; defaults preserve legacy)
    seed: int = 0
    deterministic: bool = False
    config: str = ""  # path to a YAML config, "" means none

    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


# --------------------------------------------------------------------------- #
# Placeholder sub-configs (model/rule/feature versions are addressed in later
# phases; P01 only carries the version strings).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    version: str = "legacy"  # legacy | v2 (v2 arrives in P05)


@dataclass(frozen=True)
class RuleConfig:
    ruleset: str = "legacy"  # legacy | standard (P02 adds the rule engine)


@dataclass(frozen=True)
class CheckpointConfig:
    # P01 Slice 3 fills the manifest schema; this placeholder carries the
    # feature/rule identifiers used for compatibility checks.
    feature_version: str = "legacy"
    ruleset_id: str = "legacy"


# --------------------------------------------------------------------------- #
# Evaluation -- mirrors evaluate.py's argparse defaults
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluationConfig:
    landlord: str = "baselines/douzero_ADP/landlord.ckpt"
    landlord_up: str = "baselines/sl/landlord_up.ckpt"
    landlord_down: str = "baselines/sl/landlord_down.ckpt"
    eval_data: str = "eval_data.pkl"
    num_workers: int = 5
    gpu_device: str = ""
