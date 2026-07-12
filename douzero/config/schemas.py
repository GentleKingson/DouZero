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
    # P04 widens to allow "factorized" (deployment-only); default is "legacy".
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
    # Version identifiers. P01 only supported "legacy"; P02 widened ruleset,
    # P03 widened feature_version, P04 widened model_version. Defaults stay
    # "legacy" so existing behavior is unchanged. Carried through the config so
    # --config + explicit CLI never silently drop them, and so the checkpoint
    # manifest records the effective versions.
    feature_version: str = "legacy"
    ruleset: str = "legacy"
    model_version: str = "legacy"

    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


# --------------------------------------------------------------------------- #
# Placeholder sub-configs (model/rule/feature versions are addressed in later
# phases; P01 only carries the version strings).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """Model architecture selection (P05 widens to ``v2``).

    ``version`` selects the model family:

    - ``"legacy"`` (default): the original role-specific LSTM+MLP value
      models. Preserved unchanged for backward compatibility.
    - ``"factorized"`` (P04): a deployment-only, checkpoint-compatible
      forward that is numerically equivalent to ``legacy`` under the same
      weights (encodes the shared state/history once per decision).
    - ``"v2"`` (P05): the shared state/action model with role embeddings,
      a Transformer (or LSTM) history encoder, and multi-head outputs
      (win probability + conditional scores). Enabled by
      ``model_version=v2`` + ``feature_version=v2``.

    The remaining fields configure the V2 architecture only; they are
    ignored by the legacy and factorized paths. Defaults match
    ``configs/enhanced.yaml`` and the V2 model constructor defaults.
    """

    version: str = "legacy"

    # --- V2 architecture knobs (P05). Defaults keep the model small enough
    # to run a forward/backward smoke test on CPU while still representing a
    # credible shared backbone. Tuned values belong in configs/, not here.
    hidden_size: int = 256
    history_encoder: str = "transformer"  # transformer | lstm
    history_layers: int = 4
    history_heads: int = 8
    role_embedding_dim: int = 32
    # Auxiliary heads are gated by config so ablations can disable them
    # (P09 attaches more; P05 keeps the structural skeleton).
    belief_enabled: bool = False
    human_prior_enabled: bool = False


@dataclass(frozen=True)
class RuleConfig:
    """Rule configuration (P02).

    ``ruleset`` is the CLI/config version string (``"legacy"`` or
    ``"standard"``). ``ruleset_id`` is the identity recorded in the checkpoint
    manifest; it mirrors ``ruleset`` for P02 (P16 may append a stable hash).
    """
    ruleset: str = "legacy"  # legacy | standard (P02 adds the rule engine)
    ruleset_id: str = "legacy"


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
