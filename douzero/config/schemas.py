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
# Loss (P06 multi-objective training)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossConfig:
    """Multi-objective loss weights and Huber deltas (P06).

    All weights default to 0.0 so the legacy training path (which uses the
    single-head MSE in ``douzero.dmc.compute_loss``) is unchanged when this
    config is absent. The V2 trainer (:mod:`douzero.training.v2_trainer`)
    constructs its own :class:`~douzero.training.losses.LossConfig` from
    these fields and runs the multi-objective combination.

    P06 r1: ``score_target_transform`` selects whether the conditional
    score heads are supervised against the raw team score or its
    ``sign(s)·log1p(|s|)`` transform. The two are mutually exclusive (a
    single head cannot fit both scales at once). ``score_clamp`` must match
    the model's head clamp so the raw target stays inside what the heads
    can represent.
    """

    lambda_win: float = 0.0
    lambda_score: float = 0.0
    lambda_uncertainty: float = 0.0
    lambda_bc: float = 0.0  # P08: listwise BC auxiliary weight (default off)
    lambda_min_turns: float = 0.0
    lambda_regain_initiative: float = 0.0
    lambda_teammate_finish: float = 0.0
    lambda_spring: float = 0.0
    lambda_structure: float = 0.0
    score_delta: float = 1.0
    score_target_transform: str = "raw"  # "raw" | "signed_log"
    score_clamp: float = 32.0


# --------------------------------------------------------------------------- #
# Decision policy (P06 action selection)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionPolicyConfig:
    """Configuration for the V2 deployment decision policy (P06).

    Carried so a checkpoint manifest can audit which decision mode a model
    was trained / evaluated under. Defaults preserve the P05 behaviour
    (``pure_win`` with zero tolerance, deterministic).
    """

    mode: str = "pure_win"
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    risk_penalty: float = 0.0
    prior_alpha: float = 0.0


# --------------------------------------------------------------------------- #
# Behaviour cloning (P08 human-data prior)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BCConfig:
    """Human-data behaviour-cloning configuration (P08).

    SINGLE SOURCE OF TRUTH (Blocker 3): the BC auxiliary loss is enabled **iff**
    ``loss.lambda_bc > 0``. This block carries only the BC-specific settings
    (data path, temperature, label smoothing, weight schedule). There is no
    separate ``enabled`` flag or duplicate ``lambda_bc`` here — both were
    removed because they could silently disagree with ``loss.lambda_bc`` and
    leave BC off when the user thought it was on.

    ``schedule`` selects how ``loss.lambda_bc`` evolves over RL training (the
    BC pretraining path itself ignores it and uses the pretrain_bc.py CLI
    weight):

    - ``"constant"`` (default): the weight is fixed.
    - ``"linear_decay"``: the weight linearly decays to ``schedule_floor``
      over ``schedule_steps`` (but is NOT forced below the floor); the default
      floor keeps a residual prior so the model never fully forgets the human
      signal.
    """

    data_path: str = ""  # validated canonical JSONL (human games)
    temperature: float = 1.0
    label_smoothing: float = 0.0
    skill_weight_clip: float = 10.0
    schedule: str = "constant"  # "constant" | "linear_decay"
    schedule_steps: int = 0
    schedule_floor: float = 0.0

    def __post_init__(self) -> None:
        import math

        # Blocker: label_smoothing must be finite and in [0, 1) — values >= 1
        # silently corrupt the listwise loss (negative target probs), and
        # NaN/Inf propagate. Validate at the config boundary so a bad YAML
        # fails at load, not silently mid-training.
        if isinstance(self.label_smoothing, bool) or not isinstance(
            self.label_smoothing, (int, float)
        ):
            raise ValueError(
                f"BCConfig.label_smoothing must be a number, got "
                f"{type(self.label_smoothing).__name__}"
            )
        if not math.isfinite(self.label_smoothing):
            raise ValueError(
                f"BCConfig.label_smoothing must be finite, got {self.label_smoothing}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"BCConfig.label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(
                f"BCConfig.temperature must be positive finite, got {self.temperature}"
            )
        if not math.isfinite(self.skill_weight_clip) or self.skill_weight_clip <= 0.0:
            raise ValueError(
                f"BCConfig.skill_weight_clip must be positive finite, "
                f"got {self.skill_weight_clip}"
            )
        if not isinstance(self.schedule_steps, int) or isinstance(
            self.schedule_steps, bool
        ) or self.schedule_steps < 0:
            raise ValueError(
                f"BCConfig.schedule_steps must be a non-negative int, "
                f"got {self.schedule_steps}"
            )
        if (
            not math.isfinite(self.schedule_floor)
            or self.schedule_floor < 0.0
        ):
            raise ValueError(
                f"BCConfig.schedule_floor must be non-negative finite, "
                f"got {self.schedule_floor}"
            )


@dataclass(frozen=True)
class DistillationConfig:
    """P10 privileged-teacher/public-student distillation settings.

    ``enabled=False`` is the compatibility default. The legacy trainer and
    deployment agent never consume this block; the dedicated P10 training
    entry points do.
    """

    enabled: bool = False
    teacher_checkpoint: str = ""
    dataset_path: str = ""
    cache_path: str = ""
    batch_size: int = 32
    distillation_temperature: float = 2.0
    top_k: int = 4
    lambda_kl: float = 1.0
    lambda_rank: float = 0.25
    lambda_teacher_win: float = 0.5
    lambda_teacher_score: float = 0.25
    lambda_supervised_win: float = 1.0
    lambda_supervised_score: float = 0.5

    def __post_init__(self) -> None:
        import math

        if not isinstance(self.enabled, bool):
            raise TypeError(
                f"DistillationConfig.enabled must be bool, got "
                f"{type(self.enabled).__name__}"
            )
        for name in ("teacher_checkpoint", "dataset_path", "cache_path"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(
                    f"DistillationConfig.{name} must be str, got "
                    f"{type(getattr(self, name)).__name__}"
                )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError(
                f"DistillationConfig.batch_size must be a positive int, "
                f"got {self.batch_size!r}"
            )
        if (
            isinstance(self.distillation_temperature, bool)
            or not isinstance(self.distillation_temperature, (int, float))
            or not math.isfinite(self.distillation_temperature)
            or self.distillation_temperature <= 0.0
        ):
            raise ValueError(
                f"DistillationConfig.distillation_temperature must be positive "
                f"finite, got {self.distillation_temperature}"
            )
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise ValueError(
                f"DistillationConfig.top_k must be a positive int, got {self.top_k!r}"
            )
        for name in (
            "lambda_kl", "lambda_rank", "lambda_teacher_win",
            "lambda_teacher_score", "lambda_supervised_win",
            "lambda_supervised_score",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"DistillationConfig.{name} must be non-negative finite, "
                    f"got {value}"
                )


# --------------------------------------------------------------------------- #
# Model architecture (P05 widens to ``v2``; referenced by TrainingConfig)
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
    strategy_features_enabled: bool = False
    strategy_hand_enabled: bool = True
    strategy_structure_enabled: bool = True
    strategy_control_enabled: bool = True
    strategy_cooperation_enabled: bool = True
    strategy_risk_enabled: bool = True
    strategy_aux_enabled: bool = False
    strategy_node_budget: int = 500
    strategy_time_budget_ms: int = 0


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
    # P06 multi-objective training + decision policy. Defaults preserve the
    # legacy single-target path; the V2 trainer reads these to construct its
    # LossConfig / DecisionConfig.
    loss: LossConfig = field(default_factory=LossConfig)
    decision_policy: DecisionPolicyConfig = field(default_factory=DecisionPolicyConfig)
    # P06 r5: V2 model architecture block. The legacy train.py path ignores
    # this; train_v2.py reads it via ModelV2Config.from_model_config().
    model: ModelConfig = field(default_factory=ModelConfig)
    # P08: behaviour-cloning prior configuration. Defaults preserve the
    # pre-P08 path (BC disabled). Consumed by pretrain_bc.py and the optional
    # BC auxiliary loss hook in the V2 trainer.
    bc: BCConfig = field(default_factory=BCConfig)
    # P10: opt-in and consumed only by the dedicated distillation tools.
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


# --------------------------------------------------------------------------- #
# Placeholder sub-configs (model/rule/feature versions are addressed in later
# phases; P01 only carries the version strings).
# --------------------------------------------------------------------------- #
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
