"""Frozen snapshot of the legacy (pre-P01) defaults.

Used by tests to assert that the new config plumbing produces values identical
to the original argparse defaults, and by ``configs/legacy.yaml`` parity tests.
Any change to a default here (or in schemas.py) is a behavior change that must
be intentional and documented.
"""

from __future__ import annotations

from douzero.config.schemas import (
    EvaluationConfig,
    OptimizerConfig,
    TrainingConfig,
)

# The exact defaults of douzero/dmc/arguments.py, captured at P01 baseline.
LEGACY_TRAINING_DEFAULTS = TrainingConfig(
    xpid="douzero",
    save_interval=30,
    checkpoint_sidecar_retention=2,
    objective="adp",
    actor_device_cpu=False,
    gpu_devices="0",
    num_actor_devices=1,
    num_actors=5,
    training_device="0",
    load_model=False,
    disable_checkpoint=False,
    savedir="douzero_checkpoints",
    total_frames=100000000000,
    exp_epsilon=0.01,
    batch_size=32,
    unroll_length=100,
    num_buffers=50,
    num_threads=4,
    max_grad_norm=40.0,
    seed=0,
    deterministic=False,
    config="",
    feature_version="legacy",
    ruleset="legacy",
    model_version="legacy",
    optimizer=OptimizerConfig(
        learning_rate=0.0001,
        alpha=0.99,
        momentum=0,
        epsilon=1e-5,
    ),
)

# The exact defaults of evaluate.py, captured at P01 baseline.
LEGACY_EVALUATION_DEFAULTS = EvaluationConfig(
    landlord="baselines/douzero_ADP/landlord.ckpt",
    landlord_up="baselines/sl/landlord_up.ckpt",
    landlord_down="baselines/sl/landlord_down.ckpt",
    eval_data="eval_data.pkl",
    num_workers=5,
    gpu_device="",
)


class LegacyConfig:
    """Aggregate of the frozen legacy defaults (read-only reference).

    Attributes are the frozen dataclass instances above. This class is not
    meant to be instantiated for runtime config; use ``TrainingConfig`` /
    ``EvaluationConfig`` directly. It exists as a named comparison target.
    """

    training: TrainingConfig = LEGACY_TRAINING_DEFAULTS
    evaluation: EvaluationConfig = LEGACY_EVALUATION_DEFAULTS
