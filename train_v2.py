"""V2 multi-objective training entry point (P06).

This is the single-process trainer CLI for the P06 acceptance criterion:
"极短训练能完成一次优化且参数变化" (a very short training run completes one
optimizer step and changes the parameters). It loads a YAML config
(``configs/enhanced.yaml`` by convention), builds a
:class:`~douzero.models_v2.model.ModelV2`, and runs
:class:`~douzero.training.v2_trainer.V2Trainer`.

The legacy :file:`train.py` path is unchanged. This entry point exists so
the multi-objective loss + decision-policy + team-perspective labels can be
exercised without touching the legacy multiprocessing actor/learner.

CPU-only (P06). GPU + multiprocessing is P14.

Precedence (P06 r1 fix)
-----------------------
``explicit CLI flag > YAML value > V2 built-in default``.

The argparse parser uses ``default=argparse.SUPPRESS`` for every trainer
knob so that an ABSENT CLI flag does NOT shadow the YAML value (the classic
"argparse default clobbers YAML" bug). Only flags the user actually typed
appear in the parsed namespace. This makes a YAML ``lambda_win: 0`` (or
``batch_size: 64``, etc.) actually take effect even when the CLI does not
repeat it.

Usage
-----
::

    python train_v2.py --config configs/enhanced.yaml --episodes 8 \\
        --optimizer_steps 2 --seed 0

CPU smoke (no GPU required):

::

    python train_v2.py --episodes 4 --optimizer_steps 1 --seed 0
"""

from __future__ import annotations

import argparse
import os

from douzero.config import load_config
from douzero.runtime import maybe_set_global_deterministic, set_global_seed


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with SUPPRESS defaults for trainer knobs.

    Every trainer knob uses ``default=argparse.SUPPRESS`` so an absent flag
    does NOT shadow the YAML value. ``--config`` and the boolean
    ``--deterministic`` are the only exceptions (they have real defaults).
    """
    parser = argparse.ArgumentParser(
        description="DouZero V2 multi-objective trainer (P06, CPU-only)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a YAML training config (e.g. configs/enhanced.yaml). "
        "When empty, the V2 trainer uses its built-in defaults.",
    )
    # Trainer knobs: SUPPRESS so absent flags do not clobber YAML.
    parser.add_argument("--episodes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--optimizer_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--buffer_capacity", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--exp_epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rmsprop_alpha", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rmsprop_momentum", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rmsprop_epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_grad_norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_steps_per_episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable full deterministic mode (torch.use_deterministic_algorithms).",
    )
    return parser


def _load_yaml_config(path: str):
    """Load a TrainingConfig from YAML, or return None when path is empty."""
    if not path:
        return None
    return load_config(path)


def _resolve_ruleset(cfg):
    """Resolve the ruleset for the trainer.

    P06's V2Trainer only accepts ``ruleset=None`` (the legacy card-play-only
    env), because ``Env`` treats ANY non-None ``RuleSet`` as standard mode
    (entering the bidding phase), and the trainer has no bidding driver.
    A YAML ``ruleset: standard`` is surfaced to the trainer as
    ``RuleSet.standard()`` so the trainer's gate raises a precise error
    rather than silently mis-driving bidding.
    """
    from douzero.env.rules import RuleSet

    if cfg is None:
        return None
    if cfg.ruleset == "standard":
        # The trainer will reject this at construction with a precise error.
        return RuleSet.standard()
    # legacy or any other value: return None so Env runs in legacy card-play
    # mode (no bidding). The trainer accepts None.
    return None


def _build_loss_config(cfg):
    """Build the trainer's LossConfig honouring explicit YAML weights.

    P06 r1 fix: a YAML ``lambda_win: 0`` is preserved as 0 (it disables the
    term per the LossConfig contract). The previous r0 code converted 0
    back to 1.0, silently breaking the "λ=0 disables" contract. When no
    YAML config is provided we fall back to the V2 multi-objective default
    (``LossConfig()``).
    """
    from douzero.training import LossConfig

    if cfg is None:
        return LossConfig()
    return LossConfig(
        lambda_win=cfg.loss.lambda_win,
        lambda_score=cfg.loss.lambda_score,
        lambda_uncertainty=cfg.loss.lambda_uncertainty,
        score_delta=cfg.loss.score_delta,
        score_target_transform=cfg.loss.score_target_transform,
        score_clamp=cfg.loss.score_clamp,
    )


def _build_decision_config(cfg):
    from douzero.training import DecisionConfig

    if cfg is None:
        return DecisionConfig()
    return DecisionConfig(
        mode=cfg.decision_policy.mode,
        abs_tol=cfg.decision_policy.abs_tol,
        rel_tol=cfg.decision_policy.rel_tol,
        risk_penalty=cfg.decision_policy.risk_penalty,
    )


def _build_model_cfg(cfg):
    """Build a ModelV2Config from the YAML model_version block (P06 r1)."""
    from douzero.models_v2.config import ModelV2Config

    if cfg is None:
        return ModelV2Config()
    # The legacy TrainingConfig does not carry the full ModelConfig
    # architecture block (only the version strings); use the V2 defaults
    # but honour score_clamp so the head clamp matches the loss target
    # clamp exactly.
    return ModelV2Config(score_clamp=cfg.loss.score_clamp)


def main() -> None:
    args = _build_parser().parse_args()

    # CPU-only (P06). Force CUDA invisible so the model never accidentally
    # lands on a GPU the trainer does not move tensors to.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    yaml_cfg = _load_yaml_config(args.config)

    # Resolve the seed/deterministic with CLI > YAML > default precedence.
    # Both --seed and --deterministic use real argparse defaults, so when
    # the YAML is present and the CLI did not override, prefer the YAML.
    if yaml_cfg is not None:
        seed = args.seed if "seed" in vars(args) else yaml_cfg.seed
        deterministic = (
            args.deterministic if "deterministic" in vars(args) else yaml_cfg.deterministic
        )
    else:
        seed = args.seed if "seed" in vars(args) else 0
        deterministic = args.deterministic
    set_global_seed(seed)
    maybe_set_global_deterministic(deterministic)

    # Build the V2 model from the schema + config (honouring score_clamp).
    import torch

    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    torch.manual_seed(seed)
    schema = build_v2_schema()
    model_cfg = _build_model_cfg(yaml_cfg)
    model = ModelV2(schema, model_cfg)

    # Resolve TrainerConfig with CLI > YAML > built-in-default precedence.
    # CLI overrides come from vars(args); YAML values come from yaml_cfg;
    # built-in defaults come from TrainerConfig().
    from douzero.training import TrainerConfig, V2Trainer

    defaults = TrainerConfig()

    def pick(name, yaml_obj=None):
        if name in vars(args):
            return getattr(args, name)
        if yaml_obj is not None and hasattr(yaml_obj, name):
            return getattr(yaml_obj, name)
        return getattr(defaults, name)

    trainer_cfg = TrainerConfig(
        seed=seed,
        rng_seed=seed,
        max_episodes=pick("episodes", yaml_cfg),
        optimizer_steps=pick("optimizer_steps", yaml_cfg),
        batch_size=pick("batch_size", yaml_cfg),
        buffer_capacity=pick("buffer_capacity", yaml_cfg),
        exp_epsilon=pick("exp_epsilon", yaml_cfg),
        learning_rate=pick(
            "learning_rate",
            getattr(yaml_cfg, "optimizer", None) if yaml_cfg else None,
        ),
        rmsprop_alpha=pick(
            "alpha",
            getattr(yaml_cfg, "optimizer", None) if yaml_cfg else None,
        ),
        rmsprop_momentum=pick(
            "momentum",
            getattr(yaml_cfg, "optimizer", None) if yaml_cfg else None,
        ),
        rmsprop_epsilon=pick(
            "epsilon",
            getattr(yaml_cfg, "optimizer", None) if yaml_cfg else None,
        ),
        max_grad_norm=pick("max_grad_norm", yaml_cfg),
        max_steps_per_episode=pick("max_steps_per_episode", yaml_cfg),
    )

    ruleset = _resolve_ruleset(yaml_cfg)
    loss_cfg = _build_loss_config(yaml_cfg)
    decision_cfg = _build_decision_config(yaml_cfg)

    trainer = V2Trainer(
        model,
        ruleset=ruleset,
        loss_config=loss_cfg,
        decision_config=decision_cfg,
        config=trainer_cfg,
    )

    print(
        f"[train_v2] model={type(model).__name__} "
        f"params={sum(p.numel() for p in model.parameters())} "
        f"score_clamp={model_cfg.score_clamp} "
        f"loss_cfg={loss_cfg.to_dict()} "
        f"decision={decision_cfg.to_dict()} "
        f"trainer=batch_size={trainer_cfg.batch_size} "
        f"lr={trainer_cfg.learning_rate} "
        f"epsilon={trainer_cfg.exp_epsilon} "
        f"ruleset={trainer.ruleset.ruleset_id if trainer.ruleset else 'legacy'}"
    )
    stats = trainer.train()
    print(
        f"[train_v2] episodes_completed={stats.episodes_completed} "
        f"transitions={stats.transitions_collected} "
        f"optimizer_steps={stats.optimizer_steps} "
        f"parameters_changed={getattr(trainer, 'stats_last_run_changed', 'unknown')} "
        f"last_loss={stats.last_loss} "
        f"grad_norm={stats.grad_norm_last_step:.4f} "
        f"p_win_mean={stats.p_win_mean:.4f}"
    )


if __name__ == "__main__":
    main()
