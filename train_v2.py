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

Usage
-----
::

    python train_v2.py --config configs/enhanced.yaml --episodes 8 \\
        --optimizer_steps 2 --seed 0

CPU smoke (no GPU required):

::

    python train_v2.py --episodes 4 --optimizer_steps 1 --seed 0 \\
        --checkpoint_dir /tmp/douzero_v2_smoke
"""

from __future__ import annotations

import argparse
import os

from douzero.config import load_config
from douzero.runtime import maybe_set_global_deterministic, set_global_seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DouZero V2 multi-objective trainer (P06)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a YAML training config (e.g. configs/enhanced.yaml). "
        "When empty, the V2 trainer uses its built-in defaults.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Number of self-play episodes to collect before each optimizer run.",
    )
    parser.add_argument(
        "--optimizer_steps",
        type=int,
        default=1,
        help="Number of optimizer steps per training invocation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Minibatch size for each optimizer step.",
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=4096,
        help="Replay buffer capacity (in transitions).",
    )
    parser.add_argument(
        "--exp_epsilon",
        type=float,
        default=0.3,
        help="Training-time exploration epsilon (epsilon-greedy). "
        "Evaluation is always deterministic (epsilon=0).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="RMSprop learning rate.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=40.0,
        help="Gradient norm clip.",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=600,
        help="Hard step cap per episode (safety against pathological loops).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Drives the global seed and the trainer's RNG.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable full deterministic mode (torch.use_deterministic_algorithms).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Directory to save V2 checkpoints. Empty disables saving.",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="",
        help="CUDA_VISIBLE_DEVICES override (empty = CPU).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_device)
    set_global_seed(args.seed)
    maybe_set_global_deterministic(args.deterministic)

    # Load YAML config (if provided) for the loss + decision_policy blocks.
    cfg = None
    if args.config:
        cfg = load_config(args.config)

    # Build the V2 model from the default schema + config.
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    torch_manual_seed = __import__("torch").manual_seed
    torch_manual_seed(args.seed)
    schema = build_v2_schema()
    model_cfg = ModelV2Config()
    model = ModelV2(schema, model_cfg)

    # Build the trainer's LossConfig / DecisionConfig from the YAML config
    # (if provided), else use the V2 multi-objective defaults.
    from douzero.training import (
        DecisionConfig,
        LossConfig,
        TrainerConfig,
        V2Trainer,
    )

    if cfg is not None:
        loss_cfg = LossConfig(
            lambda_win=cfg.loss.lambda_win if cfg.loss.lambda_win > 0 else 1.0,
            lambda_score=cfg.loss.lambda_score if cfg.loss.lambda_score > 0 else 0.5,
            lambda_log=cfg.loss.lambda_log,
            lambda_uncertainty=cfg.loss.lambda_uncertainty,
            score_delta=cfg.loss.score_delta,
            log_score_delta=cfg.loss.log_score_delta,
        )
        decision_cfg = DecisionConfig(
            mode=cfg.decision_policy.mode,
            abs_tol=cfg.decision_policy.abs_tol,
            rel_tol=cfg.decision_policy.rel_tol,
            risk_penalty=cfg.decision_policy.risk_penalty,
        )
    else:
        loss_cfg = LossConfig()
        decision_cfg = DecisionConfig()

    trainer_cfg = TrainerConfig(
        seed=args.seed,
        rng_seed=args.seed,
        max_episodes=args.episodes,
        optimizer_steps=args.optimizer_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        exp_epsilon=args.exp_epsilon,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        max_steps_per_episode=args.max_steps_per_episode,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = V2Trainer(
        model,
        ruleset=None,
        loss_config=loss_cfg,
        decision_config=decision_cfg,
        config=trainer_cfg,
    )

    print(
        f"[train_v2] model={type(model).__name__} "
        f"params={sum(p.numel() for p in model.parameters())} "
        f"loss_cfg={loss_cfg.to_dict()} "
        f"decision={decision_cfg.to_dict()}"
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
