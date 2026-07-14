"""Train an independently-versioned P12 opening coach from fresh labels."""

from __future__ import annotations

import argparse
import random


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the P12 opening coach")
    parser.add_argument("--labels", required=True, help="Coach-label JSONL path")
    parser.add_argument("--output", required=True, help="Output coach .pt path")
    parser.add_argument("--policy_version", required=True)
    parser.add_argument("--current_policy_step", required=True, type=int)
    parser.add_argument("--max_label_age_steps", type=int, default=100000)
    parser.add_argument("--ruleset", choices=("legacy", "standard"), default="legacy")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    import torch

    from douzero.coach import (
        CoachLabelStore,
        CoachModel,
        CoachModelConfig,
        calibration_metrics,
        save_coach_checkpoint,
        train_coach,
    )
    from douzero.env.rules import RuleSet

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ruleset = RuleSet.legacy() if args.ruleset == "legacy" else RuleSet.standard()
    labels = CoachLabelStore(args.labels).load_fresh(
        policy_version=args.policy_version,
        current_policy_step=args.current_policy_step,
        max_age_steps=args.max_label_age_steps,
    )
    if not labels:
        raise ValueError("no fresh labels matched the requested policy window")
    mismatched = [
        label.opening.opening_id
        for label in labels
        if label.opening.ruleset_obj.stable_hash() != ruleset.stable_hash()
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} fresh labels use a different RuleSet; "
            "coach training never silently mixes rule identities"
        )

    # Stable holdout: every fifth content-addressed opening. Small smoke
    # datasets use the training set for reporting and say so in stdout.
    ordered = sorted(labels, key=lambda item: item.opening.opening_id)
    validation = ordered[::5] if len(ordered) >= 5 else []
    validation_ids = {item.opening.opening_id for item in validation}
    training = [item for item in ordered if item.opening.opening_id not in validation_ids]
    if not training:
        training = ordered
    model = CoachModel(CoachModelConfig(hidden_size=args.hidden_size))
    losses = train_coach(
        model,
        training,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    calibration_set = validation or training
    probabilities = [
        model.predict(item.opening, item.policy_version) for item in calibration_set
    ]
    metrics = calibration_metrics(
        probabilities,
        [item.landlord_win for item in calibration_set],
    )
    metrics["holdout"] = float(bool(validation))
    save_coach_checkpoint(
        args.output,
        model,
        policy_version=args.policy_version,
        policy_step=args.current_policy_step,
        ruleset_hash=ruleset.stable_hash(),
        calibration=metrics,
    )
    print(
        f"[train_coach] fresh_labels={len(labels)} train={len(training)} "
        f"validation={len(validation)} epochs={args.epochs} "
        f"last_loss={losses[-1]:.6f} brier={metrics['brier']:.6f} "
        f"ece={metrics['ece']:.6f} output={args.output}"
    )


if __name__ == "__main__":
    main()
