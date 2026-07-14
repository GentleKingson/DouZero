"""Bounded teacher training and strict privileged checkpoint I/O."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from douzero.checkpoint.io import CheckpointCompatibilityError, _validate_manifest
from douzero.checkpoint.manifest import (
    CURRENT_SCHEMA_VERSION,
    MODEL_ACCESS_PRIVILEGED,
    CheckpointManifest,
    build_manifest,
)
from douzero.env.rules import RuleSet

from .dataset import OfflineDistillationSample, load_offline_dataset
from .teacher_model import TeacherModel, state_dict_sha256

TEACHER_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TeacherTrainConfig:
    """Small supervised teacher-training loop settings."""

    learning_rate: float = 1e-3
    max_grad_norm: float = 10.0
    lambda_policy: float = 1.0
    lambda_win: float = 1.0
    lambda_score: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "learning_rate", "max_grad_norm", "lambda_policy", "lambda_win",
            "lambda_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite, got {value}")


def teacher_supervised_loss(
    model: TeacherModel,
    sample: OfflineDistillationSample,
    config: TeacherTrainConfig,
) -> torch.Tensor:
    """Monte-Carlo multi-objective teacher loss for one chosen legal action."""

    output = model(
        sample.public_inputs,
        sample.privileged_observation,
        action_keys=sample.action_keys,
    )
    index = sample.action_index
    target_index = torch.tensor([index], device=output.action_logits.device)
    policy = F.cross_entropy(output.action_logits.squeeze(-1).unsqueeze(0), target_index)
    target_win = output.win_logit.new_tensor([sample.target_win])
    win = F.binary_cross_entropy_with_logits(
        output.win_logit[index].reshape(1), target_win
    )
    target_score = output.expected_score.new_tensor([sample.target_score])
    score = F.smooth_l1_loss(
        output.expected_score[index].reshape(1), target_score
    )
    total = (
        config.lambda_policy * policy
        + config.lambda_win * win
        + config.lambda_score * score
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("teacher loss is NaN or Inf")
    return total


class TeacherTrainer:
    """Single-process trainer for offline perfect-information samples."""

    def __init__(
        self,
        model: TeacherModel,
        config: TeacherTrainConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TeacherTrainConfig()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def train_epoch(self, samples: Iterable[OfflineDistillationSample]) -> float:
        """Run one bounded epoch and return mean pre-update sample loss."""

        self.model.train()
        losses: list[float] = []
        for sample in samples:
            self.optimizer.zero_grad(set_to_none=True)
            loss = teacher_supervised_loss(self.model, sample, self.config)
            losses.append(float(loss.detach().cpu()))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("teacher gradient norm is NaN or Inf")
            self.optimizer.step()
        if not losses:
            raise ValueError("teacher training requires at least one sample")
        return sum(losses) / len(losses)


def save_teacher_checkpoint(
    path: str | os.PathLike[str],
    model: TeacherModel,
    *,
    ruleset: RuleSet,
    effective_config: dict | None = None,
    frames: int = 0,
) -> CheckpointManifest:
    """Atomically save a manifest-bearing, privileged-only teacher bundle."""

    if not isinstance(ruleset, RuleSet):
        raise TypeError("ruleset must be a RuleSet")
    flags = {
        "feature_version": "v2",
        "model_version": model.model_version,
        "ruleset": ruleset.ruleset_id,
        **(effective_config or {}),
    }
    manifest = build_manifest(
        flags,
        frames=frames,
        position_frames={},
        checkpoint_kind="privileged_teacher",
    )
    object.__setattr__(manifest, "model_version", model.model_version)
    object.__setattr__(manifest, "feature_version", "v2")
    object.__setattr__(manifest, "ruleset_id", ruleset.ruleset_id)
    object.__setattr__(manifest, "ruleset_version", ruleset.ruleset_version)
    object.__setattr__(manifest, "ruleset_hash", ruleset.stable_hash())
    object.__setattr__(manifest, "model_access", MODEL_ACCESS_PRIVILEGED)
    bundle = {
        "checkpoint_version": TEACHER_CHECKPOINT_VERSION,
        "teacher_state_dict": model.state_dict(),
        "manifest": manifest.to_dict(),
        "feature_schema_hash": model.schema.stable_hash(),
        "teacher_config_hash": model.config_hash(),
        "teacher_config": asdict(model.config),
        "public_model_config": model.public_model.config.compatibility_dict(),
        "teacher_model_sha": state_dict_sha256(model),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, destination)
    return manifest


def load_teacher_checkpoint(
    path: str | os.PathLike[str],
    model: TeacherModel,
    *,
    expected_ruleset: RuleSet,
    allow_unsafe_pickle: bool = False,
) -> CheckpointManifest:
    """Strictly load a privileged teacher; never accepts a public checkpoint."""

    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=not allow_unsafe_pickle,
    )
    if not isinstance(bundle, dict) or "manifest" not in bundle:
        raise CheckpointCompatibilityError("teacher checkpoint has no manifest")
    if bundle.get("checkpoint_version") != TEACHER_CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError("teacher checkpoint_version mismatch")
    manifest = CheckpointManifest.from_dict(bundle["manifest"])
    _validate_manifest(
        manifest,
        expected_schema_version=CURRENT_SCHEMA_VERSION,
        expected_model_version=model.model_version,
        expected_feature_version="v2",
        expected_ruleset_id=expected_ruleset.ruleset_id,
        expected_checkpoint_kind="privileged_teacher",
        path=str(path),
        expected_ruleset_version=expected_ruleset.ruleset_version,
        expected_ruleset_hash=expected_ruleset.stable_hash(),
        expected_model_access=MODEL_ACCESS_PRIVILEGED,
    )
    if bundle.get("feature_schema_hash") != model.schema.stable_hash():
        raise CheckpointCompatibilityError("teacher feature_schema_hash mismatch")
    if bundle.get("teacher_config_hash") != model.config_hash():
        raise CheckpointCompatibilityError("teacher_config_hash mismatch")
    state_dict = bundle.get("teacher_state_dict")
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError("teacher_state_dict is missing or malformed")
    model.load_state_dict(state_dict, strict=True)
    if bundle.get("teacher_model_sha") != state_dict_sha256(model):
        raise CheckpointCompatibilityError("teacher_model_sha mismatch after strict load")
    model.eval()
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a P10 privileged teacher")
    parser.add_argument("--config", required=True, help="YAML with distillation.enabled=true")
    parser.add_argument("--dataset", required=True, help="Offline distillation dataset")
    parser.add_argument("--output", required=True, help="Teacher checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    from douzero.config import load_config
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    cfg = load_config(args.config)
    if not cfg.distillation.enabled:
        raise ValueError(
            "distillation.enabled is false; refusing to start privileged teacher training"
        )
    if cfg.feature_version != "v2" or cfg.model_version != "v2":
        raise ValueError("teacher training requires feature_version=v2 and model_version=v2")
    ruleset = RuleSet.legacy() if cfg.ruleset == "legacy" else RuleSet.standard()
    schema = build_v2_schema()
    public_cfg = ModelV2Config.from_model_config(cfg.model)
    teacher = TeacherModel(ModelV2(schema, public_cfg))
    dataset = load_offline_dataset(
        args.dataset,
        expected_feature_schema_hash=schema.stable_hash(),
        expected_ruleset_hash=ruleset.stable_hash(),
    )
    trainer = TeacherTrainer(
        teacher, TeacherTrainConfig(learning_rate=args.learning_rate)
    )
    last_loss = float("nan")
    for _ in range(args.epochs):
        last_loss = trainer.train_epoch(dataset)
    save_teacher_checkpoint(
        args.output,
        teacher,
        ruleset=ruleset,
        effective_config={"distillation": asdict(cfg.distillation)},
        frames=args.epochs * len(dataset),
    )
    print(
        f"[train_teacher] samples={len(dataset)} epochs={args.epochs} "
        f"last_loss={last_loss:.6f} output={args.output}"
    )


if __name__ == "__main__":
    main()
