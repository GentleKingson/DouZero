"""Public-student optimizer driven by an optional privileged teacher."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from douzero.models_v2.model import ModelV2

from .cache import TeacherCache, TeacherCacheIdentity
from .dataset import OfflineDistillationSample, load_offline_dataset
from .losses import (
    DistillationLossComponents,
    DistillationLossConfig,
    distillation_loss,
)
from .teacher_model import TeacherModel, TeacherOutput, forward_public_model, state_dict_sha256


@dataclass(frozen=True)
class StudentTrainConfig:
    """Optimizer settings for the bounded student distillation loop."""

    learning_rate: float = 1e-4
    max_grad_norm: float = 10.0
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive finite")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive finite")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive int")


class StudentDistiller:
    """Train Model V2 without ever adding hidden-hand fields to its forward API."""

    def __init__(
        self,
        student: ModelV2,
        *,
        teacher: TeacherModel | None = None,
        loss_config: DistillationLossConfig | None = None,
        train_config: StudentTrainConfig | None = None,
        teacher_cache: TeacherCache | None = None,
    ) -> None:
        if student.config.belief_enabled:
            raise ValueError(
                "StudentDistiller currently requires belief_enabled=False; the "
                "offline dataset does not carry public belief posterior features"
            )
        self.student = student
        self.loss_config = loss_config or DistillationLossConfig()
        self.train_config = train_config or StudentTrainConfig()
        if self.loss_config.score_clamp != self.student.config.score_clamp:
            raise ValueError(
                "distillation score_clamp must match public Model V2 config: "
                f"{self.loss_config.score_clamp} != {self.student.config.score_clamp}"
            )
        if (
            self.loss_config.score_target_transform
            != self.student.config.score_target_transform
        ):
            raise ValueError(
                "distillation score_target_transform must match public Model V2 config: "
                f"{self.loss_config.score_target_transform!r} != "
                f"{self.student.config.score_target_transform!r}"
            )
        if self.loss_config.enabled and teacher is None:
            raise ValueError("enabled distillation requires a TeacherModel")
        if not self.loss_config.enabled and teacher is not None:
            raise ValueError(
                "distillation is disabled but a teacher was supplied; omit the "
                "teacher so the privileged path is fully disconnected"
            )
        if teacher_cache is not None and teacher is None:
            raise ValueError("teacher_cache requires enabled distillation and a teacher")
        self.teacher = teacher
        self.teacher_cache = teacher_cache
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.train_config.learning_rate
        )
        self.optimizer_steps_last_epoch = 0
        self.max_batch_size_last_epoch = 0

    def _teacher_output(self, sample: OfflineDistillationSample) -> TeacherOutput | None:
        if self.teacher is None:
            return None
        if self.teacher_cache is not None:
            cached = self.teacher_cache.get(sample)
            if cached is not None:
                return cached
        self.teacher.eval()
        with torch.inference_mode():
            output = self.teacher(
                sample.public_inputs,
                sample.privileged_observation,
                action_keys=sample.action_keys,
            ).detached_cpu()
        if self.teacher_cache is not None:
            self.teacher_cache.put(sample, output)
        return output

    def loss_for_sample(
        self,
        sample: OfflineDistillationSample,
    ) -> DistillationLossComponents:
        """Compute one sample loss; disabled mode never reads privileged data."""

        student_output = forward_public_model(self.student, sample.public_inputs)
        teacher_output = self._teacher_output(sample) if self.loss_config.enabled else None
        return distillation_loss(
            student_output,
            sample.action_keys,
            action_index=sample.action_index,
            target_win=sample.target_win,
            target_score=sample.target_score,
            teacher=teacher_output,
            config=self.loss_config,
        )

    def _train_batch(self, samples: list[OfflineDistillationSample]) -> float:
        """Forward, backpropagate, and release one bounded minibatch graph."""

        self.optimizer.zero_grad(set_to_none=True)
        components = [self.loss_for_sample(sample) for sample in samples]
        total = torch.stack([component.total for component in components]).mean()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("student distillation loss is NaN or Inf")
        before = float(total.detach().cpu())
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            self.train_config.max_grad_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("student gradient norm is NaN or Inf")
        self.optimizer.step()
        return before

    def train_epoch(self, samples: Iterable[OfflineDistillationSample]) -> float:
        """Train with independent bounded minibatch graphs and optimizer steps."""

        self.student.train()
        self.optimizer_steps_last_epoch = 0
        self.max_batch_size_last_epoch = 0
        weighted_loss = 0.0
        sample_count = 0
        batch: list[OfflineDistillationSample] = []
        for sample in samples:
            batch.append(sample)
            if len(batch) < self.train_config.batch_size:
                continue
            batch_loss = self._train_batch(batch)
            weighted_loss += batch_loss * len(batch)
            sample_count += len(batch)
            self.optimizer_steps_last_epoch += 1
            self.max_batch_size_last_epoch = max(
                self.max_batch_size_last_epoch, len(batch)
            )
            batch = []
        if batch:
            batch_loss = self._train_batch(batch)
            weighted_loss += batch_loss * len(batch)
            sample_count += len(batch)
            self.optimizer_steps_last_epoch += 1
            self.max_batch_size_last_epoch = max(
                self.max_batch_size_last_epoch, len(batch)
            )
        if sample_count == 0:
            raise ValueError("student distillation requires at least one sample")
        return weighted_loss / sample_count


def build_teacher_cache(
    path: str,
    *,
    student: ModelV2,
    teacher: TeacherModel,
    ruleset_hash: str,
) -> TeacherCache:
    """Construct a cache bound to schema, rules, and exact teacher weights."""

    return TeacherCache(
        path,
        TeacherCacheIdentity(
            feature_schema_hash=student.schema.stable_hash(),
            ruleset_hash=ruleset_hash,
            teacher_model_sha=state_dict_sha256(teacher),
            teacher_config_hash=teacher.config_hash(),
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill a P10 public student")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--teacher", default="")
    parser.add_argument("--output", required=True, help="Public policy .ckpt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    from douzero.config import load_config
    from douzero.env.rules import RuleSet
    from douzero.models_v2.config import ModelV2Config
    from douzero.observation.schema import build_v2_schema

    from .export import export_public_student
    from .train_teacher import load_teacher_checkpoint

    cfg = load_config(args.config)
    distill_cfg = cfg.distillation
    if not distill_cfg.enabled:
        raise ValueError(
            "distillation.enabled is false; refusing to construct a privileged teacher"
        )
    dataset_path = args.dataset or distill_cfg.dataset_path
    teacher_path = args.teacher or distill_cfg.teacher_checkpoint
    if not dataset_path or not teacher_path:
        raise ValueError("dataset and teacher checkpoint paths are required")
    ruleset = RuleSet.legacy() if cfg.ruleset == "legacy" else RuleSet.standard()
    schema = build_v2_schema()
    model_cfg = ModelV2Config.from_training_config(cfg)
    student = ModelV2(schema, model_cfg)
    teacher = TeacherModel(ModelV2(schema, model_cfg))
    load_teacher_checkpoint(teacher_path, teacher, expected_ruleset=ruleset)
    dataset = load_offline_dataset(
        dataset_path,
        expected_feature_schema_hash=schema.stable_hash(),
        expected_ruleset_hash=ruleset.stable_hash(),
    )
    cache = None
    if distill_cfg.cache_path:
        cache = build_teacher_cache(
            distill_cfg.cache_path,
            student=student,
            teacher=teacher,
            ruleset_hash=ruleset.stable_hash(),
        )
    distiller = StudentDistiller(
        student,
        teacher=teacher,
        loss_config=DistillationLossConfig.from_training_config(cfg),
        train_config=StudentTrainConfig(
            learning_rate=args.learning_rate,
            batch_size=(
                distill_cfg.batch_size
                if args.batch_size is None
                else args.batch_size
            ),
        ),
        teacher_cache=cache,
    )
    last_loss = float("nan")
    for _ in range(args.epochs):
        last_loss = distiller.train_epoch(dataset)
    if cache is not None:
        cache.save()
    export_public_student(
        args.output,
        student,
        ruleset=ruleset,
        flags={"distillation": asdict(distill_cfg)},
    )
    print(
        f"[distill_student] samples={len(dataset)} epochs={args.epochs} "
        f"last_loss={last_loss:.6f} output={args.output}"
    )


if __name__ == "__main__":
    main()
