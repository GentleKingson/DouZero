"""Canonical action alignment and configurable P10 distillation losses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from douzero.models_v2.output import ModelOutput

from .teacher_model import ActionKey, TeacherOutput


@dataclass(frozen=True)
class DistillationLossConfig:
    """Weights for teacher imitation and retained terminal supervision."""

    enabled: bool = False
    temperature: float = 2.0
    top_k: int = 4
    ranking_margin: float = 0.1
    lambda_kl: float = 1.0
    lambda_rank: float = 0.25
    lambda_teacher_win: float = 0.5
    lambda_teacher_score: float = 0.25
    lambda_supervised_win: float = 1.0
    lambda_supervised_score: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0.0
        ):
            raise ValueError(f"temperature must be positive finite, got {self.temperature}")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise ValueError(f"top_k must be a positive int, got {self.top_k!r}")
        if not math.isfinite(self.ranking_margin) or self.ranking_margin < 0.0:
            raise ValueError(
                f"ranking_margin must be non-negative finite, got {self.ranking_margin}"
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
                raise ValueError(f"{name} must be non-negative finite, got {value}")

    @classmethod
    def from_training_config(cls, config) -> "DistillationLossConfig":
        """Bridge the repository YAML DistillationConfig into the loss API."""

        return cls(
            enabled=config.enabled,
            temperature=config.distillation_temperature,
            top_k=config.top_k,
            lambda_kl=config.lambda_kl,
            lambda_rank=config.lambda_rank,
            lambda_teacher_win=config.lambda_teacher_win,
            lambda_teacher_score=config.lambda_teacher_score,
            lambda_supervised_win=config.lambda_supervised_win,
            lambda_supervised_score=config.lambda_supervised_score,
        )


@dataclass
class DistillationLossComponents:
    """Differentiable total plus detached per-term metrics."""

    total: torch.Tensor
    kl: float
    ranking: float
    teacher_win: float
    teacher_score: float
    supervised_win: float
    supervised_score: float

    def as_log_dict(self) -> dict[str, float]:
        return {
            "distill_total": float(self.total.detach().cpu()),
            "distill_kl": self.kl,
            "distill_ranking": self.ranking,
            "distill_teacher_win": self.teacher_win,
            "distill_teacher_score": self.teacher_score,
            "distill_supervised_win": self.supervised_win,
            "distill_supervised_score": self.supervised_score,
        }


def align_teacher_output(
    teacher: TeacherOutput,
    student_action_keys: tuple[ActionKey, ...],
) -> TeacherOutput:
    """Reorder teacher rows to the student's canonical action-key order."""

    student_keys = tuple(tuple(key) for key in student_action_keys)
    if len(set(student_keys)) != len(student_keys):
        raise ValueError("student_action_keys must be unique")
    teacher_index = {key: index for index, key in enumerate(teacher.action_keys)}
    missing = set(student_keys) - set(teacher_index)
    extra = set(teacher_index) - set(student_keys)
    if missing or extra:
        raise ValueError(
            f"teacher/student legal-action key mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    index = torch.tensor(
        [teacher_index[key] for key in student_keys],
        dtype=torch.long,
        device=teacher.win_logit.device,
    )
    return TeacherOutput(
        action_keys=student_keys,
        win_logit=teacher.win_logit.index_select(0, index),
        p_win=teacher.p_win.index_select(0, index),
        expected_score=teacher.expected_score.index_select(0, index),
        action_logits=teacher.action_logits.index_select(0, index),
        action_mask=teacher.action_mask.index_select(0, index),
    )


def _student_policy_logits(output: ModelOutput) -> torch.Tensor:
    return output.prior_logit if output.prior_logit is not None else output.win_logit


def distillation_loss(
    student: ModelOutput,
    student_action_keys: tuple[ActionKey, ...],
    *,
    action_index: int,
    target_win: float,
    target_score: float,
    teacher: TeacherOutput | None,
    config: DistillationLossConfig | None = None,
) -> DistillationLossComponents:
    """Combine teacher imitation with the student's real terminal labels."""

    cfg = config or DistillationLossConfig()
    n = student.num_actions
    if len(student_action_keys) != n:
        raise ValueError(
            f"student_action_keys has {len(student_action_keys)} rows, output has {n}"
        )
    if not 0 <= action_index < n or not bool(student.action_mask[action_index]):
        raise ValueError(f"action_index {action_index} is not a valid student action")

    zero = student.win_logit.sum() * 0.0
    kl = zero
    ranking = zero
    teacher_win = zero
    teacher_score = zero

    if cfg.enabled:
        if teacher is None:
            raise ValueError("distillation is enabled but no teacher output was supplied")
        aligned = align_teacher_output(teacher, student_action_keys)
        if not bool(torch.equal(aligned.action_mask.cpu(), student.action_mask.cpu())):
            raise ValueError("teacher/student action masks disagree after canonical alignment")
        device = student.win_logit.device
        teacher_logits = aligned.action_logits.to(device=device).squeeze(-1).detach()
        student_logits = _student_policy_logits(student).squeeze(-1)
        valid = student.action_mask
        temperature = cfg.temperature
        teacher_probs = F.softmax(teacher_logits[valid] / temperature, dim=0)
        student_log_probs = F.log_softmax(student_logits[valid] / temperature, dim=0)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (
            temperature * temperature
        )

        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        top_k = min(cfg.top_k, int(valid_indices.numel()))
        teacher_order = valid_indices[
            torch.argsort(teacher_logits[valid_indices], descending=True)[:top_k]
        ]
        pair_terms = []
        for left in range(top_k):
            for right in range(left + 1, top_k):
                pair_terms.append(
                    F.relu(
                        student_logits.new_tensor(cfg.ranking_margin)
                        - (student_logits[teacher_order[left]] - student_logits[teacher_order[right]])
                    )
                )
        if pair_terms:
            ranking = torch.stack(pair_terms).mean()

        teacher_win = F.mse_loss(
            student.p_win[valid], aligned.p_win.to(device=device)[valid].detach()
        )
        teacher_score = F.smooth_l1_loss(
            student.score_mean[valid],
            aligned.expected_score.to(device=device)[valid].detach(),
        )
    elif teacher is not None:
        raise ValueError(
            "teacher output was supplied while distillation is disabled; omit it "
            "to guarantee the feature flag fully removes privileged supervision"
        )

    target_win_tensor = student.win_logit.new_tensor(float(target_win)).reshape(1)
    target_score_tensor = student.score_mean.new_tensor(float(target_score)).reshape(1)
    supervised_win = F.binary_cross_entropy_with_logits(
        student.win_logit[action_index].reshape(1), target_win_tensor
    )
    supervised_score = F.smooth_l1_loss(
        student.score_mean[action_index].reshape(1), target_score_tensor
    )
    total = (
        cfg.lambda_kl * kl
        + cfg.lambda_rank * ranking
        + cfg.lambda_teacher_win * teacher_win
        + cfg.lambda_teacher_score * teacher_score
        + cfg.lambda_supervised_win * supervised_win
        + cfg.lambda_supervised_score * supervised_score
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("distillation loss is NaN or Inf")
    return DistillationLossComponents(
        total=total,
        kl=float(kl.detach().cpu()),
        ranking=float(ranking.detach().cpu()),
        teacher_win=float(teacher_win.detach().cpu()),
        teacher_score=float(teacher_score.detach().cpu()),
        supervised_win=float(supervised_win.detach().cpu()),
        supervised_score=float(supervised_score.detach().cpu()),
    )
