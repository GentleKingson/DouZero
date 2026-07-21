"""Listwise H3 Oracle guidance losses over real environment legal actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from douzero.distillation.losses import align_teacher_output
from douzero.distillation.teacher_model import ActionKey, TeacherOutput

from ..output import V3HybridModelOutput
from .guidance_config import OracleGuidanceLossConfig


@dataclass(frozen=True)
class OracleGuidanceLoss:
    total: torch.Tensor
    kl: torch.Tensor
    ranking: torch.Tensor
    chosen_value: torch.Tensor
    agreement: float
    value_error_abs: float


def oracle_guidance_loss(
    student: V3HybridModelOutput,
    student_action_keys: tuple[ActionKey, ...],
    teacher: TeacherOutput,
    *,
    chosen_action_index: int,
    temperature: float,
    config: OracleGuidanceLossConfig | None = None,
) -> OracleGuidanceLoss:
    """Guide one decision after exact reuse of P10 canonical alignment."""

    cfg = config or OracleGuidanceLossConfig()
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError("temperature must be positive and finite")
    if len(student_action_keys) != student.num_actions:
        raise ValueError("student action key count does not match output rows")
    if (
        isinstance(chosen_action_index, bool)
        or not isinstance(chosen_action_index, int)
        or not 0 <= chosen_action_index < student.num_actions
        or not bool(student.action_mask[chosen_action_index])
    ):
        raise ValueError("chosen_action_index is not a real legal action")
    aligned = align_teacher_output(teacher, student_action_keys)
    aligned_mask = aligned.action_mask.to(device=student.action_mask.device)
    if not bool(torch.equal(aligned_mask, student.action_mask)):
        raise ValueError("Oracle/student action masks disagree after alignment")

    valid = student.action_mask
    student_logits = student.dmc_q.squeeze(-1)
    teacher_logits = aligned.action_logits.to(student_logits.device).squeeze(-1).detach()
    zero = student_logits[valid].sum() * 0.0
    kl = zero
    ranking = zero
    chosen_value = zero

    if cfg.lambda_kl > 0.0:
        teacher_probs = F.softmax(teacher_logits[valid] / temperature, dim=0)
        student_log_probs = F.log_softmax(student_logits[valid] / temperature, dim=0)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (
            temperature * temperature
        )

    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if cfg.lambda_ranking > 0.0:
        top_k = min(cfg.top_k, int(valid_indices.numel()))
        teacher_order = valid_indices[
            torch.argsort(teacher_logits[valid_indices], descending=True)[:top_k]
        ]
        pairs = []
        for left in range(top_k):
            for right in range(left + 1, top_k):
                pairs.append(
                    F.relu(
                        student_logits.new_tensor(cfg.ranking_margin)
                        - student_logits[teacher_order[left]]
                        + student_logits[teacher_order[right]]
                    )
                )
        if pairs:
            ranking = torch.stack(pairs).mean()

    if cfg.lambda_chosen_value > 0.0:
        chosen_value = F.mse_loss(
            student.dmc_q[chosen_action_index],
            aligned.action_logits.to(student.dmc_q.device)[chosen_action_index].detach(),
        )
    total = (
        cfg.lambda_kl * kl
        + cfg.lambda_ranking * ranking
        + cfg.lambda_chosen_value * chosen_value
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Oracle guidance loss is NaN or Inf")
    student_choice = int(torch.argmax(student_logits.masked_fill(~valid, -torch.inf)).item())
    teacher_choice = int(torch.argmax(teacher_logits.masked_fill(~valid, -torch.inf)).item())
    value_error = float(
        torch.abs(student.dmc_q[chosen_action_index].detach() - teacher_logits[chosen_action_index])
        .cpu()
        .item()
    )
    return OracleGuidanceLoss(
        total=total,
        kl=kl,
        ranking=ranking,
        chosen_value=chosen_value,
        agreement=float(student_choice == teacher_choice),
        value_error_abs=value_error,
    )
