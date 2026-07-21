from __future__ import annotations

import torch
from torch.nn import functional as F

from .models import GPU_V3_ROLES


GPU_V3_LEGACY_STATE_WIDTH = 5 * 162 + 430
GPU_V3_LEGACY_ACTION_WIDTH = 54


def _single_decision_policy_kl(
    student_values, teacher_values, temperature
):
    """Return KL for one decision without treating actions as batch items."""
    teacher_distribution = F.softmax(
        teacher_values.unsqueeze(0) / temperature, dim=1
    )
    student_log_distribution = F.log_softmax(
        student_values.unsqueeze(0) / temperature, dim=1
    )
    return F.kl_div(
        student_log_distribution,
        teacher_distribution,
        reduction="batchmean",
    ) * temperature ** 2


def build_legacy_student_inputs(
    position,
    z_single,
    x_state_single,
    x_action,
    *,
    action_bucket=None,
):
    """Pad one public Legacy factorized decision for a gpu_v3 student."""
    if position not in GPU_V3_ROLES:
        raise ValueError(f"unknown Legacy role {position!r}")
    if z_single.shape != (1, 5, 162) or x_state_single.ndim != 2:
        raise ValueError("invalid Legacy teacher state inputs")
    if x_state_single.shape[0] != 1 or x_state_single.shape[1] not in {319, 430}:
        raise ValueError("invalid Legacy role state width")
    if x_action.ndim != 2 or x_action.shape[1] != GPU_V3_LEGACY_ACTION_WIDTH:
        raise ValueError("invalid Legacy action inputs")
    count = x_action.shape[0]
    bucket = action_bucket or count
    if count < 1 or bucket < count:
        raise ValueError("action bucket must cover every legal action")

    state = x_state_single.new_zeros(1, GPU_V3_LEGACY_STATE_WIDTH)
    state[:, :5 * 162] = z_single.flatten(1)
    state[:, 5 * 162:5 * 162 + x_state_single.shape[1]] = x_state_single
    actions = x_action.new_zeros(1, bucket, GPU_V3_LEGACY_ACTION_WIDTH)
    actions[0, :count] = x_action
    mask = torch.arange(bucket, device=x_action.device)[None, :] < count
    role_ids = torch.tensor(
        [GPU_V3_ROLES.index(position)], dtype=torch.long, device=x_action.device
    )
    return state, actions, mask, role_ids


def legacy_teacher_distillation_loss(
    teacher,
    student,
    position,
    z_single,
    x_state_single,
    x_action,
    *,
    action_bucket=None,
    temperature=1.0,
    kl_weight=0.25,
):
    """Distill a frozen Legacy role model into an isolated gpu_v3 student."""
    if temperature <= 0 or kl_weight < 0:
        raise ValueError("distillation temperature/weight is invalid")
    with torch.no_grad():
        teacher_values = teacher.forward_factorized(
            z_single, x_state_single, x_action, return_value=True
        )["values"].squeeze(-1)
    inputs = build_legacy_student_inputs(
        position,
        z_single,
        x_state_single,
        x_action,
        action_bucket=action_bucket,
    )
    student_values = student(*inputs)[0, :x_action.shape[0]]
    value_loss = F.mse_loss(student_values, teacher_values)
    policy_loss = _single_decision_policy_kl(
        student_values, teacher_values, temperature
    )
    loss = value_loss + kl_weight * policy_loss
    return loss, {
        "value_loss": value_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "teacher_argmax": teacher_values.argmax().detach(),
        "student_argmax": student_values.argmax().detach(),
    }
