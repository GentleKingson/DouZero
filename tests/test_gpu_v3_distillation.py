import pytest
import torch

from douzero.dmc.models_factorized import factorized_model_dict
from douzero.gpu_v3 import GPUV3Config, SharedTrunkRoleHeads
from douzero.gpu_v3.config import SHARED_TRUNK_ROLE_HEADS
from douzero.gpu_v3.distillation import (
    GPU_V3_LEGACY_ACTION_WIDTH,
    GPU_V3_LEGACY_STATE_WIDTH,
    _single_decision_policy_kl,
    build_legacy_student_inputs,
    legacy_teacher_distillation_loss,
)


@pytest.mark.parametrize("position,state_width", [
    ("landlord", 319),
    ("landlord_up", 430),
    ("landlord_down", 430),
])
def test_legacy_student_input_contract(position, state_width):
    inputs = build_legacy_student_inputs(
        position,
        torch.randn(1, 5, 162),
        torch.randn(1, state_width),
        torch.randn(7, GPU_V3_LEGACY_ACTION_WIDTH),
        action_bucket=16,
    )
    state, actions, mask, role_ids = inputs
    assert state.shape == (1, GPU_V3_LEGACY_STATE_WIDTH)
    assert actions.shape == (1, 16, GPU_V3_LEGACY_ACTION_WIDTH)
    assert mask.sum() == 7
    assert role_ids.item() in {0, 1, 2}


def test_legacy_teacher_distillation_updates_only_student():
    teacher = factorized_model_dict["landlord"]().eval()
    config = GPUV3Config(
        architecture=SHARED_TRUNK_ROLE_HEADS,
        hidden_size=32,
        action_hidden_size=16,
        trunk_layers=1,
        role_head_layers=1,
    )
    student = SharedTrunkRoleHeads(
        GPU_V3_LEGACY_STATE_WIDTH,
        GPU_V3_LEGACY_ACTION_WIDTH,
        config,
    )
    loss, metrics = legacy_teacher_distillation_loss(
        teacher,
        student,
        "landlord",
        torch.randn(1, 5, 162),
        torch.randn(1, 319),
        torch.randn(9, GPU_V3_LEGACY_ACTION_WIDTH),
        action_bucket=16,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert set(metrics) == {
        "value_loss", "policy_loss", "teacher_argmax", "student_argmax"
    }


def test_policy_kl_does_not_shrink_when_action_set_is_duplicated():
    teacher = torch.tensor([2.0, -1.0])
    student = torch.tensor([0.5, 0.0], requires_grad=True)
    base = _single_decision_policy_kl(student, teacher, temperature=1.0)
    duplicated = _single_decision_policy_kl(
        student.repeat(10), teacher.repeat(10), temperature=1.0
    )

    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=0),
        torch.softmax(teacher, dim=0),
        reduction="sum",
    )
    assert base.item() == pytest.approx(expected.item())
    assert duplicated.item() == pytest.approx(base.item())
