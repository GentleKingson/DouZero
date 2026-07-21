import pytest
import torch

from douzero.gpu_v3 import (
    GPUV3Config,
    IndependentRoleDualTower,
    SharedTrunkRoleHeads,
)
from douzero.gpu_v3.config import SHARED_TRUNK_ROLE_HEADS


def _model(device="cpu"):
    config = GPUV3Config(
        hidden_size=32,
        action_hidden_size=16,
        trunk_layers=2,
        role_head_layers=1,
    )
    return IndependentRoleDualTower(20, 8, config).to(device)


def test_independent_role_dual_tower_shapes_masks_and_backward():
    model = _model()
    state = torch.randn(6, 20)
    actions = torch.randn(6, 11, 8)
    mask = torch.arange(11)[None, :] < torch.tensor([1, 3, 5, 7, 9, 11])[:, None]
    roles = torch.tensor([0, 1, 2, 0, 1, 2])
    values = model(state, actions, mask, roles)
    assert values.shape == (6, 11)
    assert torch.isneginf(values[~mask]).all()
    values[mask].sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_independent_role_towers_do_not_share_parameters():
    model = _model()
    pointers = {
        role: next(model.state_towers[role].parameters()).data_ptr()
        for role in ("landlord", "landlord_up", "landlord_down")
    }
    assert len(set(pointers.values())) == 3


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_independent_role_dual_tower_cuda_batch():
    model = _model("cuda")
    state = torch.randn(12, 20, device="cuda")
    actions = torch.randn(12, 64, 8, device="cuda")
    mask = torch.ones(12, 64, dtype=torch.bool, device="cuda")
    roles = torch.arange(12, device="cuda") % 3
    values = model(state, actions, mask, roles)
    assert values.is_cuda and torch.isfinite(values).all()


def test_shared_trunk_role_heads_shapes_masks_and_parameter_reduction():
    config = GPUV3Config(
        architecture=SHARED_TRUNK_ROLE_HEADS,
        hidden_size=32,
        action_hidden_size=16,
        trunk_layers=2,
        role_head_layers=1,
    )
    shared = SharedTrunkRoleHeads(20, 8, config)
    independent = _model()
    state = torch.randn(6, 20)
    actions = torch.randn(6, 12, 8)
    mask = torch.ones(6, 12, dtype=torch.bool)
    roles = torch.tensor([0, 1, 2, 0, 1, 2])
    values = shared(state, actions, mask, roles)
    assert values.shape == (6, 12)
    assert torch.isfinite(values).all()
    assert shared.parameter_count() < independent.parameter_count()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shared_trunk_role_heads_cuda_batch():
    config = GPUV3Config(
        architecture=SHARED_TRUNK_ROLE_HEADS,
        hidden_size=64,
        action_hidden_size=32,
        trunk_layers=2,
        role_head_layers=1,
    )
    model = SharedTrunkRoleHeads(20, 8, config).cuda()
    values = model(
        torch.randn(12, 20, device="cuda"),
        torch.randn(12, 64, 8, device="cuda"),
        torch.ones(12, 64, dtype=torch.bool, device="cuda"),
        torch.arange(12, device="cuda") % 3,
    )
    assert values.is_cuda and torch.isfinite(values).all()
