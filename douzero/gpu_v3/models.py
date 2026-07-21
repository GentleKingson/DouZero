from __future__ import annotations

import torch
from torch import nn

from .config import (
    GPUV3Config,
    INDEPENDENT_ROLE_DUAL_TOWER,
    SHARED_TRUNK_ROLE_HEADS,
)


GPU_V3_ROLES = ("landlord", "landlord_up", "landlord_down")


def _tower(input_width, output_width, layers, dropout):
    modules = [nn.Linear(input_width, output_width), nn.LayerNorm(output_width), nn.GELU()]
    for _ in range(layers - 1):
        modules.extend([
            nn.Linear(output_width, output_width),
            nn.LayerNorm(output_width),
            nn.GELU(),
        ])
        if dropout:
            modules.append(nn.Dropout(dropout))
    return nn.Sequential(*modules)


class IndependentRoleDualTower(nn.Module):
    """Padded GPU batch model with independent state/action towers per role."""

    model_version = "gpu_v3"

    def __init__(self, state_width: int, action_width: int, config=None):
        super().__init__()
        self.config = config or GPUV3Config()
        if self.config.architecture != INDEPENDENT_ROLE_DUAL_TOWER:
            raise ValueError("IndependentRoleDualTower requires its matching architecture")
        if state_width < 1 or action_width < 1:
            raise ValueError("gpu_v3 input widths must be positive")
        self.state_width = int(state_width)
        self.action_width = int(action_width)
        cfg = self.config
        self.state_towers = nn.ModuleDict({
            role: _tower(state_width, cfg.hidden_size, cfg.trunk_layers, cfg.dropout)
            for role in GPU_V3_ROLES
        })
        self.action_towers = nn.ModuleDict({
            role: _tower(
                action_width,
                cfg.action_hidden_size,
                cfg.trunk_layers,
                cfg.dropout,
            )
            for role in GPU_V3_ROLES
        })
        fusion_width = cfg.hidden_size + cfg.action_hidden_size
        self.value_heads = nn.ModuleDict({
            role: nn.Sequential(
                _tower(fusion_width, cfg.hidden_size, cfg.role_head_layers, cfg.dropout),
                nn.Linear(cfg.hidden_size, 1),
            )
            for role in GPU_V3_ROLES
        })

    def forward(self, state, actions, action_mask, role_ids):
        if state.ndim != 2 or state.shape[1] != self.state_width:
            raise ValueError("invalid gpu_v3 state batch")
        if actions.ndim != 3 or actions.shape[2] != self.action_width:
            raise ValueError("invalid gpu_v3 action batch")
        if actions.shape[:2] != action_mask.shape or state.shape[0] != actions.shape[0]:
            raise ValueError("gpu_v3 batch dimensions do not match")
        if role_ids.shape != (state.shape[0],):
            raise ValueError("gpu_v3 role_ids must have shape [batch]")
        if not bool(((role_ids >= 0) & (role_ids < len(GPU_V3_ROLES))).all()):
            raise ValueError("gpu_v3 role id is out of range")

        batch, action_count = actions.shape[:2]
        values = state.new_empty(batch, action_count)
        for role_id, role in enumerate(GPU_V3_ROLES):
            rows = torch.nonzero(role_ids == role_id, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            role_state = self.state_towers[role](state.index_select(0, rows))
            role_actions = actions.index_select(0, rows)
            action_embedding = self.action_towers[role](role_actions)
            shared = role_state[:, None, :].expand(-1, action_count, -1)
            role_values = self.value_heads[role](
                torch.cat((shared, action_embedding), dim=-1)
            ).squeeze(-1)
            values.index_copy_(0, rows, role_values)
        return values.masked_fill(~action_mask, -torch.inf)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class SharedTrunkRoleHeads(nn.Module):
    """GPU-dense shared state/action towers with small role-specific heads."""

    model_version = "gpu_v3"

    def __init__(self, state_width: int, action_width: int, config=None):
        super().__init__()
        self.config = config or GPUV3Config(architecture=SHARED_TRUNK_ROLE_HEADS)
        if self.config.architecture != SHARED_TRUNK_ROLE_HEADS:
            raise ValueError("SharedTrunkRoleHeads requires its matching architecture")
        if state_width < 1 or action_width < 1:
            raise ValueError("gpu_v3 input widths must be positive")
        self.state_width = int(state_width)
        self.action_width = int(action_width)
        cfg = self.config
        self.state_tower = _tower(
            state_width, cfg.hidden_size, cfg.trunk_layers, cfg.dropout
        )
        self.action_tower = _tower(
            action_width,
            cfg.action_hidden_size,
            cfg.trunk_layers,
            cfg.dropout,
        )
        fusion_width = cfg.hidden_size + cfg.action_hidden_size
        self.role_heads = nn.ModuleDict({
            role: nn.Sequential(
                _tower(fusion_width, cfg.hidden_size, cfg.role_head_layers, cfg.dropout),
                nn.Linear(cfg.hidden_size, 1),
            )
            for role in GPU_V3_ROLES
        })

    def forward(self, state, actions, action_mask, role_ids):
        if state.ndim != 2 or state.shape[1] != self.state_width:
            raise ValueError("invalid gpu_v3 state batch")
        if actions.ndim != 3 or actions.shape[2] != self.action_width:
            raise ValueError("invalid gpu_v3 action batch")
        if actions.shape[:2] != action_mask.shape or state.shape[0] != actions.shape[0]:
            raise ValueError("gpu_v3 batch dimensions do not match")
        if role_ids.shape != (state.shape[0],):
            raise ValueError("gpu_v3 role_ids must have shape [batch]")
        if not bool(((role_ids >= 0) & (role_ids < len(GPU_V3_ROLES))).all()):
            raise ValueError("gpu_v3 role id is out of range")

        batch, action_count = actions.shape[:2]
        state_embedding = self.state_tower(state)
        action_embedding = self.action_tower(actions)
        fused = torch.cat((
            state_embedding[:, None, :].expand(-1, action_count, -1),
            action_embedding,
        ), dim=-1)
        values = state.new_empty(batch, action_count)
        for role_id, role in enumerate(GPU_V3_ROLES):
            rows = torch.nonzero(role_ids == role_id, as_tuple=False).flatten()
            if rows.numel():
                role_values = self.role_heads[role](
                    fused.index_select(0, rows)
                ).squeeze(-1)
                values.index_copy_(0, rows, role_values)
        return values.masked_fill(~action_mask, -torch.inf)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
