"""Public-only padded torch.export wrapper for the H1 card-play model."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from douzero.models_v2.batch import ModelInputBundle

from .model import V3HybridModel
from .config import BELIEF_FEEDBACK_NONE


class ExportableV3HybridModel(nn.Module):
    """Tensor-only public wrapper with a fixed physical role."""

    def __init__(self, model: V3HybridModel, acting_role: str) -> None:
        super().__init__()
        if model.config.belief_feedback != BELIEF_FEEDBACK_NONE:
            raise ValueError(
                "H1 tensor export cannot omit an enabled H4 belief model"
            )
        if model.config.strategy_features_enabled or model.config.style_enabled:
            raise ValueError(
                "H1 tensor export cannot omit enabled H6 strategy/style inputs"
            )
        model.role_index(acting_role)
        self.model = model
        self.acting_role = acting_role

    def forward(
        self,
        state_cards: torch.Tensor,
        state_context_flat: torch.Tensor,
        context_cards: torch.Tensor,
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Call the tensor-only graph directly. Runtime container validation is
        # performed before this wrapper is invoked and must not become a
        # data-dependent Python guard inside torch.export.
        shared = self.model._shared_scalar(
            tuple(state_cards.unbind(0)),
            state_context_flat,
            tuple(context_cards.unbind(0)),
            context_flat,
            history_tokens,
            history_key_padding_mask,
            action_features,
            None,
            None,
        )
        adapted = self.model.role_adapters[self.acting_role](shared)
        output = self.model.role_heads[self.acting_role](adapted)
        torch._assert_async(
            action_mask.any(), "export input requires a valid action"
        )
        if self.model.config.nan_guard:
            for name, value in output.items():
                torch._assert_async(
                    torch.isfinite(value[action_mask]).all(),
                    f"exported V3 {name} contains NaN or Inf",
                )
        return (
            output["dmc_q"],
            output["win_logit"],
            output["score_if_win"],
            output["score_if_loss"],
            output["p_win"],
            output["score_mean"],
            action_mask,
        )


def padded_export_inputs(
    bundle: "ModelInputBundle", max_actions: int
) -> tuple[torch.Tensor, ...]:
    count, width = bundle.action_features.shape
    if max_actions < count:
        raise ValueError("max_actions must not truncate legal actions")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    actions = bundle.action_features.new_zeros((max_actions, width))
    mask = bundle.action_mask.new_zeros(max_actions)
    actions[:count] = bundle.action_features
    mask[:count] = bundle.action_mask
    return (
        torch.stack(bundle.state_card_vectors),
        bundle.state_context_flat,
        torch.stack(bundle.context_card_vectors),
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        actions,
        mask,
    )


def export_v3_hybrid_padded(
    model: V3HybridModel,
    bundle: "ModelInputBundle",
    path: str | Path,
    *,
    acting_role: str,
    max_actions: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> float:
    """Export, reload, and compare every public H1 output before publishing."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    inputs = padded_export_inputs(bundle, max_actions)
    training_modes = tuple(
        (module, module.training) for module in model.modules()
    )
    wrapper = ExportableV3HybridModel(model, acting_role)
    try:
        wrapper.eval()
        with torch.inference_mode():
            expected = wrapper(*inputs)
        program = torch.export.export(wrapper, inputs)
        torch.export.save(program, temporary)
        reloaded = torch.export.load(temporary).module()
        with torch.inference_mode():
            actual = reloaded(*inputs)
        aligned = all(
            torch.allclose(left, right, atol=atol, rtol=rtol)
            for left, right in zip(expected[:-1], actual[:-1])
        ) and torch.equal(expected[-1], actual[-1])
        if not aligned:
            raise RuntimeError("reloaded V3 export output mismatch")
        max_error = max(
            (float((left - right).abs().max().item())
             for left, right in zip(expected[:-1], actual[:-1])),
            default=0.0,
        )
        os.replace(temporary, output)
        return max_error
    finally:
        for module, training in training_modes:
            module.training = training
        temporary.unlink(missing_ok=True)
