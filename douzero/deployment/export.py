"""Padded-action torch.export path with numerical alignment reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from douzero.models_v2.batch import ModelInputBundle
    from douzero.models_v2.model import ModelV2


@dataclass(frozen=True)
class ExportReport:
    """Machine-readable result of an export capability/alignment check."""

    success: bool
    backend: str
    max_actions: int
    atol: float
    rtol: float
    max_abs_error: float | None
    message: str


class ExportableModelV2(nn.Module):
    """Tensor-only wrapper with a fixed role and padded legal-action batch."""

    def __init__(self, model: "ModelV2", acting_role: str) -> None:
        super().__init__()
        unsupported = []
        if model.config.belief_enabled:
            unsupported.append("belief_enabled")
        if model.config.strategy_features_enabled:
            unsupported.append("strategy_features_enabled")
        if model.config.style_enabled:
            unsupported.append("style_enabled")
        if unsupported:
            raise ValueError(
                "torch.export wrapper does not yet support: " + ", ".join(unsupported)
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
        """Return stable tensor outputs for a padded action batch."""

        out = self.model(
            tuple(state_cards.unbind(0)),
            state_context_flat,
            tuple(context_cards.unbind(0)),
            context_flat,
            history_tokens,
            history_key_padding_mask,
            action_features,
            action_mask,
            self.acting_role,
        )
        return (
            out.win_logit,
            out.score_if_win,
            out.score_if_loss,
            out.p_win,
            out.score_mean,
            out.action_mask,
        )


def _padded_inputs(bundle, max_actions: int) -> tuple[torch.Tensor, ...]:
    n_actions, action_width = bundle.action_features.shape
    if n_actions > max_actions:
        raise ValueError(
            f"observation has {n_actions} actions, exceeding max_actions={max_actions}"
        )
    action_features = torch.zeros(
        (max_actions, action_width),
        dtype=bundle.action_features.dtype,
        device=bundle.action_features.device,
    )
    action_mask = torch.zeros(
        max_actions, dtype=torch.bool, device=bundle.action_features.device
    )
    action_features[:n_actions] = bundle.action_features
    action_mask[:n_actions] = bundle.action_mask
    return (
        torch.stack(bundle.state_card_vectors),
        bundle.state_context_flat,
        torch.stack(bundle.context_card_vectors),
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        action_features,
        action_mask,
    )


def export_padded_model(
    model: "ModelV2",
    bundle: "ModelInputBundle",
    output_path: str | Path,
    *,
    acting_role: str,
    max_actions: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ExportReport:
    """Export, reload, and numerically compare a padded-action Model V2."""

    output = Path(output_path)
    report_path = output.with_suffix(output.suffix + ".report.json")
    try:
        wrapper = ExportableModelV2(model.eval(), acting_role).eval()
        inputs = _padded_inputs(bundle, max_actions)
        with torch.inference_mode():
            expected = wrapper(*inputs)
        exported = torch.export.export(wrapper, inputs)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(exported, output)
        loaded = torch.export.load(output).module()
        with torch.inference_mode():
            actual = loaded(*inputs)
        errors = [
            float((lhs - rhs).abs().max().item())
            for lhs, rhs in zip(expected[:-1], actual[:-1])
        ]
        aligned = all(
            torch.allclose(lhs, rhs, atol=atol, rtol=rtol)
            for lhs, rhs in zip(expected[:-1], actual[:-1])
        ) and torch.equal(expected[-1], actual[-1])
        report = ExportReport(
            success=aligned,
            backend="torch.export",
            max_actions=max_actions,
            atol=atol,
            rtol=rtol,
            max_abs_error=max(errors, default=0.0),
            message="export aligned" if aligned else "export output mismatch",
        )
    except Exception as exc:  # capability reporting is intentionally non-fatal
        report = ExportReport(
            success=False,
            backend="torch.export",
            max_actions=max_actions,
            atol=atol,
            rtol=rtol,
            max_abs_error=None,
            message=f"{type(exc).__name__}: {exc}",
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return report
