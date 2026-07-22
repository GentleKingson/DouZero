"""Structured public-policy outputs for V3 Hybrid H1."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_VALUE_NAMES = (
    "dmc_q",
    "win_logit",
    "score_if_win",
    "score_if_loss",
    "p_win",
    "score_mean",
)

_OPTIONAL_ACTION_NAMES = (
    "prior_logit",
    "min_turns_after",
    "regain_initiative_logit",
    "teammate_finish_logit",
    "spring_probability_logit",
    "structure_cost",
)


@dataclass(frozen=True)
class V3HybridModelOutput:
    """One scalar per legal-action row for each independent value semantic."""

    dmc_q: torch.Tensor
    win_logit: torch.Tensor
    score_if_win: torch.Tensor
    score_if_loss: torch.Tensor
    p_win: torch.Tensor
    score_mean: torch.Tensor
    action_mask: torch.Tensor
    prior_logit: torch.Tensor | None = None
    min_turns_after: torch.Tensor | None = None
    regain_initiative_logit: torch.Tensor | None = None
    teammate_finish_logit: torch.Tensor | None = None
    spring_probability_logit: torch.Tensor | None = None
    structure_cost: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.dmc_q.ndim != 2 or self.dmc_q.shape[-1] != 1:
            raise ValueError("dmc_q must have shape (A, 1)")
        count = self.dmc_q.shape[0]
        if count < 1:
            raise ValueError("V3 output requires at least one action")
        for name in _VALUE_NAMES[1:]:
            if getattr(self, name).shape != self.dmc_q.shape:
                raise ValueError(f"{name} shape must match dmc_q")
        for name in _OPTIONAL_ACTION_NAMES:
            value = getattr(self, name)
            if value is not None and value.shape != self.dmc_q.shape:
                raise ValueError(f"{name} shape must match dmc_q")
        if self.action_mask.shape != (count,) or self.action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be bool with shape (A,)")
        if not bool(self.action_mask.any()):
            raise ValueError("V3 output requires at least one valid action")

    @property
    def num_actions(self) -> int:
        return int(self.dmc_q.shape[0])

    def masked(self, name: str) -> torch.Tensor:
        if name not in ("dmc_q", "win_logit", "score_mean"):
            raise ValueError(f"unsupported selection output {name!r}")
        values = getattr(self, name).squeeze(-1).clone()
        values[~self.action_mask] = float("-inf")
        return values

    def argmax(self, name: str = "dmc_q") -> int:
        return int(torch.argmax(self.masked(name)).item())


@dataclass(frozen=True)
class BatchedV3HybridModelOutput:
    """Padded H1 output with shape ``(B, A, 1)`` and an authoritative mask."""

    dmc_q: torch.Tensor
    win_logit: torch.Tensor
    score_if_win: torch.Tensor
    score_if_loss: torch.Tensor
    p_win: torch.Tensor
    score_mean: torch.Tensor
    action_mask: torch.Tensor
    prior_logit: torch.Tensor | None = None
    min_turns_after: torch.Tensor | None = None
    regain_initiative_logit: torch.Tensor | None = None
    teammate_finish_logit: torch.Tensor | None = None
    spring_probability_logit: torch.Tensor | None = None
    structure_cost: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.dmc_q.ndim != 3 or self.dmc_q.shape[-1] != 1:
            raise ValueError("batched dmc_q must have shape (B, A, 1)")
        batch, actions = self.dmc_q.shape[:2]
        if batch < 1 or actions < 1:
            raise ValueError("batched V3 output must not be empty")
        for name in _VALUE_NAMES[1:]:
            if getattr(self, name).shape != self.dmc_q.shape:
                raise ValueError(f"{name} shape must match dmc_q")
        for name in _OPTIONAL_ACTION_NAMES:
            value = getattr(self, name)
            if value is not None and value.shape != self.dmc_q.shape:
                raise ValueError(f"{name} shape must match dmc_q")
        if self.action_mask.shape != (batch, actions):
            raise ValueError("batched action_mask must have shape (B, A)")
        if self.action_mask.dtype != torch.bool:
            raise ValueError("batched action_mask must have bool dtype")
        if not bool(self.action_mask.any(dim=1).all()):
            raise ValueError("each decision must contain a valid action")

    @property
    def batch_size(self) -> int:
        return int(self.dmc_q.shape[0])

    def select(self, index: int) -> V3HybridModelOutput:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an int")
        if not 0 <= index < self.batch_size:
            raise IndexError("batch index is outside the output")
        optional = {
            name: None if getattr(self, name) is None else getattr(self, name)[index]
            for name in _OPTIONAL_ACTION_NAMES
        }
        return V3HybridModelOutput(
            **{name: getattr(self, name)[index] for name in _VALUE_NAMES},
            action_mask=self.action_mask[index],
            **optional,
        )

    def gather_chosen(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if indices.shape != (self.batch_size,) or indices.dtype != torch.long:
            raise ValueError("chosen indices must be long with shape (B,)")
        rows = torch.arange(self.batch_size, device=indices.device)
        if not bool(
            ((indices >= 0) & (indices < self.action_mask.shape[1])).all()
        ):
            raise ValueError("chosen index is outside padded action range")
        if not bool(self.action_mask[rows, indices].all()):
            raise ValueError("chosen index references a padded action")
        gathered = {
            name: getattr(self, name)[rows, indices]
            for name in _VALUE_NAMES
        }
        gathered.update({
            name: getattr(self, name)[rows, indices]
            for name in _OPTIONAL_ACTION_NAMES
            if getattr(self, name) is not None
        })
        return gathered
