"""Structured model output for Model V2 (P05).

A :class:`ModelOutput` is the typed return value of
:class:`~douzero.models_v2.model.ModelV2.forward`. It carries the multi-head
tensors plus the legal-action mask used for selection, so a decision policy
(P06) or an evaluation harness can read everything it needs from one object
rather than unpacking a dict by string keys.

The tensors are raw (no softmax); consumers apply masks and reductions as
needed. This keeps the model output faithful to what the heads produced and
makes the decision policy explicit about how it converts logits to a choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOutput:
    """Typed multi-head output for one decision.

    All tensors share their leading dim ``N`` (the number of legal actions, or
    the padded batch size). ``win_logit`` / ``score_if_win`` / ``score_if_loss``
    / ``p_win`` / ``score_mean`` have shape ``(N, 1)``. ``action_mask`` has
    shape ``(N,)`` and is ``True`` for a valid action, ``False`` for padding.

    A padded action batch (N > num real actions) is supported so variable
    legal-action counts can be batched across decisions. Padded rows must be
    masked out before selection (the model does NOT guarantee finite outputs on
    fully-masked rows — see the head clamp for partial safety, but selection
    must respect the mask).
    """

    win_logit: torch.Tensor
    score_if_win: torch.Tensor
    score_if_loss: torch.Tensor
    p_win: torch.Tensor
    score_mean: torch.Tensor
    action_mask: torch.Tensor

    def __post_init__(self) -> None:
        n = self.win_logit.shape[0]
        # Bug #6: a ModelOutput with zero action rows is invalid. The model
        # must never produce one (forward rejects zero actions), and a caller
        # must never construct one (there is nothing to select from).
        if n == 0:
            raise ValueError(
                "ModelOutput cannot have zero action rows; a decision with no "
                "legal actions is undefined."
            )
        for name in ("score_if_win", "score_if_loss", "p_win", "score_mean"):
            t = getattr(self, name)
            if t.shape[0] != n or t.shape[-1] != 1:
                raise ValueError(
                    f"{name} must have shape ({n}, 1), got {tuple(t.shape)}"
                )
        if self.win_logit.shape[-1] != 1:
            raise ValueError(
                f"win_logit must have trailing dim 1, got {self.win_logit.shape[-1]}"
            )
        if self.action_mask.shape != (n,):
            raise ValueError(
                f"action_mask must have shape ({n},), got {tuple(self.action_mask.shape)}"
            )
        if self.action_mask.dtype != torch.bool:
            raise ValueError(
                f"action_mask must be bool, got {self.action_mask.dtype}"
            )

    @property
    def num_actions(self) -> int:
        """The (padded) number of action rows."""
        return self.win_logit.shape[0]

    def selected_win_logit(self) -> torch.Tensor:
        """Return ``win_logit`` with padded rows set to -inf.

        For argmax-over-actions selection (the deterministic decision policy).
        Raises if there are zero valid actions (a caller error — a decision
        with no legal actions is undefined).
        """
        if not bool(self.action_mask.any()):
            raise ValueError("cannot select from zero valid actions")
        masked = self.win_logit.squeeze(-1).clone()
        masked[~self.action_mask] = float("-inf")
        return masked

    def argmax_win(self) -> int:
        """Index of the highest-win-probability VALID action."""
        return int(torch.argmax(self.selected_win_logit()).item())
