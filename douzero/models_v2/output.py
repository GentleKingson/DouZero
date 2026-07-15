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
class BiddingModelOutput:
    """Learned bid output; outcome values use the landlord-side perspective."""

    bid_logits: torch.Tensor
    bid_action_mask: torch.Tensor
    landlord_win_logit: torch.Tensor
    expected_landlord_score: torch.Tensor
    uncertainty: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.bid_logits.shape != (4,):
            raise ValueError(
                f"bid_logits must have shape (4,), got {tuple(self.bid_logits.shape)}"
            )
        if self.bid_action_mask.shape != (4,) or self.bid_action_mask.dtype != torch.bool:
            raise ValueError("bid_action_mask must be bool with shape (4,)")
        if not bool(self.bid_action_mask.any()):
            raise ValueError("bidding output must contain a legal action")
        for name in ("landlord_win_logit", "expected_landlord_score"):
            if getattr(self, name).numel() != 1:
                raise ValueError(f"{name} must be scalar")
        if self.uncertainty is not None and self.uncertainty.numel() != 1:
            raise ValueError("uncertainty must be scalar when present")

    def masked_bid_logits(self) -> torch.Tensor:
        masked = self.bid_logits.clone()
        masked[~self.bid_action_mask] = float("-inf")
        return masked

    def argmax_bid(self) -> int:
        return int(torch.argmax(self.masked_bid_logits()).item())


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
    # P08: optional listwise policy-prior logits over the N legal actions. This
    # head is only present when ``ModelV2`` is built with
    # ``human_prior_enabled=True``; it is ``None`` otherwise (so a prior-
    # disabled checkpoint produces an output without a dummy tensor, and a
    # ``pure_prior`` decision mode fails loudly when the head is absent).
    prior_logit: torch.Tensor | None = None
    min_turns_after: torch.Tensor | None = None
    regain_initiative_logit: torch.Tensor | None = None
    teammate_finish_logit: torch.Tensor | None = None
    spring_probability_logit: torch.Tensor | None = None
    structure_cost: torch.Tensor | None = None

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
        # P08: validate the optional prior head shape when present. Require
        # EXACTLY 2D (N, 1) — a (N, 2, 1) tensor would pass a trailing-dim-only
        # check but is semantically wrong (round 5 non-blocking hardening).
        if self.prior_logit is not None:
            if self.prior_logit.ndim != 2 or self.prior_logit.shape != (n, 1):
                raise ValueError(
                    f"prior_logit must have shape ({n}, 1), got "
                    f"{tuple(self.prior_logit.shape)}"
                )
        aux_names = (
            "min_turns_after",
            "regain_initiative_logit",
            "teammate_finish_logit",
            "spring_probability_logit",
            "structure_cost",
        )
        present = [getattr(self, name) is not None for name in aux_names]
        if any(present) and not all(present):
            raise ValueError(
                "strategy auxiliary outputs must be all present or all absent"
            )
        if all(present):
            for name in aux_names:
                tensor = getattr(self, name)
                if tensor.ndim != 2 or tensor.shape != (n, 1):
                    raise ValueError(
                        f"{name} must have shape ({n}, 1), got {tuple(tensor.shape)}"
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

    def selected_prior_logit(self) -> torch.Tensor:
        """Return ``prior_logit`` squeezed to ``(N,)`` with padded rows -inf.

        For argmax-over-actions selection under the ``pure_prior`` decision mode
        (P08 ablation). Raises ``ValueError`` if the prior head is absent
        (``prior_logit is None``) — a prior-driven decision requires a model
        built with ``human_prior_enabled=True``.
        """
        if self.prior_logit is None:
            raise ValueError(
                "pure_prior selection requires a model built with "
                "human_prior_enabled=True; this ModelOutput has no prior head."
            )
        if not bool(self.action_mask.any()):
            raise ValueError("cannot select from zero valid actions")
        masked = self.prior_logit.squeeze(-1).clone()
        masked[~self.action_mask] = float("-inf")
        return masked

    def argmax_prior(self) -> int:
        """Index of the highest-prior-logit VALID action (P08 ablation)."""
        return int(torch.argmax(self.selected_prior_logit()).item())
