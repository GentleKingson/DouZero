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
    """Learned bid values plus landlord-perspective outcome auxiliaries.

    Each legal ``bid_logits`` entry is selected by argmax and behavior-trained
    as that physical bidder's eventual team-win logit. Explicit rule examples
    may also initialize their relative ranking with masked CE.
    """

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
        has_legal_action = self.bid_action_mask.any()
        if self.bid_action_mask.device.type == "cuda":
            torch._assert_async(
                has_legal_action, "bidding output must contain a legal action"
            )
        elif not bool(has_legal_action):
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
class BatchedBiddingOutput:
    """Batched learned-bidding values with a leading decision dimension."""

    bid_logits: torch.Tensor
    bid_action_mask: torch.Tensor
    landlord_win_logit: torch.Tensor
    expected_landlord_score: torch.Tensor
    uncertainty: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.bid_logits.ndim != 2 or self.bid_logits.shape[1] != 4:
            raise ValueError("batched bid_logits must have shape (B, 4)")
        batch_size = self.bid_logits.shape[0]
        if batch_size < 1:
            raise ValueError("batched bidding output must not be empty")
        if (
            self.bid_action_mask.shape != self.bid_logits.shape
            or self.bid_action_mask.dtype != torch.bool
        ):
            raise ValueError("batched bid_action_mask must be bool with shape (B, 4)")
        for name in ("landlord_win_logit", "expected_landlord_score"):
            if getattr(self, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape (B,)")
        if self.uncertainty is not None and self.uncertainty.shape != (batch_size,):
            raise ValueError("batched uncertainty must have shape (B,) when present")
        legal_rows = self.bid_action_mask.any(dim=1).all()
        if self.bid_action_mask.device.type == "cuda":
            torch._assert_async(
                legal_rows, "every batched bidding decision must contain a legal action"
            )
        elif not bool(legal_rows):
            raise ValueError(
                "every batched bidding decision must contain a legal action"
            )

    @property
    def batch_size(self) -> int:
        return int(self.bid_logits.shape[0])

    def masked_bid_logits(self) -> torch.Tensor:
        return self.bid_logits.masked_fill(~self.bid_action_mask, float("-inf"))

    def argmax_bids(self) -> torch.Tensor:
        """Return one legal bid index per row without synchronizing CUDA."""

        return torch.argmax(self.masked_bid_logits(), dim=1)

    def select(self, index: int) -> BiddingModelOutput:
        """Expose one row through the legacy scalar inference contract."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("bidding output index must be an int")
        if not 0 <= index < self.batch_size:
            raise IndexError("bidding output index is outside the batch")
        return BiddingModelOutput(
            bid_logits=self.bid_logits[index],
            bid_action_mask=self.bid_action_mask[index],
            landlord_win_logit=self.landlord_win_logit[index],
            expected_landlord_score=self.expected_landlord_score[index],
            uncertainty=(
                None if self.uncertainty is None else self.uncertainty[index]
            ),
        )


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

    def argmax_prior(self) -> int:
        """Index of the highest-prior-logit VALID action (P08 ablation)."""
        return int(torch.argmax(self.selected_prior_logit()).item())

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


@dataclass(frozen=True)
class BatchedModelOutput:
    """Model V2 output for padded decisions, with shape ``(B, A, 1)``."""

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
        if self.win_logit.ndim != 3 or self.win_logit.shape[-1] != 1:
            raise ValueError("batched win_logit must have shape (B, A, 1)")
        batch_actions = self.win_logit.shape[:2]
        if self.action_mask.shape != batch_actions or self.action_mask.dtype != torch.bool:
            raise ValueError("batched action_mask must be bool with shape (B, A)")
        legal_rows = self.action_mask.any(dim=1).all()
        if self.action_mask.device.type == "cuda":
            torch._assert_async(
                legal_rows, "every batched decision must contain a legal action"
            )
        elif not bool(legal_rows):
            raise ValueError("every batched decision must contain a legal action")
        names = ("score_if_win", "score_if_loss", "p_win", "score_mean")
        for name in names:
            if getattr(self, name).shape != self.win_logit.shape:
                raise ValueError(f"{name} shape must match win_logit")
        optional = (
            "prior_logit", "min_turns_after", "regain_initiative_logit",
            "teammate_finish_logit", "spring_probability_logit", "structure_cost",
        )
        for name in optional:
            value = getattr(self, name)
            if value is not None and value.shape != self.win_logit.shape:
                raise ValueError(f"{name} shape must match win_logit")

    def gather_chosen(self, indices: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Gather one legal action per decision without exposing padding."""
        if indices.shape != (self.win_logit.shape[0],) or indices.dtype != torch.long:
            raise ValueError("chosen indices must be long with shape (B,)")
        rows = torch.arange(indices.shape[0], device=indices.device)
        in_range = ((indices >= 0) & (indices < self.action_mask.shape[1])).all()
        if indices.device.type == "cuda":
            torch._assert_async(
                in_range, "chosen action index is outside the padded action range"
            )
            indices = indices.clamp(0, self.action_mask.shape[1] - 1)
            torch._assert_async(
                self.action_mask[rows, indices].all(),
                "chosen action index points at padding",
            )
        else:
            if not bool(in_range):
                raise ValueError("chosen action index is outside the padded action range")
            if not bool(self.action_mask[rows, indices].all()):
                raise ValueError("chosen action index points at padding")

        def gather(value):
            return None if value is None else value[rows, indices]

        return {
            "win_logit": gather(self.win_logit),
            "score_if_win": gather(self.score_if_win),
            "score_if_loss": gather(self.score_if_loss),
            "prior_logit": gather(self.prior_logit),
            "min_turns_after": gather(self.min_turns_after),
            "regain_initiative_logit": gather(self.regain_initiative_logit),
            "teammate_finish_logit": gather(self.teammate_finish_logit),
            "spring_probability_logit": gather(self.spring_probability_logit),
            "structure_cost": gather(self.structure_cost),
        }
