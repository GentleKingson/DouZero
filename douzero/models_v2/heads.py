"""Value heads for Model V2 (P05).

Produces the multi-head output from the fused action representations. The heads
follow the project's sign convention (AGENTS.md "Rewards, targets, and action
selection"):

- ``win_logit`` -> ``p_win = sigmoid(win_logit)`` is the win probability from
  the *current acting player's team* perspective. A farmer win is positive for
  both farmer roles.
- ``score_if_win`` is the conditional final signed score from the acting team's
  perspective, given a win. Supervised only on won-episode samples (P06).
- ``score_if_loss`` is the conditional final signed score given a loss.
  Supervised only on lost-episode samples (P06).
- ``score_mean`` is a derived convenience: the model's expected score under its
  own win probability. It is NOT an independent head (no separate loss); it
  exists so a decision policy can read a single field.

Sign convention is centralized here: all score outputs are "acting-team
perspective, positive = good for the acting team". The loss module (P06) is
responsible for converting terminal labels into this perspective before
computing the head losses; the heads themselves are perspective-agnostic.

Numerical stability
-------------------
The score heads are clamped to ``[-score_clamp, score_clamp]`` so a wild
initialization cannot emit Inf that would poison the multi-objective loss. The
clamp is applied to the raw head output (a finite linear projection), so it is
a no-op for well-behaved weights and a safety net for the tail.
"""

from __future__ import annotations

import torch
from torch import nn


class ValueHeads(nn.Module):
    """Multi-head value output over N fused action representations.

    Parameters
    ----------
    hidden_size:
        Width of the fused action representations.
    score_clamp:
        Symmetric clamp magnitude applied to the score heads (finite-output
        safety net). Must be positive.
    """

    def __init__(self, hidden_size: int, score_clamp: float = 32.0) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if score_clamp <= 0.0:
            raise ValueError(f"score_clamp must be positive, got {score_clamp}")
        self.hidden_size = hidden_size
        self.score_clamp = float(score_clamp)

        # Win head: a single logit per action. BCEWithLogitsLoss trains it.
        self.win_head = nn.Linear(hidden_size, 1)
        # Conditional score heads: a scalar per action, clamped after projection.
        # Separate heads (not one head + a sign flag) so win/loss can have
        # different magnitude tails (a big-bomb loss is a different shape than
        # a big-bomb win).
        self.score_win_head = nn.Linear(hidden_size, 1)
        self.score_loss_head = nn.Linear(hidden_size, 1)

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score N fused action representations.

        Parameters
        ----------
        fused:
            Shape ``(N, hidden_size)`` — the role-conditioned fused action
            representations from :class:`~douzero.models_v2.fusion.StateActionFusion`.

        Returns
        -------
        dict
            Keys: ``win_logit`` (N, 1), ``score_if_win`` (N, 1),
            ``score_if_loss`` (N, 1), ``p_win`` (N, 1) = sigmoid(win_logit),
            ``score_mean`` (N, 1) = p_win * score_if_win + (1-p_win) * score_if_loss.
            All values are finite by construction (score heads are clamped).
        """
        if fused.shape[-1] != self.hidden_size:
            raise ValueError(
                f"fused trailing dim {fused.shape[-1]} != hidden_size {self.hidden_size}"
            )

        win_logit = self.win_head(fused)  # (N, 1)
        score_if_win = torch.clamp(
            self.score_win_head(fused), -self.score_clamp, self.score_clamp
        )
        score_if_loss = torch.clamp(
            self.score_loss_head(fused), -self.score_clamp, self.score_clamp
        )

        p_win = torch.sigmoid(win_logit)
        # Detach p_win from the score-mean computation: score_mean is a derived
        # readout for the decision policy, NOT a loss target. The conditional
        # heads are trained by their own masked losses (P06); mixing the p_win
        # gradient back through the conditional heads would couple them.
        score_mean = p_win.detach() * score_if_win + (1.0 - p_win.detach()) * score_if_loss

        return {
            "win_logit": win_logit,
            "score_if_win": score_if_win,
            "score_if_loss": score_if_loss,
            "p_win": p_win,
            "score_mean": score_mean,
        }


class PriorHead(nn.Module):
    """Listwise policy-prior head over N fused action representations (P08).

    Produces one prior logit per legal action: ``prior_logit`` has shape
    ``(N, 1)``. The head is trained by a listwise cross-entropy over the N
    legal actions against the recorded human action index (behaviour cloning).
    It is the imperfect-information-safe way to inject a human-play prior: the
    head scores the *current* legal-action list (variable N), never a global
    action class id, exactly as AGENTS.md requires.

    The head reads only the fused action representation (derived from the
    public observation). It never sees hidden hands or the privileged human
    label at inference; the label is consumed only by the BC loss.

    Parameters
    ----------
    hidden_size:
        Width of the fused action representations. Must be positive.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = hidden_size
        self.prior_head = nn.Linear(hidden_size, 1)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Return the per-action prior logits, shape ``(N, 1)``.

        The logits are raw (no log-softmax); the listwise cross-entropy loss
        applies ``F.cross_entropy`` directly, which is numerically stable. The
        decision policy masks padded rows before argmax.
        """
        if fused.shape[-1] != self.hidden_size:
            raise ValueError(
                f"fused trailing dim {fused.shape[-1]} != hidden_size "
                f"{self.hidden_size}"
            )
        return self.prior_head(fused)
