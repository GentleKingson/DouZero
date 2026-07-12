"""Action encoder for Model V2 (P05).

Encodes each legal action's feature row (74-wide: 54 card counts + 16 move-type
one-hot + main_rank + length + is_pass + is_bomb) into a dense embedding.

The legacy model consumed the 54-wide card vector only (the action's cards);
the V2 encoder additionally consumes the structured move-type/rank/length/bomb
features so the model can distinguish "a single 3" from "a single Ace" beyond
their raw card counts, and can treat pass/bomb specially.

The encoder is role-agnostic and state-agnostic: it maps one action's feature
row to one embedding. State-action fusion (:mod:`douzero.models_v2.fusion`)
combines the per-action embedding with the shared state/history trunk. This
keeps the action encoder cheap and reusable, and matches the factorized
contract (P04): the action path runs once per candidate, the state path once
per decision.

Pass handling
-------------
A pass action has an all-zero card vector but ``is_pass=1``. The encoder does
NOT special-case it at the linear level (the ``is_pass`` feature carries the
signal); the spec mentions a "pass embedding" as an option, but a learned
linear projection over the structured feature row subsumes it without an extra
embedding table that could be missed for a rare action. This is a deliberate,
testable design choice.
"""

from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    """Project one action's feature row into the hidden space.

    Parameters
    ----------
    action_width:
        Width of the raw action feature row (74 for the canonical V2 schema).
        Derived from the schema, not hard-coded.
    hidden_size:
        Output embedding width.
    """

    def __init__(self, action_width: int, hidden_size: int) -> None:
        super().__init__()
        if action_width <= 0:
            raise ValueError(f"action_width must be positive, got {action_width}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.action_width = action_width
        self.hidden_size = hidden_size
        self.proj = nn.Linear(action_width, hidden_size)

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """Embed a batch of action feature rows.

        Parameters
        ----------
        action_features:
            Shape ``(N, action_width)`` float, one row per legal action. ``N``
            may be 1 (single legal action) or many.

        Returns
        -------
        torch.Tensor
            Shape ``(N, hidden_size)``.
        """
        if action_features.ndim != 2:
            raise ValueError(
                f"action_features must be 2-D (N, action_width), got shape {tuple(action_features.shape)}"
            )
        if action_features.shape[-1] != self.action_width:
            raise ValueError(
                f"action_features trailing dim {action_features.shape[-1]} != "
                f"action_width {self.action_width}"
            )
        return self.proj(action_features.float())
