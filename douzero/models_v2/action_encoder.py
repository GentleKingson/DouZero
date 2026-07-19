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

    def __init__(
        self,
        action_width: int,
        hidden_size: int,
        strategy_width: int = 0,
    ) -> None:
        super().__init__()
        if action_width <= 0:
            raise ValueError(f"action_width must be positive, got {action_width}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if strategy_width < 0:
            raise ValueError(f"strategy_width must be non-negative, got {strategy_width}")
        self.action_width = action_width
        self.hidden_size = hidden_size
        self.strategy_width = strategy_width
        self.proj = nn.Linear(action_width + strategy_width, hidden_size)

    def forward(
        self,
        action_features: torch.Tensor,
        strategy_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if action_features.ndim not in (2, 3):
            raise ValueError(
                "action_features must be (N, action_width) or "
                f"(B, A, action_width), got shape {tuple(action_features.shape)}"
            )
        if action_features.shape[-1] != self.action_width:
            raise ValueError(
                f"action_features trailing dim {action_features.shape[-1]} != "
                f"action_width {self.action_width}"
            )
        if self.strategy_width == 0:
            if strategy_features is not None:
                raise ValueError(
                    "strategy_features were passed to a strategy-disabled ActionEncoder"
                )
            combined = action_features.float()
        else:
            if strategy_features is None:
                raise ValueError(
                    "strategy_features are required by a strategy-enabled ActionEncoder"
                )
            expected = (*action_features.shape[:-1], self.strategy_width)
            if tuple(strategy_features.shape) != expected:
                raise ValueError(
                    f"strategy_features must have shape {expected}, got "
                    f"{tuple(strategy_features.shape)}"
                )
            combined = torch.cat(
                [
                    action_features.float(),
                    strategy_features.to(
                        device=action_features.device,
                        dtype=torch.float32,
                    ),
                ],
                dim=-1,
            )
        return self.proj(combined)
