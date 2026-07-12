"""State encoder for Model V2 (P05).

Encodes the per-decision *state* into a single trunk vector, ONCE per decision
(not per legal action). This is the V2 analogue of the P04 factorized
``x_state_single`` path, extended to consume the structured V2 schema blocks
rather than the legacy flat vector.

Inputs (all from :class:`~douzero.observation.encode_v2.ObservationV2`):

- :class:`StateBlock` — 11 sub-arrays: my hand, the public unseen pool, each
  role's played pile, the last move, three remaining-count one-hots, the bomb
  counter one-hot, and the acting-role one-hot.
- :class:`PublicContextBlock` — 7 sub-arrays: bottom-card revealed/unplayed
  identity, the final bid one-hot, the phase one-hot, the rocket count, the
  total multiplier (int32), and the ruleset-id one-hot.

The encoder embeds the card-set sub-blocks through the shared
:class:`~douzero.models_v2.card_encoder.MultiCardSetEncoder`, flattens the
small one-hot/count sub-blocks, and projects the concatenation into the trunk
width via a small MLP with residual + LayerNorm.

Role conditioning
-----------------
The acting role is ONE of the state sub-blocks (an 6-wide one-hot). The encoder
does not apply a separate role embedding here — that lives in the fusion
(:mod:`douzero.models_v2.fusion`) as a learned ``role_embedding`` table, so the
state trunk stays role-agnostic and the role signal is injected where it
modulates state-action scoring (AGENTS.md: preserve landlord/farmer positional
differences through role embeddings or heads).
"""

from __future__ import annotations

import torch
from torch import nn

from .card_encoder import MultiCardSetEncoder


class StateEncoder(nn.Module):
    """Encode the V2 state block + public context into one trunk vector.

    Parameters
    ----------
    card_vector_dim:
        Width of each card-count vector (54). The card-set sub-blocks each have
        this trailing width.
    context_width:
        Total width of the *non-card* state fields (the one-hot/count sub-blocks
        of :class:`StateBlock` and :class:`PublicContextBlock`). This is derived
        from the schema so a field-width change surfaces as a shape mismatch.
    hidden_size:
        Trunk width.
    """

    def __init__(self, card_vector_dim: int, context_width: int, hidden_size: int) -> None:
        super().__init__()
        if card_vector_dim <= 0:
            raise ValueError(f"card_vector_dim must be positive, got {card_vector_dim}")
        if context_width <= 0:
            raise ValueError(f"context_width must be positive, got {context_width}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.card_vector_dim = card_vector_dim
        self.context_width = context_width
        self.hidden_size = hidden_size

        self.card_encoder = MultiCardSetEncoder(card_vector_dim, hidden_size)
        # Project the concatenation of (card-set embeddings + flat context)
        # into the trunk. There is no residual shortcut here because the input
        # width differs from hidden_size; the fusion stack carries residuals.
        self.input_width = hidden_size + context_width
        self.proj = nn.Linear(self.input_width, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the state into a single trunk vector.

        Parameters
        ----------
        card_vectors:
            Tuple of card-count tensors, each shape ``(card_vector_dim,)`` float.
            Order is fixed by the caller (the model) and must match the order
            used to compute ``context_width`` at construction. All are embedded
            through the shared card projection and summed.
        context_flat:
            Shape ``(context_width,)`` float — the flattened non-card state +
            public-context fields.

        Returns
        -------
        torch.Tensor
            Shape ``(hidden_size,)`` — the role-agnostic state trunk.
        """
        if context_flat.shape[-1] != self.context_width:
            raise ValueError(
                f"context_flat trailing dim {context_flat.shape[-1]} != "
                f"context_width {self.context_width}"
            )
        if not card_vectors:
            raise ValueError("StateEncoder.forward requires at least one card vector")
        embeddings = self.card_encoder(*card_vectors)
        # Sum the card-set embeddings into one hidden-wide vector. Summing (vs
        # concatenating) keeps the trunk width fixed regardless of how many
        # card sets the schema carries, so adding a future public field does
        # not change the head input width.
        card_summary = torch.stack(embeddings, dim=0).sum(dim=0)
        merged = torch.cat([card_summary, context_flat.float()], dim=-1)
        return self.norm(self.proj(merged))
