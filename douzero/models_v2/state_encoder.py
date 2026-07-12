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

Field-identity preservation (the key correctness property)
----------------------------------------------------------
The encoder MUST preserve which card set is which. Swapping ``my_hand`` and
``other_hand`` (or any two card fields) must change the state trunk — otherwise
the model cannot distinguish "cards I hold" from "cards the opponents hold",
which is a catastrophic imperfect-information error. A prior version summed
all card-set embeddings into one vector, which discarded field identity
entirely (the sum is invariant under any permutation of the summed fields).

This implementation keeps field identity by concatenating the per-field
embeddings in a FIXED schema order before projecting. The schema order is
fixed at construction time (the caller passes the field names in order), so
swapping two callers' inputs changes which embedding lands in which
concatenation slot, which changes the trunk.

The non-card state fields (one-hots, counts) are likewise kept in their fixed
schema order and concatenated, never summed.

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

from .card_encoder import CardSetEncoder


class StateEncoder(nn.Module):
    """Encode the V2 state card fields + flat context into one trunk vector.

    Parameters
    ----------
    card_vector_dim:
        Width of each card-count vector (54). The card-set fields each have
        this trailing width.
    num_card_fields:
        The EXACT number of card-set fields the encoder will receive, in a
        fixed order. Used to size the per-field projection. Must match the
        number of card vectors passed at construction / forward time.
    flat_context_width:
        Total width of the *non-card* state + public-context fields (the
        one-hot/count sub-blocks). Derived from the schema so a field-width
        change surfaces as a shape mismatch.
    hidden_size:
        Trunk width.
    """

    def __init__(
        self,
        card_vector_dim: int,
        num_card_fields: int,
        flat_context_width: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if card_vector_dim <= 0:
            raise ValueError(f"card_vector_dim must be positive, got {card_vector_dim}")
        if num_card_fields <= 0:
            raise ValueError(f"num_card_fields must be positive, got {num_card_fields}")
        if flat_context_width <= 0:
            raise ValueError(f"flat_context_width must be positive, got {flat_context_width}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.card_vector_dim = card_vector_dim
        self.num_card_fields = num_card_fields
        self.flat_context_width = flat_context_width
        self.hidden_size = hidden_size

        # One shared card-set projection (shared weights = "share general card
        # knowledge"), applied independently to each card field. Sharing does
        # NOT discard field identity: the outputs are CONCATENATED in fixed
        # field order, not summed.
        self.card_encoder = CardSetEncoder(card_vector_dim, hidden_size)
        # Concatenate: [card_field_0_emb, card_field_1_emb, ..., flat_context].
        # num_card_fields * hidden_size (preserves which field is which) +
        # flat_context_width (the non-card fields, also order-preserving).
        self.input_width = num_card_fields * hidden_size + flat_context_width
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
            Tuple of card-count tensors, each shape ``(card_vector_dim,)``
            float. Order is FIXED (the caller / model.py passes them in schema
            order). Swapping two entries changes the concatenation order and
            therefore changes the trunk — field identity is preserved.
        context_flat:
            Shape ``(flat_context_width,)`` float — the flattened non-card
            state + public-context fields, in their fixed schema order.

        Returns
        -------
        torch.Tensor
            Shape ``(hidden_size,)`` — the role-agnostic state trunk.
        """
        if context_flat.shape[-1] != self.flat_context_width:
            raise ValueError(
                f"context_flat trailing dim {context_flat.shape[-1]} != "
                f"flat_context_width {self.flat_context_width}"
            )
        if len(card_vectors) != self.num_card_fields:
            raise ValueError(
                f"expected {self.num_card_fields} card vectors, got "
                f"{len(card_vectors)}"
            )
        # Embed each card field with the SHARED projection. Each result keeps
        # its position in the list, so concatenation preserves field identity.
        embeddings = [self.card_encoder(cv) for cv in card_vectors]
        # Concatenate per-field embeddings + flat context. NOT a sum: a sum
        # would be invariant under field permutation and lose field identity.
        merged = torch.cat([*embeddings, context_flat.float()], dim=-1)
        return self.norm(self.proj(merged))
