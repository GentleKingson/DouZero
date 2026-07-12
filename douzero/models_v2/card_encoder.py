"""Card / hand encoders for Model V2 (P05).

The legacy observation encodes every card set (a hand, a played pile, a move)
as a 54-wide int8 *count* vector: slot ``i`` holds how many copies of canonical
card ``i`` are present (``douzero.observation.cards.cards_to_vector``). This
module turns those raw count vectors into dense embeddings.

Two encoders are exposed:

- :class:`CardSetEncoder`: one card set -> one embedding vector. Used for the
  fixed state sub-blocks (my hand, each role's played pile, the last move) and
  for the public bottom-card identity.
- :class:`MultiCardSetEncoder`: a stack of card-set vectors -> a stack of
  embeddings, sharing the same projection. Used by the state encoder to fold
  several card sets into the state trunk without re-instantiating weights.

The projection is a single ``Linear`` layer (no nonlinearity at the encoder
boundary — the fusion stack applies normalization and nonlinearity). Count
inputs are small integers (0..4), so a linear projection is sufficient and
keeps the parameter count modest. The encoder is role-agnostic; role
differentiation happens in the fusion via the role embedding.
"""

from __future__ import annotations

import torch
from torch import nn


class CardSetEncoder(nn.Module):
    """Project a single 54-wide card-count vector into the hidden space.

    Parameters
    ----------
    card_vector_dim:
        Width of the raw count vector (54 for the canonical encoding). This is
        pulled from the schema, not hard-coded.
    hidden_size:
        Output embedding width (the model trunk width).
    """

    def __init__(self, card_vector_dim: int, hidden_size: int) -> None:
        super().__init__()
        if card_vector_dim <= 0:
            raise ValueError(f"card_vector_dim must be positive, got {card_vector_dim}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.card_vector_dim = card_vector_dim
        self.hidden_size = hidden_size
        self.proj = nn.Linear(card_vector_dim, hidden_size)

    def forward(self, card_vector: torch.Tensor) -> torch.Tensor:
        """Project a card-count tensor into the hidden space.

        Parameters
        ----------
        card_vector:
            Shape ``(..., card_vector_dim)`` float. The trailing dim is the
            count vector; any leading dims are preserved (batched or unbatched).

        Returns
        -------
        torch.Tensor
            Shape ``(..., hidden_size)``.
        """
        if card_vector.shape[-1] != self.card_vector_dim:
            raise ValueError(
                f"card_vector trailing dim {card_vector.shape[-1]} != "
                f"card_vector_dim {self.card_vector_dim}"
            )
        return self.proj(card_vector.float())


class MultiCardSetEncoder(nn.Module):
    """Project several card-count vectors through a SHARED projection.

    Used by the state encoder to embed my-hand / played-piles / last-move /
    bottom-cards with one set of weights, then combine them. Sharing is the
    "share general card knowledge" requirement (AGENTS.md "Model rules"):
    a single concept of "what does this set of cards look like" is learned
    once and reused everywhere.
    """

    def __init__(self, card_vector_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = CardSetEncoder(card_vector_dim, hidden_size)

    @property
    def card_vector_dim(self) -> int:
        return self.encoder.card_vector_dim

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    def forward(self, *card_vectors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Embed each card vector with the shared projection.

        All inputs must share their trailing dim (``card_vector_dim``) but may
        differ in leading dims. Returns one embedding per input, in order.
        """
        if not card_vectors:
            raise ValueError("MultiCardSetEncoder.forward requires at least one input")
        return tuple(self.encoder(cv) for cv in card_vectors)
