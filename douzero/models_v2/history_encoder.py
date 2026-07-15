"""History encoders for Model V2 (P05).

Encodes the bounded public action-history token sequence into a single context
embedding per decision. Two backends are provided:

- :class:`TransformerHistoryEncoder`: a small ``nn.TransformerEncoder`` stack.
  This is the default. It uses an explicit padding mask so padded history
  slots never affect the output (AGENTS.md: "Masked actions or history tokens
  must not affect valid outputs").
- :class:`LSTMHistoryEncoder`: a unidirectional LSTM over the token sequence,
  using the last non-padded timestep as the summary. Provided as a lighter
  fallback (``history_encoder: lstm``).

Both backends:

- encode each token (``token_width`` int8 features) into ``hidden_size`` via a
  linear projection,
- add a learned positional embedding (the Transformer needs positions; the LSTM
  consumes order implicitly, but the positional embedding is harmless and keeps
  the two backends interchangeable),
- apply a ``LayerNorm`` before returning (stabilizes the trunk input),
- return a single ``(hidden_size,)`` summary vector for the sequence.

Masking contract
----------------
The encoder accepts ``key_padding_mask`` in the PyTorch convention: a boolean
tensor of shape ``(max_history_len,)`` where ``True`` means *padding* (ignore).
This is exactly :attr:`HistoryTokenBatch.key_padding_mask`, so no conversion is
needed at the call site. With an all-padding sequence (no history yet, e.g. the
first move of a game) both backends fall back to the positional embedding of
the first slot rather than producing NaNs.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import HISTORY_ENCODER_LSTM, HISTORY_ENCODER_TRANSFORMER


class _HistoryBase(nn.Module):
    """Shared token projection + positional embedding + output norm."""

    def __init__(self, token_width: int, hidden_size: int, max_history_len: int) -> None:
        super().__init__()
        if token_width <= 0:
            raise ValueError(f"token_width must be positive, got {token_width}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if max_history_len <= 0:
            raise ValueError(f"max_history_len must be positive, got {max_history_len}")
        self.token_width = token_width
        self.hidden_size = hidden_size
        self.max_history_len = max_history_len

        self.input_proj = nn.Linear(token_width, hidden_size)
        # Learned absolute positions. Even the LSTM benefits from an explicit
        # position signal because it summarises via the last step and a purely
        # content-based summary loses recency ordering under padding.
        self.pos_embed = nn.Embedding(max_history_len, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project tokens ``(S, token_width)`` -> ``(S, hidden_size)`` + positions."""
        if tokens.shape[-1] != self.token_width:
            raise ValueError(
                f"tokens trailing dim {tokens.shape[-1]} != token_width {self.token_width}"
            )
        if tokens.shape[0] != self.max_history_len:
            raise ValueError(
                f"tokens seq len {tokens.shape[0]} != max_history_len {self.max_history_len}"
            )
        positions = torch.arange(self.max_history_len, device=tokens.device)
        return self.input_proj(tokens.float()) + self.pos_embed(positions)


class TransformerHistoryEncoder(_HistoryBase):
    """Transformer encoder over the history token sequence.

    Parameters
    ----------
    token_width, hidden_size, max_history_len:
        See :class:`_HistoryBase`.
    num_layers, num_heads, dropout:
        Transformer stack depth, attention head count, and dropout. Dropout is
        0 by default (deterministic inference); it is a no-op under
        ``model.eval()``.
    """

    def __init__(
        self,
        token_width: int,
        hidden_size: int,
        max_history_len: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(token_width, hidden_size, max_history_len)
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.num_layers = num_layers
        self.num_heads = num_heads

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            # batch_first so the tensor shape is (S, H) with no batch dim, which
            # matches the per-decision encoder contract (state/history encoded
            # once per decision, not per legal action).
            batch_first=False,
            # ReLU/GELU are both fine; ReLU matches the legacy MLP activations.
            activation="relu",
            # LayerNorm in the pre-norm position for training stability.
            norm_first=True,
        )
        # enable_nested_tensor=False silences a cosmetic PyTorch UserWarning
        # ("enable_nested_tensor is True, but self.use_nested_tensor is False
        # because encoder_layer.norm_first was True"). The nested-tensor path
        # is never taken under norm_first=True, so disabling it explicitly is
        # both correct and keeps test output clean.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        # Learn how to pool the sequence into a single vector. A query-less
        # mean-pool over valid tokens is the safe default; a learned attention
        # pool is overkill for P05 and adds a NaN risk under all-padding input.
        # We do masked mean-pool in forward() instead.

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode the history sequence into a single summary vector.

        Parameters
        ----------
        tokens:
            Shape ``(max_history_len, token_width)`` float. Padded slots are
            expected to be all-zero (the encoder contract).
        key_padding_mask:
            Shape ``(max_history_len,)`` bool, ``True`` for PADDING (PyTorch
            convention). May be all-True (no real history).

        Returns
        -------
        torch.Tensor
            Shape ``(hidden_size,)``. The masked-mean summary of the encoded
            sequence, LayerNorm-stabilized.
        """
        embedded = self._embed_tokens(tokens)  # (S, H)
        # Transformer expects (S, H) with batch_first=False; here "batch" is the
        # single decision. key_padding_mask is (S,) -> broadcast to (1, S).
        kpm = key_padding_mask.reshape(1, -1)
        encoded = self.transformer(embedded.unsqueeze(1), mask=None,
                                   src_key_padding_mask=kpm)
        encoded = encoded.squeeze(1)  # (S, H)

        # Masked mean over valid positions. Avoid division by zero when every
        # slot is padding (first move of a game): fall back to the first slot's
        # positional embedding so the output stays finite and well-defined.
        valid = (~key_padding_mask).to(encoded.dtype)  # (S,), 1 for real
        denom = valid.sum()
        # Keep the empty-history branch tensor-native so torch.export can trace
        # it without guarding on ``Tensor.item()``. clamp_min prevents a zero
        # divisor; torch.where selects the deterministic empty token exactly
        # when every position is padding.
        mean = (encoded * valid.unsqueeze(-1)).sum(dim=0) / denom.clamp_min(1.0)
        pooled = torch.where(denom > 0, mean, embedded[0])
        return self.out_norm(pooled)


class LSTMHistoryEncoder(_HistoryBase):
    """LSTM history encoder (lighter fallback).

    Runs a unidirectional LSTM over the token sequence and takes the hidden
    state at the last REAL position as the summary. Padded slots are zeroed in
    the input so they carry no information, and the summary is read from the
    final valid step rather than the final slot.
    """

    def __init__(
        self,
        token_width: int,
        hidden_size: int,
        max_history_len: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(token_width, hidden_size, max_history_len)
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode the history sequence into a single summary vector.

        See :meth:`TransformerHistoryEncoder.forward` for the contract. The LSTM
        summary is the hidden state at the last valid position.
        """
        embedded = self._embed_tokens(tokens)  # (S, H)
        # Zero out padded slots before the LSTM so they cannot inject the
        # positional embedding of a padding position into the recurrent state.
        valid = (~key_padding_mask).to(embedded.dtype).unsqueeze(-1)  # (S, 1)
        embedded = embedded * valid
        # (S, H) -> (S, 1, H) for batch_first=False (batch dim is 1 = one decision).
        out, (h_n, _) = self.lstm(embedded.unsqueeze(1))
        out = out.squeeze(1)  # (S, H)

        # Summary: hidden state at the last valid position. If all padding,
        # fall back to slot 0's output (which is positional-only here).
        valid_counts = valid.squeeze(-1)
        last_valid_index = (valid_counts.sum().to(torch.long) - 1).clamp_min(0)
        pooled = out[last_valid_index]
        return self.out_norm(pooled)


def build_history_encoder(
    token_width: int,
    hidden_size: int,
    max_history_len: int,
    *,
    backend: str = HISTORY_ENCODER_TRANSFORMER,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.0,
) -> nn.Module:
    """Factory selecting the history encoder backend.

    Raises ``ValueError`` for an unknown backend so a config typo fails loudly
    at construction rather than producing a silent default.
    """
    if backend == HISTORY_ENCODER_TRANSFORMER:
        return TransformerHistoryEncoder(
            token_width=token_width,
            hidden_size=hidden_size,
            max_history_len=max_history_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    if backend == HISTORY_ENCODER_LSTM:
        return LSTMHistoryEncoder(
            token_width=token_width,
            hidden_size=hidden_size,
            max_history_len=max_history_len,
            num_layers=max(num_layers, 1),
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown history encoder backend {backend!r}. "
        f"Supported: {sorted([HISTORY_ENCODER_TRANSFORMER, HISTORY_ENCODER_LSTM])}"
    )
