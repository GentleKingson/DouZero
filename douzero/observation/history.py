"""History token sequence for observation V2 (P03).

The legacy encoder keeps only the last 15 actions, packed into a ``5×162``
matrix fed to an LSTM. P03 requires a configurable, full-history token sequence
with an explicit padding mask, so a Transformer / belief model can consume the
complete public action history.

Each token is a flat vector assembled from the schema's ``history_token_fields``
(see :mod:`douzero.observation.schema`). The required fields per the P03 spec
are: ``actor_role``, ``is_pass``, ``move_type``, ``main_rank``, ``length``,
``card_count``, ``cards_encoding``, ``cards_left_after``, ``bomb_flag``,
``phase``, plus a ``valid`` padding mask.

A :class:`HistoryTokenBatch` is a pure container of numpy arrays; it holds no
privileged information. The sequence is right-padded (real tokens first, then
zero padding) and the mask marks which entries are real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cards import cards_to_vector
from .schema import (
    BOMB_ONEHOT_WIDTH,
    CARD_VECTOR_DIM,
    MOVE_TYPE_ONEHOT_WIDTH,
    SEAT_ONEHOT_WIDTH,
    FeatureSchemaManifest,
)
from .seats import ALL_ROLES

#: Game-phase integer codes used inside history tokens. These mirror the
#: string phases in ``douzero.env.rules`` but are stored as compact ints.
PHASE_CODE_BIDDING: int = 0
PHASE_CODE_PLAYING: int = 1
PHASE_CODE_REVEAL_BOTTOM: int = 2


@dataclass(frozen=True)
class HistoryMove:
    """One decoded public action, the raw material for one history token.

    ``actor_role`` is the absolute role label of the player who made the move.
    ``cards`` is the move (empty list = pass). ``cards_left_after`` is that
    actor's remaining-card count after the move. ``is_bomb`` flags bombs and the
    rocket. ``phase`` is the integer phase code in effect when the move was
    made. All fields are public.
    """

    actor_role: str
    cards: tuple[int, ...]
    is_pass: bool
    move_type: int
    main_rank: int
    length: int
    card_count: int
    cards_left_after: int
    is_bomb: bool
    phase: int


@dataclass(frozen=True)
class HistoryTokenBatch:
    """Encoded history-token tensor + padding mask.

    ``tokens`` has shape ``(max_history_len, token_width)`` int8 (read-only).
    Padding slots are all-zero.

    Mask conventions (item 6 — explicit and PyTorch-compatible):

    - ``valid_mask``: int8, shape ``(max_history_len,)``, ``1`` for a real
      token and ``0`` for padding.
    - ``key_padding_mask``: bool, shape ``(max_history_len,)``, ``True`` for
      PADDING and ``False`` for a real token (the PyTorch Transformer convention
      where True means "ignore this position"). It is the exact boolean
      negation of ``valid_mask``.

    Bounded-history contract (item 6 / AGENTS.md "bounded history"):

    - ``original_length``: the number of public moves in the full game history
      before truncation.
    - ``was_truncated``: True iff ``original_length > max_history_len`` (some
      oldest moves were dropped).
    - ``truncation_side``: always ``"left"`` for this schema — oldest moves are
      dropped, real tokens are left-aligned. Changing this would bump
      ``TRUNCATION_SEMANTICS_VERSION``.
    - ``num_real``: the count of real tokens actually encoded
      (``min(original_length, max_history_len)``).
    """

    tokens: np.ndarray
    valid_mask: np.ndarray
    num_real: int
    max_history_len: int
    token_width: int
    original_length: int
    was_truncated: bool
    truncation_side: str = "left"

    def __post_init__(self) -> None:
        if self.tokens.shape != (self.max_history_len, self.token_width):
            raise ValueError(
                f"tokens shape {self.tokens.shape} != "
                f"({self.max_history_len}, {self.token_width})"
            )
        if self.valid_mask.shape != (self.max_history_len,):
            raise ValueError(
                f"valid_mask shape {self.valid_mask.shape} != "
                f"({self.max_history_len},)"
            )
        if not (0 <= self.num_real <= self.max_history_len):
            raise ValueError(
                f"num_real {self.num_real} out of range "
                f"[0, {self.max_history_len}]"
            )
        if self.truncation_side != "left":
            raise ValueError(
                f"truncation_side must be 'left' for this schema, got "
                f"{self.truncation_side!r}"
            )
        # Freeze the mask array (item 5: deep immutability).
        vm = np.asarray(self.valid_mask, dtype=np.int8)
        if vm.flags.writeable:
            vm = vm.copy()
        vm.setflags(write=False)
        object.__setattr__(self, "valid_mask", vm)

    @property
    def key_padding_mask(self) -> np.ndarray:
        """Boolean mask, True at PADDING positions (PyTorch convention).

        ``key_padding_mask[i] is True`` means position ``i`` is padding and
        should be ignored by attention. It is the boolean negation of
        ``valid_mask``.
        """
        return self.valid_mask == 0

    @property
    def mask(self) -> np.ndarray:
        """Backward-compatible alias for :attr:`valid_mask` (1=valid, 0=pad)."""
        return self.valid_mask


def _role_onehot(role: str) -> np.ndarray:
    """One-hot encode an absolute role over the SEAT_ONEHOT_WIDTH layout.

    The first three slots correspond to the three absolute roles in canonical
    order; the remaining slots (teammate/opponent/etc.) are reserved and left
    zero here because a history token records the *absolute* actor role.
    """
    vec = np.zeros(SEAT_ONEHOT_WIDTH, dtype=np.int8)
    if role in ALL_ROLES:
        vec[ALL_ROLES.index(role)] = 1
    return vec


def encode_history_token(move: HistoryMove) -> np.ndarray:
    """Encode one :class:`HistoryMove` into a flat int8 token vector.

    The field order MUST match :func:`build_v2_schema`'s
    ``history_token_fields`` (schema.py). Tests assert this correspondence.
    """
    parts: list[np.ndarray] = [
        _role_onehot(move.actor_role),
        np.array([1 if move.is_pass else 0], dtype=np.int8),
        _move_type_onehot(move.move_type),
        np.array([move.main_rank], dtype=np.int8),
        np.array([move.length], dtype=np.int8),
        np.array([move.card_count], dtype=np.int8),
        cards_to_vector(move.cards),
        np.array([move.cards_left_after], dtype=np.int8),
        np.array([1 if move.is_bomb else 0], dtype=np.int8),
        np.array([move.phase], dtype=np.int8),
        np.array([1], dtype=np.int8),  # valid padding mask
    ]
    return np.concatenate(parts).astype(np.int8)


def _move_type_onehot(move_type: int) -> np.ndarray:
    vec = np.zeros(MOVE_TYPE_ONEHOT_WIDTH, dtype=np.int8)
    if 0 <= move_type < MOVE_TYPE_ONEHOT_WIDTH:
        vec[move_type] = 1
    return vec


def encode_history(
    moves: Sequence[HistoryMove],
    schema: FeatureSchemaManifest,
) -> HistoryTokenBatch:
    """Encode a sequence of public moves into a padded :class:`HistoryTokenBatch`.

    Bounded history: only the most recent ``schema.max_history_len`` moves are
    kept (older moves are dropped — left-truncation), matching the "bounded
    history" contract. Real tokens are placed at the start (left-aligned); the
    remainder is all-zero padding with a zero ``valid_mask``.

    ``original_length`` records the full move count before truncation, and
    ``was_truncated`` is True iff some oldest moves were dropped.
    """
    max_len = schema.max_history_len
    width = _history_token_width(schema)
    tokens = np.zeros((max_len, width), dtype=np.int8)
    valid_mask = np.zeros(max_len, dtype=np.int8)

    original_length = len(moves)
    # Keep the most recent max_len moves (drop oldest beyond the cap).
    recent = list(moves)[-max_len:] if max_len > 0 else []
    for i, move in enumerate(recent):
        tokens[i, :] = encode_history_token(move)
        valid_mask[i] = 1

    # Freeze the token buffer so callers cannot mutate the encoded history.
    tokens.setflags(write=False)

    return HistoryTokenBatch(
        tokens=tokens,
        valid_mask=valid_mask,
        num_real=len(recent),
        max_history_len=max_len,
        token_width=width,
        original_length=original_length,
        was_truncated=(original_length > max_len),
        truncation_side="left",
    )


def _history_token_width(schema: FeatureSchemaManifest) -> int:
    """Total flat width of one history token, derived from the schema.

    Sums every ``history_token_fields`` entry — including the trailing
    ``valid`` mask slot, which lives inside the token vector itself (so a model
    reading ``tokens`` sees the mask per-position without a separate array).
    """
    width = 0
    for spec in schema.history_token_fields:
        w = 1
        for d in spec.shape:
            w *= d
        width += w
    return width
