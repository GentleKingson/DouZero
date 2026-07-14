"""Leakage-safe other-player statistics derived from public actions only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Sequence

import numpy as np

from douzero.env.move_detector import get_move_type
from douzero.env.utils import TYPE_4_BOMB, TYPE_5_KING_BOMB
from douzero.observation.seats import ALL_ROLES

STYLE_FEATURE_VERSION = "public-other-player-style-v1"
STYLE_NUM_OTHER_PLAYERS = 2
STYLE_PER_PLAYER_WIDTH = 8
STYLE_FEATURE_WIDTH = STYLE_NUM_OTHER_PLAYERS * STYLE_PER_PLAYER_WIDTH

_STYLE_FIELDS = (
    "observed",
    "turn_count",
    "pass_rate",
    "high_card_rate",
    "bomb_rate",
    "split_rate",
    "mean_action_size",
    "mean_rank",
)
STYLE_LAYOUT_HASH = hashlib.sha256(
    json.dumps(
        {
            "version": STYLE_FEATURE_VERSION,
            "other_players": STYLE_NUM_OTHER_PLAYERS,
            "fields": _STYLE_FIELDS,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def _history(public: Any) -> tuple[tuple[int, ...], ...]:
    history = getattr(public, "action_history", ())
    return tuple(tuple(sorted(int(card) for card in action)) for action in history)


def _other_player_roles(acting_role: str) -> tuple[str, str]:
    if acting_role not in ALL_ROLES:
        raise ValueError(
            f"unknown acting_role {acting_role!r}; expected one of {ALL_ROLES}"
        )
    return tuple(role for role in ALL_ROLES if role != acting_role)  # type: ignore[return-value]


def _role_features(
    history: Sequence[Sequence[int]],
    role: str,
) -> np.ndarray:
    role_index = ALL_ROLES.index(role)
    actions = [
        tuple(action)
        for turn_index, action in enumerate(history)
        if turn_index % len(ALL_ROLES) == role_index
    ]
    if not actions:
        return np.zeros(STYLE_PER_PLAYER_WIDTH, dtype=np.float32)

    passes = sum(not action for action in actions)
    non_pass = [action for action in actions if action]
    played_cards = [card for action in non_pass for card in action]
    high_cards = sum(card >= 17 for card in played_cards)
    bombs = 0
    split_actions = 0
    seen_ranks: Counter[int] = Counter()
    for action in non_pass:
        move_type = int(get_move_type(list(action))["type"])
        bombs += move_type in (TYPE_4_BOMB, TYPE_5_KING_BOMB)
        counts = Counter(action)
        if any(seen_ranks[rank] > 0 for rank in counts):
            split_actions += 1
        seen_ranks.update(counts)

    turns = len(actions)
    played_count = len(played_cards)
    non_pass_count = len(non_pass)
    return np.asarray(
        [
            1.0,
            min(turns, 20) / 20.0,
            passes / turns,
            high_cards / max(1, played_count),
            bombs / max(1, non_pass_count),
            split_actions / max(1, non_pass_count),
            min(8.0, played_count / max(1, non_pass_count)) / 8.0,
            (sum(played_cards) / max(1, played_count)) / 30.0,
        ],
        dtype=np.float32,
    )


def build_style_features(public: Any) -> np.ndarray:
    """Return a fixed-width vector for the acting role's two other players.

    The input is intentionally duck-typed to the public observation surface.
    Only ``acting_role`` and ``action_history`` are read. Hidden hands, player
    identifiers, and persistent account identity are neither accepted nor
    inferred. For a farmer, one row is the teammate and one is the landlord;
    rows remain in canonical seat order. A player with no observed turn
    receives an all-zero row; the
    trainable :class:`~douzero.style.encoder.StyleEncoder` maps that row to a
    learned cold-start embedding.
    """

    acting_role = str(public.acting_role)
    history = _history(public)
    rows = [
        _role_features(history, role)
        for role in _other_player_roles(acting_role)
    ]
    out = np.concatenate(rows).astype(np.float32)
    if out.shape != (STYLE_FEATURE_WIDTH,):
        raise RuntimeError(
            f"style feature shape {out.shape} != ({STYLE_FEATURE_WIDTH},)"
        )
    return out
