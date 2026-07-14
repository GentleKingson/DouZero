"""Bounded minimum-turn hand decomposition for legal DouDizhu moves."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from douzero.env.move_detector import get_move_type
from douzero.env.move_generator import MovesGener
from douzero.env.utils import TYPE_0_PASS, TYPE_15_WRONG, TYPE_4_BOMB, TYPE_5_KING_BOMB


@dataclass(frozen=True)
class DecompositionResult:
    """Minimum/upper-bound turns together with bounded-search diagnostics."""

    min_turns: int
    exact: bool
    nodes_visited: int
    fallback_used: bool


class _BudgetExceeded(RuntimeError):
    pass


def _canonical_cards(cards) -> tuple[int, ...]:
    out = tuple(sorted(int(card) for card in cards))
    if len(out) > 20:
        raise ValueError(f"hand_decomposition supports at most 20 cards, got {len(out)}")
    return out


@lru_cache(maxsize=131_072)
def _candidate_moves(hand: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return unique non-pass legal moves in stable best-first order."""

    if not hand:
        return ()
    unique: set[tuple[int, ...]] = set()
    for move in MovesGener(list(hand)).gen_moves():
        key = tuple(sorted(move))
        move_type = get_move_type(list(key))["type"]
        if move_type not in (TYPE_0_PASS, TYPE_15_WRONG):
            unique.add(key)

    def order_key(move: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        move_type = get_move_type(list(move))["type"]
        bomb_penalty = int(move_type in (TYPE_4_BOMB, TYPE_5_KING_BOMB))
        return (bomb_penalty, -len(move), move)

    return tuple(sorted(unique, key=order_key))


def _subtract(hand: tuple[int, ...], move: tuple[int, ...]) -> tuple[int, ...]:
    remaining = list(hand)
    for card in move:
        remaining.remove(card)
    return tuple(remaining)


@lru_cache(maxsize=131_072)
def _heuristic_turns(hand: tuple[int, ...]) -> int:
    """Deterministic legal-play upper bound used after any budget timeout."""

    turns = 0
    remaining = hand
    while remaining:
        moves = _candidate_moves(remaining)
        if not moves:  # Defensive: every non-empty hand has a single-card move.
            return turns + len(remaining)
        remaining = _subtract(remaining, moves[0])
        turns += 1
    return turns


def hand_decomposition(
    cards,
    *,
    node_budget: int = 500,
    time_budget_ms: int = 0,
) -> DecompositionResult:
    """Compute minimum turns, or a deterministic legal upper bound on timeout.

    The exact solver is memoized dynamic programming over remaining-hand
    tuples.  ``node_budget`` is the deterministic primary bound.  An optional
    wall-clock bound is available for latency-sensitive deployment; if either
    bound fires, the returned value is the same deterministic greedy upper
    bound for the original hand, rather than a partially explored result.
    """

    hand = _canonical_cards(cards)
    if isinstance(node_budget, bool) or node_budget <= 0:
        raise ValueError(f"node_budget must be a positive int, got {node_budget!r}")
    if isinstance(time_budget_ms, bool) or time_budget_ms < 0:
        raise ValueError(f"time_budget_ms must be non-negative, got {time_budget_ms!r}")
    if not hand:
        return DecompositionResult(0, True, 1, False)

    deadline = (
        time.monotonic() + float(time_budget_ms) / 1000.0
        if time_budget_ms > 0
        else None
    )
    nodes = 0
    memo: dict[tuple[int, ...], int] = {(): 0}

    def solve(state: tuple[int, ...]) -> int:
        nonlocal nodes
        cached = memo.get(state)
        if cached is not None:
            return cached
        nodes += 1
        if nodes > node_budget or (deadline is not None and time.monotonic() >= deadline):
            raise _BudgetExceeded
        best = _heuristic_turns(state)
        for move in _candidate_moves(state):
            if len(move) == len(state):
                best = 1
                break
            best = min(best, 1 + solve(_subtract(state, move)))
            if best == 1:
                break
        memo[state] = best
        return best

    try:
        result = solve(hand)
    except _BudgetExceeded:
        return DecompositionResult(_heuristic_turns(hand), False, nodes, True)
    return DecompositionResult(result, True, nodes, False)
