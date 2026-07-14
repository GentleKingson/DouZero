"""Bounded minimum-turn hand decomposition for legal DouDizhu moves."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

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


@dataclass
class _SearchBudget:
    """Shared cooperative node/deadline budget for one decomposition call."""

    node_limit: int
    deadline: float | None
    clock: Callable[[], float]
    nodes: int = 0

    def check_deadline(self) -> None:
        if self.deadline is not None and self.clock() >= self.deadline:
            raise _BudgetExceeded

    def visit_node(self) -> None:
        self.check_deadline()
        self.nodes += 1
        if self.nodes > self.node_limit:
            raise _BudgetExceeded


def _canonical_cards(cards) -> tuple[int, ...]:
    out = tuple(sorted(int(card) for card in cards))
    if len(out) > 20:
        raise ValueError(f"hand_decomposition supports at most 20 cards, got {len(out)}")
    return out


def _generate_candidate_moves(
    hand: tuple[int, ...], budget: _SearchBudget | None
) -> tuple[tuple[int, ...], ...]:
    """Generate stable legal moves, cooperatively checking ``budget``."""
    if not hand:
        return ()
    unique: set[tuple[int, ...]] = set()
    check = budget.check_deadline if budget is not None else None
    for move in MovesGener(list(hand), budget_check=check).gen_moves():
        if budget is not None:
            budget.check_deadline()
        key = tuple(sorted(move))
        move_type = get_move_type(list(key))["type"]
        if move_type not in (TYPE_0_PASS, TYPE_15_WRONG):
            unique.add(key)

    def order_key(move: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        move_type = get_move_type(list(move))["type"]
        bomb_penalty = int(move_type in (TYPE_4_BOMB, TYPE_5_KING_BOMB))
        return (bomb_penalty, -len(move), move)

    if budget is not None:
        budget.check_deadline()
    ordered = tuple(sorted(unique, key=order_key))
    if budget is not None:
        budget.check_deadline()
    return ordered


@lru_cache(maxsize=131_072)
def _candidate_moves_cached(hand: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Cache only searches without a wall-clock budget."""

    return _generate_candidate_moves(hand, None)


def _subtract(hand: tuple[int, ...], move: tuple[int, ...]) -> tuple[int, ...]:
    remaining = list(hand)
    for card in move:
        remaining.remove(card)
    return tuple(remaining)


def _linear_turn_upper_bound(hand: tuple[int, ...]) -> int:
    """O(n) legal upper bound that never enumerates compound moves.

    Each rank group is a legal single/pair/triple/bomb.  When both jokers are
    present they form one rocket instead of two singles.  The result is not a
    strength heuristic; it is a fixed-cost, deterministic timeout fallback.
    """

    counts = Counter(hand)
    turns = len(counts)
    if counts.get(20, 0) == 1 and counts.get(30, 0) == 1:
        turns -= 1
    return turns


def hand_decomposition(
    cards,
    *,
    node_budget: int = 500,
    time_budget_ms: int = 0,
    clock: Callable[[], float] | None = None,
) -> DecompositionResult:
    """Compute minimum turns, or a deterministic legal upper bound on timeout.

    The exact solver is memoized dynamic programming over remaining-hand
    tuples.  ``node_budget`` is the deterministic primary bound.  An optional
    cooperative wall-clock deadline is available for latency-sensitive
    deployment and is checked inside every combinatorial enumeration loop as
    well as at DP state boundaries. Overrun is limited to the small fixed-cost
    primitive between adjacent checks rather than a complete state expansion.
    If either bound fires, the fallback is an O(n), deterministic legal upper
    bound and never calls the combinatorial move generator again.

    ``clock`` is injectable for deterministic deadline tests. Results from a
    non-zero time budget are intentionally not stored in a permanent cache.
    """

    hand = _canonical_cards(cards)
    if isinstance(node_budget, bool) or node_budget <= 0:
        raise ValueError(f"node_budget must be a positive int, got {node_budget!r}")
    if isinstance(time_budget_ms, bool) or time_budget_ms < 0:
        raise ValueError(f"time_budget_ms must be non-negative, got {time_budget_ms!r}")
    if not hand:
        return DecompositionResult(0, True, 1, False)

    active_clock = clock or time.monotonic
    deadline = (
        active_clock() + float(time_budget_ms) / 1000.0
        if time_budget_ms > 0
        else None
    )
    budget = _SearchBudget(node_budget, deadline, active_clock)
    memo: dict[tuple[int, ...], int] = {(): 0}

    def solve(state: tuple[int, ...]) -> int:
        cached = memo.get(state)
        if cached is not None:
            return cached
        budget.visit_node()
        best = _linear_turn_upper_bound(state)
        moves = (
            _candidate_moves_cached(state)
            if deadline is None
            else _generate_candidate_moves(state, budget)
        )
        for move in moves:
            budget.check_deadline()
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
        return DecompositionResult(
            _linear_turn_upper_bound(hand), False, budget.nodes, True
        )
    return DecompositionResult(result, True, budget.nodes, False)
