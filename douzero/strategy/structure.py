"""Card-structure and opportunity-cost features for one legal action."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from douzero.env.move_detector import get_move_type
from douzero.env.move_generator import MovesGener
from douzero.env.utils import (
    TYPE_4_BOMB,
    TYPE_5_KING_BOMB,
    TYPE_8_SERIAL_SINGLE,
    TYPE_9_SERIAL_PAIR,
    TYPE_10_SERIAL_TRIPLE,
    TYPE_13_4_2,
    TYPE_14_4_22,
)


@dataclass(frozen=True)
class HandStructure:
    singles: int
    pairs: int
    triples: int
    straights: int
    serial_pairs: int
    airplanes: int
    bombs: int


@dataclass(frozen=True)
class ActionStructureCost:
    """Before/after structure deltas and explicit strategic opportunity costs."""

    single_delta: int
    pair_delta: int
    triple_delta: int
    straight_delta: int
    serial_pair_delta: int
    airplane_delta: int
    bomb_delta: int
    bomb_break_cost: float
    joker_pair_break: float
    high_control_card_cost: float
    total: float


def _count_unique(moves, move_type: int) -> int:
    return len({tuple(sorted(move)) for move in moves if get_move_type(sorted(move))["type"] == move_type})


def describe_hand_structure(cards) -> HandStructure:
    """Return a deterministic structural summary of a hand."""

    hand = tuple(sorted(int(card) for card in cards))
    counts = Counter(hand)
    generated = MovesGener(list(hand)) if hand else None
    return HandStructure(
        singles=sum(value == 1 for value in counts.values()),
        pairs=sum(value == 2 for value in counts.values()),
        triples=sum(value == 3 for value in counts.values()),
        straights=(
            _count_unique(generated.gen_type_8_serial_single(), TYPE_8_SERIAL_SINGLE)
            if generated else 0
        ),
        serial_pairs=(
            _count_unique(generated.gen_type_9_serial_pair(), TYPE_9_SERIAL_PAIR)
            if generated else 0
        ),
        airplanes=(
            _count_unique(generated.gen_type_10_serial_triple(), TYPE_10_SERIAL_TRIPLE)
            if generated else 0
        ),
        bombs=sum(value == 4 for value in counts.values())
        + int(20 in counts and 30 in counts),
    )

def _remaining_hand(hand: tuple[int, ...], action: tuple[int, ...]) -> tuple[int, ...]:
    remaining = list(hand)
    try:
        for card in action:
            remaining.remove(card)
    except ValueError as exc:
        raise ValueError(f"action {action!r} is not a subset of hand {hand!r}") from exc
    return tuple(remaining)


def action_structure_cost(handcards, action) -> ActionStructureCost:
    """Measure structural damage caused by a legal action.

    This function never declares an action illegal.  In particular, four-with-
    two remains available but pays a bomb opportunity cost because the same
    four cards could have been retained as a bomb.
    """

    hand = tuple(sorted(int(card) for card in handcards))
    move = tuple(sorted(int(card) for card in action))
    remaining = _remaining_hand(hand, move)
    before = describe_hand_structure(hand)
    after = describe_hand_structure(remaining)
    deltas = {
        name: getattr(after, name) - getattr(before, name)
        for name in (
            "singles", "pairs", "triples", "straights", "serial_pairs",
            "airplanes", "bombs",
        )
    }

    hand_counts = Counter(hand)
    move_counts = Counter(move)
    move_type = get_move_type(list(move))["type"]
    broken_bombs = sum(
        1 for rank, count in hand_counts.items()
        if count == 4 and 0 < move_counts[rank] < 4
    )
    if move_type in (TYPE_13_4_2, TYPE_14_4_22):
        broken_bombs += 1
    joker_pair_break = float(
        20 in hand_counts and 30 in hand_counts
        and ((20 in move_counts) ^ (30 in move_counts))
    )
    high_weights = {14: 0.25, 17: 0.5, 20: 0.75, 30: 1.0}
    high_cost = sum(high_weights.get(card, 0.0) for card in move)
    if move_type in (TYPE_4_BOMB, TYPE_5_KING_BOMB):
        # Playing a bomb is an explicit opportunity cost, not "breaking" it.
        bomb_opportunity = 1.0
    else:
        bomb_opportunity = 0.0

    fragmentation = (
        max(0, deltas["singles"]) + 0.75 * max(0, deltas["pairs"])
        + 0.5 * max(0, deltas["triples"])
        + max(0, -deltas["straights"])
        + max(0, -deltas["serial_pairs"])
        + max(0, -deltas["airplanes"])
    )
    total = (
        fragmentation + 2.5 * broken_bombs + 2.0 * joker_pair_break
        + high_cost + 1.5 * bomb_opportunity
    )
    return ActionStructureCost(
        single_delta=deltas["singles"],
        pair_delta=deltas["pairs"],
        triple_delta=deltas["triples"],
        straight_delta=deltas["straights"],
        serial_pair_delta=deltas["serial_pairs"],
        airplane_delta=deltas["airplanes"],
        bomb_delta=deltas["bombs"],
        bomb_break_cost=float(broken_bombs),
        joker_pair_break=joker_pair_break,
        high_control_card_cost=float(high_cost),
        total=float(total),
    )
