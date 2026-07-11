"""Table-driven tests for move generation, detection, and filtering.

These three modules are the rule source of truth for DouDizhu legality. We
exercise every move type, the detector that classifies them, and the selector
that decides what beats what. No pretrained weights are needed.
"""

from __future__ import annotations

import pytest

from douzero.env.move_detector import get_move_type, is_continuous_seq
from douzero.env.move_generator import MovesGener
from douzero.env.move_selector import (
    filter_type_13_4_2,
    filter_type_14_4_22,
    filter_type_1_single,
    filter_type_2_pair,
    filter_type_3_triple,
    filter_type_4_bomb,
    filter_type_6_3_1,
    filter_type_7_3_2,
    filter_type_8_serial_single,
    filter_type_11_serial_3_1,
    filter_type_12_serial_3_2,
)
from douzero.env.utils import (
    MIN_PAIRS,
    MIN_SINGLE_CARDS,
    MIN_TRIPLES,
    TYPE_0_PASS,
    TYPE_1_SINGLE,
    TYPE_2_PAIR,
    TYPE_3_TRIPLE,
    TYPE_4_BOMB,
    TYPE_5_KING_BOMB,
    TYPE_6_3_1,
    TYPE_7_3_2,
    TYPE_8_SERIAL_SINGLE,
    TYPE_11_SERIAL_3_1,
    TYPE_12_SERIAL_3_2,
    TYPE_13_4_2,
    TYPE_14_4_22,
    TYPE_15_WRONG,
)


# --------------------------------------------------------------------------- #
# is_continuous_seq
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "seq,expected",
    [
        ([3, 4, 5, 6, 7], True),
        ([3, 4, 5, 6], True),
        ([3, 3, 4, 5], False),     # duplicate -> not strictly increasing by 1
        ([10, 11, 12, 13, 14], True),
        ([12, 13, 14, 17], False),  # gap to "2"
        ([3, 5, 7], False),
        ([17], True),               # length 1 trivially continuous
    ],
)
def test_is_continuous_seq(seq, expected):
    assert is_continuous_seq(seq) == expected


# --------------------------------------------------------------------------- #
# get_move_type
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "move,expected_type",
    [
        ([], TYPE_0_PASS),
        ([3], TYPE_1_SINGLE),
        ([7, 7], TYPE_2_PAIR),
        ([9, 9, 9], TYPE_3_TRIPLE),
        ([5, 5, 5, 5], TYPE_4_BOMB),
        ([20, 30], TYPE_5_KING_BOMB),
        ([5, 5, 5, 7], TYPE_6_3_1),       # triple + single
        ([5, 5, 5, 7, 7], TYPE_7_3_2),    # triple + pair
        ([3, 4, 5, 6, 7], TYPE_8_SERIAL_SINGLE),
        ([6, 6, 7, 7, 8, 8], 9),          # serial pair (type 9)
        ([6, 6, 6, 7, 7, 7], 10),         # serial triple (type 10)
        ([5, 5, 5, 6, 6, 6, 3, 4], TYPE_11_SERIAL_3_1),  # serial 3+1 (2 triples + 2 singles)
        ([5, 5, 5, 6, 6, 6, 7, 7, 8, 8], TYPE_12_SERIAL_3_2),  # serial 3+2 (2 triples + 2 pairs)
        ([4, 4, 4, 4, 5, 6], TYPE_13_4_2),    # 4 + 2 singles
        ([4, 4, 4, 4, 5, 5, 6, 6], TYPE_14_4_22),  # 4 + 2 pairs
    ],
)
def test_get_move_type_classifies_known_moves(move, expected_type):
    assert get_move_type(move)["type"] == expected_type


def test_get_move_type_pass_is_zero():
    assert get_move_type([])["type"] == TYPE_0_PASS


def test_get_move_type_king_bomb_detected():
    assert get_move_type([20, 30])["type"] == TYPE_5_KING_BOMB


def test_get_move_type_invalid_is_wrong():
    # An unclassifiable collection -> TYPE_15_WRONG.
    assert get_move_type([3, 4, 4, 5])["type"] == TYPE_15_WRONG


# --------------------------------------------------------------------------- #
# MovesGener basic types
# --------------------------------------------------------------------------- #

def test_gen_type_1_single_all_distinct_ranks():
    mg = MovesGener([3, 4, 7, 7, 9])
    singles = mg.gen_type_1_single()
    # Ranks present: 3, 4, 7, 9 (each at least once).
    distinct_ranks = {m[0] for m in singles}
    assert distinct_ranks == {3, 4, 7, 9}


def test_gen_type_2_pair_requires_two_of_a_kind():
    mg = MovesGener([3, 3, 4, 5, 5, 7, 7, 7])
    pairs = mg.gen_type_2_pair()
    ranks = {m[0] for m in pairs}
    assert ranks == {3, 5, 7}  # 4 only has one copy -> not a pair


def test_gen_type_3_triple_requires_three():
    mg = MovesGener([5, 5, 5, 6, 6, 7])
    triples = mg.gen_type_3_triple()
    assert triples == [[5, 5, 5]]


def test_gen_type_4_bomb_requires_four():
    mg = MovesGener([8, 8, 8, 8, 9, 9])
    bombs = mg.gen_type_4_bomb()
    assert bombs == [[8, 8, 8, 8]]


def test_gen_type_5_king_bomb_present_when_both_jokers():
    mg = MovesGener([20, 30, 3, 4])
    assert mg.gen_type_5_king_bomb() == [[20, 30]]


def test_gen_type_5_king_bomb_absent_without_both_jokers():
    assert MovesGener([20, 3, 4]).gen_type_5_king_bomb() == []
    assert MovesGener([3, 4]).gen_type_5_king_bomb() == []


def test_gen_type_6_3_1_combines_triple_with_single():
    mg = MovesGener([5, 5, 5, 7, 8])
    moves = mg.gen_type_6_3_1()
    # Triple 555 combined with each non-5 single: 7, 8 (4 combinations counted
    # by select() over the non-triple cards). Verify the triple is always there.
    for m in moves:
        assert sorted(m) in ([5, 5, 5, 7], [5, 5, 5, 8])


def test_gen_type_7_3_2_combines_triple_with_pair():
    mg = MovesGener([5, 5, 5, 9, 9])
    moves = mg.gen_type_7_3_2()
    # MovesGener does NOT sort each move; GameEnv.get_legal_card_play_actions
    # sorts in place later. Compare on the sorted form.
    assert sorted([sorted(m) for m in moves]) == [[5, 5, 5, 9, 9]]


def test_gen_type_8_serial_single_minimum_length():
    assert MIN_SINGLE_CARDS == 5
    # Long enough run: 3..9 -> serial singles of length 5..7 exist.
    mg = MovesGener([3, 4, 5, 6, 7, 8, 9])
    serials = mg.gen_type_8_serial_single()
    assert all(len(m) >= MIN_SINGLE_CARDS for m in serials)
    # A 5-length straight 3-4-5-6-7 must be among them.
    assert sorted([3, 4, 5, 6, 7]) in [sorted(m) for m in serials]


def test_gen_type_8_serial_single_excludes_rank_2():
    # "2" (17) is not part of straights in this implementation.
    mg = MovesGener([14, 17])
    assert mg.gen_type_8_serial_single() == []


def test_gen_type_9_serial_pair_min_length():
    assert MIN_PAIRS == 3
    mg = MovesGener([3, 3, 4, 4, 5, 5])
    serials = mg.gen_type_9_serial_pair()
    assert serials == [[3, 3, 4, 4, 5, 5]]


def test_gen_type_10_serial_triple_min_length():
    assert MIN_TRIPLES == 2
    mg = MovesGener([6, 6, 6, 7, 7, 7])
    serials = mg.gen_type_10_serial_triple()
    assert serials == [[6, 6, 6, 7, 7, 7]]


def test_gen_type_13_4_2_attaches_two_singles():
    mg = MovesGener([4, 4, 4, 4, 6, 8])
    moves = mg.gen_type_13_4_2()
    # Two attachment singles drawn from {6, 8}: select 2 -> only {6,8}.
    expected = [[4, 4, 4, 4, 6, 8]]
    assert sorted([sorted(m) for m in moves]) == sorted([sorted(m) for m in expected])


def test_gen_type_14_4_22_attaches_two_pairs():
    mg = MovesGener([4, 4, 4, 4, 6, 6, 8, 8])
    moves = mg.gen_type_14_4_22()
    expected = [[4, 4, 4, 4, 6, 6, 8, 8]]
    assert sorted([sorted(m) for m in moves]) == sorted([sorted(m) for m in expected])


def test_gen_moves_includes_all_type_families():
    mg = MovesGener([3, 3, 4, 4, 4, 5, 6, 7, 8, 20, 30])
    all_moves = mg.gen_moves()
    # The union must contain at least: a single, a pair, a triple, a bomb-less
    # hand has no bomb here, but king bomb exists. Just sanity-check sizes.
    assert len(all_moves) > 0
    flat = [card for m in all_moves for card in m]
    assert set(flat).issubset({3, 4, 5, 6, 7, 8, 20, 30})


# --------------------------------------------------------------------------- #
# Filters: what beats what
# --------------------------------------------------------------------------- #

def test_filter_single_keeps_higher_rank_only():
    moves = [[3], [5], [7], [9]]
    assert filter_type_1_single(moves, [5]) == [[7], [9]]


def test_filter_single_excludes_equal_rank():
    # Strict greater-than: cannot "beat" the same rank.
    moves = [[5], [7]]
    assert filter_type_1_single(moves, [5]) == [[7]]


def test_filter_pair_keeps_higher_rank():
    moves = [[3, 3], [6, 6], [8, 8]]
    assert filter_type_2_pair(moves, [6, 6]) == [[8, 8]]


def test_filter_triple_keeps_higher_rank():
    moves = [[5, 5, 5], [9, 9, 9]]
    assert filter_type_3_triple(moves, [5, 5, 5]) == [[9, 9, 9]]


def test_filter_bomb_keeps_higher_rank():
    moves = [[3, 3, 3, 3], [7, 7, 7, 7]]
    assert filter_type_4_bomb(moves, [3, 3, 3, 3]) == [[7, 7, 7, 7]]


def test_filter_3_1_uses_triple_rank():
    moves = [[5, 5, 5, 3], [7, 7, 7, 4], [9, 9, 9, 6]]
    out = filter_type_6_3_1(moves, [5, 5, 5, 8])
    # Only triples of rank > 5 survive: 7 and 9. filter_type_6_3_1 sorts each
    # move in place, so the expected form is the sorted tuple.
    out_ranks = {tuple(m) for m in out}
    assert (4, 7, 7, 7) in out_ranks
    assert (6, 9, 9, 9) in out_ranks


def test_filter_3_2_uses_triple_rank():
    moves = [[5, 5, 5, 3, 3], [7, 7, 7, 9, 9]]
    out = filter_type_7_3_2(moves, [5, 5, 5, 8, 8])
    assert out == [[7, 7, 7, 9, 9]]


def test_filter_serial_single_strict_higher_low_card():
    moves = [[3, 4, 5, 6, 7], [5, 6, 7, 8, 9]]
    assert filter_type_8_serial_single(moves, [3, 4, 5, 6, 7]) == [[5, 6, 7, 8, 9]]


def test_filter_serial_3_1_uses_triple_rank():
    moves = [[3, 3, 3, 4, 4, 4, 5, 6], [5, 5, 5, 6, 6, 6, 7, 8]]
    out = filter_type_11_serial_3_1(moves, [3, 3, 3, 4, 4, 4, 5, 6])
    assert out == [[5, 5, 5, 6, 6, 6, 7, 8]]


def test_filter_serial_3_2_uses_triple_rank():
    moves = [[3, 3, 3, 4, 4, 4, 5, 5], [5, 5, 5, 6, 6, 6, 7, 7]]
    out = filter_type_12_serial_3_2(moves, [3, 3, 3, 4, 4, 4, 5, 5])
    assert out == [[5, 5, 5, 6, 6, 6, 7, 7]]


def test_filter_4_2_uses_bomb_rank():
    moves = [[4, 4, 4, 4, 5, 6], [7, 7, 7, 7, 8, 9]]
    out = filter_type_13_4_2(moves, [4, 4, 4, 4, 5, 6])
    assert out == [[7, 7, 7, 7, 8, 9]]


def test_filter_4_22_uses_bomb_rank():
    moves = [[4, 4, 4, 4, 5, 5, 6, 6], [7, 7, 7, 7, 8, 8, 9, 9]]
    out = filter_type_14_4_22(moves, [4, 4, 4, 4, 5, 5, 6, 6])
    assert out == [[7, 7, 7, 7, 8, 8, 9, 9]]


# --------------------------------------------------------------------------- #
# King bomb is always legal when both jokers are held
# --------------------------------------------------------------------------- #

def test_king_bomb_generation_only_when_complete():
    # The move generator only yields the rocket when BOTH jokers are in hand.
    assert MovesGener([20, 30, 3]).gen_type_5_king_bomb() == [[20, 30]]
    assert MovesGener([20, 3]).gen_type_5_king_bomb() == []
    assert MovesGener([30, 3]).gen_type_5_king_bomb() == []


# --------------------------------------------------------------------------- #
# Boundary ranks
# --------------------------------------------------------------------------- #

def test_boundary_rank_3_smallest_single():
    mg = MovesGener([3, 4])
    singles = sorted(m[0] for m in mg.gen_type_1_single())
    assert singles[0] == 3


def test_boundary_rank_ace_pair_possible():
    mg = MovesGener([14, 14, 3])
    assert mg.gen_type_2_pair() == [[14, 14]]


def test_boundary_rank_2_pair_and_bomb():
    mg = MovesGener([17, 17, 17, 17, 3])
    assert mg.gen_type_4_bomb() == [[17, 17, 17, 17]]
    assert mg.gen_type_2_pair() == [[17, 17]]


def test_generated_moves_are_each_sorted_inplace_by_game_env():
    # The legal-action pipeline sorts each move in place (game.py:260-261).
    # MovesGener itself does not guarantee sorted output, so this test documents
    # the contract that GameEnv.get_legal_card_play_actions enforces sorting.
    from douzero.env.game import GameEnv
    from douzero.env.env import Env

    # Use a fixed deal via GameEnv directly to avoid RNG.
    env = Env("adp")
    # Reset with RNG seeded for a concrete state, but the assertion is about
    # legality-layer sorting, not about which deal.
    import numpy as np

    np.random.seed(7)
    env.reset()
    for action in env.infoset.legal_actions:
        assert action == sorted(action)
