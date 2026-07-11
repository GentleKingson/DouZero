"""Tests for the low-level card/array encoders in ``douzero.env.env``.

These encoders are the foundation of every observation; if they drift, the
frozen baseline is meaningless. We assert exact shape/dtype and round-trip
semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from douzero.env.env import (
    Card2Column,
    NumOnes2Array,
    _action_seq_list2array,
    _cards2array,
    _get_one_hot_array,
    _get_one_hot_bomb,
    _process_action_seq,
    deck,
)


# --------------------------------------------------------------------------- #
# Deck
# --------------------------------------------------------------------------- #

def test_deck_has_54_cards():
    assert len(deck) == 54


def test_deck_rank_counts():
    # Four copies of 3..14, four copies of 17, one small joker, one big joker.
    counts = {rank: deck.count(rank) for rank in set(deck)}
    for rank in range(3, 15):
        assert counts[rank] == 4, f"rank {rank} should appear 4 times"
    assert counts[17] == 4
    assert counts[20] == 1  # small joker
    assert counts[30] == 1  # big joker


def test_card2column_covers_all_non_joker_ranks():
    # 13 columns: ranks 3..14 and 17. 16 is intentionally unused.
    assert set(Card2Column.keys()) == set(range(3, 15)) | {17}
    assert set(Card2Column.values()) == set(range(13))


def test_numones2array_is_left_aligned():
    assert NumOnes2Array[0].tolist() == [0, 0, 0, 0]
    assert NumOnes2Array[1].tolist() == [1, 0, 0, 0]
    assert NumOnes2Array[2].tolist() == [1, 1, 0, 0]
    assert NumOnes2Array[3].tolist() == [1, 1, 1, 0]
    assert NumOnes2Array[4].tolist() == [1, 1, 1, 1]


# --------------------------------------------------------------------------- #
# _cards2array
# --------------------------------------------------------------------------- #

def test_cards2array_shape_and_dtype():
    arr = _cards2array([3, 3, 3])
    assert arr.shape == (54,)
    assert arr.dtype == np.int8


def test_cards2array_empty_is_all_zero():
    arr = _cards2array([])
    assert arr.shape == (54,)
    assert np.all(arr == 0)
    assert arr.dtype == np.int8


def test_cards2array_single_rank_occupies_four_consecutive_slots():
    # Fortran-flatten of a 4x13 matrix is COLUMN-major: rank 3 -> slots [0:4].
    arr = _cards2array([3, 3, 3])
    assert arr[0:4].tolist() == [1, 1, 1, 0]
    # All other rank slots must be zero.
    assert arr[4:].sum() == 0


def test_cards2array_rank_to_slot_offset_is_column_index_times_four():
    # Each column occupies 4 consecutive slots; rank r -> Card2Column[r]*4.
    for rank in [3, 4, 14, 17]:
        for n in range(1, deck.count(rank) + 1):
            cards = [rank] * n
            arr = _cards2array(cards)
            base = Card2Column[rank] * 4
            assert arr[base : base + 4].tolist() == NumOnes2Array[n].tolist(), (
                f"rank {rank} x{n} slot layout wrong"
            )


def test_cards2array_jokers_in_last_two_slots():
    small = _cards2array([20])
    big = _cards2array([30])
    rocket = _cards2array([20, 30])
    assert small[52] == 1 and small[53] == 0
    assert big[52] == 0 and big[53] == 1
    assert rocket[52] == 1 and rocket[53] == 1
    # Jokers do not touch the 4x13 rank matrix.
    assert small[:52].sum() == 0
    assert big[:52].sum() == 0


def test_cards2array_roundtrip_via_count():
    # _cards2array is not invertible by name, but the per-rank multiplicity must
    # survive the encoding. We reconstruct counts and compare.
    hand = [3, 3, 5, 7, 7, 7, 14, 17, 17, 20, 30]
    arr = _cards2array(hand)
    reconstructed = []
    for rank, col in Card2Column.items():
        ones = int(arr[col * 4 : col * 4 + 4].sum())
        reconstructed.extend([rank] * ones)
    if arr[52] == 1:
        reconstructed.append(20)
    if arr[53] == 1:
        reconstructed.append(30)
    assert sorted(reconstructed) == sorted(hand)


# --------------------------------------------------------------------------- #
# one-hot helpers
# --------------------------------------------------------------------------- #

def test_get_one_hot_array_length_and_index():
    oh = _get_one_hot_array(5, 17)
    assert oh.shape == (17,)
    assert int(oh.sum()) == 1
    assert int(np.argmax(oh)) == 4  # index = num_left_cards - 1


@pytest.mark.parametrize("num_left,max_len", [(1, 17), (17, 17), (1, 20), (20, 20)])
def test_get_one_hot_array_boundary_indices(num_left, max_len):
    oh = _get_one_hot_array(num_left, max_len)
    assert oh.shape == (max_len,)
    assert int(np.argmax(oh)) == num_left - 1


def test_get_one_hot_bomb_length_and_index():
    for bomb_num in range(15):
        oh = _get_one_hot_bomb(bomb_num)
        assert oh.shape == (15,)
        assert int(oh.sum()) == 1
        assert int(np.argmax(oh)) == bomb_num


# --------------------------------------------------------------------------- #
# history encoding
# --------------------------------------------------------------------------- #

def test_action_seq_list2array_shape():
    seq = _process_action_seq([[3], [4, 4], [5, 5, 5]])
    arr = _action_seq_list2array(seq)
    assert arr.shape == (5, 162)


def test_action_seq_list2array_pads_to_15_moves():
    # 3 real moves + 12 padding rows = 15; reshape to (5, 162).
    seq = _process_action_seq([[3], [4, 4], [5, 5, 5]])
    assert len(seq) == 15
    # Padding rows (indices 0..11) are empty -> _cards2array -> all zeros.
    for row in seq[:12]:
        assert row == []


def test_process_action_seq_takes_last_15():
    moves = [[r] for r in range(3, 25)]  # 22 single moves
    out = _process_action_seq(moves, length=15)
    assert len(out) == 15
    # The most-recent 15 are retained (and in order).
    expected_tail = moves[-15:]
    assert out == expected_tail


def test_action_seq_list2array_first_move_lands_in_last_round_block():
    # 15 moves -> 5 blocks of 3; move index 0 is in block 0 (row 0 of the 5x162).
    moves = [[3]] + [[] for _ in range(14)]
    seq = _process_action_seq(moves)
    arr = _action_seq_list2array(seq)
    # The single "3" occupies rank-3 slots [0:4] of row 0 (within block 0).
    assert arr[0, 0:4].tolist() == [1, 0, 0, 0]
