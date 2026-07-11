"""Tests for the evaluation data format and legacy adapter (P02 Slice 4).

Covers:
- legacy data generation (4-key dict, no format_version)
- standard data generation (deck, first_bidder, ruleset_id, format_version=2)
- legacy adapter reads old format
- legacy adapter reads new format
- format/ruleset mismatch raises precise errors
- standard data is reproducible (same seed → same deck)
- generate_eval_data smoke test (legacy and standard)
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from douzero.evaluation.legacy_data_adapter import (
    deal_standard_deck,
    is_legacy_format,
    is_standard_format,
    load_eval_data,
)
from generate_eval_data import generate, generate_standard


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def test_legacy_generate_produces_4_key_dict():
    """Legacy generate() must produce the original 4-key format."""
    np.random.seed(42)
    data = generate()
    assert set(data.keys()) == {
        'landlord', 'landlord_up', 'landlord_down', 'three_landlord_cards'
    }
    assert len(data['landlord']) == 20
    assert len(data['landlord_up']) == 17
    assert len(data['landlord_down']) == 17
    assert len(data['three_landlord_cards']) == 3
    # No format_version key in legacy.
    assert 'format_version' not in data
    assert 'ruleset_id' not in data


def test_standard_generate_produces_v2_format():
    """Standard generate_standard() must produce the v2 format with rule identity."""
    np.random.seed(42)
    data = generate_standard()
    assert data['format_version'] == 2
    assert data['schema_version'] == 1
    assert data['ruleset_id'] == 'standard'
    assert data['ruleset_version'] == 'standard-v1'
    # Full SHA-256 hash (64 hex chars).
    assert len(data['ruleset_hash']) == 64
    assert len(data['deck']) == 54
    # first_bidder uses neutral seat labels.
    assert data['first_bidder'] in ('0', '1', '2')
    assert 'bidding_order' in data
    assert len(data['bidding_order']) == 3
    assert sorted(data['bidding_order']) == ['0', '1', '2']


def test_standard_deck_is_valid_54_cards():
    """The standard deck must be a valid 54-card deck."""
    np.random.seed(42)
    data = generate_standard()
    from collections import Counter
    counts = Counter(data['deck'])
    for rank in range(3, 15):
        assert counts[rank] == 4
    assert counts[17] == 4
    assert counts[20] == 1
    assert counts[30] == 1
    assert sum(counts.values()) == 54


def test_standard_data_reproducible_with_same_seed():
    """Same numpy seed → same deck."""
    np.random.seed(42)
    a = generate_standard()
    np.random.seed(42)
    b = generate_standard()
    assert a['deck'] == b['deck']


def test_standard_data_differs_with_different_seed():
    """Different seeds → different decks (with overwhelming probability)."""
    np.random.seed(42)
    a = generate_standard()
    np.random.seed(99)
    b = generate_standard()
    assert a['deck'] != b['deck']


# --------------------------------------------------------------------------- #
# Legacy adapter
# --------------------------------------------------------------------------- #
def test_load_legacy_data(tmp_path):
    """load_eval_data reads legacy format with ruleset='legacy'."""
    np.random.seed(42)
    data = [generate() for _ in range(5)]
    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    loaded = load_eval_data(str(pkl), ruleset="legacy")
    assert len(loaded) == 5
    assert set(loaded[0].keys()) == {
        'landlord', 'landlord_up', 'landlord_down', 'three_landlord_cards'
    }


def test_load_standard_data(tmp_path):
    """load_eval_data reads standard format with ruleset='standard'."""
    np.random.seed(42)
    data = [generate_standard() for _ in range(5)]
    pkl = tmp_path / "standard.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    loaded = load_eval_data(str(pkl), ruleset="standard")
    assert len(loaded) == 5
    assert loaded[0]['format_version'] == 2


def test_load_legacy_data_rejects_standard_request(tmp_path):
    """Legacy data + ruleset='standard' must raise."""
    np.random.seed(42)
    data = [generate() for _ in range(3)]
    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="legacy format"):
        load_eval_data(str(pkl), ruleset="standard")


def test_load_standard_data_rejects_legacy_request(tmp_path):
    """Standard data + ruleset='legacy' must raise."""
    np.random.seed(42)
    data = [generate_standard() for _ in range(3)]
    pkl = tmp_path / "standard.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="standard format"):
        load_eval_data(str(pkl), ruleset="legacy")


def test_is_legacy_format():
    np.random.seed(42)
    assert is_legacy_format([generate()])
    assert not is_legacy_format([generate_standard()])


def test_is_standard_format():
    np.random.seed(42)
    assert is_standard_format([generate_standard()])
    assert not is_standard_format([generate()])


def test_load_empty_list_returns_empty(tmp_path):
    """An empty dataset must load without error."""
    pkl = tmp_path / "empty.pkl"
    with open(pkl, "wb") as f:
        pickle.dump([], f)
    assert load_eval_data(str(pkl), ruleset="legacy") == []


def test_load_non_list_raises(tmp_path):
    """A non-list dataset must raise."""
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"not": "a list"}, f)
    with pytest.raises(TypeError, match="list"):
        load_eval_data(str(pkl))


# --------------------------------------------------------------------------- #
# deal_standard_deck helper
# --------------------------------------------------------------------------- #
def test_deal_standard_deck_splits_correctly():
    """deal_standard_deck must produce 17+17+17+3."""
    deck = list(range(3, 15)) * 4 + [17] * 4 + [20, 30]
    dealt = deal_standard_deck(deck)
    assert len(dealt['landlord']) == 17
    assert len(dealt['landlord_up']) == 17
    assert len(dealt['landlord_down']) == 17
    assert len(dealt['three_landlord_cards']) == 3
    # Total = 54.
    total = sum(len(v) for v in dealt.values())
    assert total == 54


def test_deal_standard_deck_wrong_size_raises():
    with pytest.raises(ValueError, match="54 cards"):
        deal_standard_deck([1, 2, 3])


# --------------------------------------------------------------------------- #
# generate_eval_data smoke test (subprocess)
# --------------------------------------------------------------------------- #
def test_generate_eval_data_smoke_legacy(tmp_path):
    """generate_eval_data.py --ruleset legacy must produce a valid pickle."""
    import subprocess
    import sys
    out = tmp_path / "eval_legacy"
    result = subprocess.run(
        [sys.executable, "generate_eval_data.py",
         "--output", str(out), "--num_games", "12", "--ruleset", "legacy"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    pkl = str(out) + ".pkl"
    data = load_eval_data(pkl, ruleset="legacy")
    assert len(data) == 12
    assert is_legacy_format(data)


def test_generate_eval_data_smoke_standard(tmp_path):
    """generate_eval_data.py --ruleset standard must produce a valid pickle."""
    import subprocess
    import sys
    out = tmp_path / "eval_standard"
    result = subprocess.run(
        [sys.executable, "generate_eval_data.py",
         "--output", str(out), "--num_games", "12", "--ruleset", "standard"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    pkl = str(out) + ".pkl"
    data = load_eval_data(pkl, ruleset="standard")
    assert len(data) == 12
    assert is_standard_format(data)
    # Each deal must have a valid 54-card deck.
    from collections import Counter
    for deal in data:
        counts = Counter(deal['deck'])
        for rank in range(3, 15):
            assert counts[rank] == 4
        assert counts[17] == 4
        assert counts[20] == 1
        assert counts[30] == 1


# --------------------------------------------------------------------------- #
# schema_version and ruleset_hash validation
# --------------------------------------------------------------------------- #
def test_standard_data_missing_schema_version_rejected(tmp_path):
    """A v2 dataset without schema_version must be rejected."""
    np.random.seed(42)
    data = [generate_standard()]
    del data[0]['schema_version']
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="schema_version"):
        load_eval_data(str(pkl), ruleset="standard")


def test_standard_data_unknown_schema_version_rejected(tmp_path):
    """A v2 dataset with an unsupported schema_version must be rejected."""
    np.random.seed(42)
    data = [generate_standard()]
    data[0]['schema_version'] = 99
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="schema_version"):
        load_eval_data(str(pkl), ruleset="standard")


def test_standard_data_ruleset_hash_mismatch_rejected(tmp_path):
    """A v2 dataset with a wrong ruleset_hash must be rejected."""
    from douzero.env.rules import RuleSet
    np.random.seed(42)
    data = [generate_standard()]
    # Use a valid-length but wrong hash (64 hex chars).
    data[0]['ruleset_hash'] = "0" * 64
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="ruleset_hash"):
        load_eval_data(str(pkl), ruleset="standard",
                       expected_ruleset=RuleSet.standard())


def test_standard_data_bad_deck_rejected(tmp_path):
    """A v2 dataset with an invalid deck must be rejected."""
    np.random.seed(42)
    data = [generate_standard()]
    data[0]['deck'] = [3] * 54  # all rank 3 — invalid
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="Rank"):
        load_eval_data(str(pkl), ruleset="standard")


def test_standard_data_short_deck_rejected(tmp_path):
    """A v2 dataset with a short deck must be rejected."""
    np.random.seed(42)
    data = [generate_standard()]
    data[0]['deck'] = data[0]['deck'][:50]
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="54 cards"):
        load_eval_data(str(pkl), ruleset="standard")


# --------------------------------------------------------------------------- #
# Worker-count reproducibility
# --------------------------------------------------------------------------- #
def test_worker_count_reproducibility(tmp_path):
    """num_workers=1 and num_workers>1 must produce identical results.

    The per-game seed is derived from eval_seed + global_game_index +
    deck_hash, so the same game always produces the same bidding sequence
    and card-play regardless of which worker processes it.
    """
    import subprocess
    import sys

    # Generate 12 standard deals with a fixed seed.
    np.random.seed(42)
    data = [generate_standard() for _ in range(12)]
    pkl = tmp_path / "eval_std.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)

    def run(num_workers):
        result = subprocess.run(
            [sys.executable, "evaluate.py",
             "--landlord", "random", "--landlord_up", "random",
             "--landlord_down", "random",
             "--eval_data", str(pkl), "--num_workers", str(num_workers),
             "--ruleset", "standard", "--eval_seed", "42"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        return result.stdout

    out1 = run(1)
    out2 = run(2)
    out3 = run(3)
    # All three runs must produce identical output.
    assert out1 == out2, f"1-worker != 2-worker:\n{out1}\n{out2}"
    assert out1 == out3, f"1-worker != 3-worker:\n{out1}\n{out3}"


# --------------------------------------------------------------------------- #
# Mixed dataset rejection
# --------------------------------------------------------------------------- #
def test_mixed_legacy_and_standard_rejected(tmp_path):
    """A dataset mixing legacy and v2 records must be rejected."""
    np.random.seed(42)
    legacy_deal = generate()
    standard_deal = generate_standard()
    mixed = [standard_deal, legacy_deal]
    pkl = tmp_path / "mixed.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(mixed, f)
    with pytest.raises(ValueError, match="different format"):
        load_eval_data(str(pkl), ruleset="standard")


def test_evaluate_cli_calls_adapter_and_rejects_wrong_format(tmp_path):
    """evaluate.py --ruleset standard with legacy data must fail up front."""
    import subprocess
    import sys

    # Generate legacy data.
    np.random.seed(42)
    legacy_data = [generate() for _ in range(3)]
    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(legacy_data, f)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--landlord", "random", "--landlord_up", "random",
         "--landlord_down", "random",
         "--eval_data", str(pkl), "--num_workers", "1",
         "--ruleset", "standard", "--eval_seed", "42"],
        capture_output=True, text=True, timeout=30,
    )
    # Must fail (not hang, not crash inside a worker).
    assert result.returncode != 0
    # The error must mention the format mismatch.
    assert "legacy format" in result.stderr or "legacy format" in result.stdout


def test_evaluate_cli_legacy_data_with_legacy_ruleset_succeeds(tmp_path):
    """evaluate.py --ruleset legacy with legacy data must succeed."""
    import subprocess
    import sys

    np.random.seed(42)
    legacy_data = [generate() for _ in range(6)]
    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(legacy_data, f)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--landlord", "random", "--landlord_up", "random",
         "--landlord_down", "random",
         "--eval_data", str(pkl), "--num_workers", "1",
         "--ruleset", "legacy"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "WP results" in result.stdout


# --------------------------------------------------------------------------- #
# first_bidder actually changes bidding order
# --------------------------------------------------------------------------- #
def test_first_bidder_changes_bidding_order():
    """Different first_bidder values must produce different bidding orders."""
    from douzero.env.rules import RuleSet
    from douzero.env.game import GameEnv

    class Stub:
        def __init__(self):
            self.action = None
        def set_action(self, a):
            self.action = a
        def act(self, infoset):
            return infoset.legal_actions[0]

    players = {p: Stub() for p in ["landlord", "landlord_up", "landlord_down"]}
    env = GameEnv(players, ruleset=RuleSet.standard())

    deck = []
    for rank in range(3, 15):
        deck.extend([rank] * 4)
    deck.extend([17] * 4)
    deck.extend([20, 30])

    data = {
        'landlord': sorted(deck[:17]),
        'landlord_up': sorted(deck[17:34]),
        'landlord_down': sorted(deck[34:51]),
        'three_landlord_cards': sorted(deck[51:54]),
    }

    # first_bidder = "0" → order [0, 1, 2]
    env.card_play_init_standard(data, bidding_order=["0", "1", "2"])
    assert env.bidding_order == ["0", "1", "2"]
    assert env.acting_player_position == "0"

    # first_bidder = "1" → order [1, 2, 0]
    env.reset()
    env.card_play_init_standard(data, bidding_order=["1", "2", "0"])
    assert env.bidding_order == ["1", "2", "0"]
    assert env.acting_player_position == "1"

    # first_bidder = "2" → order [2, 0, 1]
    env.reset()
    env.card_play_init_standard(data, bidding_order=["2", "0", "1"])
    assert env.bidding_order == ["2", "0", "1"]
    assert env.acting_player_position == "2"


def test_invalid_bidding_order_rejected_by_adapter(tmp_path):
    """A bidding_order that is not a permutation of ['0','1','2'] must be rejected."""
    np.random.seed(42)
    data = [generate_standard()]
    data[0]['bidding_order'] = ['0', '1', '1']  # duplicate
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="permutation"):
        load_eval_data(str(pkl), ruleset="standard")


# --------------------------------------------------------------------------- #
# Missing ruleset_hash in v2 record is rejected
# --------------------------------------------------------------------------- #
def test_missing_ruleset_hash_rejected(tmp_path):
    """A v2 record without ruleset_hash must be rejected (not optional)."""
    np.random.seed(42)
    data = [generate_standard()]
    del data[0]['ruleset_hash']
    pkl = tmp_path / "bad.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="ruleset_hash"):
        load_eval_data(str(pkl), ruleset="standard")


# --------------------------------------------------------------------------- #
# Custom RuleSet affects GameResult
# --------------------------------------------------------------------------- #
def test_custom_ruleset_multiplier_affects_game_result():
    """A custom RuleSet with different multipliers must produce different scores."""
    from douzero.env.scoring import compute_game_result
    from douzero.env.rules import RuleSet

    rs_default = RuleSet.standard()
    rs_custom = RuleSet.from_dict({
        "ruleset_id": "standard",
        "bomb_multiplier": 3,  # default is 2
    })
    result_default = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=1,
        rocket_count=0,
        bid_value=1,
        ruleset=rs_default,
    )
    result_custom = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=1,
        rocket_count=0,
        bid_value=1,
        ruleset=rs_custom,
    )
    # Default: 2^1=2 multiplier; custom: 3^1=3 multiplier.
    assert result_default.total_multiplier == 2
    assert result_custom.total_multiplier == 3
    assert result_default.landlord_score != result_custom.landlord_score


# --------------------------------------------------------------------------- #
# max_multiplier breakdown consistency
# --------------------------------------------------------------------------- #
def test_max_multiplier_breakdown_consistency():
    """breakdown must be consistent: uncapped vs capped total_multiplier."""
    from douzero.env.scoring import compute_game_result
    from douzero.env.rules import RuleSet

    rs = RuleSet.from_dict({"ruleset_id": "standard", "max_multiplier": 4})
    result = compute_game_result(
        played_cards={},
        action_counts={"landlord": 5, "landlord_up": 3, "landlord_down": 3},
        winner_position="landlord",
        bomb_count=3,  # 2^3 = 8 > cap of 4
        rocket_count=0,
        bid_value=1,
        ruleset=rs,
    )
    bd = result.multiplier_breakdown
    # uncapped_multiplier (8) != total_multiplier (capped to 4).
    assert bd["uncapped_multiplier"] == 8
    assert bd["total_multiplier"] == 4
    assert result.total_multiplier == 4
    assert "max_multiplier_cap" in bd and bd["max_multiplier_cap"] == 4
