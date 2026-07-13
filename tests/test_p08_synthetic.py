"""P08 synthetic game generator tests."""

from __future__ import annotations

import pytest

from douzero.env.rules import RuleSet
from douzero.human_data import HumanGameRecord, read_jsonl, write_jsonl
from douzero.human_data.synthetic import (
    generate_synthetic_record,
    generate_synthetic_records,
)


class TestSyntheticGenerator:
    def test_single_record_is_well_formed(self):
        rec = generate_synthetic_record("syn-001", seed=42)
        assert isinstance(rec, HumanGameRecord)
        assert rec.game_id == "syn-001"
        assert rec.ruleset_id == "legacy"
        assert rec.ruleset_hash == RuleSet.legacy().stable_hash()
        # Legacy cardplay conservation: 20 + 17 + 17 hands, 3 bottom cards.
        assert len(rec.initial_hands["landlord"]) == 20
        assert len(rec.initial_hands["landlord_up"]) == 17
        assert len(rec.initial_hands["landlord_down"]) == 17
        assert len(rec.initial_hands["three_landlord_cards"]) == 3
        assert len(rec.bottom_cards) == 3
        # Action history is non-empty and uses canonical roles.
        assert len(rec.action_history) > 0
        for pos, cards in rec.action_history:
            assert pos in ("landlord", "landlord_down", "landlord_up")
            assert isinstance(cards, tuple)
        # Terminal result is populated.
        assert rec.final_result["winner_team"] in ("landlord", "farmer")

    def test_seed_reproducibility(self):
        r1 = generate_synthetic_record("a", seed=7)
        r2 = generate_synthetic_record("a", seed=7)
        assert r1.to_dict() == r2.to_dict()

    def test_different_seeds_yield_different_deals(self):
        r1 = generate_synthetic_record("a", seed=1)
        r2 = generate_synthetic_record("b", seed=2)
        assert r1.initial_hands["landlord"] != r2.initial_hands["landlord"]

    def test_generate_records_stream_is_deterministic(self):
        batch1 = list(generate_synthetic_records(num_games=3, base_seed=100))
        batch2 = list(generate_synthetic_records(num_games=3, base_seed=100))
        assert len(batch1) == 3
        assert [r.to_dict() for r in batch1] == [r.to_dict() for r in batch2]
        # Distinct game ids.
        assert len({r.game_id for r in batch1}) == 3

    def test_generated_records_roundtrip_through_jsonl(self, tmp_path):
        recs = list(generate_synthetic_records(num_games=2, base_seed=5))
        path = str(tmp_path / "syn.jsonl")
        n = write_jsonl(recs, path)
        assert n == 2
        loaded = list(read_jsonl(path))
        assert [r.to_dict() for r in loaded] == [r.to_dict() for r in recs]

    def test_negative_num_games_rejected(self):
        with pytest.raises(ValueError):
            list(generate_synthetic_records(num_games=-1, base_seed=1))

    def test_generated_actions_use_sorted_tuples(self):
        rec = generate_synthetic_record("syn-sort", seed=11)
        for _pos, cards in rec.action_history:
            assert list(cards) == sorted(cards)
