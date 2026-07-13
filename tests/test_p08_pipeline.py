"""P08 validation-by-replay, quarantine, ingest, and split tests."""

from __future__ import annotations

import json

import pytest

from douzero.human_data import HumanGameRecord, write_jsonl
from douzero.human_data.ingest import (
    IngestError,
    dedupe_by_game_id,
    ingest_batch,
    ingest_record,
)
from douzero.human_data.split import (
    Split,
    SplitConfig,
    SplitError,
    split_records,
    split_stats,
)
from douzero.human_data.synthetic import (
    generate_synthetic_record,
    generate_synthetic_records,
)
from douzero.human_data.validate import (
    ReplayValidationError,
    ValidationReport,
    validate_deal_conservation,
    validate_record,
    validate_records,
)


# --------------------------------------------------------------------------- #
# Replay validation
# --------------------------------------------------------------------------- #
class TestValidateRecord:
    def test_synthetic_record_is_valid(self):
        rec = generate_synthetic_record("syn-v1", seed=3)
        result = validate_record(rec)
        assert result.ok
        assert result.reason == "ok"
        assert result.error == ""

    def test_batch_partitions_into_valid_and_quarantined(self):
        recs = [generate_synthetic_record(f"s{i}", seed=i) for i in range(3)]
        report = validate_records(recs)
        assert isinstance(report, ValidationReport)
        assert report.total == 3
        assert report.num_valid == 3
        assert report.num_quarantined == 0

    def test_illegal_action_is_quarantined_not_silently_dropped(self):
        """A record whose action is not legal is quarantined with a reason."""
        rec = generate_synthetic_record("syn-v2", seed=4)
        # Corrupt the first action to an illegal card combo (a single high card
        # is usually legal, so instead play a card the landlord does not hold).
        good_pos, _good_cards = rec.action_history[0]
        bad = (good_pos, (30, 20))  # rocket that the role likely cannot play
        corrupted = HumanGameRecord(
            game_id=rec.game_id,
            ruleset_id=rec.ruleset_id,
            ruleset_version=rec.ruleset_version,
            ruleset_hash=rec.ruleset_hash,
            seats=rec.seats,
            initial_hands=rec.initial_hands,
            bottom_cards=rec.bottom_cards,
            bidding_history=rec.bidding_history,
            action_history=(bad,) + rec.action_history[1:],
            final_result=rec.final_result,
            player_skill_weight=rec.player_skill_weight,
            source_metadata=rec.source_metadata,
            timestamp=rec.timestamp,
        )
        result = validate_record(corrupted)
        assert not result.ok
        assert result.reason in ("illegal_action", "turn_order_mismatch")
        # The quarantined record is still carried (never silently dropped).
        assert result.record.game_id == corrupted.game_id

    def test_deal_conservation_rejects_short_landlord(self):
        rec = generate_synthetic_record("syn-v3", seed=5)
        bad_hands = dict(rec.initial_hands)
        bad_hands = {k: list(v) for k, v in bad_hands.items()}
        bad_hands["landlord"] = bad_hands["landlord"][:19]  # 19 cards
        bad = HumanGameRecord(
            game_id=rec.game_id,
            ruleset_id=rec.ruleset_id,
            ruleset_version=rec.ruleset_version,
            ruleset_hash=rec.ruleset_hash,
            seats=rec.seats,
            initial_hands=bad_hands,
            bottom_cards=rec.bottom_cards,
            bidding_history=rec.bidding_history,
            action_history=rec.action_history,
            final_result=rec.final_result,
        )
        with pytest.raises(ReplayValidationError):
            validate_deal_conservation(bad)

    def test_winner_mismatch_is_caught(self):
        rec = generate_synthetic_record("syn-v4", seed=6)
        flipped = "farmer" if rec.final_result["winner_team"] == "landlord" else "landlord"
        bad_result = dict(rec.final_result)
        bad_result["winner_team"] = flipped
        bad = HumanGameRecord(
            game_id=rec.game_id,
            ruleset_id=rec.ruleset_id,
            ruleset_version=rec.ruleset_version,
            ruleset_hash=rec.ruleset_hash,
            seats=rec.seats,
            initial_hands=rec.initial_hands,
            bottom_cards=rec.bottom_cards,
            bidding_history=rec.bidding_history,
            action_history=rec.action_history,
            final_result=bad_result,
        )
        result = validate_record(bad)
        assert not result.ok
        assert result.reason == "winner_mismatch"


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
class TestIngest:
    def test_ingest_record_runs_adapter_and_sanitizes(self):
        rec = generate_synthetic_record("ing-1", seed=1)

        def adapter(raw):
            return HumanGameRecord(
                game_id=raw["game_id"],
                ruleset_id=rec.ruleset_id,
                ruleset_version=rec.ruleset_version,
                ruleset_hash=rec.ruleset_hash,
                seats=rec.seats,
                initial_hands=rec.initial_hands,
                bottom_cards=rec.bottom_cards,
                action_history=rec.action_history,
                final_result=rec.final_result,
                source_metadata={"source": "test", "user_id": "LEAK"},
            )

        # The adapter here intentionally leaves a forbidden key; ingest must
        # reject it even though the adapter produced a valid record otherwise.
        with pytest.raises(IngestError):
            ingest_record({"game_id": "ing-1"}, adapter)

    def test_ingest_record_clean_adapter_succeeds(self):
        rec = generate_synthetic_record("ing-2", seed=2)

        def adapter(raw):
            return HumanGameRecord(
                game_id=raw["game_id"],
                ruleset_id=rec.ruleset_id,
                ruleset_version=rec.ruleset_version,
                ruleset_hash=rec.ruleset_hash,
                seats=rec.seats,
                initial_hands=rec.initial_hands,
                bottom_cards=rec.bottom_cards,
                action_history=rec.action_history,
                final_result=rec.final_result,
            )

        out = ingest_record({"game_id": "ing-2"}, adapter)
        assert out.game_id == "ing-2"

    def test_dedupe_by_game_id_keeps_first(self):
        a = generate_synthetic_record("dup", seed=1)
        b = generate_synthetic_record("dup", seed=2)  # same id, diff content
        out = list(dedupe_by_game_id([a, b]))
        assert len(out) == 1
        assert out[0].to_dict() == a.to_dict()

    def test_ingest_batch_sorts_and_dedupes(self):
        base = generate_synthetic_record("z", seed=1)

        def adapter(raw):
            return HumanGameRecord(
                game_id=raw["id"],
                ruleset_id=base.ruleset_id,
                ruleset_version=base.ruleset_version,
                ruleset_hash=base.ruleset_hash,
                seats=base.seats,
                initial_hands=base.initial_hands,
                bottom_cards=base.bottom_cards,
                action_history=base.action_history,
                final_result=base.final_result,
            )

        raws = [{"id": "c"}, {"id": "a"}, {"id": "b"}, {"id": "a"}]
        out = ingest_batch(raws, adapter)
        assert [r.game_id for r in out] == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
class TestSplit:
    def _make(self, n: int, base_seed: int = 0):
        return list(generate_synthetic_records(num_games=n, base_seed=base_seed))

    def test_split_covers_all_records_exactly_once(self):
        recs = self._make(20)
        split = split_records(recs, SplitConfig(val_ratio=0.25, seed=1))
        all_ids = set(split.all_game_ids)
        assert all_ids == {r.game_id for r in recs}
        assert len(split.train) + len(split.val) + len(split.test) == 20

    def test_split_has_no_game_id_overlap(self):
        recs = self._make(20)
        split = split_records(recs, SplitConfig(val_ratio=0.2, test_ratio=0.1))
        # assert_no_overlap is called inside split_records; double-check here.
        split.assert_no_overlap()

    def test_split_is_deterministic_for_same_seed(self):
        recs = self._make(20)
        s1 = split_records(recs, SplitConfig(val_ratio=0.2, seed=42))
        s2 = split_records(recs, SplitConfig(val_ratio=0.2, seed=42))
        assert s1.all_game_ids == s2.all_game_ids
        assert [r.game_id for r in s1.train] == [r.game_id for r in s2.train]

    def test_split_rejects_duplicate_input(self):
        a = generate_synthetic_record("dup", seed=1)
        b = generate_synthetic_record("dup", seed=2)
        with pytest.raises(SplitError):
            split_records([a, b])

    def test_split_rejects_invalid_ratios(self):
        with pytest.raises(SplitError):
            SplitConfig(val_ratio=0.6, test_ratio=0.5)
        with pytest.raises(SplitError):
            SplitConfig(val_ratio=-0.1)

    def test_split_stats_reports_team_counts(self):
        recs = self._make(20)
        split = split_records(recs, SplitConfig(val_ratio=0.2))
        stats = split_stats(split)
        for name in ("train", "val", "test"):
            assert "total" in stats[name]
            assert stats[name]["total"] == sum(
                v for k, v in stats[name].items() if k != "total"
            )
