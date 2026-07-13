"""P08 human-game record schema, serialization, and adapters tests."""

from __future__ import annotations

import json

import pytest

from douzero.env.rules import RuleSet
from douzero.human_data import (
    CANONICAL_FORMAT_VERSION,
    HUMAN_RECORD_KIND,
    HUMAN_RECORD_SCHEMA_VERSION,
    HumanGameRecord,
    RecordValidationError,
    read_jsonl,
    record_from_dict,
    record_from_jsonl_line,
    write_jsonl,
)
from douzero.human_data.adapters import (
    Adapter,
    assert_no_forbidden_metadata,
    audit_source_metadata,
)


# --------------------------------------------------------------------------- #
# Minimal-record factory
# --------------------------------------------------------------------------- #
def _minimal_payload(*, game_id: str = "g1") -> dict:
    rs = RuleSet.legacy()
    return {
        "format_version": CANONICAL_FORMAT_VERSION,
        "schema_version": HUMAN_RECORD_SCHEMA_VERSION,
        "kind": HUMAN_RECORD_KIND,
        "game_id": game_id,
        "ruleset_id": rs.ruleset_id,
        "ruleset_version": rs.ruleset_version,
        "ruleset_hash": rs.stable_hash(),
        "seats": ["landlord", "landlord_down", "landlord_up"],
        "initial_hands": {
            "landlord": [3, 3, 4, 4],
            "landlord_up": [5, 5, 6, 6],
            "landlord_down": [7, 7, 8, 8],
            "three_landlord_cards": [3, 4, 5],
        },
        "bottom_cards": [3, 4, 5],
        "bidding_history": [],
        "action_history": [
            ["landlord", [3, 3]],
            ["landlord_down", [7]],
        ],
        "final_result": {
            "winner_team": "landlord",
            "winner_position": "landlord",
        },
        "player_skill_weight": {"landlord": 1.0},
        "source_metadata": {"source": "test"},
        "timestamp": "2026-01",
    }


class TestSchemaConstruction:
    def test_minimal_record_constructs_and_stamps_kind(self):
        rec = record_from_dict(_minimal_payload())
        assert rec.game_id == "g1"
        assert rec.kind == HUMAN_RECORD_KIND
        assert rec.format_version == CANONICAL_FORMAT_VERSION
        assert rec.schema_version == HUMAN_RECORD_SCHEMA_VERSION
        # Card tuples are canonical (sorted, immutable).
        assert rec.initial_hands["landlord"] == (3, 3, 4, 4)
        assert rec.bottom_cards == (3, 4, 5)
        assert rec.action_history[0] == ("landlord", (3, 3))
        # Mapping fields are read-only MappingProxyType.
        with pytest.raises(TypeError):
            rec.final_result["x"] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            rec.initial_hands["landlord"] = (1,)  # type: ignore[index]

    def test_winner_team_validation_rejects_unknown(self):
        payload = _minimal_payload()
        payload["final_result"]["winner_team"] = "nobody"
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_negative_card_rejected(self):
        payload = _minimal_payload()
        payload["initial_hands"]["landlord"] = [3, -1]
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_bool_card_rejected(self):
        payload = _minimal_payload()
        payload["action_history"][0][1].append(True)  # type: ignore[attr-defined]
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_negative_skill_weight_rejected(self):
        payload = _minimal_payload()
        payload["player_skill_weight"]["landlord"] = -0.5
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_missing_required_key_rejected(self):
        payload = _minimal_payload()
        del payload["action_history"]
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)


class TestEnvelopeRejection:
    def test_wrong_kind_rejected(self):
        payload = _minimal_payload()
        payload["kind"] = "something_else"
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_wrong_format_version_rejected(self):
        payload = _minimal_payload()
        payload["format_version"] = 999
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_wrong_schema_version_rejected(self):
        payload = _minimal_payload()
        payload["schema_version"] = 999
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_non_mapping_rejected(self):
        with pytest.raises(RecordValidationError):
            record_from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


class TestSerialization:
    def test_jsonl_line_roundtrip(self):
        rec = record_from_dict(_minimal_payload())
        line = rec.to_jsonl_line()
        assert "\n" not in line
        rec2 = record_from_jsonl_line(line)
        assert rec2.to_dict() == rec.to_dict()

    def test_dict_roundtrip_is_json_serializable(self):
        rec = record_from_dict(_minimal_payload())
        d = rec.to_dict()
        # Must be plain JSON-serializable (no numpy/tuples leaking).
        s = json.dumps(d, sort_keys=True)
        d2 = json.loads(s)
        rec2 = record_from_dict(d2)
        assert rec2.game_id == rec.game_id

    def test_empty_jsonl_line_rejected(self):
        with pytest.raises(RecordValidationError):
            record_from_jsonl_line("   ")

    def test_invalid_json_rejected(self):
        with pytest.raises(RecordValidationError):
            record_from_jsonl_line("{not json")

    def test_write_and_read_jsonl(self, tmp_path):
        recs = [
            record_from_dict(_minimal_payload(game_id=f"g{i}"))
            for i in range(3)
        ]
        path = str(tmp_path / "games.jsonl")
        n = write_jsonl(recs, path)
        assert n == 3
        loaded = list(read_jsonl(path))
        assert len(loaded) == 3
        assert [r.game_id for r in loaded] == ["g0", "g1", "g2"]
        assert loaded[0].to_dict() == recs[0].to_dict()


class TestAdapters:
    def test_adapter_protocol_is_runtime_checkable(self):
        def my_adapter(raw):
            return record_from_dict(_minimal_payload(game_id=raw["id"]))

        assert isinstance(my_adapter, Adapter)

    def test_audit_drops_forbidden_keys(self):
        cleaned = audit_source_metadata(
            {"source": "x", "user_id": 123, "email": "a@b.c", "keep": 1}
        )
        assert "user_id" not in cleaned
        assert "email" not in cleaned
        assert cleaned["source"] == "x"
        assert cleaned["keep"] == 1

    def test_audit_drops_credential_like_keys(self):
        cleaned = audit_source_metadata(
            {"auth_token": "abc", "session_cookie": "z", "source": "ok"}
        )
        assert "auth_token" not in cleaned
        assert "session_cookie" not in cleaned
        assert cleaned == {"source": "ok"}

    def test_assert_no_forbidden_metadata_raises(self):
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata({"user_id": 1})

    def test_assert_no_forbidden_metadata_passes_clean(self):
        assert_no_forbidden_metadata({"source": "ok"})  # no raise
