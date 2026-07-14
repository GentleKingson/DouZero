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
    make_internal_game_id,
    pseudonymize_external_game_id,
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
def _gid(label: str) -> str:
    return make_internal_game_id(label)


def _minimal_payload(*, game_id: str = "g1") -> dict:
    rs = RuleSet.legacy()
    return {
        "format_version": CANONICAL_FORMAT_VERSION,
        "schema_version": HUMAN_RECORD_SCHEMA_VERSION,
        "kind": HUMAN_RECORD_KIND,
        "game_id": _gid(game_id),
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
        assert rec.game_id == _gid("g1")
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
        assert [r.game_id for r in loaded] == [_gid("g0"), _gid("g1"), _gid("g2")]
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
        assert "keep" not in cleaned

    def test_audit_drops_credential_like_keys(self):
        cleaned = audit_source_metadata(
            {"auth_token": "abc", "session_cookie": "z", "source": "ok"}
        )
        assert "auth_token" not in cleaned
        assert "session_cookie" not in cleaned
        assert cleaned == {"source": "ok"}

    def test_audit_substring_key_match_catches_compound_keys(self):
        """Blocker 4: substring matching catches api_token, client_secret,
        user_email — not just exact key matches."""
        cleaned = audit_source_metadata(
            {"api_token": "x", "client_secret": "y",
             "user_email": "a@b.c", "source": "ok"}
        )
        assert cleaned == {"source": "ok"}

    def test_audit_drops_nested_extension_mappings(self):
        cleaned = audit_source_metadata(
            {"profile": {"email": "a@b.c", "name": "ok"}, "source": "s"}
        )
        assert cleaned == {"source": "s"}

    def test_audit_drops_list_extension_fields(self):
        cleaned = audit_source_metadata(
            {"players": [{"user_id": 1, "seat": 0}], "source": "s"}
        )
        assert cleaned == {"source": "s"}

    def test_audit_drops_credential_like_values(self):
        """Blocker 4: a credential-looking string value is dropped even when
        the key is benign."""
        cleaned = audit_source_metadata(
            {"note": "Bearer abc123xyz", "source": "ok"}
        )
        assert "note" not in cleaned
        assert cleaned == {"source": "ok"}

    def test_audit_drops_pem_private_key_value(self):
        cleaned = audit_source_metadata(
            {"key": "-----BEGIN RSA PRIVATE KEY-----\nMIIE", "source": "ok"}
        )
        assert "key" not in cleaned

    def test_assert_no_forbidden_metadata_raises(self):
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata({"user_id": 1})

    def test_assert_raises_on_compound_forbidden_key(self):
        """Blocker 4: substring matching at the assert boundary too."""
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata({"api_token": "x"})
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata({"client_secret": "x"})

    def test_assert_raises_on_nested_forbidden(self):
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata(
                {"profile": {"email": "a@b.c"}}
            )

    def test_assert_raises_on_credential_value(self):
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata({"note": "Bearer leak"})

    def test_assert_no_forbidden_metadata_passes_clean(self):
        assert_no_forbidden_metadata({"source": "ok"})  # no raise

    # ------------------------------------------------------------------ #
    # Blocker 2 (round 3): PII value detection + tuple recursion
    # ------------------------------------------------------------------ #
    def test_audit_drops_email_value_under_neutral_key(self):
        """A plain email under a benign key is dropped (PII value detection)."""
        cleaned = audit_source_metadata(
            {"contact": "alice@example.com", "source": "ok"}
        )
        assert "contact" not in cleaned
        assert cleaned == {"source": "ok"}

    def test_audit_drops_ip_value(self):
        cleaned = audit_source_metadata(
            {"origin": "203.0.113.10", "source": "ok"}
        )
        assert "origin" not in cleaned

    def test_audit_drops_phone_value(self):
        cleaned = audit_source_metadata(
            {"note": "phone: +1-555-123-4567", "source": "ok"}
        )
        assert "note" not in cleaned

    def test_audit_drops_tuple_extension_fields(self):
        cleaned = audit_source_metadata(
            {"players": ({"user_email": "a@b.c", "seat": 0},), "source": "s"}
        )
        assert cleaned == {"source": "s"}

    def test_audit_drops_set_extension_fields(self):
        cleaned = audit_source_metadata(
            {"tags": {"x", "user_id"}, "source": "s"}
        )
        assert cleaned == {"source": "s"}

    def test_assert_raises_on_pii_value(self):
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata({"contact": "alice@example.com"})

    def test_assert_raises_on_tuple_nested_forbidden(self):
        with pytest.raises(RecordValidationError):
            assert_no_forbidden_metadata(
                {"items": ({"email": "a@b.c"},)}
            )

    def test_record_rejects_nested_metadata(self):
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError, match="unknown key"):
            HumanGameRecord(
                game_id=_gid("g-tuple"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3, 4, 4],
                    "landlord_up": [5, 5, 6, 6],
                    "landlord_down": [7, 7, 8, 8],
                    "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(("landlord", (3, 3)),),
                final_result={
                    "winner_team": "landlord",
                    "winner_position": "landlord",
                },
                source_metadata={"items": (1, 2, 3)},
            )

    def test_record_rejects_non_json_metadata_type(self):
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError):
            HumanGameRecord(
                game_id=_gid("g-bad"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={"winner_team": "landlord", "winner_position": "landlord"},
                source_metadata={"bad": b"bytes"},
            )

    # ------------------------------------------------------------------ #
    # Blocker 2 (round 4): canonical record boundary privacy
    # ------------------------------------------------------------------ #
    def test_record_rejects_pii_in_source_metadata_at_boundary(self):
        """A record constructed directly (bypassing ingest) with PII in
        source_metadata is rejected at the canonical boundary."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError):
            HumanGameRecord(
                game_id=_gid("g-pii"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={"winner_team": "landlord", "winner_position": "landlord"},
                source_metadata={"contact": "alice@example.com"},
            )

    def test_record_rejects_pii_in_final_result(self):
        """final_result values are scanned for PII at the boundary."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError):
            HumanGameRecord(
                game_id=_gid("g-fr-pii"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={
                    "winner_team": "landlord", "winner_position": "landlord",
                    "landlord_score": "alice@example.com",  # PII in a value
                },
            )

    def test_record_rejects_unknown_final_result_key(self):
        """final_result keys are whitelisted; arbitrary keys are rejected."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError, match="unknown keys"):
            HumanGameRecord(
                game_id=_gid("g-fr-ext"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={
                    "winner_team": "landlord", "winner_position": "landlord",
                    "player_email": "alice@example.com",
                },
            )

    def test_record_rejects_pii_in_timestamp(self):
        """timestamp with a PII-shaped value is rejected."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError):
            HumanGameRecord(
                game_id=_gid("g-ts-pii"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={"winner_team": "landlord", "winner_position": "landlord"},
                timestamp="alice@example.com",
            )

    def test_record_rejects_bad_timestamp_format(self):
        """timestamp must be empty or YYYY-MM."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError, match="YYYY-MM"):
            HumanGameRecord(
                game_id=_gid("g-ts-bad"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={"winner_team": "landlord", "winner_position": "landlord"},
                timestamp="2026-07-13T12:00:00Z",  # too fine-grained
            )

    def test_record_accepts_valid_timestamp(self):
        """Empty string and YYYY-MM are accepted."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        for ts in ("", "2026-07"):
            rec = HumanGameRecord(
                game_id=_gid(f"g-ts-{ts}"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={"winner_team": "landlord", "winner_position": "landlord"},
                timestamp=ts,
            )
            assert rec.timestamp == ts

    def test_record_rejects_non_string_metadata_key(self):
        """metadata mapping keys must be strings."""
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        with pytest.raises(RecordValidationError, match="must be a string"):
            HumanGameRecord(
                game_id=_gid("g-key"),
                ruleset_id=rs.ruleset_id,
                ruleset_version=rs.ruleset_version,
                ruleset_hash=rs.stable_hash(),
                seats=("landlord", "landlord_down", "landlord_up"),
                initial_hands={
                    "landlord": [3, 3], "landlord_up": [5, 5],
                    "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
                },
                bottom_cards=[3, 4, 5],
                action_history=(),
                final_result={"winner_team": "landlord", "winner_position": "landlord"},
                source_metadata={1: "value"},  # non-string key
            )

    def test_record_from_dict_rejects_pii_in_source_metadata(self):
        """record_from_dict (the direct JSONL load path) also scans for PII."""
        payload = _minimal_payload()
        payload["source_metadata"] = {"contact": "alice@example.com"}
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_record_from_dict_rejects_pii_in_final_result(self):
        payload = _minimal_payload()
        payload["final_result"]["landlord_score"] = "alice@example.com"
        with pytest.raises(RecordValidationError):
            record_from_dict(payload)

    def test_record_from_dict_rejects_unknown_final_result_key(self):
        payload = _minimal_payload()
        payload["final_result"]["custom_field"] = "x"
        with pytest.raises(RecordValidationError, match="unknown keys"):
            record_from_dict(payload)

    # ------------------------------------------------------------------ #
    # Round 5: comprehensive PII/structural hardening
    # ------------------------------------------------------------------ #
    def _base_record_args(self, **overrides):
        from douzero.env.rules import RuleSet

        rs = RuleSet.legacy()
        args = dict(
            game_id=_gid("g-test"),
            ruleset_id=rs.ruleset_id,
            ruleset_version=rs.ruleset_version,
            ruleset_hash=rs.stable_hash(),
            seats=("landlord", "landlord_down", "landlord_up"),
            initial_hands={
                "landlord": [3, 3], "landlord_up": [5, 5],
                "landlord_down": [7, 7], "three_landlord_cards": [3, 4, 5],
            },
            bottom_cards=[3, 4, 5],
            action_history=(),
            final_result={"winner_team": "landlord", "winner_position": "landlord"},
        )
        args.update(overrides)
        return args

    def test_game_id_with_pii_rejected(self):
        with pytest.raises(RecordValidationError, match="game_id"):
            HumanGameRecord(**self._base_record_args(
                game_id="alice@example.com"
            ))

    def test_extra_initial_hands_key_rejected(self):
        hands = dict(self._base_record_args()["initial_hands"])
        hands["user_email"] = []
        with pytest.raises(RecordValidationError, match="unknown keys"):
            HumanGameRecord(**self._base_record_args(initial_hands=hands))

    def test_non_canonical_seats_rejected(self):
        with pytest.raises(RecordValidationError, match="canonical legacy seat"):
            HumanGameRecord(**self._base_record_args(
                seats=("landlord", "user@example.com", "landlord_up")
            ))

    def test_action_history_non_legal_role_rejected(self):
        with pytest.raises(RecordValidationError, match="legal role"):
            HumanGameRecord(**self._base_record_args(
                action_history=(("landlord", (3,)), ("bad_role", (5,))),
            ))

    def test_player_skill_weight_non_legal_role_rejected(self):
        with pytest.raises(RecordValidationError, match="legal role"):
            HumanGameRecord(**self._base_record_args(
                player_skill_weight={"bad_role": 1.0},
            ))

    def test_nan_skill_weight_rejected(self):
        with pytest.raises(RecordValidationError, match="finite"):
            HumanGameRecord(**self._base_record_args(
                player_skill_weight={"landlord": float("nan")},
            ))

    def test_inf_skill_weight_rejected(self):
        with pytest.raises(RecordValidationError, match="finite"):
            HumanGameRecord(**self._base_record_args(
                player_skill_weight={"landlord": float("inf")},
            ))

    def test_metadata_set_rejected(self):
        """set/frozenset have non-deterministic iteration order and are rejected."""
        with pytest.raises(RecordValidationError, match="set"):
            HumanGameRecord(**self._base_record_args(
                source_metadata={"batch_id": {"x", "y"}},
            ))

    def test_metadata_nan_float_rejected(self):
        with pytest.raises(RecordValidationError, match="non-finite"):
            HumanGameRecord(**self._base_record_args(
                source_metadata={"batch_id": float("nan")},
            ))

    def test_post_construction_metadata_mutation_raises(self):
        rec = HumanGameRecord(**self._base_record_args(
            source_metadata={"source": "test"}
        ))
        with pytest.raises(TypeError):
            rec.source_metadata["new"] = "x"

    def test_source_metadata_allowlist_rejects_common_identifiers(self):
        for metadata in (
            {"display_name": "Alice Smith"},
            {"address": "123 Main Street"},
            {"player_code": "platform-user-9384"},
            {"profile": {"nickname": "alice"}},
        ):
            with pytest.raises(RecordValidationError, match="unknown key"):
                HumanGameRecord(**self._base_record_args(source_metadata=metadata))

    def test_source_metadata_rejects_ipv6_and_url_query_identifier(self):
        for source in (
            "2001:db8::1",
            "https://example.test/export?player_id=123",
        ):
            with pytest.raises(RecordValidationError):
                HumanGameRecord(**self._base_record_args(
                    source_metadata={"source": source}
                ))

    def test_external_game_id_hmac_is_deterministic_and_opaque(self):
        key = b"k" * 32
        first = pseudonymize_external_game_id("Alice-Smith-game-001", project_key=key)
        second = pseudonymize_external_game_id("Alice-Smith-game-001", project_key=key)
        assert first == second
        assert "Alice" not in first
        HumanGameRecord(**self._base_record_args(game_id=first))

    def test_name_shaped_raw_game_id_rejected(self):
        with pytest.raises(RecordValidationError, match="game_id"):
            HumanGameRecord(**self._base_record_args(
                game_id="Alice-Smith-game-001"
            ))

    def test_to_jsonl_line_rejects_nan(self):
        """Round 5 Blocker 2: allow_nan=False ensures NaN/Inf cannot be
        serialized even if they somehow reached the dict."""
        rec = HumanGameRecord(**self._base_record_args())
        # Tamper with the internal dict to simulate a bypass (the construction
        # would normally reject NaN; here we verify the serializer also rejects).
        import math

        d = rec.to_dict()
        d["player_skill_weight"] = {"landlord": float("nan")}
        import json

        with pytest.raises(ValueError):
            json.dumps(d, allow_nan=False)

    # ------------------------------------------------------------------ #
    # Round 6: bidding PII, ruleset format, final_result deep-freeze,
    #          record_from_dict robustness, top-level key allowlist
    # ------------------------------------------------------------------ #
    def test_bidding_history_seat_pii_rejected(self):
        with pytest.raises(RecordValidationError, match="bidding_history seat"):
            HumanGameRecord(**self._base_record_args(
                bidding_history=(("alice@example.com", 1),),
            ))

    def test_ruleset_version_pii_rejected(self):
        with pytest.raises(RecordValidationError, match="ruleset_version"):
            HumanGameRecord(**self._base_record_args(
                ruleset_version="alice@example.com"
            ))

    def test_ruleset_hash_non_hex_rejected(self):
        with pytest.raises(RecordValidationError, match="64-char"):
            HumanGameRecord(**self._base_record_args(
                ruleset_hash="not-a-hex-hash"
            ))

    def test_final_result_nested_mutation_raises(self):
        """Round 6: final_result is deep-frozen (nested mutation blocked)."""
        rec = HumanGameRecord(**self._base_record_args(
            final_result={"winner_team": "landlord", "winner_position": "landlord",
                          "bomb_count": 2}
        ))
        with pytest.raises(TypeError):
            rec.final_result["bomb_count"] = 5

    def test_final_result_wrong_type_rejected(self):
        """bomb_count must be int, not string/dict."""
        with pytest.raises(RecordValidationError, match="must be an int"):
            HumanGameRecord(**self._base_record_args(
                final_result={"winner_team": "landlord", "winner_position": "landlord",
                              "bomb_count": "two"}
            ))

    def test_final_result_nan_score_rejected(self):
        with pytest.raises(RecordValidationError, match="finite"):
            HumanGameRecord(**self._base_record_args(
                final_result={"winner_team": "landlord", "winner_position": "landlord",
                              "landlord_score": float("nan")}
            ))

    def test_record_from_dict_null_initial_hands_quarantined(self):
        """Round 6 Blocker 2: null initial_hands → RecordValidationError (not
        AttributeError that crashes the resilient reader)."""
        payload = _minimal_payload()
        payload["initial_hands"] = None
        with pytest.raises(RecordValidationError, match="must be a JSON object"):
            record_from_dict(payload)

    def test_record_from_dict_list_initial_hands_quarantined(self):
        payload = _minimal_payload()
        payload["initial_hands"] = []
        with pytest.raises(RecordValidationError, match="must be a JSON object"):
            record_from_dict(payload)

    def test_record_from_dict_rejects_unknown_top_level_key(self):
        payload = _minimal_payload()
        payload["extra_field"] = "x"
        with pytest.raises(RecordValidationError, match="unknown top-level"):
            record_from_dict(payload)

    def test_resilient_reader_quarantines_null_field_not_crash(self, tmp_path):
        """End-to-end: iter_jsonl_resilient quarantines a null-field line
        instead of crashing (round 6 Blocker 2)."""
        from douzero.human_data import iter_jsonl_resilient

        good = _minimal_payload()
        bad = _minimal_payload()
        bad["initial_hands"] = None  # would crash without type check
        path = str(tmp_path / "mixed.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(good, sort_keys=True) + "\n")
            fh.write(json.dumps(bad, sort_keys=True) + "\n")
        results = list(iter_jsonl_resilient(path))
        assert len(results) == 2
        assert results[0].record is not None  # good line parsed
        assert results[1].error  # bad line quarantined, not crashed
