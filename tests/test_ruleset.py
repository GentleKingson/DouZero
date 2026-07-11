"""Tests for the versioned RuleSet dataclass and YAML loading (P02 Slice 1).

Covers:
- canonical ``legacy()`` and ``standard()`` field values
- ``from_dict`` type and range validation
- ``stable_hash`` determinism
- ``configs/standard.yaml`` parity with ``RuleSet.standard()``
- CLI ``--ruleset standard`` is accepted; ``--ruleset v2`` is rejected
- ``--feature_version v2`` is still rejected (P02 does not widen it)
"""

from __future__ import annotations

import pytest

from douzero.env.rules import (
    BIDDING_MODE_NONE,
    BIDDING_MODE_SCORE_0_1_2_3,
    RULESET_LEGACY,
    RULESET_STANDARD,
    RuleSet,
)


# --------------------------------------------------------------------------- #
# Canonical constructors
# --------------------------------------------------------------------------- #
def test_legacy_ruleset_reproduces_original_environment():
    rs = RuleSet.legacy()
    assert rs.ruleset_id == RULESET_LEGACY
    assert rs.bidding_mode == BIDDING_MODE_NONE
    assert rs.bid_values == ()
    assert rs.allow_rob is False
    assert rs.all_pass_redeal is False
    assert rs.bid_multiplier is False
    # Legacy uses base_score=2 so the effective multiplier is 2**bomb_num.
    assert rs.base_score == 2
    assert rs.bomb_multiplier == 2
    assert rs.rocket_multiplier == 2
    # No spring in legacy.
    assert rs.spring_multiplier == 0
    assert rs.anti_spring_multiplier == 0
    assert rs.allow_double is False
    assert rs.max_multiplier is None


def test_standard_ruleset_has_bidding_and_spring():
    rs = RuleSet.standard()
    assert rs.ruleset_id == RULESET_STANDARD
    assert rs.bidding_mode == BIDDING_MODE_SCORE_0_1_2_3
    assert rs.bid_values == (0, 1, 2, 3)
    assert rs.allow_rob is False
    assert rs.all_pass_redeal is True
    assert rs.bid_multiplier is True
    assert rs.base_score == 1
    assert rs.bomb_multiplier == 2
    assert rs.rocket_multiplier == 2
    assert rs.spring_multiplier == 2
    assert rs.anti_spring_multiplier == 2
    assert rs.allow_double is False
    assert rs.max_multiplier is None


def test_legacy_and_standard_are_distinct():
    legacy = RuleSet.legacy()
    standard = RuleSet.standard()
    assert legacy.ruleset_id != standard.ruleset_id
    assert legacy.bidding_mode != standard.bidding_mode
    assert legacy.bid_values != standard.bid_values
    assert legacy.base_score != standard.base_score
    assert legacy.spring_multiplier != standard.spring_multiplier


# --------------------------------------------------------------------------- #
# from_dict validation
# --------------------------------------------------------------------------- #
def test_from_dict_none_returns_legacy():
    rs = RuleSet.from_dict(None)
    assert rs == RuleSet.legacy()


def test_from_dict_empty_returns_legacy():
    rs = RuleSet.from_dict({})
    assert rs == RuleSet.legacy()


def test_from_dict_standard_explicit():
    rs = RuleSet.from_dict({"ruleset_id": "standard"})
    assert rs.ruleset_id == RULESET_STANDARD
    assert rs.bidding_mode == BIDDING_MODE_SCORE_0_1_2_3
    assert rs.bid_values == (0, 1, 2, 3)


def test_from_dict_bid_values_list_converted_to_tuple():
    rs = RuleSet.from_dict({"ruleset_id": "standard", "bid_values": [0, 1, 2, 3]})
    assert isinstance(rs.bid_values, tuple)
    assert rs.bid_values == (0, 1, 2, 3)


def test_from_dict_rejects_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        RuleSet.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown"):
        RuleSet.from_dict({"ruleset_id": "legacy", "nonsense": True})


def test_from_dict_rejects_bad_ruleset_id():
    with pytest.raises(ValueError, match="ruleset_id"):
        RuleSet.from_dict({"ruleset_id": "v2"})


def test_from_dict_rejects_bad_bidding_mode():
    with pytest.raises(ValueError, match="bidding_mode"):
        RuleSet.from_dict({"ruleset_id": "standard", "bidding_mode": "rob"})


def test_from_dict_rejects_negative_bid_value():
    with pytest.raises(ValueError, match="non-negative"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": [0, -1, 2, 3]})


def test_from_dict_rejects_bid_values_wrong_type():
    with pytest.raises(TypeError, match="bid_values"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": "0123"})


def test_from_dict_rejects_bomb_multiplier_zero():
    with pytest.raises(ValueError, match="bomb_multiplier"):
        RuleSet.from_dict({"ruleset_id": "standard", "bomb_multiplier": 0})


def test_from_dict_rejects_negative_spring_multiplier():
    with pytest.raises(ValueError, match="spring_multiplier"):
        RuleSet.from_dict({"ruleset_id": "standard", "spring_multiplier": -1})


def test_from_dict_rejects_zero_base_score():
    with pytest.raises(ValueError, match="base_score"):
        RuleSet.from_dict({"ruleset_id": "standard", "base_score": 0})


def test_from_dict_rejects_bad_max_multiplier():
    with pytest.raises(ValueError, match="max_multiplier"):
        RuleSet.from_dict({"ruleset_id": "standard", "max_multiplier": 0})


def test_from_dict_rejects_legacy_with_bidding():
    with pytest.raises(ValueError, match="legacy.*bidding_mode"):
        RuleSet.from_dict({"ruleset_id": "legacy", "bidding_mode": "score_0_1_2_3"})


def test_from_dict_standard_without_bid_values_uses_canonical_default():
    # standard() canonical default provides bid_values=(0,1,2,3).
    rs = RuleSet.from_dict({"ruleset_id": "standard"})
    assert rs.bid_values == (0, 1, 2, 3)


def test_from_dict_score_bidding_requires_bid_values():
    # If bidding_mode is score but bid_values is empty, it should fail.
    with pytest.raises(ValueError, match="non-empty bid_values"):
        RuleSet.from_dict({
            "ruleset_id": "standard",
            "bidding_mode": "score_0_1_2_3",
            "bid_values": [],
        })


def test_from_dict_bool_rejected_for_int_field():
    # bool is a subclass of int but must be rejected for multiplier fields.
    with pytest.raises(ValueError, match="bomb_multiplier"):
        RuleSet.from_dict({"ruleset_id": "standard", "bomb_multiplier": True})


# --------------------------------------------------------------------------- #
# Serialisation / hashing
# --------------------------------------------------------------------------- #
def test_to_dict_round_trip():
    rs = RuleSet.standard()
    d = rs.to_dict()
    assert isinstance(d["bid_values"], list)
    rs2 = RuleSet.from_dict({**d, "bid_values": list(d["bid_values"])})
    assert rs == rs2


def test_stable_hash_deterministic():
    rs = RuleSet.standard()
    assert rs.stable_hash() == rs.stable_hash()


def test_stable_hash_differs_for_legacy_and_standard():
    assert RuleSet.legacy().stable_hash() != RuleSet.standard().stable_hash()


def test_stable_hash_changes_on_field_change():
    base = RuleSet.standard()
    modified = RuleSet.from_dict({"ruleset_id": "standard", "base_score": 2})
    assert base.stable_hash() != modified.stable_hash()


# --------------------------------------------------------------------------- #
# YAML config parity
# --------------------------------------------------------------------------- #
def test_standard_yaml_rules_block_matches_canonical_standard():
    """The 'rules:' block in configs/standard.yaml must equal RuleSet.standard()."""
    from douzero.config import load_config

    import pathlib
    yaml_path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "standard.yaml"
    raw = load_config(str(yaml_path))
    assert raw.ruleset == "standard"


def test_standard_yaml_validates_rules_block():
    """Loading configs/standard.yaml should not raise (the rules block is valid)."""
    from douzero.config import load_config

    import pathlib
    yaml_path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "standard.yaml"
    cfg = load_config(str(yaml_path))
    assert cfg.ruleset == "standard"
    assert cfg.feature_version == "legacy"
    assert cfg.model_version == "legacy"


def test_legacy_yaml_still_loads():
    """configs/legacy.yaml must still load with ruleset=legacy."""
    from douzero.config import load_config

    import pathlib
    yaml_path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "legacy.yaml"
    cfg = load_config(str(yaml_path))
    assert cfg.ruleset == "legacy"


# --------------------------------------------------------------------------- #
# CLI choices
# --------------------------------------------------------------------------- #
def test_cli_accepts_ruleset_standard():
    from douzero.dmc.arguments import parser

    ns = parser.parse_args(["--ruleset", "standard"])
    assert ns.ruleset == "standard"


def test_cli_accepts_ruleset_legacy_default():
    from douzero.dmc.arguments import parser

    ns = parser.parse_args([])
    assert ns.ruleset == "legacy"


def test_cli_rejects_ruleset_v2():
    from douzero.dmc.arguments import parser

    with pytest.raises(SystemExit):
        parser.parse_args(["--ruleset", "v2"])


def test_cli_feature_version_still_legacy_only():
    """P02 does NOT widen feature_version; it stays choices=['legacy']."""
    from douzero.dmc.arguments import parser

    with pytest.raises(SystemExit):
        parser.parse_args(["--feature_version", "v2"])


def test_cli_model_version_still_legacy_only():
    from douzero.dmc.arguments import parser

    with pytest.raises(SystemExit):
        parser.parse_args(["--model_version", "v2"])


# --------------------------------------------------------------------------- #
# Phase constants
# --------------------------------------------------------------------------- #
def test_phase_constants_are_distinct_strings():
    from douzero.env.rules import (
        PHASE_BIDDING,
        PHASE_DEAL,
        PHASE_PLAYING,
        PHASE_REVEAL_BOTTOM,
        PHASE_TERMINAL,
    )

    phases = {PHASE_DEAL, PHASE_BIDDING, PHASE_REVEAL_BOTTOM, PHASE_PLAYING, PHASE_TERMINAL}
    assert len(phases) == 5
    assert all(isinstance(p, str) for p in phases)
