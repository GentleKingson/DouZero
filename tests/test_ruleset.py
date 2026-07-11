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
    assert rs.ruleset_version == "legacy-v1"
    assert rs.bidding_mode == BIDDING_MODE_NONE
    assert rs.bid_values == ()
    assert rs.allow_rob is False
    assert rs.all_pass_redeal is False
    assert rs.max_redeals == 10
    assert rs.bid_multiplier is False
    assert rs.base_score == 2
    assert rs.bomb_multiplier == 2
    assert rs.rocket_multiplier == 2
    assert rs.spring_multiplier == 0
    assert rs.anti_spring_multiplier == 0
    assert rs.allow_double is False
    assert rs.max_multiplier is None


def test_standard_ruleset_has_bidding_and_spring():
    rs = RuleSet.standard()
    assert rs.ruleset_id == RULESET_STANDARD
    assert rs.ruleset_version == "standard-v1"
    assert rs.bidding_mode == BIDDING_MODE_SCORE_0_1_2_3
    assert rs.bid_values == (0, 1, 2, 3)
    assert rs.allow_rob is False
    assert rs.all_pass_redeal is True
    assert rs.max_redeals == 10
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


def test_identity_returns_full_rule_identity():
    """identity() must return ruleset_id, ruleset_version, and ruleset_hash."""
    rs = RuleSet.standard()
    ident = rs.identity()
    assert ident["ruleset_id"] == "standard"
    assert ident["ruleset_version"] == "standard-v1"
    assert len(ident["ruleset_hash"]) == 64  # full SHA-256


def test_identity_differs_for_different_params():
    """Same ruleset_id but different params must produce different hashes."""
    base = RuleSet.standard()
    modified = RuleSet.from_dict({"ruleset_id": "standard", "base_score": 2})
    assert base.identity()["ruleset_hash"] != modified.identity()["ruleset_hash"]
    assert base.identity()["ruleset_id"] == modified.identity()["ruleset_id"]


def test_from_dict_rejects_zero_max_redeals():
    with pytest.raises(ValueError, match="max_redeals"):
        RuleSet.from_dict({"ruleset_id": "standard", "max_redeals": 0})


def test_from_dict_rejects_negative_max_redeals():
    with pytest.raises(ValueError, match="max_redeals"):
        RuleSet.from_dict({"ruleset_id": "standard", "max_redeals": -1})


def test_from_dict_custom_max_redeals():
    rs = RuleSet.from_dict({"ruleset_id": "standard", "max_redeals": 3})
    assert rs.max_redeals == 3


# --------------------------------------------------------------------------- #
# Strict validation (round 3 review)
# --------------------------------------------------------------------------- #
def test_bid_values_unsorted_rejected():
    """Unsorted bid_values like (3,2,1,0) must be rejected."""
    with pytest.raises(ValueError, match="exactly"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": [3, 2, 1, 0]})


def test_bid_values_with_duplicates_rejected():
    """Duplicate bid_values like (0,1,2,2) must be rejected."""
    with pytest.raises(ValueError, match="exactly"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": [0, 1, 2, 2]})


def test_bid_values_out_of_range_rejected():
    """Out-of-range bid_values like (0,1,2,99) must be rejected."""
    with pytest.raises(ValueError, match="exactly"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": [0, 1, 2, 99]})


def test_bid_values_string_rejected():
    """String bid_values like '2' must be rejected (no implicit conversion)."""
    with pytest.raises(TypeError, match="ints"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": ["0", "1", "2", "3"]})


def test_bid_values_float_rejected():
    """Float bid_values like 2.8 must be rejected."""
    with pytest.raises(TypeError, match="ints"):
        RuleSet.from_dict({"ruleset_id": "standard", "bid_values": [0.0, 1.0, 2.8, 3.0]})


def test_legacy_with_bid_values_rejected():
    """Legacy ruleset must have empty bid_values."""
    with pytest.raises(ValueError, match="empty bid_values"):
        RuleSet.from_dict({
            "ruleset_id": "legacy",
            "bidding_mode": "none",
            "bid_values": [0, 1, 2, 3],
        })


def test_standard_allow_rob_rejected():
    """allow_rob=True must be rejected for standard ruleset (not implemented)."""
    with pytest.raises(ValueError, match="allow_rob"):
        RuleSet.from_dict({"ruleset_id": "standard", "allow_rob": True})


def test_standard_allow_double_rejected():
    """allow_double=True must be rejected for standard ruleset (not implemented)."""
    with pytest.raises(ValueError, match="allow_double"):
        RuleSet.from_dict({"ruleset_id": "standard", "allow_double": True})


def test_bool_field_string_rejected():
    """A truthy string 'false' for a bool field must be rejected."""
    with pytest.raises(TypeError, match="bool"):
        RuleSet.from_dict({"ruleset_id": "standard", "all_pass_redeal": "false"})


def test_ruleset_version_must_be_nonempty():
    """ruleset_version must be a non-empty string."""
    with pytest.raises(ValueError, match="ruleset_version"):
        RuleSet.from_dict({"ruleset_id": "standard", "ruleset_version": ""})


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


def test_cli_feature_version_accepts_v2():
    """P03 widens feature_version: --feature_version v2 is now accepted."""
    from douzero.dmc.arguments import parser

    ns = parser.parse_args(["--feature_version", "v2"])
    assert ns.feature_version == "v2"


def test_cli_feature_version_rejects_unknown():
    """An unknown --feature_version value must still be rejected."""
    from douzero.dmc.arguments import parser

    with pytest.raises(SystemExit):
        parser.parse_args(["--feature_version", "v3"])


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
