"""H0 contract tests for the isolated V3 Hybrid program."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from douzero.config.loader import load_config
from douzero.observation.schema import build_v2_schema
from douzero.v3_hybrid import (
    V3_HYBRID_CHECKPOINT_KIND,
    V3_HYBRID_CONTRACT_VERSION,
    V3_HYBRID_FEATURE_VERSION,
    V3_HYBRID_LOSS_TERMS,
    V3_HYBRID_MODEL_VERSION,
    V3_HYBRID_OBSERVATION_SCHEMA_HASH,
    V3_HYBRID_OBSERVATION_SCHEMA_VERSION,
    V3HybridCompatibilityIdentity,
    assert_v3_hybrid_compatible,
    v3_hybrid_semantic_contract,
)


def _identity(**overrides):
    sections = {
        "ruleset": {
            "ruleset_id": "standard",
            "ruleset_version": "standard-v1",
            "ruleset_hash": "a" * 64,
        },
        "feature_flags": {"belief": False, "oracle": False},
        "model_graph": {"version": "role_residual_h1"},
        "output_semantics": {"q": "scalar_dmc", "win_score": "v2"},
        "optimizer_config": {"kind": "rmsprop", "learning_rate": 0.0001},
        "loss_config": dict(V3_HYBRID_LOSS_TERMS),
        "loss_schedules": {
            name: {"kind": "constant"} for name in V3_HYBRID_LOSS_TERMS
        },
        "belief_layout": {"version": "disabled"},
        "cooperation_mixer": {"version": "disabled"},
        "trainer_config": {"batch_size": 32, "policy_lag_limit": 0},
        "training_topology": {"version": "single_process"},
    }
    sections.update(overrides)
    return V3HybridCompatibilityIdentity(**sections)


def test_canonical_name_and_public_observation_contract_are_frozen():
    contract = v3_hybrid_semantic_contract()
    schema = build_v2_schema()
    assert V3_HYBRID_MODEL_VERSION == "v3_hybrid"
    assert V3_HYBRID_FEATURE_VERSION == "v2"
    assert V3_HYBRID_CHECKPOINT_KIND == "public_policy"
    assert V3_HYBRID_CONTRACT_VERSION == "v3-hybrid-h0-contract-v1"
    assert V3_HYBRID_OBSERVATION_SCHEMA_VERSION == schema.schema_version
    assert V3_HYBRID_OBSERVATION_SCHEMA_HASH == schema.stable_hash()
    assert contract["observation"]["schema_hash"] == schema.stable_hash()
    assert contract["observation"]["legal_action_authority"] == (
        "environment_rules_engine_only_v1"
    )
    assert contract["deployment"]["strength_without_formal_evaluation"] == (
        "playing strength not measured"
    )


def test_complete_identity_is_canonical_and_every_section_changes_the_hash():
    baseline = _identity()
    assert len(baseline.stable_hash()) == 64
    with pytest.raises(TypeError):
        baseline.loss_config["new"] = "not mutable"
    for section in (
        "feature_flags",
        "model_graph",
        "output_semantics",
        "optimizer_config",
        "loss_config",
        "loss_schedules",
        "belief_layout",
        "cooperation_mixer",
        "trainer_config",
        "training_topology",
    ):
        changed = copy.deepcopy(baseline.compatibility_dict()[section])
        changed["h0_hash_probe"] = section
        assert _identity(**{section: changed}).stable_hash() != baseline.stable_hash()


def test_identity_rejects_partial_unknown_nonfinite_and_hash_mismatch():
    expected = _identity()
    payload = expected.compatibility_dict()
    with pytest.raises(ValueError, match="missing=.*training_topology"):
        V3HybridCompatibilityIdentity.from_dict({
            key: value for key, value in payload.items() if key != "training_topology"
        })
    with pytest.raises(ValueError, match="unknown=.*surprise"):
        V3HybridCompatibilityIdentity.from_dict({**payload, "surprise": {}})
    with pytest.raises(ValueError, match="finite canonical JSON"):
        _identity(loss_config={"lambda_dmc": float("nan")})
    with pytest.raises(ValueError, match="payload/hash mismatch"):
        assert_v3_hybrid_compatible(expected, payload, actual_hash="0" * 64)
    changed = _identity(training_topology={"version": "async_single_gpu"})
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        assert_v3_hybrid_compatible(
            expected,
            changed.compatibility_dict(),
            actual_hash=changed.stable_hash(),
        )


def test_h0_does_not_enable_v3_in_legacy_or_v2_config_paths(tmp_path):
    config = tmp_path / "v3.yaml"
    config.write_text(
        "feature_version: v2\nmodel_version: v3_hybrid\n"
        "model:\n  version: v3_hybrid\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="unsupported value 'v3_hybrid'"):
        load_config(str(config))


def test_public_contract_has_no_privileged_import_side_effect():
    probe = (
        "import sys; "
        "assert 'douzero.observation.privileged' not in sys.modules; "
        "import douzero.v3_hybrid; "
        "assert 'douzero.observation.privileged' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
    forbidden = v3_hybrid_semantic_contract()["deployment"]
    assert "all_handcards" in forbidden["forbidden_payloads"]
