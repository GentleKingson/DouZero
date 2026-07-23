from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from douzero.v3_hybrid.formal_config import (
    FormalConfigError,
    FormalExperimentConfig,
    V3_FORMAL_INITIAL_CHECKPOINT_SCHEMA,
    canonical_hash,
    freeze_formal_config,
    load_formal_config,
    validate_initial_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "v3_formal"
RUNNABLE_CONFIGS = tuple(sorted(CONFIG_DIR.glob("*.yaml")))


def _resolved(name: str = "v3_role_legacy.yaml") -> dict[str, object]:
    return load_formal_config(CONFIG_DIR / name).resolved_dict()


def test_complete_supported_matrix_loads_and_is_exact() -> None:
    assert len(RUNNABLE_CONFIGS) == 15
    observed = {
        (config.variant, config.ruleset["id"])
        for config in map(load_formal_config, RUNNABLE_CONFIGS)
    }
    assert observed == {
        ("legacy_a1", "legacy"),
        *(('model_v2', ruleset) for ruleset in ("legacy", "standard")),
        *((variant, ruleset) for variant in (
            "v3_role", "v3_admc", "v3_oracle", "v3_belief",
            "v3_farmer_cooperation", "v3_full_hybrid",
        ) for ruleset in ("legacy", "standard")),
    }


def test_metadata_does_not_change_training_semantics() -> None:
    payload = _resolved()
    original = FormalExperimentConfig.from_dict(payload).identity_dict()
    payload["metadata"]["description"] = "A non-semantic report label"
    changed = FormalExperimentConfig.from_dict(payload).identity_dict()
    assert original["config_sha256"] != changed["config_sha256"]
    assert original["training_semantics_hash"] == changed["training_semantics_hash"]
    assert original["workload_hash"] == changed["workload_hash"]


@pytest.mark.parametrize(
    ("section", "field"),
    [("runtime", "batch_size"), ("budgets", "pilot")],
)
def test_workload_fields_change_workload_hash(section: str, field: str) -> None:
    payload = _resolved()
    original = FormalExperimentConfig.from_dict(payload).identity_dict()
    if section == "runtime":
        payload[section][field] += 1
    else:
        payload[section][field]["sample_budget"] += 1
    changed = FormalExperimentConfig.from_dict(payload).identity_dict()
    assert original["workload_hash"] != changed["workload_hash"]


def test_model_and_loss_fields_change_training_semantics_hash() -> None:
    payload = _resolved()
    original = FormalExperimentConfig.from_dict(payload).identity_dict()
    payload["model"]["config"]["hidden_size"] = 384
    changed = FormalExperimentConfig.from_dict(payload).identity_dict()
    assert original["training_semantics_hash"] != changed["training_semantics_hash"]

    payload = _resolved()
    payload["losses"]["weights"]["dmc"] = 0.5
    changed_loss = FormalExperimentConfig.from_dict(payload).identity_dict()
    assert original["training_semantics_hash"] != changed_loss["training_semantics_hash"]


def test_unsupported_combination_fails_closed() -> None:
    payload = _resolved("legacy_a1.yaml")
    standard = load_formal_config(CONFIG_DIR / "model_v2_standard.yaml")
    payload["ruleset"] = dict(standard.ruleset)
    with pytest.raises(FormalConfigError, match="unsupported variant/ruleset"):
        FormalExperimentConfig.from_dict(payload)


@pytest.mark.parametrize("mutation", ["old_schema", "missing", "unknown"])
def test_schema_drift_fails_closed(mutation: str) -> None:
    payload = _resolved()
    if mutation == "old_schema":
        payload["schema_version"] = "v3-formal-experiment-config-v0"
    elif mutation == "missing":
        del payload["runtime"]
    else:
        payload["surprise"] = True
    with pytest.raises(FormalConfigError):
        FormalExperimentConfig.from_dict(payload)


def test_duplicate_yaml_key_and_include_traversal_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: one\nschema_version: two\n", encoding="utf-8")
    with pytest.raises(FormalConfigError, match="duplicate YAML key"):
        load_formal_config(duplicate)

    traversal = tmp_path / "traversal.yaml"
    traversal.write_text("extends: ../outside.yaml\n", encoding="utf-8")
    with pytest.raises(FormalConfigError, match="sibling"):
        load_formal_config(traversal)


def test_repeated_freeze_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    freeze_formal_config(CONFIG_DIR / "v3_full_hybrid_standard.yaml", first)
    freeze_formal_config(CONFIG_DIR / "v3_full_hybrid_standard.yaml", second)
    for name in ("resolved_config.json", "identity.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert b"NaN" not in (first / name).read_bytes()
        assert b"Infinity" not in (first / name).read_bytes()


def test_checkpoint_identity_accepts_exact_and_rejects_cross_family(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initial.ckpt"
    checkpoint.write_bytes(b"immutable test checkpoint")
    payload = _resolved()
    payload["initialization"] = {
        "kind": "checkpoint",
        "source": "test-fixture",
        "seed": 101,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_kind": "v3_hybrid_public",
    }
    config = FormalExperimentConfig.from_dict(payload)
    manifest = {
        "schema_version": V3_FORMAL_INITIAL_CHECKPOINT_SCHEMA,
        "checkpoint_kind": "v3_hybrid_public",
        "model_family": config.model["family"],
        "model_hash": config.identity_dict()["model_hash"],
        "ruleset_hash": config.ruleset["hash"],
    }
    sidecar = checkpoint.with_name(checkpoint.name + ".manifest.json")
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    validate_initial_checkpoint(config)

    manifest["model_family"] = "model_v2"
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FormalConfigError, match="identity is incompatible"):
        validate_initial_checkpoint(config)


def test_legacy_v2_and_v3_model_identities_do_not_alias() -> None:
    identities = {
        load_formal_config(CONFIG_DIR / name).identity_dict()["model_hash"]
        for name in ("legacy_a1.yaml", "model_v2_legacy.yaml", "v3_role_legacy.yaml")
    }
    assert len(identities) == 3


def test_release_claims_remain_closed() -> None:
    identity = load_formal_config(
        CONFIG_DIR / "v3_full_hybrid_standard.yaml"
    ).identity_dict()
    assert identity["release_candidate"] == "NONE"
    assert identity["release_status"] == "NOT READY"
    assert identity["playing_strength"] == "NOT MEASURED"
