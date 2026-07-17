"""P16 deployment manifest, package, export, and runtime hardening tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from douzero.deployment import (
    MODEL_ABI_VERSION,
    ModelManifest,
    ModelPackageError,
    build_model_manifest,
    create_model_package,
    export_padded_model,
    load_model_package,
    model_implementation_hash,
    verify_model_package,
)
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.evaluation.deep_agent import DeepAgentV2
from douzero.search import SearchConfig
from douzero.models_v2 import (
    ModelV2,
    ModelV2Config,
    observation_to_model_inputs,
)
from douzero.observation import build_v2_schema, get_obs_v2


_FROZEN_P16_FORMAT1_MANIFEST = {
    "format_version": 1,
    "model_abi_version": "model-v2-deployment-1",
    "implementation_hash": "1" * 64,
    "model_version": "v2",
    "feature_version": "v2",
    "feature_schema_hash": "2" * 64,
    "model_config_hash": "3" * 64,
    "ruleset_id": "legacy",
    "ruleset_hash": "4" * 64,
    "git_sha": "5" * 40,
    "training_config_hash": "6" * 64,
    "role_support": ["landlord", "landlord_up", "landlord_down"],
    "belief_enabled": False,
    "search_compatible": False,
    "public_or_privileged": "public",
    "dtype": "float32",
    "required_package_versions": {
        "python": ">=3.11",
        "numpy": ">=1.24",
        "torch": ">=2.0",
        "douzero": "==0.1.0",
    },
    "weights_sha256": "7" * 64,
}


def _refresh_package_checksums(package):
    names = sorted(
        path.name for path in package.iterdir() if path.name != "SHA256SUMS"
    )
    lines = [
        f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names
    ]
    (package / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


@pytest.fixture
def p16_runtime():
    torch.manual_seed(1600)
    schema = build_v2_schema(max_history_len=8)
    config = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        role_embedding_dim=8,
        mlp_layers=1,
        nan_guard=False,
    )
    ruleset = RuleSet.legacy()
    model = ModelV2(schema, config).eval()
    env = Env("adp")
    env.reset()
    obs = get_obs_v2(env.infoset, ruleset=ruleset, schema=schema)
    return model, schema, config, ruleset, obs


def test_format1_package_rejection_names_matching_runtime_and_rebuild(tmp_path):
    """A frozen P16 manifest gets migration guidance before P17 file checks."""

    package = tmp_path / "p16-format1"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(_FROZEN_P16_FORMAT1_MANIFEST), encoding="utf-8"
    )

    with pytest.raises(
        ModelPackageError,
        match=(
            "format-1 P16 package requires its matching P16 runtime.*"
            "rebuild a new package from the original manifest-bearing public checkpoint"
        ),
    ):
        verify_model_package(package)


def test_model_manifest_roundtrip_and_complete_fields(p16_runtime):
    model, _, _, ruleset, _ = p16_runtime
    manifest = build_model_manifest(
        model, ruleset, training_config={"seed": 1600}, search_compatible=True
    )
    assert ModelManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.role_support == ("landlord", "landlord_up", "landlord_down")
    assert manifest.public_or_privileged == "public"
    assert manifest.feature_schema_hash == model.schema.stable_hash()
    assert manifest.model_config_hash == model.config.stable_hash()
    assert manifest.model_abi_version == MODEL_ABI_VERSION
    assert manifest.implementation_hash == model_implementation_hash()
    assert manifest.git_sha != "unknown"
    assert set(manifest.required_package_versions) == {
        "python",
        "numpy",
        "torch",
        "douzero",
    }


def test_manifest_rejects_unknown_or_missing_fields(p16_runtime):
    model, _, _, ruleset, _ = p16_runtime
    raw = build_model_manifest(model, ruleset).to_dict()
    raw["surprise"] = True
    with pytest.raises(ValueError, match="field mismatch"):
        ModelManifest.from_dict(raw)
    raw.pop("surprise")
    raw.pop("feature_version")
    with pytest.raises(ValueError, match="field mismatch"):
        ModelManifest.from_dict(raw)


def test_release_manifest_rejects_unknown_git_sha(monkeypatch, p16_runtime):
    model, _, _, ruleset, _ = p16_runtime
    monkeypatch.setattr("douzero.deployment.manifest.git_sha", lambda: "unknown")
    with pytest.raises(ValueError, match="known git_sha"):
        build_model_manifest(model, ruleset)


def test_package_roundtrip_clean_inference_and_checksums(tmp_path, p16_runtime):
    model, schema, config, ruleset, obs = p16_runtime
    package = tmp_path / "release"
    manifest = create_model_package(
        package,
        model,
        ruleset,
        training_config={"seed": 1600, "dataset": "synthetic-test"},
    )
    assert {path.name for path in package.iterdir()} >= {
        "weights.pt", "manifest.json", "ruleset.json", "feature_schema.json",
        "README.md", "THIRD_PARTY_NOTICES", "SHA256SUMS",
    }
    assert verify_model_package(
        package,
        expected_ruleset=ruleset,
        expected_schema_hash=schema.stable_hash(),
        expected_model_config_hash=config.stable_hash(),
    ) == manifest

    loaded = load_model_package(
        package, schema=schema, ruleset=ruleset, config=config, device="cpu"
    )
    original_agent = DeepAgentV2("landlord", model, ruleset, device="cpu")
    loaded_agent = DeepAgentV2("landlord", loaded, ruleset, device="cpu")
    assert loaded_agent.act_v2(obs) == original_agent.act_v2(obs)


def test_package_rejects_runtime_implementation_drift(
    tmp_path, monkeypatch, p16_runtime
):
    model, _, _, ruleset, _ = p16_runtime
    package = tmp_path / "release"
    create_model_package(package, model, ruleset)
    monkeypatch.setattr(
        "douzero.deployment.package.model_implementation_hash", lambda: "f" * 64
    )
    with pytest.raises(ModelPackageError, match="implementation hash mismatch"):
        verify_model_package(package)


def test_package_rejects_wrong_ruleset_feature_config_and_tamper(tmp_path, p16_runtime):
    model, schema, config, ruleset, _ = p16_runtime
    package = tmp_path / "release"
    create_model_package(package, model, ruleset)
    with pytest.raises(ModelPackageError, match="runtime ruleset"):
        verify_model_package(package, expected_ruleset=RuleSet.standard())
    with pytest.raises(ModelPackageError, match="feature schema"):
        verify_model_package(package, expected_schema_hash=build_v2_schema(9).stable_hash())
    other_config = ModelV2Config(
        hidden_size=32, history_layers=1, history_heads=4,
        role_embedding_dim=8, mlp_layers=1, nan_guard=True,
    )
    with pytest.raises(ModelPackageError, match="model config"):
        verify_model_package(package, expected_model_config_hash=other_config.stable_hash())

    manifest_path = package / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["feature_version"] = "wrong-version"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    _refresh_package_checksums(package)
    with pytest.raises(ModelPackageError, match="feature_version"):
        verify_model_package(package)


def test_package_rejects_replaced_valid_v2_sidecar(tmp_path, p16_runtime):
    from douzero.checkpoint import save_v2_position_weights

    model, _, config, ruleset, _ = p16_runtime
    package = tmp_path / "release"
    create_model_package(package, model, ruleset)

    other_schema = build_v2_schema(max_history_len=9)
    other_model = ModelV2(other_schema, config).eval()
    weights_path = package / "weights.pt"
    save_v2_position_weights(str(weights_path), other_model, ruleset=ruleset)

    manifest_path = package / "manifest.json"
    outer = json.loads(manifest_path.read_text(encoding="utf-8"))
    outer["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(outer), encoding="utf-8")
    _refresh_package_checksums(package)

    with pytest.raises(ModelPackageError, match="checkpoint identity"):
        verify_model_package(package)

    with (package / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ModelPackageError, match="checksum mismatch"):
        verify_model_package(package)


def test_production_loader_rejects_privileged_manifest(tmp_path, p16_runtime):
    model, _, _, ruleset, _ = p16_runtime
    package = tmp_path / "release"
    create_model_package(package, model, ruleset)
    manifest_path = package / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["public_or_privileged"] = "privileged"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ModelPackageError, match="training-only"):
        verify_model_package(package)


def test_export_padded_action_batch_aligns(tmp_path, p16_runtime):
    model, _, _, _, obs = p16_runtime
    bundle = observation_to_model_inputs(obs)
    report = export_padded_model(
        model,
        bundle,
        tmp_path / "model.pt2",
        acting_role=obs.public.acting_role,
        max_actions=len(obs.actions.legal_actions) + 2,
    )
    assert report.success, report.message
    assert report.max_abs_error is not None and report.max_abs_error <= 1e-5
    assert (tmp_path / "model.pt2").is_file()
    assert (tmp_path / "model.pt2.report.json").is_file()


def test_failed_export_removes_stale_target_and_temporary_file(tmp_path, p16_runtime):
    model, schema, config, _, obs = p16_runtime
    unsupported = ModelV2(schema, replace(config, belief_enabled=True)).eval()
    bundle = observation_to_model_inputs(obs)
    output = tmp_path / "model.pt2"
    output.write_bytes(b"stale export")

    report = export_padded_model(
        unsupported,
        bundle,
        output,
        acting_role=obs.public.acting_role,
        max_actions=len(obs.actions.legal_actions),
    )

    assert report.success is False
    assert "belief_enabled" in report.message
    assert not output.exists()
    assert not list(tmp_path.glob(".model.pt2.*.tmp"))


def test_alignment_failure_never_publishes_export(tmp_path, p16_runtime):
    model, _, _, _, obs = p16_runtime
    bundle = observation_to_model_inputs(obs)
    output = tmp_path / "misaligned.pt2"
    output.write_bytes(b"stale export")

    report = export_padded_model(
        model,
        bundle,
        output,
        acting_role=obs.public.acting_role,
        max_actions=len(obs.actions.legal_actions),
        atol=-1.0,
        rtol=-1.0,
    )

    assert report.success is False
    assert not output.exists()
    assert not list(tmp_path.glob(".misaligned.pt2.*.tmp"))


def test_search_compatible_manifest_is_enforced(p16_runtime):
    model, _, _, ruleset, _ = p16_runtime
    search = SearchConfig(enabled=True)
    belief_stub = torch.nn.Identity()
    model.deployment_manifest = SimpleNamespace(search_compatible=False)
    with pytest.raises(ValueError, match="search_compatible is false"):
        DeepAgentV2(
            "landlord",
            model,
            ruleset,
            belief_model=belief_stub,
            search_config=search,
            device="cpu",
        )

    model.deployment_manifest = SimpleNamespace(search_compatible=True)
    agent = DeepAgentV2(
        "landlord",
        model,
        ruleset,
        belief_model=belief_stub,
        search_config=search,
        device="cpu",
    )
    assert agent.search_config.enabled is True


def test_agent_explicit_device_explanation_and_exception_fallback(p16_runtime):
    model, _, _, ruleset, obs = p16_runtime
    agent = DeepAgentV2(
        "landlord",
        model,
        ruleset,
        device="cpu",
        deterministic=True,
        explanations_enabled=True,
    )
    action, explanation = agent.act_v2(obs, return_explanation=True)
    assert action in obs.actions.legal_actions
    assert explanation["source"] == "model"
    assert explanation["selected_index"] < len(obs.actions.legal_actions)
    assert len(explanation["p_win"]) == len(obs.actions.legal_actions)
    assert agent.device == torch.device("cpu")

    def fail_forward(*args, **kwargs):
        raise RuntimeError("synthetic inference failure")

    model.forward = fail_forward
    fallback_action, fallback = agent.act_v2(obs, return_explanation=True)
    assert fallback_action in obs.actions.legal_actions
    assert fallback["source"] == "conservative"
    assert fallback["fallback_reason"] == "RuntimeError"


def test_package_refuses_nonempty_output(tmp_path, p16_runtime):
    model, _, _, ruleset, _ = p16_runtime
    output = tmp_path / "release"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ModelPackageError, match="not empty"):
        create_model_package(output, model, ruleset)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"
