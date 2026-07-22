"""Strict public-only H8 package tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

import douzero.v3_hybrid.release_package as release_package_module
from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.env.rules import RuleSet
from douzero.observation import build_v2_schema
from douzero.v3_hybrid import (
    BELIEF_FEEDBACK_FARMERS,
    V3BeliefPolicy,
    V3HybridModel,
    V3HybridModelConfig,
    V3ModelPackageError,
    create_v3_public_model_package,
    save_v3_h4_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
    verify_v3_public_model_package,
)


def _runtime():
    schema = build_v2_schema()
    config = V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
    )
    ruleset = RuleSet.legacy()
    return schema, config, ruleset, V3HybridModel(schema, config)


def _package(tmp_path):
    schema, config, ruleset, model = _runtime()
    checkpoint = tmp_path / "public.pt"
    save_v3_hybrid_public_checkpoint(checkpoint, model, ruleset=ruleset)
    output = tmp_path / "package"
    manifest = create_v3_public_model_package(
        output,
        checkpoint,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        source_git_sha="a" * 40,
        decision_config={"action_selection": "argmax_dmc_q", "temperature": 0.0},
        search_compatible=False,
    )
    return output, manifest, schema, config, ruleset


def _refresh_checksums(package) -> None:
    names = sorted(path.name for path in package.iterdir() if path.name != "SHA256SUMS")
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )


def test_not_ready_package_is_public_only_strict_and_truthful(tmp_path) -> None:
    package, manifest, schema, config, ruleset = _package(tmp_path)
    assert manifest["access"] == "public"
    assert manifest["release_candidate"] == "NONE"
    assert manifest["release_status"] == "NOT READY"
    assert "not measured" in manifest["playing_strength"].lower()
    assert verify_v3_public_model_package(
        package,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        expected_source_git_sha="a" * 40,
        expected_search_compatible=False,
    ) == manifest
    names = {path.name.lower() for path in package.iterdir()}
    assert not any(
        token in name
        for name in names
        for token in ("oracle", "teacher", "privileged", "replay", "optimizer", "mixer")
    )


def test_package_rejects_tamper_unknown_files_and_runtime_drift(tmp_path) -> None:
    package, _, schema, config, ruleset = _package(tmp_path)
    with (package / "model_card.md").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(V3ModelPackageError, match="checksum mismatch"):
        verify_v3_public_model_package(
            package, schema=schema, ruleset=ruleset, model_config=config
        )
    (package / "model_card.md").write_text("replacement\n", encoding="utf-8")
    _refresh_checksums(package)
    (package / "optimizer.pt").write_bytes(b"forbidden")
    with pytest.raises(V3ModelPackageError, match="package files mismatch"):
        verify_v3_public_model_package(
            package, schema=schema, ruleset=ruleset, model_config=config
        )
    (package / "optimizer.pt").unlink()
    other = V3HybridModelConfig(
        hidden_size=32, history_layers=1, history_heads=4,
        shared_fusion_layers=1, landlord_adapter_layers=1,
        farmer_adapter_layers=1,
    )
    with pytest.raises(V3ModelPackageError, match="runtime feature/model identity mismatch"):
        verify_v3_public_model_package(
            package, schema=schema, ruleset=ruleset, model_config=other
        )


def test_package_rejects_false_ready_claim_even_with_recomputed_checksums(tmp_path) -> None:
    package, _, schema, config, ruleset = _package(tmp_path)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_candidate"] = "v3_full_hybrid_search_on"
    manifest["release_status"] = "READY"
    manifest["playing_strength"] = "MEASURED"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    report_path = package / "formal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["release_candidate"] = manifest["release_candidate"]
    report["release_status"] = "READY"
    report["playing_strength"] = "MEASURED"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    _refresh_checksums(package)
    with pytest.raises(V3ModelPackageError, match="recomputed packaged evidence"):
        verify_v3_public_model_package(
            package, schema=schema, ruleset=ruleset, model_config=config
        )


@pytest.mark.parametrize(
    "decision_config",
    [
        {"action_selection": "argmax_dmc_q", "api_token": "secret"},
        {"action_selection": "argmax_dmc_q", "model_path": "/private/model.pt"},
    ],
)
def test_package_rejects_secret_or_path_bearing_decision_config(
    tmp_path, decision_config
) -> None:
    schema, config, ruleset, model = _runtime()
    checkpoint = tmp_path / "public.pt"
    save_v3_hybrid_public_checkpoint(checkpoint, model, ruleset=ruleset)
    with pytest.raises(V3ModelPackageError, match="unsupported fields"):
        create_v3_public_model_package(
            tmp_path / "package",
            checkpoint,
            schema=schema,
            ruleset=ruleset,
            model_config=config,
            source_git_sha="a" * 40,
            decision_config=decision_config,
            search_compatible=False,
        )


def test_package_rejects_evidence_from_a_stale_source_head(
    tmp_path, monkeypatch
) -> None:
    schema, config, ruleset, model = _runtime()
    checkpoint = tmp_path / "public.pt"
    save_v3_hybrid_public_checkpoint(checkpoint, model, ruleset=ruleset)
    monkeypatch.setattr(
        release_package_module,
        "validate_h8_formal_evidence",
        lambda _payload: {
            "schema_version": release_package_module.H8_REPORT_SCHEMA_VERSION,
            "evidence_sha256": "d" * 64,
            "support_matrix_hash": release_package_module.h8a_support_matrix_hash(),
            "development_status": "INCOMPLETE",
            "release_candidate": "NONE",
            "release_status": "NOT READY",
            "playing_strength": "NOT MEASURED",
            "training_run_count": 0,
            "evaluation_count": 0,
            "required_variants": [],
            "issues": ["test fixture"],
        },
    )
    with pytest.raises(V3ModelPackageError, match="source SHA"):
        create_v3_public_model_package(
            tmp_path / "package",
            checkpoint,
            schema=schema,
            ruleset=ruleset,
            model_config=config,
            source_git_sha="a" * 40,
            decision_config={"action_selection": "argmax_dmc_q"},
            search_compatible=False,
            formal_evidence={"experiment_identity": {"git_sha": "b" * 40}},
        )


def test_deployment_import_graph_excludes_training_only_modules() -> None:
    code = (
        "import sys; import douzero.v3_hybrid.release_package; "
        "forbidden=('training.oracle','training.cooperation','training.h5_learner'); "
        "assert not any(any(x in name for x in forbidden) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_package_round_trips_complete_h6_public_graph(tmp_path) -> None:
    schema = build_v2_schema()
    ruleset = RuleSet.standard()
    config = V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
        human_prior_enabled=True,
        strategy_features_enabled=True,
        strategy_aux_enabled=True,
        style_enabled=True,
        style_embedding_dim=8,
        bidding_enabled=True,
        bidding_hidden_size=12,
        bidding_uncertainty_enabled=True,
    )
    checkpoint = tmp_path / "h6-public.pt"
    save_v3_hybrid_public_checkpoint(
        checkpoint, V3HybridModel(schema, config), ruleset=ruleset
    )
    package = tmp_path / "h6-package"
    create_v3_public_model_package(
        package,
        checkpoint,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        source_git_sha="b" * 40,
        decision_config={"action_selection": "argmax_dmc_q"},
        search_compatible=True,
    )
    verify_v3_public_model_package(
        package,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        expected_source_git_sha="b" * 40,
        expected_search_compatible=True,
    )


def test_package_round_trips_h4_public_belief_graph(tmp_path) -> None:
    schema = build_v2_schema()
    ruleset = RuleSet.legacy()
    config = V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
        belief_feedback=BELIEF_FEEDBACK_FARMERS,
    )
    belief_config = BeliefConfig(hidden_size=16, num_layers=1)
    policy = V3BeliefPolicy(
        V3HybridModel(schema, config),
        BeliefModel(belief_config),
        ruleset=ruleset,
    )
    checkpoint = tmp_path / "h4-public.pt"
    save_v3_h4_public_checkpoint(checkpoint, policy)
    package = tmp_path / "h4-package"
    create_v3_public_model_package(
        package,
        checkpoint,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        belief_config=belief_config,
        source_git_sha="c" * 40,
        decision_config={"action_selection": "argmax_dmc_q"},
        search_compatible=False,
    )
    verify_v3_public_model_package(
        package,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        belief_config=belief_config,
        expected_source_git_sha="c" * 40,
    )


def test_ready_package_selects_matching_checkpoint_and_search_mode(
    tmp_path, monkeypatch
) -> None:
    schema, config, ruleset, model = _runtime()
    checkpoint = tmp_path / "public.pt"
    save_v3_hybrid_public_checkpoint(checkpoint, model, ruleset=ruleset)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema_version": release_package_module.H8_REPORT_SCHEMA_VERSION,
        "evidence_sha256": "d" * 64,
        "support_matrix_hash": release_package_module.h8a_support_matrix_hash(),
        "development_status": "COMPLETE",
        "release_candidate": "v3_full_hybrid",
        "release_status": "READY",
        "playing_strength": "MEASURED",
        "training_run_count": 2,
        "evaluation_count": 2,
        "required_variants": ["v3_full_hybrid"],
        "issues": [],
    }
    monkeypatch.setattr(
        release_package_module,
        "validate_h8_formal_evidence",
        lambda _payload: report,
    )
    evidence = {
        "experiment_identity": {"git_sha": "a" * 40},
        "evaluations": [
            {
                "variant": "v3_full_hybrid",
                "tier": "promotion",
                "search_enabled": False,
                "ruleset": ruleset.identity(),
                "model_checkpoint_sha256": "e" * 64,
            },
            {
                "variant": "v3_full_hybrid",
                "tier": "promotion",
                "search_enabled": False,
                "ruleset": ruleset.identity(),
                "model_checkpoint_sha256": checkpoint_sha,
            },
        ],
    }
    package = tmp_path / "selected-package"
    manifest = create_v3_public_model_package(
        package,
        checkpoint,
        schema=schema,
        ruleset=ruleset,
        model_config=config,
        source_git_sha="a" * 40,
        decision_config={"action_selection": "argmax_dmc_q"},
        search_compatible=False,
        formal_evidence=evidence,
    )
    assert manifest["release_status"] == "READY"
    assert json.loads(
        (package / "formal_evidence.json").read_text(encoding="utf-8")
    )["payload"] == evidence

    alternate = tmp_path / "alternate-public.pt"
    save_v3_hybrid_public_checkpoint(
        alternate, V3HybridModel(schema, config), ruleset=ruleset
    )
    (package / "public_checkpoint.pt").write_bytes(alternate.read_bytes())
    manifest_path = package / "manifest.json"
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["checkpoint_sha256"] = hashlib.sha256(
        alternate.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(tampered_manifest, sort_keys=True), encoding="utf-8"
    )
    _refresh_checksums(package)
    with pytest.raises(V3ModelPackageError, match="packaged checkpoint"):
        verify_v3_public_model_package(
            package,
            schema=schema,
            ruleset=ruleset,
            model_config=config,
        )

    with pytest.raises(V3ModelPackageError, match="packaged search mode"):
        create_v3_public_model_package(
            tmp_path / "wrong-search-package",
            checkpoint,
            schema=schema,
            ruleset=ruleset,
            model_config=config,
            source_git_sha="a" * 40,
            decision_config={"action_selection": "argmax_dmc_q"},
            search_compatible=True,
            formal_evidence=evidence,
        )
