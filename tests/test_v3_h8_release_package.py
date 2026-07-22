"""Strict public-only H8 package tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from douzero.env.rules import RuleSet
from douzero.observation import build_v2_schema
from douzero.v3_hybrid import (
    V3HybridModel,
    V3HybridModelConfig,
    V3ModelPackageError,
    create_v3_public_model_package,
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
    assert "not measured" in manifest["playing_strength"]
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
    path = package / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["release_candidate"] = "v3_full_hybrid_search_on"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _refresh_checksums(package)
    with pytest.raises(V3ModelPackageError, match="release/strength claim"):
        verify_v3_public_model_package(
            package, schema=schema, ruleset=ruleset, model_config=config
        )


def test_deployment_import_graph_excludes_training_only_modules() -> None:
    code = (
        "import sys; import douzero.v3_hybrid.release_package; "
        "forbidden=('training.oracle','training.cooperation','training.h5_learner'); "
        "assert not any(any(x in name for x in forbidden) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
