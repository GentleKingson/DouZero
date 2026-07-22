"""Strict public-only V3 Hybrid package creation and verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from douzero.belief.model import BeliefConfig
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.observation.schema import FeatureSchemaManifest

from .belief_checkpoint import (
    V3_H4_PUBLIC_CHECKPOINT_FORMAT,
    load_v3_h4_public_checkpoint,
)
from .checkpoint import (
    V3_HYBRID_H1_CHECKPOINT_FORMAT,
    load_v3_hybrid_public_checkpoint,
)
from .config import BELIEF_FEEDBACK_NONE, V3HybridModelConfig
from .contract import V3_HYBRID_MODEL_VERSION
from .formal_evidence import (
    H8_REPORT_SCHEMA_VERSION,
    H8EvidenceError,
    REQUIRED_VARIANTS,
    h8a_support_matrix_hash,
    validate_h8_formal_evidence,
)

V3_H8_PACKAGE_FORMAT = "v3-hybrid-h8-public-package-v1"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FILES = frozenset({
    "public_checkpoint.pt", "manifest.json", "ruleset.json",
    "feature_schema.json", "model_config.json", "decision_config.json",
    "model_card.md", "training_summary.md", "benchmark_summary.md",
    "evaluation_summary.md", "known_limitations.md", "rollback.md",
    "formal_evidence.json", "formal_report.json", "LICENSE",
    "THIRD_PARTY_NOTICES", "SHA256SUMS",
})
_FORBIDDEN_FILE_TOKENS = (
    "oracle", "teacher", "privileged", "hidden", "replay", "optimizer",
    "mixer", "human_id", "cache", "secret",
)


class V3ModelPackageError(RuntimeError):
    """Raised when a V3 package is incomplete, corrupt, or incompatible."""


def _validate_decision_config(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"action_selection", "temperature"}
    unknown = set(value) - allowed
    if unknown:
        raise V3ModelPackageError(
            f"decision_config contains unsupported fields: {sorted(unknown)}"
        )
    if value.get("action_selection") != "argmax_dmc_q":
        raise V3ModelPackageError(
            "decision_config.action_selection must be 'argmax_dmc_q'"
        )
    temperature = value.get("temperature", 0.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) != 0.0
    ):
        raise V3ModelPackageError(
            "decision_config.temperature must be finite and zero for argmax_dmc_q"
        )
    return {
        "action_selection": "argmax_dmc_q",
        "temperature": float(temperature),
    }


def _empty_formal_report() -> dict[str, Any]:
    return {
        "schema_version": H8_REPORT_SCHEMA_VERSION,
        "evidence_sha256": "0" * 64,
        "support_matrix_hash": h8a_support_matrix_hash(),
        "development_status": "INCOMPLETE",
        "release_candidate": "NONE",
        "release_status": "NOT READY",
        "playing_strength": "NOT MEASURED",
        "training_run_count": 0,
        "evaluation_count": 0,
        "required_variants": list(REQUIRED_VARIANTS),
        "issues": ["formal evidence was not supplied"],
    }


def _assert_ready_checkpoint_binding(
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    ruleset: RuleSet,
    search_compatible: bool,
) -> None:
    evaluations = evidence.get("evaluations")
    if not isinstance(evaluations, list):
        raise V3ModelPackageError("READY evidence has no evaluation rows")
    candidate_rows = [
        row for row in evaluations
        if isinstance(row, Mapping)
        and row.get("variant") == report["release_candidate"]
        and row.get("tier") == "promotion"
        and row.get("search_enabled") is search_compatible
        and row.get("ruleset") == ruleset.identity()
    ]
    if not any(
        row.get("model_checkpoint_sha256") == checkpoint_sha256
        for row in candidate_rows
    ):
        raise V3ModelPackageError(
            "formal candidate evidence for the packaged search mode is not "
            "bound to the packaged checkpoint"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise V3ModelPackageError(f"invalid JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise V3ModelPackageError(f"{path.name} must contain an object")
    return value


def _default_documents(report: Mapping[str, Any]) -> dict[str, str]:
    ready = report["release_status"] == "READY"
    candidate = report["release_candidate"]
    strength = "MEASURED" if ready else "NOT MEASURED"
    return {
        "model_card.md": (
            "# DouZero V3 Hybrid Model Card\n\n"
            f"- Release candidate: **{candidate}**\n"
            f"- Release status: **{report['release_status']}**\n"
            f"- Playing strength: **{strength}**\n\n"
            "This package contains only public-policy inference assets.\n"
        ),
        "training_summary.md": (
            "# Training Summary\n\n"
            f"Validated formal training rows: {report['training_run_count']}.\n"
        ),
        "benchmark_summary.md": (
            "# Benchmark Summary\n\n"
            "Formal matched benchmark evidence is bound by the evidence SHA-256 "
            "in `manifest.json`.\n"
        ),
        "evaluation_summary.md": (
            "# Evaluation Summary\n\n"
            f"Status: **{strength}**\n\n"
            f"Validated formal evaluation rows: {report['evaluation_count']}.\n"
        ),
        "known_limitations.md": (
            "# Known Limitations\n\n"
            + ("See the formal evaluation evidence and role-specific confidence intervals.\n"
               if ready else
               "Formal promotion gates are incomplete. This package is not a release candidate.\n")
        ),
        "rollback.md": (
            "# Rollback\n\n"
            "Stop new routing, restore the previous checksummed public package, "
            "and preserve this package and its evidence for diagnosis.\n"
        ),
    }


def _checkpoint_format(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise V3ModelPackageError(f"cannot safely inspect public checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("format"), str):
        raise V3ModelPackageError("public checkpoint has no recognized format")
    return payload["format"], payload


def _strict_load_checkpoint(
    checkpoint: Path,
    *,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
    model_config: V3HybridModelConfig,
    belief_config: BeliefConfig | None,
) -> str:
    checkpoint_format, _ = _checkpoint_format(checkpoint)
    try:
        if checkpoint_format == V3_HYBRID_H1_CHECKPOINT_FORMAT:
            if belief_config is not None:
                raise V3ModelPackageError(
                    "H1 public checkpoint cannot carry a belief configuration"
                )
            load_v3_hybrid_public_checkpoint(
                checkpoint, schema=schema, ruleset=ruleset,
                config=model_config, device="cpu",
            )
        elif checkpoint_format == V3_H4_PUBLIC_CHECKPOINT_FORMAT:
            if belief_config is None:
                raise V3ModelPackageError(
                    "H4 public checkpoint requires its exact belief configuration"
                )
            load_v3_h4_public_checkpoint(
                checkpoint, schema=schema, ruleset=ruleset,
                model_config=model_config, belief_config=belief_config,
                device="cpu",
            )
        else:
            raise V3ModelPackageError(
                f"unsupported public checkpoint format {checkpoint_format!r}"
            )
    except (CheckpointCompatibilityError, TypeError, ValueError) as exc:
        raise V3ModelPackageError(f"public checkpoint identity mismatch: {exc}") from exc
    return checkpoint_format


def create_v3_public_model_package(
    output_dir: str | Path,
    checkpoint: str | Path,
    *,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
    model_config: V3HybridModelConfig,
    source_git_sha: str,
    decision_config: Mapping[str, Any],
    search_compatible: bool,
    belief_config: BeliefConfig | None = None,
    formal_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a checksummed package after strict public checkpoint reload."""

    if not _HEX40.fullmatch(source_git_sha):
        raise V3ModelPackageError("source_git_sha must be a full lowercase Git SHA")
    if not isinstance(search_compatible, bool):
        raise TypeError("search_compatible must be bool")
    if not isinstance(decision_config, Mapping):
        raise TypeError("decision_config must be a mapping")
    resolved_decision_config = _validate_decision_config(decision_config)
    if model_config.belief_feedback == BELIEF_FEEDBACK_NONE and belief_config is not None:
        raise V3ModelPackageError("belief-disabled policy cannot package belief config")
    if model_config.belief_feedback != BELIEF_FEEDBACK_NONE and belief_config is None:
        raise V3ModelPackageError("belief-feedback policy requires belief config")
    source = Path(checkpoint)
    if not source.is_file():
        raise V3ModelPackageError(f"checkpoint is not a file: {source}")
    checkpoint_format = _strict_load_checkpoint(
        source, schema=schema, ruleset=ruleset, model_config=model_config,
        belief_config=belief_config,
    )
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise V3ModelPackageError(f"output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if formal_evidence is None:
        report = _empty_formal_report()
        evidence_hash = "0" * 64
    else:
        report = validate_h8_formal_evidence(formal_evidence)
        evidence_hash = report["evidence_sha256"]
        if formal_evidence["experiment_identity"]["git_sha"] != source_git_sha:
            raise V3ModelPackageError("formal evidence source SHA does not match package")
        if report["release_status"] == "READY":
            _assert_ready_checkpoint_binding(
                formal_evidence,
                report,
                checkpoint_sha256=_sha256(source),
                ruleset=ruleset,
                search_compatible=search_compatible,
            )
    shutil.copyfile(source, root / "public_checkpoint.pt")
    _write_json(root / "ruleset.json", ruleset.to_dict())
    _write_json(root / "feature_schema.json", schema.to_dict())
    _write_json(root / "model_config.json", {
        "model_config_hash": model_config.stable_hash(),
        "config": model_config.to_dict(),
        "belief_config_hash": (
            belief_config.stable_hash() if belief_config is not None else "0" * 64
        ),
        "belief_config": asdict(belief_config) if belief_config is not None else None,
    })
    _write_json(root / "decision_config.json", {
        "decision_config_hash": _canonical_hash(resolved_decision_config),
        "config": resolved_decision_config,
    })
    _write_json(root / "formal_evidence.json", {
        "supplied": formal_evidence is not None,
        "payload": dict(formal_evidence) if formal_evidence is not None else None,
    })
    _write_json(root / "formal_report.json", report)
    for name, contents in _default_documents(report).items():
        (root / name).write_text(contents, encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository_root / "LICENSE", root / "LICENSE")
    shutil.copyfile(
        repository_root / "douzero" / "deployment" / "THIRD_PARTY_NOTICES",
        root / "THIRD_PARTY_NOTICES",
    )
    manifest = {
        "format": V3_H8_PACKAGE_FORMAT,
        "model_version": V3_HYBRID_MODEL_VERSION,
        "access": "public",
        "source_git_sha": source_git_sha,
        "checkpoint_kind": "public_policy",
        "checkpoint_format": checkpoint_format,
        "checkpoint_sha256": _sha256(root / "public_checkpoint.pt"),
        "feature_schema_hash": schema.stable_hash(),
        "model_config_hash": model_config.stable_hash(),
        "belief_config_hash": (
            belief_config.stable_hash() if belief_config is not None else "0" * 64
        ),
        "ruleset": ruleset.identity(),
        "decision_config_hash": _canonical_hash(resolved_decision_config),
        "search_compatible": search_compatible,
        "formal_evidence_sha256": evidence_hash,
        "release_candidate": report["release_candidate"],
        "release_status": report["release_status"],
        "playing_strength": report["playing_strength"],
    }
    _write_json(root / "manifest.json", manifest)
    checksum_names = sorted(path.name for path in root.iterdir())
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in checksum_names),
        encoding="ascii",
    )
    verify_v3_public_model_package(
        root, schema=schema, ruleset=ruleset, model_config=model_config,
        belief_config=belief_config, expected_source_git_sha=source_git_sha,
        expected_search_compatible=search_compatible,
    )
    return manifest


def verify_v3_public_model_package(
    package_dir: str | Path,
    *,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
    model_config: V3HybridModelConfig,
    belief_config: BeliefConfig | None = None,
    expected_source_git_sha: str | None = None,
    expected_search_compatible: bool | None = None,
) -> dict[str, Any]:
    """Verify exact contents, checksums, identities, and strict reloadability."""

    root = Path(package_dir)
    if not root.is_dir():
        raise V3ModelPackageError(f"package is not a directory: {root}")
    names = {path.name for path in root.iterdir()}
    if names != _FILES:
        raise V3ModelPackageError(
            f"package files mismatch: missing={sorted(_FILES - names)}, "
            f"unknown={sorted(names - _FILES)}"
        )
    forbidden = sorted(
        name for name in names
        if any(token in name.lower() for token in _FORBIDDEN_FILE_TOKENS)
    )
    if forbidden:
        raise V3ModelPackageError(f"package contains training-only files: {forbidden}")
    checksum_lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    expected_names = sorted(names - {"SHA256SUMS"})
    if len(checksum_lines) != len(expected_names):
        raise V3ModelPackageError("SHA256SUMS entry count mismatch")
    parsed: dict[str, str] = {}
    for line in checksum_lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise V3ModelPackageError("SHA256SUMS has an invalid line")
        digest, name = line[:64], line[66:]
        if not _HEX64.fullmatch(digest) or name in parsed or "/" in name:
            raise V3ModelPackageError("SHA256SUMS has an invalid entry")
        parsed[name] = digest
    if sorted(parsed) != expected_names:
        raise V3ModelPackageError("SHA256SUMS filenames mismatch")
    for name, digest in parsed.items():
        if _sha256(root / name) != digest:
            raise V3ModelPackageError(f"checksum mismatch for {name}")
    manifest = _read_json(root / "manifest.json")
    expected_manifest_fields = {
        "format", "model_version", "access", "source_git_sha", "checkpoint_kind",
        "checkpoint_format", "checkpoint_sha256", "feature_schema_hash",
        "model_config_hash", "belief_config_hash", "ruleset", "decision_config_hash",
        "search_compatible", "formal_evidence_sha256", "release_candidate",
        "release_status", "playing_strength",
    }
    if set(manifest) != expected_manifest_fields:
        raise V3ModelPackageError("manifest fields mismatch")
    if manifest["format"] != V3_H8_PACKAGE_FORMAT or manifest["model_version"] != V3_HYBRID_MODEL_VERSION:
        raise V3ModelPackageError("package format/model identity mismatch")
    if manifest["access"] != "public" or manifest["checkpoint_kind"] != "public_policy":
        raise V3ModelPackageError("package is not public-policy-only")
    if manifest["checkpoint_sha256"] != _sha256(root / "public_checkpoint.pt"):
        raise V3ModelPackageError("checkpoint digest does not match manifest")
    if manifest["feature_schema_hash"] != schema.stable_hash() or manifest["model_config_hash"] != model_config.stable_hash():
        raise V3ModelPackageError("runtime feature/model identity mismatch")
    expected_belief_hash = belief_config.stable_hash() if belief_config is not None else "0" * 64
    if manifest["belief_config_hash"] != expected_belief_hash:
        raise V3ModelPackageError("runtime belief identity mismatch")
    if manifest["ruleset"] != ruleset.identity():
        raise V3ModelPackageError("runtime ruleset identity mismatch")
    if _read_json(root / "ruleset.json") != ruleset.to_dict():
        raise V3ModelPackageError("ruleset.json identity mismatch")
    if _read_json(root / "feature_schema.json") != schema.to_dict():
        raise V3ModelPackageError("feature_schema.json identity mismatch")
    if not _HEX40.fullmatch(str(manifest["source_git_sha"])):
        raise V3ModelPackageError("manifest source Git SHA is malformed")
    if not _HEX64.fullmatch(str(manifest["formal_evidence_sha256"])):
        raise V3ModelPackageError("manifest formal evidence hash is malformed")
    if expected_source_git_sha is not None and manifest["source_git_sha"] != expected_source_git_sha:
        raise V3ModelPackageError("source Git SHA mismatch")
    if expected_search_compatible is not None and manifest["search_compatible"] is not expected_search_compatible:
        raise V3ModelPackageError("search compatibility mismatch")
    model_payload = _read_json(root / "model_config.json")
    if model_payload != {
        "model_config_hash": model_config.stable_hash(),
        "config": model_config.to_dict(),
        "belief_config_hash": expected_belief_hash,
        "belief_config": asdict(belief_config) if belief_config is not None else None,
    }:
        raise V3ModelPackageError("model_config.json identity mismatch")
    decision = _read_json(root / "decision_config.json")
    if set(decision) != {"decision_config_hash", "config"} or not isinstance(
        decision["config"], dict
    ):
        raise V3ModelPackageError("decision_config.json identity mismatch")
    resolved_decision = _validate_decision_config(decision["config"])
    if resolved_decision != decision["config"] or decision[
        "decision_config_hash"
    ] != _canonical_hash(resolved_decision):
        raise V3ModelPackageError("decision_config.json identity mismatch")
    if decision["decision_config_hash"] != manifest["decision_config_hash"]:
        raise V3ModelPackageError("decision config does not match manifest")
    report = _read_json(root / "formal_report.json")
    evidence_wrapper = _read_json(root / "formal_evidence.json")
    if set(evidence_wrapper) != {"supplied", "payload"} or not isinstance(
        evidence_wrapper["supplied"], bool
    ):
        raise V3ModelPackageError("formal_evidence.json envelope mismatch")
    if evidence_wrapper["supplied"]:
        evidence_payload = evidence_wrapper["payload"]
        if not isinstance(evidence_payload, dict):
            raise V3ModelPackageError("formal evidence payload must be an object")
        try:
            recomputed_report = validate_h8_formal_evidence(evidence_payload)
        except (H8EvidenceError, TypeError, ValueError) as exc:
            raise V3ModelPackageError(
                f"packaged formal evidence is invalid: {exc}"
            ) from exc
    else:
        if evidence_wrapper["payload"] is not None:
            raise V3ModelPackageError(
                "absent formal evidence must use a null payload"
            )
        recomputed_report = _empty_formal_report()
    expected_report_fields = {
        "schema_version", "evidence_sha256", "release_candidate",
        "release_status", "playing_strength", "required_variants",
        "training_run_count", "evaluation_count", "issues",
        "support_matrix_hash", "development_status",
    }
    if set(report) != expected_report_fields:
        raise V3ModelPackageError("formal_report.json fields mismatch")
    if report != recomputed_report:
        raise V3ModelPackageError(
            "formal report does not match recomputed packaged evidence"
        )
    for report_name, manifest_name in (
        ("evidence_sha256", "formal_evidence_sha256"),
        ("release_candidate", "release_candidate"),
        ("release_status", "release_status"),
        ("playing_strength", "playing_strength"),
    ):
        if report[report_name] != manifest[manifest_name]:
            raise V3ModelPackageError("formal report does not match manifest")
    if report["schema_version"] != H8_REPORT_SCHEMA_VERSION:
        raise V3ModelPackageError("formal report schema mismatch")
    if report["support_matrix_hash"] != h8a_support_matrix_hash():
        raise V3ModelPackageError("formal report support matrix mismatch")
    checkpoint_format = _strict_load_checkpoint(
        root / "public_checkpoint.pt", schema=schema, ruleset=ruleset,
        model_config=model_config, belief_config=belief_config,
    )
    if checkpoint_format != manifest["checkpoint_format"]:
        raise V3ModelPackageError("checkpoint format does not match manifest")
    if manifest["release_status"] != "READY" and (
        manifest["release_candidate"] != "NONE"
        or "not measured" not in str(manifest["playing_strength"]).lower()
    ):
        raise V3ModelPackageError("non-ready package makes a release/strength claim")
    if manifest["release_status"] == "READY":
        if not evidence_wrapper["supplied"]:
            raise V3ModelPackageError("READY package must carry formal evidence")
        _assert_ready_checkpoint_binding(
            evidence_payload,
            report,
            checkpoint_sha256=_sha256(root / "public_checkpoint.pt"),
            ruleset=ruleset,
            search_compatible=manifest["search_compatible"],
        )
    return manifest


__all__ = [
    "V3_H8_PACKAGE_FORMAT",
    "V3ModelPackageError",
    "create_v3_public_model_package",
    "verify_v3_public_model_package",
]
