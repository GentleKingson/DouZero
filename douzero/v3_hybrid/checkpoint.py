"""Strict public-only H1 checkpoint sidecar."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.checkpoint.manifest import (
    MODEL_ACCESS_PUBLIC,
    CheckpointManifest,
    build_manifest,
)
from douzero.env.rules import RuleSet
from douzero.observation.schema import FeatureSchemaManifest

from .config import V3HybridModelConfig
from .contract import (
    V3_HYBRID_CHECKPOINT_KIND,
    V3_HYBRID_FEATURE_VERSION,
    V3_HYBRID_LOSS_TERMS,
    V3_HYBRID_MODEL_VERSION,
    V3HybridCompatibilityIdentity,
    assert_v3_hybrid_compatible,
)
from .model import V3HybridModel

V3_HYBRID_H1_CHECKPOINT_FORMAT = "v3-hybrid-h1-public-policy-v1"

_CHECKPOINT_KEYS = frozenset({
    "format",
    "manifest",
    "state_dict",
    "feature_schema_hash",
    "model_config",
    "model_config_hash",
    "compatibility_identity",
    "compatibility_hash",
})

_FORBIDDEN_PAYLOAD_NAMES = (
    "privileged",
    "teacher",
    "oracle",
    "all_handcards",
    "hidden_hand",
    "training_labels",
)


def h1_compatibility_identity(
    config: V3HybridModelConfig,
    ruleset: RuleSet,
) -> V3HybridCompatibilityIdentity:
    """Build the complete frozen H0 identity for the H1 public sidecar."""

    disabled = {"version": "disabled_h1"}
    return V3HybridCompatibilityIdentity(
        ruleset=ruleset.identity(),
        feature_flags={
            "adaptive_dmc": False,
            "belief": False,
            "bidding": False,
            "cooperation": False,
            "human_bc": False,
            "league": False,
            "oracle": False,
            "strategy": False,
            "style": False,
        },
        model_graph=config.compatibility_dict(),
        output_semantics={
            "dmc_q": {
                "perspective": "acting_team",
                "target": "monte_carlo_return",
                "transform": config.dmc_target_transform,
                "clamp": config.dmc_target_clamp,
            },
            "win": "acting_team_probability",
            "score_if_win": "acting_team_conditional_signed_score",
            "score_if_loss": "acting_team_conditional_signed_score",
            "bidding": "not_integrated_h1",
        },
        optimizer_config={"version": "not_integrated_h1"},
        loss_config={
            "version": "not_integrated_h1",
            "available_terms": list(V3_HYBRID_LOSS_TERMS),
        },
        loss_schedules={"version": "not_integrated_h1"},
        belief_layout=disabled,
        cooperation_mixer=disabled,
        trainer_config={"version": "not_integrated_h1"},
        training_topology={"version": "model_only_h1"},
    )


def _reject_forbidden_names(state_dict: dict[str, torch.Tensor]) -> None:
    violations = sorted(
        name
        for name in state_dict
        if any(token in name.lower() for token in _FORBIDDEN_PAYLOAD_NAMES)
    )
    if violations:
        raise CheckpointCompatibilityError(
            f"public V3 checkpoint contains forbidden parameter names: {violations}"
        )


def _manifest(model: V3HybridModel, ruleset: RuleSet) -> CheckpointManifest:
    manifest = build_manifest(
        {
            "feature_version": V3_HYBRID_FEATURE_VERSION,
            "model_version": V3_HYBRID_MODEL_VERSION,
            "ruleset": ruleset.ruleset_id,
            "model_config_hash": model.config.stable_hash(),
        },
        frames=0,
        position_frames={},
        checkpoint_kind=V3_HYBRID_CHECKPOINT_KIND,
    )
    object.__setattr__(manifest, "ruleset_id", ruleset.ruleset_id)
    object.__setattr__(manifest, "ruleset_version", ruleset.ruleset_version)
    object.__setattr__(manifest, "ruleset_hash", ruleset.stable_hash())
    return manifest


def save_v3_hybrid_public_checkpoint(
    path: str | Path,
    model: V3HybridModel,
    *,
    ruleset: RuleSet,
) -> CheckpointManifest:
    """Atomically save a strict public sidecar with no training-only payload."""

    if not isinstance(model, V3HybridModel):
        raise TypeError("public checkpoint requires a V3HybridModel")
    if not isinstance(ruleset, RuleSet):
        raise TypeError("public checkpoint requires a RuleSet")
    bound_ruleset = getattr(model, "expected_ruleset_identity", None)
    requested_ruleset = (
        ruleset.ruleset_id,
        ruleset.ruleset_version,
        ruleset.stable_hash(),
    )
    if bound_ruleset is not None and bound_ruleset != requested_ruleset:
        raise ValueError(
            "cannot relabel a loaded V3 model with a different ruleset identity"
        )
    state_dict = model.state_dict()
    _reject_forbidden_names(state_dict)
    identity = h1_compatibility_identity(model.config, ruleset)
    manifest = _manifest(model, ruleset)
    bundle = {
        "format": V3_HYBRID_H1_CHECKPOINT_FORMAT,
        "manifest": manifest.to_dict(),
        "state_dict": state_dict,
        "feature_schema_hash": model.schema.stable_hash(),
        "model_config": asdict(model.config),
        "model_config_hash": model.config.stable_hash(),
        "compatibility_identity": identity.compatibility_dict(),
        "compatibility_hash": identity.stable_hash(),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(bundle, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _load_bundle(path: Path, device: str | torch.device) -> dict:
    try:
        bundle = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"unable to safely load V3 checkpoint {str(path)!r}: {exc}"
        ) from exc
    if not isinstance(bundle, dict):
        raise CheckpointCompatibilityError("V3 checkpoint must contain a dict")
    actual_keys = set(bundle)
    if actual_keys != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - actual_keys)
        extra = sorted(actual_keys - _CHECKPOINT_KEYS)
        raise CheckpointCompatibilityError(
            f"V3 checkpoint envelope mismatch: missing={missing}, extra={extra}"
        )
    return bundle


def load_v3_hybrid_public_checkpoint(
    path: str | Path,
    *,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
    config: V3HybridModelConfig,
    device: str | torch.device = "cpu",
) -> V3HybridModel:
    """Strictly validate identity before constructing and loading H1 weights."""

    if not isinstance(schema, FeatureSchemaManifest):
        raise TypeError("schema must be a FeatureSchemaManifest")
    if not isinstance(ruleset, RuleSet):
        raise TypeError("ruleset must be a RuleSet")
    if not isinstance(config, V3HybridModelConfig):
        raise TypeError("config must be a V3HybridModelConfig")
    checkpoint = Path(path)
    bundle = _load_bundle(checkpoint, device)
    if bundle["format"] != V3_HYBRID_H1_CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError("unsupported V3 Hybrid checkpoint format")
    try:
        manifest = CheckpointManifest.from_dict(bundle["manifest"])
    except Exception as exc:
        raise CheckpointCompatibilityError(f"invalid V3 manifest: {exc}") from exc
    expected_manifest = {
        "model_version": V3_HYBRID_MODEL_VERSION,
        "feature_version": V3_HYBRID_FEATURE_VERSION,
        "checkpoint_kind": V3_HYBRID_CHECKPOINT_KIND,
        "model_access": MODEL_ACCESS_PUBLIC,
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_version": ruleset.ruleset_version,
        "ruleset_hash": ruleset.stable_hash(),
    }
    for name, expected in expected_manifest.items():
        actual = getattr(manifest, name)
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"V3 checkpoint {name} mismatch: {actual!r} != {expected!r}"
            )
    if bundle["feature_schema_hash"] != schema.stable_hash():
        raise CheckpointCompatibilityError("V3 checkpoint feature schema mismatch")
    if bundle["model_config"] != asdict(config):
        raise CheckpointCompatibilityError("V3 checkpoint model config mismatch")
    if bundle["model_config_hash"] != config.stable_hash():
        raise CheckpointCompatibilityError("V3 checkpoint model config hash mismatch")
    expected_identity = h1_compatibility_identity(config, ruleset)
    try:
        assert_v3_hybrid_compatible(
            expected_identity,
            bundle["compatibility_identity"],
            actual_hash=bundle["compatibility_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"V3 checkpoint compatibility identity mismatch: {exc}"
        ) from exc
    state_dict = bundle["state_dict"]
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError("V3 state_dict must be a dict")
    _reject_forbidden_names(state_dict)
    model = V3HybridModel(schema, config).to(device)
    expected_keys = set(model.state_dict())
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        raise CheckpointCompatibilityError(
            "V3 state_dict key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"V3 state_dict shape/value mismatch: {exc}"
        ) from exc
    model.expected_ruleset_identity = (
        ruleset.ruleset_id,
        ruleset.ruleset_version,
        ruleset.stable_hash(),
    )
    model.checkpoint_manifest = manifest
    model.compatibility_identity = expected_identity
    return model
