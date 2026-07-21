"""Strict coupled public checkpoint for H4 belief-feedback policies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.env.rules import RuleSet
from douzero.observation.schema import FeatureSchemaManifest

from .belief_policy import V3BeliefPolicy
from .config import BELIEF_FEEDBACK_NONE, V3HybridModelConfig
from .contract import V3_HYBRID_MODEL_VERSION
from .model import V3HybridModel

V3_H4_PUBLIC_CHECKPOINT_FORMAT = "v3-hybrid-h4-belief-public-v1"

_KEYS = frozenset({
    "format",
    "artifact_access",
    "model_version",
    "feature_schema_hash",
    "ruleset_identity",
    "model_config",
    "model_config_hash",
    "belief_config",
    "belief_config_hash",
    "policy_identity",
    "policy_identity_hash",
    "student_state_dict",
    "belief_state_dict",
})
_FORBIDDEN = (
    "privileged", "teacher", "oracle", "all_handcards", "hidden_hand", "label"
)


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _identity(
    model_config: V3HybridModelConfig,
    belief_config: BeliefConfig,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
) -> dict[str, object]:
    return {
        "identity_version": 1,
        "model_version": V3_HYBRID_MODEL_VERSION,
        "access": "public",
        "feature_schema_hash": schema.stable_hash(),
        "ruleset": ruleset.identity(),
        "model_config_hash": model_config.stable_hash(),
        "belief_config_hash": belief_config.stable_hash(),
        "belief_layout": "public_joint_rank_count_conservative_dp_v1",
        "policy_feedback": model_config.belief_feedback,
        "posterior_gradient": "detached_before_policy_v1",
        "legal_actions": "environment_owned_rank_only_v1",
    }


def _validate_states(
    student: dict[str, torch.Tensor], belief: dict[str, torch.Tensor]
) -> None:
    if any(
        token in name.lower()
        for state in (student, belief)
        for name in state
        for token in _FORBIDDEN
    ):
        raise CheckpointCompatibilityError(
            "H4 public checkpoint contains training-only parameter names"
        )


def save_v3_h4_public_checkpoint(
    path: str | Path, policy: V3BeliefPolicy
) -> None:
    if not isinstance(policy, V3BeliefPolicy):
        raise TypeError("H4 public checkpoint requires a V3BeliefPolicy")
    student = policy.model.state_dict()
    belief = policy.belief_model.state_dict()
    _validate_states(student, belief)
    identity = _identity(
        policy.model.config,
        policy.belief_model.config,
        policy.model.schema,
        policy.ruleset,
    )
    bundle = {
        "format": V3_H4_PUBLIC_CHECKPOINT_FORMAT,
        "artifact_access": "public",
        "model_version": V3_HYBRID_MODEL_VERSION,
        "feature_schema_hash": policy.model.schema.stable_hash(),
        "ruleset_identity": policy.ruleset.identity(),
        "model_config": asdict(policy.model.config),
        "model_config_hash": policy.model.config.stable_hash(),
        "belief_config": asdict(policy.belief_model.config),
        "belief_config_hash": policy.belief_model.config.stable_hash(),
        "policy_identity": identity,
        "policy_identity_hash": _hash(identity),
        "student_state_dict": student,
        "belief_state_dict": belief,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(bundle, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_v3_h4_public_checkpoint(
    path: str | Path,
    *,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
    model_config: V3HybridModelConfig,
    belief_config: BeliefConfig,
    device: str | torch.device = "cpu",
) -> V3BeliefPolicy:
    if model_config.belief_feedback == BELIEF_FEEDBACK_NONE:
        raise CheckpointCompatibilityError(
            "H4 coupled loader requires enabled belief feedback"
        )
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"unable to safely load H4 public checkpoint: {exc}"
        ) from exc
    if not isinstance(bundle, dict) or set(bundle) != _KEYS:
        raise CheckpointCompatibilityError("H4 public checkpoint envelope mismatch")
    identity = _identity(model_config, belief_config, schema, ruleset)
    expected = {
        "format": V3_H4_PUBLIC_CHECKPOINT_FORMAT,
        "artifact_access": "public",
        "model_version": V3_HYBRID_MODEL_VERSION,
        "feature_schema_hash": schema.stable_hash(),
        "ruleset_identity": ruleset.identity(),
        "model_config": asdict(model_config),
        "model_config_hash": model_config.stable_hash(),
        "belief_config": asdict(belief_config),
        "belief_config_hash": belief_config.stable_hash(),
        "policy_identity": identity,
        "policy_identity_hash": _hash(identity),
    }
    for name, value in expected.items():
        if bundle[name] != value:
            raise CheckpointCompatibilityError(
                f"H4 public checkpoint {name} mismatch"
            )
    student_state = bundle["student_state_dict"]
    belief_state = bundle["belief_state_dict"]
    if not isinstance(student_state, dict) or not isinstance(belief_state, dict):
        raise CheckpointCompatibilityError("H4 public state dictionaries are invalid")
    _validate_states(student_state, belief_state)
    model = V3HybridModel(schema, model_config).to(device)
    belief_model = BeliefModel(belief_config).to(device)
    if set(student_state) != set(model.state_dict()):
        raise CheckpointCompatibilityError("H4 student state keys mismatch")
    if set(belief_state) != set(belief_model.state_dict()):
        raise CheckpointCompatibilityError("H4 belief state keys mismatch")
    try:
        model.load_state_dict(student_state, strict=True)
        belief_model.load_state_dict(belief_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"H4 public state shape mismatch: {exc}"
        ) from exc
    model.eval()
    belief_model.eval()
    return V3BeliefPolicy(model, belief_model, ruleset=ruleset)


__all__ = [
    "V3_H4_PUBLIC_CHECKPOINT_FORMAT",
    "load_v3_h4_public_checkpoint",
    "save_v3_h4_public_checkpoint",
]
