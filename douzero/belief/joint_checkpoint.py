"""Strict checkpoints for jointly trained belief and value models.

Frozen belief training keeps the established independent V2 and belief
checkpoint formats unchanged.  Joint and alternating training need one atomic
bundle: loading only one half would silently pair weights that were never
trained together.  This module stores both state dicts, both architecture
identities, the public feature schema, ruleset identity, optimizer state, and
training mode in a dedicated checkpoint kind.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import torch

JOINT_CHECKPOINT_SCHEMA_VERSION = 1
JOINT_CHECKPOINT_KIND = "joint_belief_value_training"
JOINT_BELIEF_MODES = frozenset({"joint", "alternating"})


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for common training wrappers."""
    current = module
    while isinstance(getattr(current, "module", None), torch.nn.Module):
        current = current.module
    return current


@dataclass(frozen=True)
class JointCheckpointManifest:
    schema_version: int
    checkpoint_kind: str
    belief_training_mode: str
    value_model_config_hash: str
    belief_model_config_hash: str
    feature_schema_hash: str
    ruleset_id: str
    ruleset_version: str
    ruleset_hash: str
    public_input_contract: str
    optimizer_included: bool
    optimizer_steps: int
    git_sha: str
    python_version: str
    torch_version: str
    platform: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identities(value_model, belief_model) -> tuple[torch.nn.Module, torch.nn.Module, str, str, str]:
    value = _unwrap(value_model)
    belief = _unwrap(belief_model)
    if not hasattr(value, "config") or not hasattr(value.config, "stable_hash"):
        raise TypeError("value_model must expose a stable ModelV2 config identity")
    if not hasattr(value, "schema") or not hasattr(value.schema, "stable_hash"):
        raise TypeError("value_model must expose a stable feature schema identity")
    if not hasattr(belief, "config") or not hasattr(belief.config, "stable_hash"):
        raise TypeError("belief_model must expose a stable BeliefConfig identity")
    return (
        value,
        belief,
        value.config.stable_hash(),
        belief.config.stable_hash(),
        value.schema.stable_hash(),
    )


def save_joint_checkpoint(
    path: str,
    value_model: torch.nn.Module,
    belief_model: torch.nn.Module,
    *,
    ruleset: object,
    belief_training_mode: str,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_steps: int = 0,
) -> JointCheckpointManifest:
    """Atomically save a coupled joint/alternating training checkpoint."""

    from douzero.env.rules import RuleSet

    if belief_training_mode not in JOINT_BELIEF_MODES:
        raise ValueError(
            "joint checkpoints require belief_training_mode 'joint' or "
            f"'alternating', got {belief_training_mode!r}. Frozen mode keeps "
            "the existing independent checkpoint formats."
        )
    if not isinstance(ruleset, RuleSet):
        raise TypeError("ruleset must be a RuleSet")
    if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int) or optimizer_steps < 0:
        raise ValueError("optimizer_steps must be a non-negative int")

    value, belief, value_hash, belief_hash, schema_hash = _identities(
        value_model, belief_model
    )
    identity = ruleset.identity()
    manifest = JointCheckpointManifest(
        schema_version=JOINT_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_kind=JOINT_CHECKPOINT_KIND,
        belief_training_mode=belief_training_mode,
        value_model_config_hash=value_hash,
        belief_model_config_hash=belief_hash,
        feature_schema_hash=schema_hash,
        ruleset_id=identity["ruleset_id"],
        ruleset_version=identity["ruleset_version"],
        ruleset_hash=identity["ruleset_hash"],
        public_input_contract="belief_input_public_v1",
        optimizer_included=optimizer is not None,
        optimizer_steps=optimizer_steps,
        git_sha=_git_sha(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        platform=platform.platform(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    bundle = {
        "manifest": manifest.to_dict(),
        "value_state_dict": value.state_dict(),
        "belief_state_dict": belief.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return manifest


def load_joint_checkpoint(
    path: str,
    value_model: torch.nn.Module,
    belief_model: torch.nn.Module,
    *,
    expected_ruleset: object,
    expected_belief_training_mode: str,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: Any = "cpu",
    allow_unsafe_pickle: bool = False,
) -> JointCheckpointManifest:
    """Validate and restore both halves of a coupled training checkpoint."""

    from douzero.env.rules import RuleSet

    if expected_belief_training_mode not in JOINT_BELIEF_MODES:
        raise ValueError("expected mode must be 'joint' or 'alternating'")
    if not isinstance(expected_ruleset, RuleSet):
        raise TypeError("expected_ruleset must be a RuleSet")
    value, belief, value_hash, belief_hash, schema_hash = _identities(
        value_model, belief_model
    )
    bundle = torch.load(
        path,
        map_location=map_location,
        weights_only=not allow_unsafe_pickle,
    )
    if not isinstance(bundle, dict):
        raise ValueError("joint checkpoint must be a dict bundle")
    required = {
        "manifest",
        "value_state_dict",
        "belief_state_dict",
        "optimizer_state_dict",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"joint checkpoint is missing keys {sorted(missing)}")
    raw = bundle["manifest"]
    if not isinstance(raw, dict):
        raise TypeError("joint checkpoint manifest must be a dict")
    try:
        manifest = JointCheckpointManifest(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid joint checkpoint manifest: {exc}") from exc
    if manifest.schema_version != JOINT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("joint checkpoint schema_version mismatch")
    if manifest.checkpoint_kind != JOINT_CHECKPOINT_KIND:
        raise ValueError("joint checkpoint kind mismatch")
    if manifest.belief_training_mode != expected_belief_training_mode:
        raise ValueError(
            "belief training mode mismatch: checkpoint "
            f"{manifest.belief_training_mode!r} != runtime "
            f"{expected_belief_training_mode!r}"
        )
    expected = {
        "value_model_config_hash": value_hash,
        "belief_model_config_hash": belief_hash,
        "feature_schema_hash": schema_hash,
    }
    for name, expected_value in expected.items():
        if getattr(manifest, name) != expected_value:
            raise ValueError(
                f"joint checkpoint {name} mismatch: "
                f"{getattr(manifest, name)!r} != {expected_value!r}"
            )
    ruleset_identity = expected_ruleset.identity()
    for name in ("ruleset_id", "ruleset_version", "ruleset_hash"):
        if getattr(manifest, name) != ruleset_identity[name]:
            raise ValueError(f"joint checkpoint {name} mismatch")
    if manifest.public_input_contract != "belief_input_public_v1":
        raise ValueError("joint checkpoint public input contract mismatch")

    value_state = bundle["value_state_dict"]
    belief_state = bundle["belief_state_dict"]
    if not isinstance(value_state, dict) or not isinstance(belief_state, dict):
        raise TypeError("joint checkpoint model states must be dictionaries")
    value.load_state_dict(value_state, strict=True)
    belief.load_state_dict(belief_state, strict=True)
    optimizer_state = bundle["optimizer_state_dict"]
    if optimizer is not None:
        if not manifest.optimizer_included or not isinstance(optimizer_state, dict):
            raise ValueError("runtime requested optimizer restore but checkpoint has none")
        optimizer.load_state_dict(optimizer_state)
    return manifest


__all__ = [
    "JOINT_CHECKPOINT_KIND",
    "JOINT_CHECKPOINT_SCHEMA_VERSION",
    "JointCheckpointManifest",
    "load_joint_checkpoint",
    "save_joint_checkpoint",
]
