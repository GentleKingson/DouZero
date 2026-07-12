"""Checkpoint save/load for the belief model (P07).

A belief checkpoint is a self-describing bundle: the model ``state_dict`` plus a
manifest carrying the architecture identity (``BeliefConfig.stable_hash()``),
the ruleset identity, the feature version, and provenance (git sha, torch
version, frame count). The loader validates the architecture hash and ruleset
so a belief checkpoint cannot be silently loaded into a mismatched model.

This is deliberately separate from :mod:`douzero.checkpoint.v2` (the value
model's checkpoint machinery): the belief model is pretrained and frozen
independently, and its identity axes are its own architecture + the public
feature schema it consumes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

#: Current belief-checkpoint manifest schema version.
BELIEF_MANIFEST_SCHEMA_VERSION: int = 1

#: The model-version string stamped into every belief checkpoint.
BELIEF_MODEL_VERSION: str = "belief_v1"

#: Bundle keys.
_STATE_DICT_KEY = "belief_state_dict"
_MANIFEST_KEY = "manifest"
_CONFIG_KEY = "belief_config"


@dataclass(frozen=True)
class BeliefManifest:
    """Provenance + identity manifest for a belief checkpoint."""

    schema_version: int
    model_version: str
    belief_config_hash: str
    belief_config: dict[str, Any]
    feature_version: str
    ruleset_id: str
    ruleset_version: str
    ruleset_hash: str
    checkpoint_kind: str
    git_sha: str
    python_version: str
    torch_version: str
    platform: str
    created_at: str
    frames: int
    effective_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_belief_checkpoint(
    path: str,
    model: "object",
    *,
    ruleset: "object | None" = None,
    feature_version: str = "v2",
    frames: int = 0,
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Save a belief model bundle with a manifest (atomic write).

    Parameters
    ----------
    path:
        Output ``.pt`` path. Written to a temp file then renamed for atomicity.
    model:
        A :class:`~douzero.belief.model.BeliefModel`.
    ruleset:
        Optional :class:`~douzero.env.rules.RuleSet`; its identity is stamped
        into the manifest. Defaults to the canonical legacy ruleset.
    feature_version:
        The public feature version the belief input was built against.
    frames:
        Number of training frames consumed so far.
    extra_config:
        Additional audit-only config (e.g. loss lambdas, learning rate).
    """
    from douzero.env.rules import RuleSet

    from .model import BeliefConfig

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module (BeliefModel)")
    config: BeliefConfig = getattr(model, "config", BeliefConfig())
    rs = ruleset if ruleset is not None else RuleSet.legacy()
    rs_ident = rs.identity() if hasattr(rs, "identity") else {
        "ruleset_id": "legacy",
        "ruleset_version": "legacy-v1",
        "ruleset_hash": rs.stable_hash() if hasattr(rs, "stable_hash") else "",
    }
    manifest = BeliefManifest(
        schema_version=BELIEF_MANIFEST_SCHEMA_VERSION,
        model_version=BELIEF_MODEL_VERSION,
        belief_config_hash=config.stable_hash(),
        belief_config=config.compatibility_dict(),
        feature_version=feature_version,
        ruleset_id=rs_ident.get("ruleset_id", "legacy"),
        ruleset_version=rs_ident.get("ruleset_version", "legacy-v1"),
        ruleset_hash=rs_ident.get("ruleset_hash", ""),
        checkpoint_kind="belief_model",
        git_sha=_git_sha(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        platform=platform.platform(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        frames=int(frames),
        effective_config=dict(extra_config or {}),
    )
    # Store ONLY the BeliefConfig constructor fields (the dataclass fields), so
    # the loader can reconstruct it via ``BeliefConfig(**fields)``. The full
    # compatibility_dict (with derived constants) is kept in the manifest for
    # the hash check; it must NOT be passed to the constructor.
    config_fields = {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }
    bundle = {
        _STATE_DICT_KEY: model.state_dict(),
        _MANIFEST_KEY: manifest.to_dict(),
        _CONFIG_KEY: config_fields,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(bundle, tmp)
    os.replace(tmp, path)


def load_belief_checkpoint(
    path: str,
    *,
    expected_belief_config: "object | None" = None,
    expected_ruleset: "object | None" = None,
    map_location: Any = "cpu",
) -> "object":
    """Load and validate a belief checkpoint, returning a ready model.

    Validates the manifest's architecture hash against ``expected_belief_config``
    (if provided) and the ruleset identity against ``expected_ruleset`` (if
    provided). Raises :class:`ValueError` on any mismatch rather than
    permissively partial-loading.

    Returns the reconstructed :class:`~douzero.belief.model.BeliefModel` in
    ``eval()`` mode.
    """
    from douzero.env.rules import RuleSet

    from .model import BeliefConfig, BeliefModel

    bundle = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(bundle, dict) or _MANIFEST_KEY not in bundle:
        raise ValueError(
            f"{path!r} is not a belief checkpoint bundle (missing "
            f"{_MANIFEST_KEY!r})."
        )
    manifest = bundle[_MANIFEST_KEY]
    if manifest.get("model_version") != BELIEF_MODEL_VERSION:
        raise ValueError(
            f"checkpoint model_version {manifest.get('model_version')!r} != "
            f"expected {BELIEF_MODEL_VERSION!r}."
        )
    # Reconstruct the config from the CONSTRUCTOR fields (stored at the bundle
    # top level under _CONFIG_KEY), NOT from ``manifest['belief_config']``
    # (which is the full compatibility dict including derived constants that
    # are not valid constructor kwargs).
    config = BeliefConfig(**bundle[_CONFIG_KEY])
    # Architecture hash check (the strict identity axis).
    runtime_hash = config.stable_hash()
    if expected_belief_config is not None:
        expected_hash = expected_belief_config.stable_hash()
        if expected_hash != manifest["belief_config_hash"]:
            raise ValueError(
                "belief config hash mismatch: checkpoint "
                f"{manifest['belief_config_hash']!r} != runtime "
                f"{expected_hash!r}. The checkpoint was trained under a "
                "different belief architecture."
            )
    if runtime_hash != manifest["belief_config_hash"]:
        raise ValueError(
            "belief config reconstructed from checkpoint does not reproduce "
            f"its manifest hash ({runtime_hash!r} vs "
            f"{manifest['belief_config_hash']!r}); the config schema drifted."
        )
    # Ruleset identity check.
    if expected_ruleset is not None:
        rs_ident = expected_ruleset.identity()
        for key in ("ruleset_id", "ruleset_version", "ruleset_hash"):
            if manifest.get(key) != rs_ident.get(key):
                raise ValueError(
                    f"ruleset {key} mismatch: checkpoint {manifest.get(key)!r} "
                    f"!= runtime {rs_ident.get(key)!r}."
                )
    model = BeliefModel(config)
    model.load_state_dict(bundle[_STATE_DICT_KEY])
    model.eval()
    return model


def manifest_hash(path: str) -> str:
    """Return a short hash of a checkpoint's manifest (for logging)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    m = bundle[_MANIFEST_KEY]
    payload = json.dumps(m, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
