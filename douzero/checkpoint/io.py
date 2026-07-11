"""Checkpoint save/load with versioned manifests (P01).

Save format: ``model.tar`` keeps the legacy six keys and adds a ``manifest``
key (stored as a plain dict, never a pickled dataclass). Legacy loaders that
ignore unknown keys are unaffected; the new loader validates the manifest.

Security: PyTorch >=2.6 defaults to ``weights_only=True``. We do NOT rely on
the version-dependent default -- every ``torch.load`` call sets it explicitly:

  - position-weights sidecar (pure state_dict): ``weights_only=True`` (safe).
  - training ``model.tar``: ``weights_only=False`` is REQUIRED because the
    bundle contains optimizer state dicts and arbitrary Python objects (stats,
    flags, manifest). This is an explicit, documented, trusted-path choice:
    the training checkpoint is a locally-produced artifact, not an untrusted
    download. The constant ``TRAINING_CHECKPOINT_TRUSTED`` documents this.

Load behavior:
  - manifest present + compatible -> return (bundle, manifest)
  - manifest present + incompatible -> raise CheckpointCompatibilityError
    (never silently partial-load)
  - manifest absent (legacy checkpoint) -> delegate to compat, manifest=None
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from douzero.checkpoint.compat import _resolve_map_location, load_legacy_model_tar
from douzero.checkpoint.manifest import (
    CHECKPOINT_KINDS,
    CURRENT_SCHEMA_VERSION,
    CheckpointManifest,
    build_manifest,
)


class CheckpointCompatibilityError(Exception):
    """Raised when a checkpoint's manifest is incompatible with the runtime.

    The message includes the offending field and expected/actual values so the
    failure is actionable. We never fall back to a permissive partial load.
    """


# Documented trust flag for the training-checkpoint load path. The model.tar
# bundle must load with weights_only=False (it contains optimizer states and
# arbitrary Python objects); this is acceptable ONLY because the checkpoint is a
# locally-produced training artifact, never an untrusted download.
TRAINING_CHECKPOINT_TRUSTED = True


def save_checkpoint(
    path: str,
    learner_models: dict,
    optimizers: dict,
    stats: dict,
    flags: argparse.Namespace | dict[str, Any] | None,
    frames: int,
    position_frames: dict[str, int],
) -> CheckpointManifest:
    """Save a model.tar with the legacy six keys PLUS a manifest.

    Returns the manifest that was stamped in. The manifest is serialized via
    ``to_dict()`` (plain dict) so it round-trips under weights_only=True loads
    of the manifest itself; the full bundle still needs weights_only=False at
    load time (see TRAINING_CHECKPOINT_TRUSTED) because of optimizer states.
    """
    manifest = build_manifest(
        flags, frames, position_frames, checkpoint_kind="training_checkpoint"
    )
    bundle = {
        "model_state_dict": {k: learner_models[k].state_dict() for k in learner_models},
        "optimizer_state_dict": {k: optimizers[k].state_dict() for k in optimizers},
        "stats": stats,
        "flags": vars(flags) if not isinstance(flags, dict) and flags is not None else (flags or {}),
        "frames": frames,
        "position_frames": position_frames,
        "manifest": manifest.to_dict(),
    }
    torch.save(bundle, path)
    return manifest


def load_checkpoint(
    path: str,
    *,
    expected_model_version: str = "legacy",
    expected_feature_version: str = "legacy",
    expected_ruleset_id: str = "legacy",
    expected_checkpoint_kind: str = "training_checkpoint",
    expected_schema_version: int = CURRENT_SCHEMA_VERSION,
    training_device: str | None = None,
) -> tuple[dict, CheckpointManifest | None]:
    """Load a model.tar, validating its manifest when present.

    Returns ``(bundle, manifest)`` where ``manifest`` is None for legacy
    checkpoints that predate manifests. Raises CheckpointCompatibilityError on
    a version mismatch (never silently partial-loads).

    The training bundle requires weights_only=False (optimizer states); this is
    the documented trusted path (TRAINING_CHECKPOINT_TRUSTED).
    """
    if not TRAINING_CHECKPOINT_TRUSTED:
        raise RuntimeError(
            "Refusing to load a training checkpoint: the trusted path is disabled."
        )
    bundle = torch.load(
        path,
        map_location=_resolve_map_location(training_device),
        weights_only=False,
    )
    if not isinstance(bundle, dict) or "manifest" not in bundle:
        # Legacy checkpoint (no manifest) -- delegate to the compat path. This
        # returns the same bundle with manifest=None.
        return load_legacy_model_tar(path, training_device=training_device)

    manifest = CheckpointManifest.from_dict(bundle["manifest"])
    _validate_manifest(manifest, expected_schema_version, expected_model_version,
                       expected_feature_version, expected_ruleset_id,
                       expected_checkpoint_kind, path)
    return bundle, manifest


def _validate_manifest(
    manifest: CheckpointManifest,
    expected_schema_version: int,
    expected_model_version: str,
    expected_feature_version: str,
    expected_ruleset_id: str,
    expected_checkpoint_kind: str,
    path: str,
) -> None:
    """Validate all version/identity fields; raise on any mismatch."""
    ctx = f"Checkpoint at {path} (git_sha={manifest.git_sha}, frames={manifest.frames})"

    if manifest.schema_version != expected_schema_version:
        raise CheckpointCompatibilityError(
            f"Checkpoint schema_version mismatch: checkpoint has "
            f"{manifest.schema_version}, runtime expects {expected_schema_version}. {ctx}"
        )
    if manifest.model_version != expected_model_version:
        raise CheckpointCompatibilityError(
            f"Checkpoint model_version mismatch: checkpoint has "
            f"{manifest.model_version!r}, runtime expects {expected_model_version!r}. {ctx}"
        )
    if manifest.feature_version != expected_feature_version:
        raise CheckpointCompatibilityError(
            f"Checkpoint feature_version mismatch: checkpoint has "
            f"{manifest.feature_version!r}, runtime expects {expected_feature_version!r}. {ctx}"
        )
    if manifest.ruleset_id != expected_ruleset_id:
        raise CheckpointCompatibilityError(
            f"Checkpoint ruleset_id mismatch: checkpoint has "
            f"{manifest.ruleset_id!r}, runtime expects {expected_ruleset_id!r}. {ctx}"
        )
    if manifest.checkpoint_kind != expected_checkpoint_kind:
        raise CheckpointCompatibilityError(
            f"Checkpoint checkpoint_kind mismatch: checkpoint has "
            f"{manifest.checkpoint_kind!r}, runtime expects {expected_checkpoint_kind!r}. "
            "A training_checkpoint cannot be loaded where a position_weights "
            f"sidecar is expected (or vice versa). {ctx}"
        )
    if manifest.checkpoint_kind not in CHECKPOINT_KINDS:
        raise CheckpointCompatibilityError(
            f"Checkpoint has unknown checkpoint_kind {manifest.checkpoint_kind!r}; "
            f"expected one of {sorted(CHECKPOINT_KINDS)}. {ctx}"
        )
