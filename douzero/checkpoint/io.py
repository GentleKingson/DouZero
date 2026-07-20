"""Checkpoint save/load with versioned manifests (P01).

Save format: ``model.tar`` keeps the legacy six keys and adds a ``manifest``
key (stored as a plain dict, never a pickled dataclass). Legacy loaders that
ignore unknown keys are unaffected; the new loader validates the manifest.

Security model: the DEFAULT load path uses ``weights_only=True`` (PyTorch's
safe unpickling mode), which restricts deserialization to tensors, primitives,
and standard containers. A P01 training bundle (RMSProp optimizer state dicts
+ plain-dict stats/flags/manifest) is loadable under ``weights_only=True``.

Some legacy or externally-produced checkpoints embed arbitrary Python objects
(e.g. a pickled Namespace, custom stats objects) that ``weights_only=True``
cannot reconstruct. For those, the caller must explicitly opt in via
``allow_unsafe_pickle=True``, which switches the load to ``weights_only=False``
and logs a warning. The default is always safe (``weights_only=True``), so an
untrusted checkpoint cannot execute arbitrary code by default.

Load behavior:
  - manifest present + compatible -> return (bundle, manifest)
  - manifest present + incompatible -> raise CheckpointCompatibilityError
    (never silently partial-load)
  - manifest absent (legacy checkpoint) -> delegate to compat, manifest=None
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from douzero.checkpoint.compat import _resolve_map_location, load_legacy_model_tar
from douzero.checkpoint.manifest import (
    CHECKPOINT_KINDS,
    CURRENT_SCHEMA_VERSION,
    MODEL_ACCESS_CLASSES,
    MODEL_ACCESS_PRIVILEGED,
    MODEL_ACCESS_PUBLIC,
    CheckpointManifest,
    build_manifest,
)

_log = logging.getLogger(__name__)


class CheckpointCompatibilityError(Exception):
    """Raised when a checkpoint's manifest is incompatible with the runtime.

    The message includes the offending field and expected/actual values so the
    failure is actionable. We never fall back to a permissive partial load.
    """


def _atomic_torch_save(bundle: dict[str, Any], path: str) -> None:
    """Durably write beside the destination, then atomically replace it."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(bundle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None

        # Persist the directory entry where the platform supports directory
        # fsync. The checkpoint is already atomically visible if this fails.
        directory_fd = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(
    path: str,
    learner_models: dict,
    optimizers: dict,
    stats: dict,
    flags: argparse.Namespace | dict[str, Any] | None,
    frames: int,
    position_frames: dict[str, int],
    runtime_state: dict[str, Any] | None = None,
) -> CheckpointManifest:
    """Save a model.tar with the legacy six keys PLUS a manifest.

    Returns the manifest that was stamped in. The manifest and all bundle
    values are tensors + plain dicts of primitives, so the saved file is
    loadable under ``weights_only=True`` (the safe default).
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
    if runtime_state is not None:
        bundle["runtime_state"] = runtime_state
    _atomic_torch_save(bundle, path)
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
    allow_unsafe_pickle: bool = False,
) -> tuple[dict, CheckpointManifest | None]:
    """Load a model.tar, validating its manifest when present.

    Returns ``(bundle, manifest)`` where ``manifest`` is None for legacy
    checkpoints that predate manifests. Raises CheckpointCompatibilityError on
    a version mismatch (never silently partial-loads).

    Security: the default is ``weights_only=True`` (safe unpickling). If the
    checkpoint embeds objects that safe mode cannot reconstruct (e.g. a pickled
    Namespace from an old or external checkpoint), the caller must explicitly
    pass ``allow_unsafe_pickle=True`` to switch to ``weights_only=False``. This
    keeps untrusted checkpoints safe by default.
    """
    weights_only = not allow_unsafe_pickle
    if not weights_only:
        _log.warning(
            "Loading checkpoint %s with weights_only=False (allow_unsafe_pickle=True). "
            "This permits arbitrary code execution via pickle; only use this for "
            "trusted, locally-produced checkpoints.", path,
        )
    bundle = torch.load(
        path,
        map_location=_resolve_map_location(training_device),
        weights_only=weights_only,
    )
    if not isinstance(bundle, dict) or "manifest" not in bundle:
        # Legacy checkpoint (no manifest) -- delegate to the compat path. This
        # returns the same bundle with manifest=None. The caller's
        # allow_unsafe_pickle choice is forwarded so the re-read is consistent.
        return load_legacy_model_tar(
            path,
            training_device=training_device,
            allow_unsafe_pickle=allow_unsafe_pickle,
            _already_loaded_bundle=bundle,
        )

    manifest = CheckpointManifest.from_dict(bundle["manifest"])

    # Compute the expected rule identity from the canonical RuleSet.
    from douzero.env.rules import RuleSet
    if expected_ruleset_id == "standard":
        expected_rs = RuleSet.standard()
    else:
        expected_rs = RuleSet.legacy()

    _validate_manifest(manifest, expected_schema_version, expected_model_version,
                       expected_feature_version, expected_ruleset_id,
                       expected_checkpoint_kind, path,
                       expected_ruleset_version=expected_rs.ruleset_version,
                       expected_ruleset_hash=expected_rs.stable_hash())
    return bundle, manifest


def _validate_manifest(
    manifest: CheckpointManifest,
    expected_schema_version: int,
    expected_model_version: str,
    expected_feature_version: str,
    expected_ruleset_id: str,
    expected_checkpoint_kind: str,
    path: str,
    expected_ruleset_version: str | None = None,
    expected_ruleset_hash: str | None = None,
    expected_model_access: str | None = None,
) -> None:
    """Validate all version/identity fields; raise on any mismatch.

    When ``expected_ruleset_version`` and ``expected_ruleset_hash`` are
    provided, they are checked against the manifest. P01 checkpoints that lack
    these fields are backfilled with the legacy identity in
    ``CheckpointManifest.from_dict``, so a legacy checkpoint with the legacy
    runtime still passes.
    """
    ctx = f"Checkpoint at {path} (git_sha={manifest.git_sha}, frames={manifest.frames})"

    if manifest.model_access not in MODEL_ACCESS_CLASSES:
        raise CheckpointCompatibilityError(
            f"Checkpoint model_access has unknown value {manifest.model_access!r}; "
            f"expected one of {sorted(MODEL_ACCESS_CLASSES)}. {ctx}"
        )
    if expected_model_access is None:
        expected_model_access = (
            MODEL_ACCESS_PRIVILEGED
            if expected_checkpoint_kind == "privileged_teacher"
            else MODEL_ACCESS_PUBLIC
        )
    if manifest.model_access != expected_model_access:
        raise CheckpointCompatibilityError(
            f"Checkpoint model_access mismatch: checkpoint has "
            f"{manifest.model_access!r}, runtime expects {expected_model_access!r}. "
            f"Privileged models are training-only and cannot be loaded as public "
            f"policies. {ctx}"
        )

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
    if expected_ruleset_version is not None and manifest.ruleset_version != expected_ruleset_version:
        raise CheckpointCompatibilityError(
            f"Checkpoint ruleset_version mismatch: checkpoint has "
            f"{manifest.ruleset_version!r}, runtime expects "
            f"{expected_ruleset_version!r}. {ctx}"
        )
    if expected_ruleset_hash is not None and manifest.ruleset_hash != expected_ruleset_hash:
        raise CheckpointCompatibilityError(
            f"Checkpoint ruleset_hash mismatch: checkpoint has "
            f"{manifest.ruleset_hash!r}, runtime expects "
            f"{expected_ruleset_hash!r}. Same ruleset_id but different rule "
            f"parameters. {ctx}"
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
