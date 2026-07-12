"""Save / load Model V2 weights with a versioned manifest (P05).

The legacy checkpoint path (``save_checkpoint`` / ``load_checkpoint`` in
:mod:`douzero.checkpoint.io`) is built around the three-role legacy /
factorized model family and the ``model.tar`` bundle layout. Model V2 has a
single shared model (not three role-specific submodules), so it needs its own
sidecar format.

This module provides:

- :func:`save_v2_checkpoint`: write a V2 ``state_dict`` + manifest to a
  ``model_v2.tar`` bundle. The manifest is stamped with
  ``model_version="v2"`` and the feature schema hash, so a future loader can
  reject a schema/config drift.
- :func:`load_v2_checkpoint`: read a V2 bundle, validate its manifest against
  the expected schema hash + model version, and return the ``state_dict`` +
  manifest. It raises :class:`CheckpointCompatibilityError` on any mismatch,
  including an attempt to load a legacy / factorized ``model.tar`` here.

A separate per-position sidecar (``save_v2_position_weights`` /
:func:`load_v2_position_weights`) is also provided so the existing
``DeepAgentV2`` deployment path (which takes a ``.ckpt`` path, mirroring the
legacy ``DeepAgent``) can load a V2 state_dict without a full bundle. The
sidecar carries a small manifest so a strict loader can still reject
mismatches.

Security: all loads default to ``weights_only=True``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from douzero.checkpoint.compat import _resolve_map_location
from douzero.checkpoint.io import CheckpointCompatibilityError, _validate_manifest
from douzero.checkpoint.manifest import (
    CURRENT_SCHEMA_VERSION,
    CheckpointManifest,
    build_manifest,
)

#: The bundle key holding the V2 state_dict.
_V2_STATE_DICT_KEY = "model_state_dict"
#: The bundle key holding the manifest dict.
_MANIFEST_KEY = "manifest"
#: The bundle key holding the (optional) ModelV2Config dict, for auditability.
_CONFIG_KEY = "model_v2_config"
#: The bundle key holding the feature schema hash the model was built against.
_SCHEMA_HASH_KEY = "feature_schema_hash"


def _coerce_flags(flags: Any) -> argparse.Namespace | dict[str, Any] | None:
    """Accept a Namespace, dict, or None for manifest building."""
    if flags is None or isinstance(flags, (dict, argparse.Namespace)):
        return flags
    raise TypeError(
        f"flags must be a Namespace, dict, or None, got {type(flags).__name__}"
    )


def save_v2_checkpoint(
    path: str,
    model: "torch.nn.Module",
    *,
    schema_hash: str,
    config_dict: dict[str, Any] | None = None,
    flags: argparse.Namespace | dict[str, Any] | None = None,
    frames: int = 0,
    position_frames: dict[str, int] | None = None,
) -> CheckpointManifest:
    """Save a Model V2 ``state_dict`` + manifest to a ``model_v2.tar`` bundle.

    Parameters
    ----------
    path:
        Output path (typically ``<savedir>/<xpid>/model_v2.tar``).
    model:
        The :class:`~douzero.models_v2.model.ModelV2` to save.
    schema_hash:
        The :attr:`FeatureSchemaManifest.stable_hash` the model was constructed
        against (i.e. ``model.schema.stable_hash()``). Stamped into the bundle
        so a loader can reject a schema drift.
    config_dict:
        Optional serializable dict of the :class:`ModelV2Config`. Auditability
        only; the loader does not reconstruct the config from it (the caller
        passes the config explicitly so construction is explicit).
    flags:
        Optional runtime flags (Namespace or dict). The manifest's
        ``model_version`` is read off this if present; otherwise it defaults to
        ``"v2"`` (see :func:`build_v2_manifest`).
    frames, position_frames:
        Training progress counters (0 for a fresh / untrained model).
    """
    manifest = build_v2_manifest(
        flags=flags,
        schema_hash=schema_hash,
        frames=frames,
        position_frames=position_frames or {},
    )
    # Force the model_version to v2 regardless of what flags carried: this is a
    # V2 bundle and the manifest must say so.
    object.__setattr__(manifest, "model_version", "v2")

    bundle = {
        _V2_STATE_DICT_KEY: model.state_dict(),
        _MANIFEST_KEY: manifest.to_dict(),
        _CONFIG_KEY: dict(config_dict) if config_dict else {},
        _SCHEMA_HASH_KEY: str(schema_hash),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return manifest


def load_v2_checkpoint(
    path: str,
    *,
    expected_schema_hash: str | None = None,
    expected_model_version: str = "v2",
    training_device: str | None = None,
    allow_unsafe_pickle: bool = False,
) -> tuple[dict, CheckpointManifest]:
    """Load a V2 bundle, validating its manifest.

    Parameters
    ----------
    path:
        Path to a ``model_v2.tar`` written by :func:`save_v2_checkpoint`.
    expected_schema_hash:
        If provided, the bundle's schema hash must match exactly. This is how a
        caller binds a checkpoint to the exact feature schema it was trained
        against. Pass ``None`` to skip the check (NOT recommended for
        production; the strict manifest loader in P16 always checks).
    expected_model_version:
        Defaults to ``"v2"``. Loading a legacy / factorized bundle here raises.
    training_device:
        Device to map tensors to (``"cpu"`` / ``"cuda:N"`` / None).
    allow_unsafe_pickle:
        See :func:`douzero.checkpoint.io.load_checkpoint`.

    Returns
    -------
    tuple
        ``(state_dict, manifest)``. The state_dict is a plain dict of tensors
        ready for ``model.load_state_dict`` (strict).
    """
    weights_only = not allow_unsafe_pickle
    bundle = torch.load(
        path,
        map_location=_resolve_map_location(training_device),
        weights_only=weights_only,
    )
    if not isinstance(bundle, dict) or _MANIFEST_KEY not in bundle:
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} is not a Model V2 bundle (no manifest "
            f"key). It may be a legacy/factorized model.tar; load it with the "
            f"legacy loader instead. A V2 weights sidecar (.ckpt) should be "
            f"loaded with load_v2_position_weights."
        )
    manifest = CheckpointManifest.from_dict(bundle[_MANIFEST_KEY])
    _validate_manifest(
        manifest,
        expected_schema_version=CURRENT_SCHEMA_VERSION,
        expected_model_version=expected_model_version,
        expected_feature_version=manifest.feature_version,
        expected_ruleset_id=manifest.ruleset_id,
        expected_checkpoint_kind=manifest.checkpoint_kind,
        path=path,
        expected_ruleset_version=manifest.ruleset_version,
        expected_ruleset_hash=manifest.ruleset_hash,
    )
    if manifest.model_version != "v2":
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} has model_version="
            f"{manifest.model_version!r}, expected 'v2'. Load legacy/"
            f"factorized checkpoints with the legacy loader."
        )
    if expected_schema_hash is not None:
        actual_hash = bundle.get(_SCHEMA_HASH_KEY)
        if actual_hash != expected_schema_hash:
            raise CheckpointCompatibilityError(
                f"V2 checkpoint at {path!r} feature_schema_hash mismatch: "
                f"checkpoint has {actual_hash!r}, runtime expects "
                f"{expected_schema_hash!r}. The model was trained against a "
                f"different observation schema."
            )
    state_dict = bundle[_V2_STATE_DICT_KEY]
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} state_dict is not a dict (got "
            f"{type(state_dict).__name__}). The bundle is malformed."
        )
    return state_dict, manifest


def save_v2_position_weights(
    path: str,
    model: "torch.nn.Module",
    *,
    schema_hash: str,
    model_version: str = "v2",
) -> None:
    """Save a V2 ``state_dict`` sidecar (``.ckpt``) for DeepAgentV2 deployment.

    This mirrors the legacy per-position ``.ckpt`` sidecar that
    :class:`~douzero.evaluation.deep_agent.DeepAgent` loads, but for the single
    shared V2 model. The sidecar is a bare ``state_dict`` (no manifest) for
    P05; the strict manifest-bearing sidecar arrives in P16. The
    :func:`~douzero.evaluation.deep_agent.load_v2_model` helper performs a
    strict key/shape match on load so a legacy ``.ckpt`` is rejected.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def build_v2_manifest(
    flags: argparse.Namespace | dict[str, Any] | None,
    *,
    schema_hash: str,
    frames: int = 0,
    position_frames: dict[str, int] | None = None,
) -> CheckpointManifest:
    """Build a V2 checkpoint manifest.

    Wraps :func:`build_manifest` and forces ``model_version="v2"`` +
    ``feature_version="v2"`` (a V2 model always consumes the V2 observation
    schema). The schema hash is recorded in the bundle (not the manifest) so it
    does not perturb the manifest's compatibility-dict hash.
    """
    flags = _coerce_flags(flags)
    manifest = build_manifest(
        flags,
        frames=frames,
        position_frames=position_frames or {},
        checkpoint_kind="training_checkpoint",
    )
    # Force the V2 identity. A caller that passed model_version="legacy" on the
    # flags would otherwise stamp the wrong version; a V2 bundle is always v2.
    object.__setattr__(manifest, "model_version", "v2")
    object.__setattr__(manifest, "feature_version", "v2")
    return manifest
