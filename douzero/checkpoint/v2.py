"""Save / load Model V2 weights with a versioned manifest (P05).

The legacy checkpoint path (``save_checkpoint`` / ``load_checkpoint`` in
:mod:`douzero.checkpoint.io`) is built around the three-role legacy /
factorized model family and the ``model.tar`` bundle layout. Model V2 has a
single shared model (not three role-specific submodules), so it needs its own
sidecar format.

Identity contract (bug #3 fix — the critical correctness property)
------------------------------------------------------------------
A V2 checkpoint is bound to FOUR identity axes, and every load validates ALL
of them against RUNTIME-SUPPLIED expectations (never against the checkpoint's
own self-reported values — a forged or corrupted checkpoint could otherwise
self-attest compatibility):

1. ``model_version == "v2"`` — rejects a legacy / factorized bundle.
2. ``feature_schema_hash`` — the observation schema the model was trained
   against. Must equal the runtime schema's ``stable_hash()``. This catches a
   same-shape-different-schema drift (e.g. a field reordered) that a pure
   shape check would miss.
3. ``ruleset_id`` / ``ruleset_version`` / ``ruleset_hash`` — the rule engine
   the model expects. A V2 model trained under the legacy ruleset must not be
   silently served under the standard ruleset (the bidding/scoring context
   fields differ).
4. ``checkpoint_kind`` — ``training_checkpoint`` vs ``public_policy``. A
   training bundle is not directly deployable as a public-policy sidecar and
   vice versa.

Security: every load defaults to ``weights_only=True`` (safe unpickling). The
position sidecar carries its own manifest so the strict identity check applies
equally to the deployment path, not just the full bundle.

Two formats
-----------
- :func:`save_v2_checkpoint` / :func:`load_v2_checkpoint` — the full
  ``model_v2.tar`` bundle (state_dict + manifest + config + schema hash). Used
  by training (P06) and full-checkpoint round-trips.
- :func:`save_v2_position_weights` / :func:`load_v2_position_weights` — the
  deployment sidecar (``.ckpt``), a small bundle with the state_dict + a
  minimal manifest. Used by :class:`DeepAgentV2`. It is manifest-bearing (not
  a bare state_dict) so the schema/model/ruleset identity is enforced at
  deployment, closing the loophole where a same-shape legacy/wrong-schema
  sidecar would load silently.
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

#: A small sentinel schema hash stamped into a manifest-less bundle so a loader
#: can distinguish "no schema hash present" from "schema hash is the empty
#: string". Never equal to a real schema hash (real hashes are SHA-256 hex).
_NO_SCHEMA_HASH = ""

#: The checkpoint kinds a V2 bundle may carry. ``public_policy`` is the
#: deployment sidecar (DeepAgentV2); ``training_checkpoint`` is the full
#: bundle. Both are in the global CHECKPOINT_KINDS set.
_V2_CHECKPOINT_KINDS = frozenset({"training_checkpoint", "public_policy"})


def _coerce_flags(flags: Any) -> argparse.Namespace | dict[str, Any] | None:
    """Accept a Namespace, dict, or None for manifest building."""
    if flags is None or isinstance(flags, (dict, argparse.Namespace)):
        return flags
    raise TypeError(
        f"flags must be a Namespace, dict, or None, got {type(flags).__name__}"
    )


def _force_v2_identity(manifest: CheckpointManifest) -> CheckpointManifest:
    """Force the V2 identity onto a manifest (in place via frozen-dataclass set).

    A V2 bundle is ALWAYS model_version="v2" + feature_version="v2", regardless
    of what the flags carried: a caller that passed model_version="legacy"
    would otherwise stamp the wrong version.
    """
    object.__setattr__(manifest, "model_version", "v2")
    object.__setattr__(manifest, "feature_version", "v2")
    return manifest


def build_v2_manifest(
    flags: argparse.Namespace | dict[str, Any] | None,
    *,
    schema_hash: str,
    frames: int = 0,
    position_frames: dict[str, int] | None = None,
    checkpoint_kind: str = "training_checkpoint",
) -> CheckpointManifest:
    """Build a V2 checkpoint manifest.

    Wraps :func:`build_manifest` and forces ``model_version="v2"`` +
    ``feature_version="v2"`` (a V2 model always consumes the V2 observation
    schema). The schema hash is recorded in the bundle (not the manifest) so it
    does not perturb the manifest's compatibility-dict hash.
    """
    if checkpoint_kind not in _V2_CHECKPOINT_KINDS:
        raise ValueError(
            f"V2 checkpoint_kind must be one of {sorted(_V2_CHECKPOINT_KINDS)}, "
            f"got {checkpoint_kind!r}"
        )
    flags = _coerce_flags(flags)
    manifest = build_manifest(
        flags,
        frames=frames,
        position_frames=position_frames or {},
        checkpoint_kind=checkpoint_kind,
    )
    return _force_v2_identity(manifest)


# --------------------------------------------------------------------------- #
# Full bundle: model_v2.tar
# --------------------------------------------------------------------------- #
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
        ``model_version`` is forced to ``"v2"`` regardless.
    frames, position_frames:
        Training progress counters (0 for a fresh / untrained model).
    """
    if not schema_hash:
        raise ValueError(
            "schema_hash is required (an empty hash cannot bind the checkpoint "
            "to a feature schema)."
        )
    manifest = build_v2_manifest(
        flags=flags,
        schema_hash=schema_hash,
        frames=frames,
        position_frames=position_frames,
        checkpoint_kind="training_checkpoint",
    )
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
    expected_schema_hash: str,
    expected_ruleset_id: str = "legacy",
    expected_checkpoint_kind: str = "training_checkpoint",
    training_device: str | None = None,
    allow_unsafe_pickle: bool = False,
) -> tuple[dict, CheckpointManifest]:
    """Load a V2 bundle, validating its manifest against RUNTIME expectations.

    Parameters
    ----------
    path:
        Path to a ``model_v2.tar`` written by :func:`save_v2_checkpoint`.
    expected_schema_hash:
        The feature schema hash the RUNTIME expects (i.e. the schema the model
        was constructed against). The bundle's schema hash must match exactly.
        Required (no default) — binding a checkpoint to a schema must be
        explicit, not skipped.
    expected_ruleset_id:
        The rule-engine identity the runtime expects (``"legacy"`` or
        ``"standard"``). Defaults to ``"legacy"``. A V2 model trained under one
        ruleset must not be served under another.
    expected_checkpoint_kind:
        Defaults to ``"training_checkpoint"``. Pass ``"public_policy"`` when
        loading a deployment sidecar via this loader.
    training_device:
        Device to map tensors to (``"cpu"`` / ``"cuda:N"`` / None).
    allow_unsafe_pickle:
        Switches to ``weights_only=False``. Default is ``True`` (safe).

    Returns
    -------
    tuple
        ``(state_dict, manifest)``. The state_dict is a plain dict of tensors
        ready for ``model.load_state_dict(..., strict=True)``.

    Raises
    ------
    CheckpointCompatibilityError
        On ANY identity mismatch (model_version, schema hash, ruleset, kind).
        The expected values come from the RUNTIME, not the checkpoint, so a
        forged/corrupted checkpoint cannot self-attest compatibility.
    """
    if not expected_schema_hash:
        raise ValueError(
            "expected_schema_hash is required: a checkpoint must be explicitly "
            "bound to a feature schema, never loaded without the check."
        )
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
            f"legacy loader instead."
        )
    manifest = CheckpointManifest.from_dict(bundle[_MANIFEST_KEY])

    # Compute the EXPECTED rule identity from the runtime-supplied ruleset id
    # (NOT from the manifest's self-reported value). This is the bug #3 fix: a
    # forged manifest cannot self-attest a different ruleset.
    from douzero.env.rules import RuleSet
    if expected_ruleset_id == "standard":
        expected_rs = RuleSet.standard()
    else:
        expected_rs = RuleSet.legacy()

    # Validate EVERY identity field. The expected values come from the runtime
    # (this function's arguments), except feature_version which is always "v2"
    # for a V2 bundle (and the manifest's own ruleset_version/hash, which must
    # match the canonical identity for the ruleset_id the manifest carries —
    # this catches a bundle that claims ruleset_id=legacy but stamps a standard
    # ruleset hash).
    _validate_manifest(
        manifest,
        expected_schema_version=CURRENT_SCHEMA_VERSION,
        expected_model_version="v2",
        expected_feature_version="v2",
        expected_ruleset_id=expected_ruleset_id,
        expected_checkpoint_kind=expected_checkpoint_kind,
        path=path,
        expected_ruleset_version=expected_rs.ruleset_version,
        expected_ruleset_hash=expected_rs.stable_hash(),
    )
    if manifest.model_version != "v2":
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} has model_version="
            f"{manifest.model_version!r}, expected 'v2'. Load legacy/"
            f"factorized checkpoints with the legacy loader."
        )

    # Schema hash check: the bundle's hash must equal the runtime's hash.
    actual_hash = bundle.get(_SCHEMA_HASH_KEY, _NO_SCHEMA_HASH)
    if actual_hash != expected_schema_hash:
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} feature_schema_hash mismatch: "
            f"checkpoint has {actual_hash!r}, runtime expects "
            f"{expected_schema_hash!r}. The model was trained against a "
            f"different observation schema (same shapes are not enough: a "
            f"field reorder/resize changes the hash)."
        )

    state_dict = bundle[_V2_STATE_DICT_KEY]
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} state_dict is not a dict (got "
            f"{type(state_dict).__name__}). The bundle is malformed."
        )
    return state_dict, manifest


# --------------------------------------------------------------------------- #
# Deployment sidecar: manifest-bearing .ckpt
# --------------------------------------------------------------------------- #
def save_v2_position_weights(
    path: str,
    model: "torch.nn.Module",
    *,
    schema_hash: str,
    ruleset_id: str = "legacy",
    flags: argparse.Namespace | dict[str, Any] | None = None,
) -> CheckpointManifest:
    """Save a manifest-bearing V2 deployment sidecar (``.ckpt``).

    Bug #3 fix: the sidecar is NOT a bare state_dict. It is a small bundle
    (state_dict + manifest + schema hash) so the strict identity check applies
    at deployment too. A same-shape legacy/wrong-schema sidecar is rejected on
    load, not silently accepted.

    Parameters
    ----------
    path:
        Output path (``.ckpt``).
    model:
        The :class:`~douzero.models_v2.model.ModelV2` to save.
    schema_hash:
        The feature schema hash the model was constructed against. Required.
    ruleset_id:
        The rule-engine identity (``"legacy"`` or ``"standard"``). Defaults to
        ``"legacy"``. Stamped into the manifest so a deployment loader can
        reject a ruleset mismatch.
    flags:
        Optional runtime flags for the manifest's effective_config.
    """
    if not schema_hash:
        raise ValueError(
            "save_v2_position_weights requires a non-empty schema_hash."
        )
    manifest = build_v2_manifest(
        flags=flags,
        schema_hash=schema_hash,
        checkpoint_kind="public_policy",
    )
    # Force the ruleset identity onto the manifest so the deployment loader
    # can reject a ruleset mismatch. build_manifest reads ruleset off flags;
    # we override here so the caller's explicit ruleset_id wins.
    from douzero.env.rules import RuleSet
    rs = RuleSet.standard() if ruleset_id == "standard" else RuleSet.legacy()
    object.__setattr__(manifest, "ruleset_id", ruleset_id)
    object.__setattr__(manifest, "ruleset_version", rs.ruleset_version)
    object.__setattr__(manifest, "ruleset_hash", rs.stable_hash())

    bundle = {
        _V2_STATE_DICT_KEY: model.state_dict(),
        _MANIFEST_KEY: manifest.to_dict(),
        _SCHEMA_HASH_KEY: str(schema_hash),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return manifest


def load_v2_position_weights(
    path: str,
    *,
    expected_schema_hash: str,
    expected_ruleset_id: str = "legacy",
    training_device: str | None = None,
    allow_unsafe_pickle: bool = False,
) -> tuple[dict, CheckpointManifest]:
    """Load a manifest-bearing V2 deployment sidecar.

    Validates the manifest's model_version, schema hash, ruleset identity, and
    checkpoint_kind (must be ``public_policy``) against RUNTIME expectations.
    Returns ``(state_dict, manifest)`` for a strict ``load_state_dict``.

    Raises ``CheckpointCompatibilityError`` on any mismatch, including:
    - a bare state_dict sidecar (no manifest) — the sidecar must carry identity;
    - a legacy/factorized ``.ckpt`` (wrong model_version or no manifest);
    - a same-shape different-schema sidecar (schema hash mismatch);
    - a wrong ruleset (ruleset_id/version/hash mismatch).
    """
    if not expected_schema_hash:
        raise ValueError(
            "expected_schema_hash is required for load_v2_position_weights."
        )
    weights_only = not allow_unsafe_pickle
    bundle = torch.load(
        path,
        map_location=_resolve_map_location(training_device),
        weights_only=weights_only,
    )

    # Reject a bare state_dict (no manifest). A bare sidecar has no identity,
    # so a same-shape wrong-schema/legacy sidecar would load silently.
    if isinstance(bundle, dict) and _MANIFEST_KEY in bundle:
        manifest = CheckpointManifest.from_dict(bundle[_MANIFEST_KEY])
        actual_hash = bundle.get(_SCHEMA_HASH_KEY, _NO_SCHEMA_HASH)
        state_dict = bundle[_V2_STATE_DICT_KEY]
    elif isinstance(bundle, dict) and all(
        isinstance(v, torch.Tensor) for v in bundle.values()
    ):
        # Bare state_dict (legacy .ckpt shape). This is NOT a V2 sidecar.
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} is a bare state_dict with no manifest. A "
            f"V2 deployment sidecar MUST carry a manifest (model_version, "
            f"schema hash, ruleset). A bare state_dict cannot be verified and "
            f"may be a legacy/factorized checkpoint. Refusing to load."
        )
    else:
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} is not a recognised bundle (no manifest "
            f"key and not a bare state_dict). The file is malformed."
        )

    from douzero.env.rules import RuleSet
    if expected_ruleset_id == "standard":
        expected_rs = RuleSet.standard()
    else:
        expected_rs = RuleSet.legacy()

    _validate_manifest(
        manifest,
        expected_schema_version=CURRENT_SCHEMA_VERSION,
        expected_model_version="v2",
        expected_feature_version="v2",
        expected_ruleset_id=expected_ruleset_id,
        expected_checkpoint_kind="public_policy",
        path=path,
        expected_ruleset_version=expected_rs.ruleset_version,
        expected_ruleset_hash=expected_rs.stable_hash(),
    )
    if actual_hash != expected_schema_hash:
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} feature_schema_hash mismatch: "
            f"checkpoint has {actual_hash!r}, runtime expects "
            f"{expected_schema_hash!r}."
        )
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} state_dict is not a dict (got "
            f"{type(state_dict).__name__})."
        )
    return state_dict, manifest
