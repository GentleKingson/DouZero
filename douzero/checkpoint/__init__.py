"""Versioned checkpoint manifests and I/O (P01).

This package adds a versioned ``CheckpointManifest`` to the training
checkpoint (``model.tar``) WITHOUT changing the existing tensor keys or their
meanings. Backward compatibility is required:

  - Legacy ``model.tar`` (no manifest) must still load (compat path).
  - New ``model.tar`` (with manifest) is validated against the runtime's
    schema_version / model_version / feature_version / ruleset_id /
    checkpoint_kind; mismatches raise a precise error rather than silently
    partial-loading.

Security: checkpoint loads default to ``weights_only=True`` (safe unpickling).
A checkpoint that embeds objects safe mode cannot reconstruct requires the
caller to explicitly pass ``allow_unsafe_pickle=True`` to
``load_checkpoint`` / ``load_legacy_model_tar``.

The legacy per-position ``{pos}_weights_{frames}.ckpt`` sidecars (bare
state_dicts consumed by DeepAgent) keep their permissive load behavior behind
an explicit opt-in; new code uses the strict loader.
"""

from douzero.checkpoint.compat import (
    load_legacy_model_tar,
    load_legacy_position_ckpt,
    load_position_state_dict_strict,
)
from douzero.checkpoint.io import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)
from douzero.checkpoint.manifest import (
    CHECKPOINT_KINDS,
    CURRENT_SCHEMA_VERSION,
    CheckpointManifest,
    build_manifest,
)
from douzero.checkpoint.v2 import (
    build_v2_manifest,
    load_v2_checkpoint,
    save_v2_checkpoint,
    save_v2_position_weights,
)

__all__ = [
    "CHECKPOINT_KINDS",
    "CURRENT_SCHEMA_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointManifest",
    "build_manifest",
    "load_checkpoint",
    "load_legacy_model_tar",
    "load_legacy_position_ckpt",
    "load_position_state_dict_strict",
    "save_checkpoint",
    # P05: Model V2 checkpoint helpers.
    "build_v2_manifest",
    "load_v2_checkpoint",
    "save_v2_checkpoint",
    "save_v2_position_weights",
]
