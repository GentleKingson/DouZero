"""Legacy checkpoint compatibility readers (P01).

These read checkpoint formats that predate the P01 manifest:

  - Legacy ``model.tar`` (the six legacy keys, no ``manifest`` key) -- returned
    as (bundle, manifest=None).
  - Legacy per-position ``{pos}_weights_{frames}.ckpt`` sidecars -- bare
    state_dicts consumed by ``DeepAgent``.

Security:
  - Position-weights sidecars are pure state_dicts, so they load with
    ``weights_only=True`` (safe, the default).
  - Legacy ``model.tar`` bundles default to ``weights_only=True``. A bundle
    that embeds arbitrary pickled Python objects (e.g. a pickled Namespace)
    that safe mode cannot reconstruct requires the caller to explicitly pass
    ``allow_unsafe_pickle=True``. This keeps untrusted checkpoints safe by
    default.

Position-checkpoint strictness: the DEFAULT position-ckpt loader now requires
exact key-set and shape match (no permissive partial load). The legacy
permissive filter used by the P00-pinned DeepAgent path is preserved behind an
explicit ``strict=False`` opt-in, so existing behavior is unchanged while new
callers get strictness by default.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

_log = logging.getLogger(__name__)


def _resolve_map_location(training_device: str | None):
    """Map a training_device flag to a torch map_location string.

    Defined here (not imported from io.py) to avoid a circular import: io.py
    imports this module's load_legacy_model_tar. io.py re-exports this helper.
    """
    if training_device is None or training_device == "cpu":
        return "cpu"
    return "cuda:" + str(training_device)


def load_legacy_model_tar(
    path: str,
    *,
    training_device: str | None = None,
    allow_unsafe_pickle: bool = False,
    _already_loaded_bundle: dict | None = None,
) -> tuple[dict, None]:
    """Read a pre-P01 model.tar (no manifest). Returns (bundle, None).

    The bundle has the six legacy keys. No compatibility validation is
    possible because there is no manifest -- callers assume legacy feature/
    rule identity.

    Security: defaults to ``weights_only=True``. A legacy bundle that embeds
    objects safe mode cannot reconstruct requires
    ``allow_unsafe_pickle=True`` (which logs a warning and uses
    ``weights_only=False``). If ``io.load_checkpoint`` already loaded the
    bundle while probing for a manifest, it passes that bundle via
    ``_already_loaded_bundle`` so we do not read the file twice.
    """
    if _already_loaded_bundle is not None:
        bundle = _already_loaded_bundle
    else:
        weights_only = not allow_unsafe_pickle
        if not weights_only:
            _log.warning(
                "Loading legacy checkpoint %s with weights_only=False "
                "(allow_unsafe_pickle=True). Only use this for trusted, "
                "locally-produced checkpoints.", path,
            )
        bundle = torch.load(
            path,
            map_location=_resolve_map_location(training_device),
            weights_only=weights_only,
        )
    if not isinstance(bundle, dict):
        raise TypeError(f"Legacy checkpoint at {path} is not a dict: {type(bundle)}")
    return bundle, None


def load_legacy_position_ckpt(path: str) -> dict:
    """Read a bare per-position state_dict sidecar.

    Uses ``weights_only=True`` (the sidecar is a pure state_dict). Returns the
    raw state_dict; the caller decides strict vs permissive loading.
    """
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    pretrained = torch.load(path, map_location=map_location, weights_only=True)
    return pretrained


def load_position_state_dict_strict(
    path: str, model_state_dict: dict
) -> dict:
    """Load a position checkpoint and STRICTLY validate keys + shapes.

    Raises ``CheckpointCompatibilityError`` if the checkpoint's key set does not
    exactly match ``model_state_dict`` or if any tensor shape differs. This is
    the safe default for new code; the legacy permissive path (filter then
    fill with random init) is deliberately NOT the default because it silently
    hides mismatches.
    """
    from douzero.checkpoint.io import CheckpointCompatibilityError

    pretrained = load_legacy_position_ckpt(path)
    ckpt_keys = set(pretrained.keys())
    model_keys = set(model_state_dict.keys())
    if ckpt_keys != model_keys:
        missing = model_keys - ckpt_keys
        extra = ckpt_keys - model_keys
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"extra keys: {sorted(extra)}")
        raise CheckpointCompatibilityError(
            f"Position checkpoint at {path} has a key-set mismatch ({'; '.join(parts)}). "
            "Refusing to partially load with random initialization."
        )
    for key in model_keys:
        ckpt_shape = tuple(pretrained[key].shape)
        model_shape = tuple(model_state_dict[key].shape)
        if ckpt_shape != model_shape:
            raise CheckpointCompatibilityError(
                f"Position checkpoint at {path} has a shape mismatch for key "
                f"{key!r}: checkpoint {ckpt_shape} vs model {model_shape}."
            )
    return pretrained
