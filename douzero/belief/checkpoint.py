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

from douzero._version import git_sha

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


def _full_git_sha() -> str:
    value = git_sha()
    if (
        len(value) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            "belief checkpoints require a full source Git SHA; build from a "
            "Git checkout or set DOUZERO_GIT_SHA"
        )
    return value


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
        git_sha=_full_git_sha(),
        python_version=platform.python_version(),
        # str() coerces torch's TorchVersion (a str subclass) to a native
        # Python str. Storing a TorchVersion object triggers an "Unsupported
        # global" error under weights_only=True loading (torch.torch_version.
        # TorchVersion is not in the safe allowlist). Mirrors the V2 manifest.
        torch_version=str(torch.__version__),
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
        item.name: getattr(config, item.name)
        for item in dataclasses.fields(BeliefConfig)
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
    expected_ruleset: "object",
    expected_feature_version: str = "v2",
    expected_belief_config: "object | None" = None,
    map_location: Any = "cpu",
    allow_unsafe_pickle: bool = False,
    require_full_git_sha: bool = False,
) -> "object":
    """Load and validate a belief checkpoint, returning a ready model.

    Security (Blocker #1): the bundle is loaded with ``weights_only=True`` by
    default, so untrusted checkpoints cannot trigger arbitrary pickle
    deserialization. ``allow_unsafe_pickle=True`` opts back into the legacy
    unpickler and should only be set for locally-produced, trusted files when a
    weights-only load fails (e.g. an older torch without full weights-only
    support).

    Validation (performed BEFORE any model construction so a malformed or
    hostile bundle is rejected at the boundary):

    - bundle is a dict with the expected keys and value types;
    - manifest ``schema_version``, ``model_version``, ``checkpoint_kind`` and
      ``feature_version`` match the expected values;
    - the reconstructed ``BeliefConfig`` reproduces the manifest's
      ``belief_config_hash`` (the strict architecture identity axis), and
      matches ``expected_belief_config`` when supplied;
    - the ruleset identity (id/version/hash) matches ``expected_ruleset``.

    ``expected_ruleset`` is REQUIRED (callers must state the rule contract they
    intend to load under). Raises :class:`ValueError` / :class:`TypeError` on
    any mismatch rather than permissively partial-loading.

    Returns the reconstructed :class:`~douzero.belief.model.BeliefModel` in
    ``eval()`` mode.
    """
    from douzero.env.rules import RuleSet

    from .model import BeliefConfig, BeliefModel

    if expected_ruleset is None:
        raise TypeError(
            "expected_ruleset is REQUIRED: a caller must state the rule "
            "contract it intends to load under. Pass a RuleSet instance."
        )
    if not isinstance(expected_ruleset, RuleSet):
        raise TypeError(
            f"expected_ruleset must be a RuleSet, got "
            f"{type(expected_ruleset).__name__}"
        )

    weights_only = not allow_unsafe_pickle
    bundle = torch.load(
        path, map_location=map_location, weights_only=weights_only
    )
    if not isinstance(bundle, dict):
        raise ValueError(
            f"{path!r} is not a belief checkpoint bundle (top-level object is "
            f"{type(bundle).__name__}, expected dict)."
        )
    for key in (_STATE_DICT_KEY, _MANIFEST_KEY, _CONFIG_KEY):
        if key not in bundle:
            raise ValueError(
                f"{path!r} is not a belief checkpoint bundle (missing "
                f"{key!r})."
            )
    manifest = bundle[_MANIFEST_KEY]
    if not isinstance(manifest, dict):
        raise ValueError(
            f"manifest must be a dict, got {type(manifest).__name__}"
        )
    config_fields = bundle[_CONFIG_KEY]
    if not isinstance(config_fields, dict):
        raise ValueError(
            f"belief config fields must be a dict, got "
            f"{type(config_fields).__name__}"
        )
    state_dict = bundle[_STATE_DICT_KEY]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"state_dict must be a dict, got {type(state_dict).__name__}"
        )

    # Identity / schema validation.
    if manifest.get("schema_version") != BELIEF_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "belief checkpoint schema_version mismatch: checkpoint "
            f"{manifest.get('schema_version')!r} != runtime "
            f"{BELIEF_MANIFEST_SCHEMA_VERSION!r}."
        )
    if manifest.get("model_version") != BELIEF_MODEL_VERSION:
        raise ValueError(
            f"checkpoint model_version {manifest.get('model_version')!r} != "
            f"expected {BELIEF_MODEL_VERSION!r}."
        )
    if manifest.get("checkpoint_kind") != "belief_model":
        raise ValueError(
            "checkpoint_kind mismatch: checkpoint "
            f"{manifest.get('checkpoint_kind')!r} != 'belief_model'. This "
            "loader only consumes belief-model checkpoints."
        )
    if manifest.get("feature_version") != expected_feature_version:
        raise ValueError(
            "feature_version mismatch: checkpoint "
            f"{manifest.get('feature_version')!r} != expected "
            f"{expected_feature_version!r}."
        )
    checkpoint_git_sha = manifest.get("git_sha")
    if (
        not isinstance(checkpoint_git_sha, str)
        or len(checkpoint_git_sha) < 7
        or len(checkpoint_git_sha) > 64
        or any(char not in "0123456789abcdef" for char in checkpoint_git_sha)
    ):
        raise ValueError(
            "belief checkpoint git_sha must be a hexadecimal source revision"
        )
    if require_full_git_sha and len(checkpoint_git_sha) not in (40, 64):
        raise ValueError(
            "belief checkpoint git_sha must be a full source Git SHA for "
            "training, evaluation, or release use"
        )

    # Reconstruct the config from the CONSTRUCTOR fields (stored at the bundle
    # top level under _CONFIG_KEY), NOT from ``manifest['belief_config']``
    # (which is the full compatibility dict including derived constants that
    # are not valid constructor kwargs).
    try:
        config = BeliefConfig(**config_fields)
    except TypeError as exc:
        raise ValueError(
            f"belief config fields in checkpoint are invalid: {exc}"
        ) from exc
    runtime_hash = config.stable_hash()
    if runtime_hash != manifest.get("belief_config_hash"):
        raise ValueError(
            "belief config reconstructed from checkpoint does not reproduce "
            f"its manifest hash ({runtime_hash!r} vs "
            f"{manifest.get('belief_config_hash')!r}); the checkpoint is "
            "corrupted or the config schema drifted."
        )
    if expected_belief_config is not None:
        expected_hash = expected_belief_config.stable_hash()
        if expected_hash != manifest["belief_config_hash"]:
            raise ValueError(
                "belief config hash mismatch: checkpoint "
                f"{manifest['belief_config_hash']!r} != runtime "
                f"{expected_hash!r}. The checkpoint was trained under a "
                "different belief architecture."
            )
    # Ruleset identity check (REQUIRED).
    rs_ident = expected_ruleset.identity()
    for key in ("ruleset_id", "ruleset_version", "ruleset_hash"):
        if manifest.get(key) != rs_ident.get(key):
            raise ValueError(
                f"ruleset {key} mismatch: checkpoint {manifest.get(key)!r} "
                f"!= runtime {rs_ident.get(key)!r}."
            )
    model = BeliefModel(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def manifest_hash(path: str, *, allow_unsafe_pickle: bool = False) -> str:
    """Return a short hash of a checkpoint's manifest (for logging).

    Loads with ``weights_only=True`` unless ``allow_unsafe_pickle=True``.
    """
    bundle = torch.load(
        path, map_location="cpu", weights_only=not allow_unsafe_pickle
    )
    m = bundle[_MANIFEST_KEY]
    payload = json.dumps(m, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
