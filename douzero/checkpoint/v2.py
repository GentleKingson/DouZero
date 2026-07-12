"""Save / load Model V2 weights with a versioned manifest (P05).

The legacy checkpoint path (``save_checkpoint`` / ``load_checkpoint`` in
:mod:`douzero.checkpoint.io`) is built around the three-role legacy /
factorized model family and the ``model.tar`` bundle layout. Model V2 has a
single shared model (not three role-specific submodules), so it needs its own
sidecar format.

Identity contract (bug #3 fix — the critical correctness property)
------------------------------------------------------------------
A V2 checkpoint is bound to FIVE identity axes, and every load validates ALL
of them against RUNTIME-SUPPLIED expectations (never against the checkpoint's
own self-reported values — a forged or corrupted checkpoint could otherwise
self-attest compatibility):

1. ``model_version == "v2"`` — rejects a legacy / factorized bundle.
2. ``feature_schema_hash`` — the observation schema the model was trained
   against. Must equal the runtime schema's ``stable_hash()``. This catches a
   same-shape-different-schema drift (e.g. a field reordered) that a pure
   shape check would miss.
3. ``model_config_hash`` — the :meth:`ModelV2Config.stable_hash` the model was
   constructed with. Must equal the runtime config's hash. This catches a
   same-shape-different-semantics config drift (e.g. ``history_heads`` 8→4
   keeps projection shapes but changes the Transformer split; ``score_clamp``
   / ``nan_guard`` change runtime behavior) that strict state_dict loading
   cannot detect.
4. ``ruleset_id`` / ``ruleset_version`` / ``ruleset_hash`` — the rule engine
   the model expects. A V2 model trained under the legacy ruleset must not be
   silently served under the standard ruleset (the bidding/scoring context
   fields differ).
5. ``checkpoint_kind`` — ``training_checkpoint`` vs ``public_policy``. A
   training bundle is not directly deployable as a public-policy sidecar and
   vice versa.

Save-side closure: :func:`save_v2_checkpoint` does NOT accept a self-reported
identity. It derives the schema/config hash from the model itself, requires
the full :class:`~douzero.env.rules.RuleSet`, and rejects any caller-supplied
``schema_hash`` / ``model_config`` that disagrees with the model's own. So a
bundle cannot be labelled with an identity that does not match the actual
weights, and the default call (no overrides) always produces a loadable file.

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
#: The bundle key holding the ModelV2Config compatibility hash (blocker #2).
#: A loader binds the checkpoint to the exact architecture config, so a
#: same-shape-different-semantics config drift (e.g. history_heads 8→4) is
#: rejected, not silently loaded.
_MODEL_CONFIG_HASH_KEY = "model_config_hash"

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
    model: "ModelV2",
    *,
    ruleset: "RuleSet",
    schema_hash: str | None = None,
    model_config: "ModelV2Config | None" = None,
    config_dict: dict[str, Any] | None = None,
    flags: argparse.Namespace | dict[str, Any] | None = None,
    frames: int = 0,
    position_frames: dict[str, int] | None = None,
) -> CheckpointManifest:
    """Save a Model V2 ``state_dict`` + manifest to a ``model_v2.tar`` bundle.

    The bundle's identity is DERIVED FROM THE MODEL, never self-reported by the
    caller. This closes the save-side loopholes (blockers #1 and #2):

    - The feature schema hash and model-config hash default to the model's own
      ``model.schema.stable_hash()`` / ``model.config.stable_hash()``. A caller
      may pass ``schema_hash`` / ``model_config`` for belt-and-braces, but if
      either disagrees with the model's own identity the save is REJECTED — a
      bundle can never be labelled with an identity that does not match the
      actual weights (e.g. stamping config-A weights with config-B's hash when
      the two happen to share parameter shapes).
    - The default call (no overrides) therefore always produces a LOADABLE
      file. Previously ``model_config=None`` wrote an empty config hash that
      the loader always rejects, so the default save produced an un-loadable
      bundle.
    - The full :class:`~douzero.env.rules.RuleSet` is REQUIRED and its complete
      identity (id + version + hash) is stamped onto the manifest. This mirrors
      :func:`save_v2_position_weights` and supports custom rule families: the
      flags-only path could only express the canonical ``legacy`` / ``standard``
      strings, never a same-id-different-parameters ruleset.

    Parameters
    ----------
    path:
        Output path (typically ``<savedir>/<xpid>/model_v2.tar``).
    model:
        The :class:`~douzero.models_v2.model.ModelV2` to save. Must carry
        ``.schema`` and ``.config`` (always present on a ``ModelV2``).
    ruleset:
        REQUIRED :class:`~douzero.env.rules.RuleSet` the model was trained
        under. Its full identity (id + version + hash) is stamped into the
        manifest. Passing ``None`` is a ``TypeError``; an unknown id is
        rejected on construction by :class:`RuleSet` itself.
    schema_hash:
        Optional override for the feature schema hash. Defaults to
        ``model.schema.stable_hash()``. If provided, it MUST equal the model's
        schema hash or the save raises ``ValueError``.
    model_config:
        Optional override :class:`~douzero.models_v2.config.ModelV2Config`.
        Defaults to ``model.config``. If provided, its :meth:`stable_hash` MUST
        equal the model's config hash or the save raises ``ValueError``.
    config_dict:
        Optional serializable dict of the :class:`ModelV2Config`. Auditability
        only; the loader does not reconstruct the config from it.
    flags:
        Optional runtime flags (Namespace or dict). The manifest's
        ``model_version`` is forced to ``"v2"`` regardless. Note the ruleset
        identity is stamped from ``ruleset``, NOT derived from ``flags``.
    frames, position_frames:
        Training progress counters (0 for a fresh / untrained model).
    """
    from douzero.env.rules import RuleSet as _RuleSet

    if not isinstance(ruleset, _RuleSet):
        raise TypeError(
            f"ruleset must be a RuleSet instance, got {type(ruleset).__name__}. "
            f"Pass the full RuleSet so the complete ruleset_hash is stamped "
            f"into the manifest, not just an id string."
        )
    # Derive the model identity from the model itself. A ModelV2 always carries
    # .schema and .config; a non-ModelV2 (or a bare nn.Module) is a caller bug.
    try:
        actual_schema_hash = model.schema.stable_hash()
        actual_config_hash = model.config.stable_hash()
    except AttributeError as exc:
        raise TypeError(
            f"save_v2_checkpoint requires a ModelV2 with .schema and .config, "
            f"got {type(model).__name__}."
        ) from exc

    # If the caller supplied an override, it MUST match the model's own
    # identity — otherwise the bundle would be mislabelled (the attack surface
    # blocker #2 calls out: stamping one model's weights with another's hash).
    if schema_hash is None:
        schema_hash = actual_schema_hash
    elif schema_hash != actual_schema_hash:
        raise ValueError(
            f"save_v2_checkpoint schema_hash mismatch: passed {schema_hash!r} "
            f"but the model's schema hash is {actual_schema_hash!r}. A bundle's "
            f"identity must match the actual model; pass schema_hash=None (the "
            f"default) to derive it from the model."
        )
    if model_config is None:
        model_config_hash = actual_config_hash
    else:
        model_config_hash = model_config.stable_hash()
        if model_config_hash != actual_config_hash:
            raise ValueError(
                f"save_v2_checkpoint model_config mismatch: the passed "
                f"ModelV2Config hash {model_config_hash!r} differs from the "
                f"model's config hash {actual_config_hash!r}. A bundle's "
                f"identity must match the actual model; same-shape weights are "
                f"not enough to relabel a model under a different config."
            )

    manifest = build_v2_manifest(
        flags=flags,
        schema_hash=schema_hash,
        frames=frames,
        position_frames=position_frames,
        checkpoint_kind="training_checkpoint",
    )
    # Stamp the FULL ruleset identity (id + version + hash) from the caller's
    # RuleSet, mirroring save_v2_position_weights. The flags path could only
    # express canonical legacy/standard; this supports custom rule families
    # (same id, different parameters -> different hash) and keeps the full
    # bundle and the deployment sidecar on ONE rule-identity API.
    object.__setattr__(manifest, "ruleset_id", ruleset.ruleset_id)
    object.__setattr__(manifest, "ruleset_version", ruleset.ruleset_version)
    object.__setattr__(manifest, "ruleset_hash", ruleset.stable_hash())

    bundle = {
        _V2_STATE_DICT_KEY: model.state_dict(),
        _MANIFEST_KEY: manifest.to_dict(),
        _CONFIG_KEY: dict(config_dict) if config_dict else {},
        _SCHEMA_HASH_KEY: str(schema_hash),
        _MODEL_CONFIG_HASH_KEY: str(model_config_hash),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return manifest


def load_v2_checkpoint(
    path: str,
    *,
    expected_schema_hash: str,
    expected_model_config_hash: str,
    expected_ruleset: "RuleSet",
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
    expected_model_config_hash:
        The :meth:`ModelV2Config.stable_hash` the runtime expects. The bundle's
        model-config hash must match exactly. Required — this closes the
        same-shape-different-semantics loophole (e.g. ``history_heads`` 8→4)
        that strict state_dict loading cannot detect.
    expected_ruleset:
        The :class:`~douzero.env.rules.RuleSet` the runtime expects. The
        bundle's ruleset_id / ruleset_version / ruleset_hash must all match.
        Passing a full RuleSet (not just an id string) supports custom rule
        families and rejects an unknown id that would otherwise be silently
        downgraded to legacy.
    expected_checkpoint_kind:
        Defaults to ``"training_checkpoint"``. Pass ``"public_policy"`` when
        loading a deployment sidecar via this loader.
    training_device:
        Device to map tensors to (``"cpu"`` / ``"cuda:N"`` / None).
    allow_unsafe_pickle:
        Switches to ``weights_only=False`` (permits arbitrary code execution
        via pickle). Default is ``False`` (safe — ``weights_only=True``).

    Returns
    -------
    tuple
        ``(state_dict, manifest)``. The state_dict is a plain dict of tensors
        ready for ``model.load_state_dict(..., strict=True)``.

    Raises
    ------
    CheckpointCompatibilityError
        On ANY identity mismatch (model_version, schema hash, model-config
        hash, ruleset id/version/hash, kind). The expected values come from
        the RUNTIME, not the checkpoint, so a forged/corrupted checkpoint
        cannot self-attest compatibility.
    """
    if not expected_schema_hash:
        raise ValueError(
            "expected_schema_hash is required: a checkpoint must be explicitly "
            "bound to a feature schema, never loaded without the check."
        )
    if not expected_model_config_hash:
        raise ValueError(
            "expected_model_config_hash is required: a checkpoint must be "
            "explicitly bound to its ModelV2Config, never loaded without the "
            "check (a same-shape-different-config drift is otherwise silent)."
        )
    from douzero.env.rules import RuleSet as _RuleSet
    if not isinstance(expected_ruleset, _RuleSet):
        raise TypeError(
            f"expected_ruleset must be a RuleSet instance, got "
            f"{type(expected_ruleset).__name__}. Pass the full RuleSet so the "
            f"complete ruleset_hash is validated, not just an id string."
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

    # Validate EVERY identity field. The expected values come from the runtime
    # (this function's arguments — the caller's RuleSet and schema/config
    # hashes), NOT from the manifest's self-reported values. A forged manifest
    # cannot self-attest a different identity.
    _validate_manifest(
        manifest,
        expected_schema_version=CURRENT_SCHEMA_VERSION,
        expected_model_version="v2",
        expected_feature_version="v2",
        expected_ruleset_id=expected_ruleset.ruleset_id,
        expected_checkpoint_kind=expected_checkpoint_kind,
        path=path,
        expected_ruleset_version=expected_ruleset.ruleset_version,
        expected_ruleset_hash=expected_ruleset.stable_hash(),
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

    # Model-config hash check (blocker #2): the bundle's config hash must equal
    # the runtime's. This catches a same-shape-different-semantics config drift
    # (e.g. history_heads 8->4 keeps projection shapes but changes the
    # Transformer split; score_clamp / nan_guard change runtime behavior) that
    # strict state_dict loading cannot detect.
    actual_cfg_hash = bundle.get(_MODEL_CONFIG_HASH_KEY, _NO_SCHEMA_HASH)
    if actual_cfg_hash != expected_model_config_hash:
        raise CheckpointCompatibilityError(
            f"V2 checkpoint at {path!r} model_config_hash mismatch: "
            f"checkpoint has {actual_cfg_hash!r}, runtime expects "
            f"{expected_model_config_hash!r}. The model was saved under a "
            f"different ModelV2Config (e.g. history_heads, score_clamp, "
            f"nan_guard). Same-shape weights are not enough."
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
    model_config: "ModelV2Config | None",
    ruleset: "RuleSet",
    flags: argparse.Namespace | dict[str, Any] | None = None,
) -> CheckpointManifest:
    """Save a manifest-bearing V2 deployment sidecar (``.ckpt``).

    The sidecar is NOT a bare state_dict. It is a small bundle
    (state_dict + manifest + schema hash + model-config hash) so the strict
    identity check applies at deployment too. A same-shape legacy/wrong-schema/
    wrong-config sidecar is rejected on load, not silently accepted.

    Parameters
    ----------
    path:
        Output path (``.ckpt``).
    model:
        The :class:`~douzero.models_v2.model.ModelV2` to save.
    schema_hash:
        The feature schema hash the model was constructed against. Required.
    model_config:
        The :class:`~douzero.models_v2.config.ModelV2Config` the model was
        constructed with. Its :meth:`~ModelV2Config.stable_hash` is stamped
        into the sidecar (blocker #2). Required.
    ruleset:
        The :class:`~douzero.env.rules.RuleSet` the model was trained under.
        Its full identity (id + version + hash) is stamped into the manifest so
        a deployment loader can reject a ruleset mismatch — including a custom
        rule family (blocker #3). Passing the full RuleSet (not just an id
        string) avoids the silent-downgrade-to-legacy loophole.
    flags:
        Optional runtime flags for the manifest's effective_config.
    """
    if not schema_hash:
        raise ValueError(
            "save_v2_position_weights requires a non-empty schema_hash."
        )
    if model_config is None:
        raise ValueError(
            "save_v2_position_weights requires the ModelV2Config (to stamp its "
            "compatibility hash)."
        )
    from douzero.env.rules import RuleSet as _RuleSet
    if not isinstance(ruleset, _RuleSet):
        raise TypeError(
            f"ruleset must be a RuleSet instance, got {type(ruleset).__name__}."
        )
    manifest = build_v2_manifest(
        flags=flags,
        schema_hash=schema_hash,
        checkpoint_kind="public_policy",
    )
    # Stamp the FULL ruleset identity (id + version + hash) from the caller's
    # RuleSet. This supports custom rule families: the complete hash
    # distinguishes same-id/different-parameters rulesets.
    object.__setattr__(manifest, "ruleset_id", ruleset.ruleset_id)
    object.__setattr__(manifest, "ruleset_version", ruleset.ruleset_version)
    object.__setattr__(manifest, "ruleset_hash", ruleset.stable_hash())

    bundle = {
        _V2_STATE_DICT_KEY: model.state_dict(),
        _MANIFEST_KEY: manifest.to_dict(),
        _SCHEMA_HASH_KEY: str(schema_hash),
        _MODEL_CONFIG_HASH_KEY: str(model_config.stable_hash()),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return manifest


def load_v2_position_weights(
    path: str,
    *,
    expected_schema_hash: str,
    expected_model_config_hash: str,
    expected_ruleset: "RuleSet",
    training_device: str | None = None,
    allow_unsafe_pickle: bool = False,
) -> tuple[dict, CheckpointManifest]:
    """Load a manifest-bearing V2 deployment sidecar.

    Validates the manifest's model_version, schema hash, model-config hash,
    ruleset identity (id + version + hash), and checkpoint_kind (must be
    ``public_policy``) against RUNTIME expectations. Returns
    ``(state_dict, manifest)`` for a strict ``load_state_dict``.

    Raises ``CheckpointCompatibilityError`` on any mismatch, including:
    - a bare state_dict sidecar (no manifest) — the sidecar must carry identity;
    - a legacy/factorized ``.ckpt`` (wrong model_version or no manifest);
    - a same-shape different-schema sidecar (schema hash mismatch);
    - a same-shape different-config sidecar (model-config hash mismatch);
    - a wrong ruleset, including a custom rule family with the same id but
      different parameters (ruleset hash mismatch);
    - an unknown ruleset id (the caller's RuleSet provides the expected id).
    """
    if not expected_schema_hash:
        raise ValueError(
            "expected_schema_hash is required for load_v2_position_weights."
        )
    if not expected_model_config_hash:
        raise ValueError(
            "expected_model_config_hash is required for load_v2_position_weights."
        )
    from douzero.env.rules import RuleSet as _RuleSet
    if not isinstance(expected_ruleset, _RuleSet):
        raise TypeError(
            f"expected_ruleset must be a RuleSet instance, got "
            f"{type(expected_ruleset).__name__}."
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
        actual_cfg_hash = bundle.get(_MODEL_CONFIG_HASH_KEY, _NO_SCHEMA_HASH)
        state_dict = bundle[_V2_STATE_DICT_KEY]
    elif isinstance(bundle, dict) and all(
        isinstance(v, torch.Tensor) for v in bundle.values()
    ):
        # Bare state_dict (legacy .ckpt shape). This is NOT a V2 sidecar.
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} is a bare state_dict with no manifest. A "
            f"V2 deployment sidecar MUST carry a manifest (model_version, "
            f"schema hash, model-config hash, ruleset). A bare state_dict "
            f"cannot be verified and may be a legacy/factorized checkpoint. "
            f"Refusing to load."
        )
    else:
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} is not a recognised bundle (no manifest "
            f"key and not a bare state_dict). The file is malformed."
        )

    _validate_manifest(
        manifest,
        expected_schema_version=CURRENT_SCHEMA_VERSION,
        expected_model_version="v2",
        expected_feature_version="v2",
        expected_ruleset_id=expected_ruleset.ruleset_id,
        expected_checkpoint_kind="public_policy",
        path=path,
        expected_ruleset_version=expected_ruleset.ruleset_version,
        expected_ruleset_hash=expected_ruleset.stable_hash(),
    )
    if actual_hash != expected_schema_hash:
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} feature_schema_hash mismatch: "
            f"checkpoint has {actual_hash!r}, runtime expects "
            f"{expected_schema_hash!r}."
        )
    if actual_cfg_hash != expected_model_config_hash:
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} model_config_hash mismatch: "
            f"checkpoint has {actual_cfg_hash!r}, runtime expects "
            f"{expected_model_config_hash!r}. The model was saved under a "
            f"different ModelV2Config."
        )
    if not isinstance(state_dict, dict):
        raise CheckpointCompatibilityError(
            f"V2 sidecar at {path!r} state_dict is not a dict (got "
            f"{type(state_dict).__name__})."
        )
    return state_dict, manifest
