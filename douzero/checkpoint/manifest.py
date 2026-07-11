"""Checkpoint manifest dataclass and builder (P01).

The manifest is stamped into ``model.tar`` alongside the existing tensor keys so
that a checkpoint records enough metadata to reject incompatible loads
(wrong feature version, wrong rule set, wrong model version, wrong kind) with a
precise error instead of a silent partial load.

The manifest is stored as a plain dict (via ``to_dict``), NEVER pickled as a
dataclass instance, so that ``torch.load(..., weights_only=True)`` can load it
(weights_only restricts unpickling to tensors + primitives + plain containers).
"""

from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

# Bumped only on a breaking change to the manifest schema itself (field
# renames/removals), NOT on every feature-version change. Feature-version
# compatibility is checked via `feature_version` / `ruleset_id` / `model_version`.
CURRENT_SCHEMA_VERSION = 1

# The set of supported checkpoint kinds. ``position_weights`` is the per-position
# eval sidecar (DeepAgent); ``training_checkpoint`` is the full model.tar bundle.
# ``privileged_teacher`` / ``public_policy`` are reserved for P10/P16 and are
# not produced by P01, but loaders reject unknown kinds so a future mismatch is
# caught rather than mis-handled.
CHECKPOINT_KINDS = frozenset({
    "training_checkpoint",
    "position_weights",
    "privileged_teacher",  # P10 reserved
    "public_policy",       # P16 reserved
})


@dataclass(frozen=True)
class CheckpointManifest:
    """Metadata stamped into a training checkpoint.

    All fields are JSON-serializable primitives or nested dicts. ``git_sha`` is
    always a string ("unknown" when git is unavailable); it is never None.
    ``checkpoint_kind`` distinguishes the four checkpoint roles so a model.tar
    cannot be loaded where a position-weights sidecar is expected (and vice
    versa).
    """

    schema_version: int
    model_version: str
    feature_version: str
    ruleset_id: str
    checkpoint_kind: str
    git_sha: str
    python_version: str
    torch_version: str
    effective_config: dict[str, Any]
    frames: int
    position_frames: dict[str, int]
    created_at: str  # ISO-8601 UTC

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict serialization (for torch.save with weights_only)."""
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "ruleset_id": self.ruleset_id,
            "checkpoint_kind": self.checkpoint_kind,
            "git_sha": self.git_sha,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "effective_config": self.effective_config,
            "frames": self.frames,
            "position_frames": dict(self.position_frames),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointManifest":
        """Reconstruct from a plain dict (the inverse of to_dict)."""
        return cls(**d)


def build_manifest(
    flags: argparse.Namespace | dict[str, Any] | None,
    frames: int,
    position_frames: dict[str, int],
    *,
    checkpoint_kind: str = "training_checkpoint",
) -> CheckpointManifest:
    """Build a manifest from runtime flags + environment metadata.

    ``flags`` may be an argparse Namespace (training) or a dict. The
    feature_version / ruleset_id / model_version default to "legacy" when not
    present on the flags object, preserving the legacy-baseline identity.
    """
    from douzero._version import environment_info, git_sha

    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(
            f"Unknown checkpoint_kind {checkpoint_kind!r}; expected one of "
            f"{sorted(CHECKPOINT_KINDS)}"
        )

    env = environment_info()

    # Read optional version identifiers from flags if present (P01 adds these
    # as argparse dests; legacy checkpoints had none).
    def _get(name: str, default: str) -> str:
        if flags is None:
            return default
        if isinstance(flags, dict):
            return str(flags.get(name, default))
        return str(getattr(flags, name, default))

    feature_version = _get("feature_version", "legacy")
    ruleset_id = _get("ruleset", "legacy")
    model_version = _get("model_version", "legacy")

    # Effective config: the full flag dict, for auditability. For a Namespace
    # we use vars(); for a dict we copy. This is informational only.
    if flags is None:
        effective: dict[str, Any] = {}
    elif isinstance(flags, dict):
        effective = dict(flags)
    else:
        effective = dict(vars(flags))

    return CheckpointManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        model_version=model_version,
        feature_version=feature_version,
        ruleset_id=ruleset_id,
        checkpoint_kind=checkpoint_kind,
        git_sha=git_sha(),
        python_version=env.get("python_version", "unknown"),
        # str() guarantees a native Python str (torch's TorchVersion is a str
        # subclass that breaks weights_only=True pickle). Belt-and-braces: the
        # environment_info() also coerces, but this keeps the manifest robust.
        torch_version=str(env.get("torch_version") or "unknown"),
        effective_config=effective,
        frames=int(frames),
        position_frames={k: int(v) for k, v in position_frames.items()},
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
