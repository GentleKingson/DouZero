"""Strict extraction of a public policy from a committed H7 training checkpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping

import torch

from douzero.env.rules import RuleSet

from .belief_checkpoint import save_v3_h4_public_checkpoint
from .belief_policy import V3BeliefPolicy
from .checkpoint import save_v3_hybrid_public_checkpoint
from .formal_config import FormalExperimentConfig
from .pilot import create_pilot_learner
from .runtime import V3_H7_CHECKPOINT_FORMAT


_H7_KEYS = frozenset({
    "format",
    "artifact_access",
    "runtime_identity",
    "runtime_hash",
    "h6_checkpoint",
    "stats",
    "rng_state",
    "served_version_offset",
    "snapshot_step",
    "long_running_state",
})


def export_h7_public_checkpoint(
    training_checkpoint: str | Path,
    output_path: str | Path,
    *,
    formal_config: FormalExperimentConfig,
) -> None:
    """Fail closed on the training envelope, then export only the public graph."""

    try:
        bundle = torch.load(
            training_checkpoint, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError(f"unable to safely load H7 training checkpoint: {exc}") from exc
    if not isinstance(bundle, Mapping) or set(bundle) != _H7_KEYS:
        raise ValueError("H7 training checkpoint envelope mismatch")
    if bundle["format"] != V3_H7_CHECKPOINT_FORMAT:
        raise ValueError("unsupported H7 training checkpoint format")
    if bundle["artifact_access"] != "privileged_training_only":
        raise ValueError("H7 training checkpoint access class mismatch")
    if not isinstance(bundle["h6_checkpoint"], Mapping):
        raise ValueError("H7 nested H6 checkpoint is invalid")

    learner, resolved = create_pilot_learner(formal_config, allow_standard=True)
    with tempfile.TemporaryDirectory(prefix="douzero-public-export-") as temporary:
        inner = Path(temporary) / "h6.pt"
        torch.save(dict(bundle["h6_checkpoint"]), inner)
        learner.load_checkpoint(inner)

    ruleset = (
        RuleSet.standard()
        if resolved.learner.topology.ruleset == "standard"
        else RuleSet.legacy()
    )
    if resolved.learner.features.belief:
        belief_model = learner.base.base.belief_model
        if belief_model is None:
            raise RuntimeError("belief-enabled learner has no public belief model")
        save_v3_h4_public_checkpoint(
            output_path,
            V3BeliefPolicy(learner.model, belief_model, ruleset=ruleset),
        )
    else:
        save_v3_hybrid_public_checkpoint(
            output_path, learner.model, ruleset=ruleset
        )


__all__ = ["export_h7_public_checkpoint"]
