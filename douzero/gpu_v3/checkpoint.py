from __future__ import annotations

from typing import Any

from douzero.checkpoint.io import (
    CheckpointCompatibilityError,
    _atomic_torch_save,
    load_checkpoint,
)
from douzero.checkpoint.manifest import build_manifest

from .config import GPUV3Config
from .identity import (
    GPU_V3_CHECKPOINT_KIND,
    GPU_V3_FEATURE_VERSION,
    GPU_V3_MODEL_VERSION,
)


def save_gpu_v3_checkpoint(
    path,
    model,
    config: GPUV3Config,
    *,
    optimizer=None,
    steps: int = 0,
    extra: dict[str, Any] | None = None,
):
    flags = {
        "model_version": GPU_V3_MODEL_VERSION,
        "feature_version": GPU_V3_FEATURE_VERSION,
        "ruleset": "legacy",
        "gpu_v3": config.to_dict(),
    }
    manifest = build_manifest(
        flags,
        frames=0,
        position_frames={},
        checkpoint_kind=GPU_V3_CHECKPOINT_KIND,
    )
    bundle = {
        "manifest": manifest.to_dict(),
        "model_state_dict": model.state_dict(),
        "gpu_v3_config": config.to_dict(),
        "gpu_v3_config_hash": config.stable_hash(),
        "steps": int(steps),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        bundle["optimizer_state_dict"] = optimizer.state_dict()
    _atomic_torch_save(bundle, path)
    return manifest


def load_gpu_v3_checkpoint(path, model, config: GPUV3Config, *, optimizer=None, device=None):
    bundle, manifest = load_checkpoint(
        path,
        expected_model_version=GPU_V3_MODEL_VERSION,
        expected_feature_version=GPU_V3_FEATURE_VERSION,
        expected_ruleset_id="legacy",
        expected_checkpoint_kind=GPU_V3_CHECKPOINT_KIND,
        training_device=device,
    )
    if manifest is None:
        raise CheckpointCompatibilityError("gpu_v3 requires a versioned manifest")
    if bundle.get("gpu_v3_config_hash") != config.stable_hash():
        raise CheckpointCompatibilityError("gpu_v3 config hash mismatch")
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    if optimizer is not None:
        if "optimizer_state_dict" not in bundle:
            raise CheckpointCompatibilityError("gpu_v3 optimizer state is missing")
        optimizer.load_state_dict(bundle["optimizer_state_dict"])
    return bundle, manifest
