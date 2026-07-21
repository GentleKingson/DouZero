import pytest
import torch

from douzero.checkpoint.io import CheckpointCompatibilityError, load_checkpoint
from douzero.gpu_v3 import (
    GPUV3Config,
    GPU_V3_CHECKPOINT_KIND,
    GPU_V3_FEATURE_VERSION,
    GPU_V3_MODEL_VERSION,
    load_gpu_v3_checkpoint,
    save_gpu_v3_checkpoint,
)


def test_gpu_v3_checkpoint_identity_and_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    config = GPUV3Config(hidden_size=64, action_hidden_size=32)
    path = tmp_path / "gpu_v3.pt"
    manifest = save_gpu_v3_checkpoint(path, model, config, steps=7)
    assert manifest.model_version == GPU_V3_MODEL_VERSION
    assert manifest.feature_version == GPU_V3_FEATURE_VERSION
    assert manifest.checkpoint_kind == GPU_V3_CHECKPOINT_KIND

    for parameter in model.parameters():
        parameter.data.zero_()
    bundle, loaded_manifest = load_gpu_v3_checkpoint(path, model, config)
    assert bundle["steps"] == 7
    assert loaded_manifest == manifest
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_gpu_v3_checkpoint_rejects_legacy_identity(tmp_path):
    model = torch.nn.Linear(4, 2)
    config = GPUV3Config(hidden_size=64, action_hidden_size=32)
    path = tmp_path / "gpu_v3.pt"
    save_gpu_v3_checkpoint(path, model, config)
    with pytest.raises(CheckpointCompatibilityError, match="model_version"):
        load_checkpoint(path, expected_model_version="legacy")


def test_gpu_v3_checkpoint_rejects_architecture_config_mismatch(tmp_path):
    model = torch.nn.Linear(4, 2)
    path = tmp_path / "gpu_v3.pt"
    save_gpu_v3_checkpoint(path, model, GPUV3Config(hidden_size=64))
    with pytest.raises(CheckpointCompatibilityError, match="config hash"):
        load_gpu_v3_checkpoint(path, model, GPUV3Config(hidden_size=128))


def test_gpu_v3_optimizer_resume_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    config = GPUV3Config(hidden_size=64)
    path = tmp_path / "gpu_v3_resume.pt"
    save_gpu_v3_checkpoint(path, model, config, optimizer=optimizer, steps=11)

    resumed_model = torch.nn.Linear(4, 2)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    bundle, _ = load_gpu_v3_checkpoint(
        path,
        resumed_model,
        config,
        optimizer=resumed_optimizer,
    )
    assert bundle["steps"] == 11
    assert resumed_optimizer.state_dict()["state"]
    for expected, actual in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)
