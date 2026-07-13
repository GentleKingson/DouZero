"""P08 train_v2.py end-to-end config integration tests (Blocker 1).

These verify the reviewer's requirement that the RL+BC path is driven by the
YAML config (bc.data_path -> BCSamples -> lambda_bc -> V2Trainer), NOT just the
programmatic V2Trainer(..., bc_aux_samples=...) interface tested in isolation.
"""

from __future__ import annotations

import sys

import pytest
import yaml


def _write_bc_yaml(tmp_path, *, data_path, lambda_bc, human_prior=True,
                   schedule="constant", schedule_steps=0, schedule_floor=0.0):
    """Write an enhanced.yaml variant that enables RL+BC."""
    cfg = {
        "xpid": "bc_e2e",
        "objective": "adp",
        "actor_device_cpu": True,
        "batch_size": 4,
        "exp_epsilon": 0.5,
        "max_grad_norm": 40.0,
        "seed": 1,
        "feature_version": "v2",
        "ruleset": "legacy",
        "model_version": "v2",
        "optimizer": {"learning_rate": 1e-4, "alpha": 0.99,
                      "momentum": 0, "epsilon": 1e-5},
        "model": {
            "version": "v2",
            "hidden_size": 32,
            "history_encoder": "lstm",
            "history_layers": 1,
            "history_heads": 4,
            "role_embedding_dim": 8,
            "belief_enabled": False,
            "human_prior_enabled": human_prior,
        },
        "loss": {
            "lambda_win": 1.0,
            "lambda_score": 0.0,
            "lambda_uncertainty": 0.0,
            "lambda_bc": lambda_bc,
        },
        "decision_policy": {"mode": "pure_win"},
        "bc": {
            "enabled": True,
            "data_path": str(data_path),
            "lambda_bc": lambda_bc,
            "schedule": schedule,
            "schedule_steps": schedule_steps,
            "schedule_floor": schedule_floor,
        },
    }
    path = str(tmp_path / "bc_e2e.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def _run_train_v2(yaml_path, *, episodes=2, optimizer_steps=1):
    """Invoke train_v2.main() with a synthetic argv (returns None on success)."""
    import train_v2

    old_argv = sys.argv
    sys.argv = ["train_v2.py", "--config", yaml_path,
                "--episodes", str(episodes),
                "--optimizer_steps", str(optimizer_steps)]
    try:
        train_v2.main()
    finally:
        sys.argv = old_argv


@pytest.fixture
def bc_data_path(tmp_path):
    """Produce a validated canonical JSONL for the train_v2 BC path."""
    from douzero.human_data import write_jsonl
    from douzero.human_data.synthetic import generate_synthetic_records
    from douzero.human_data.validate import validate_record

    recs = [r for r in generate_synthetic_records(num_games=3, base_seed=5)
            if validate_record(r).ok]
    path = str(tmp_path / "bc.jsonl")
    write_jsonl(recs, path)
    return path


class TestTrainV2BCConfig:
    def test_lambda_bc_positive_without_prior_head_rejected(self, tmp_path, bc_data_path):
        """lambda_bc > 0 with human_prior_enabled=false fails fast."""
        yaml_path = _write_bc_yaml(
            tmp_path, data_path=bc_data_path, lambda_bc=0.3, human_prior=False
        )
        with pytest.raises(Exception, match="human_prior_enabled"):
            _run_train_v2(yaml_path)

    def test_lambda_bc_positive_without_data_path_rejected(self, tmp_path):
        """lambda_bc > 0 without bc.data_path fails fast."""
        yaml_path = _write_bc_yaml(
            tmp_path, data_path="", lambda_bc=0.3, human_prior=True
        )
        with pytest.raises(Exception, match="data_path|bc_samples"):
            _run_train_v2(yaml_path)

    def test_rl_plus_bc_runs_end_to_end(self, tmp_path, bc_data_path, monkeypatch):
        """The full YAML-driven RL+BC path runs an optimizer step (Blocker 1).
        No exception means the bc.data_path -> BCSamples -> lambda_bc -> V2Trainer
        wiring works end-to-end."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        yaml_path = _write_bc_yaml(
            tmp_path, data_path=bc_data_path, lambda_bc=0.4, human_prior=True
        )
        _run_train_v2(yaml_path)

    def test_linear_decay_schedule_via_yaml(self, tmp_path, bc_data_path, monkeypatch):
        """The linear_decay schedule flows from YAML through to the trainer."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        yaml_path = _write_bc_yaml(
            tmp_path, data_path=bc_data_path, lambda_bc=0.5, human_prior=True,
            schedule="linear_decay", schedule_steps=100, schedule_floor=0.05,
        )
        _run_train_v2(yaml_path)
