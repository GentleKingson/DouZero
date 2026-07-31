from pathlib import Path

import pytest

from tools.run_v3_formal_training import build_formal_training_command


ROOT = Path(__file__).resolve().parents[1]


def _command(config, tmp_path, *, resume=False, elapsed=0.0):
    return build_formal_training_command(
        ROOT / "configs/v3_formal" / config,
        tier="development",
        seed=101,
        output=tmp_path,
        resume=resume,
        elapsed_wall_seconds=elapsed,
    )


def test_v3_command_uses_frozen_budget_and_family_cli(tmp_path):
    command, record = _command("v3_full_hybrid_standard.yaml", tmp_path)
    assert command[1].endswith("train_v3_h7.py")
    assert command[command.index("--formal-budget-tier") + 1] == "development"
    assert command[command.index("--seed") + 1] == "101"
    assert "--resume" not in command
    assert record["runtime_profile"] == "v3_h71_formal_v1"
    assert record["playing_strength"] == "NOT MEASURED"


def test_v2_command_uses_ruleset_owned_config_and_strict_resume(tmp_path):
    command, record = _command(
        "model_v2_standard.yaml", tmp_path, resume=True
    )
    assert command[1].endswith("train_v2.py")
    assert command[command.index("--config") + 1].endswith(
        "configs/standard_v2.yaml"
    )
    assert command[command.index("--max_total_optimizer_steps") + 1] == "50000"
    assert command[command.index("--max_wall_time_minutes") + 1] == "240.0"
    assert command[command.index("--v2_training_mode") + 1] == (
        "async_single_gpu"
    )
    assert command[command.index("--resume_checkpoint") + 1].endswith(
        "training-latest.json"
    )
    assert record["runtime_profile"] == "model_v2_formal_v1"


def test_legacy_command_uses_production_profile_and_remaining_wall(tmp_path):
    command, record = _command(
        "legacy_a1.yaml",
        tmp_path,
        resume=True,
        elapsed=600.0,
    )
    assert command[1].endswith("train.py")
    assert command[command.index("--config") + 1].endswith(
        "configs/legacy_single_gpu_a1.yaml"
    )
    assert command[command.index("--total_frames") + 1] == "5000000"
    assert command[command.index("--unroll_length") + 1] == "50"
    assert command[command.index("--num_actors") + 1] == "12"
    assert command[command.index("--max_wall_time_minutes") + 1] == "230.0"
    assert "--load_model" in command
    assert record["runtime_profile"] == "legacy_a1_production_v1"


def test_training_command_rejects_unfrozen_seed_and_wall_override(tmp_path):
    with pytest.raises(ValueError, match="not frozen"):
        build_formal_training_command(
            ROOT / "configs/v3_formal/v3_role_legacy.yaml",
            tier="development",
            seed=404,
            output=tmp_path,
            resume=False,
        )
    with pytest.raises(ValueError, match="restored from its checkpoint"):
        _command("v3_role_legacy.yaml", tmp_path, elapsed=1.0)
    with pytest.raises(ValueError, match="within the frozen wall ceiling"):
        _command("legacy_a1.yaml", tmp_path, elapsed=14_400.0)
