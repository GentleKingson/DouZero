"""Focused P17 release, empirical-input, GPU, and privacy tooling tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import ingest_human_games
import douzero.evaluation.p17 as p17_evaluation
from douzero.belief import BeliefConfig, BeliefModel
from douzero.belief.checkpoint import save_belief_checkpoint
from douzero.deployment import (
    CURRENT_MODEL_FORMAT_VERSION,
    ModelPackageError,
    create_model_package,
    load_model_package,
    verify_model_package,
)
from douzero.env.game import GameEnv
from douzero.env.env import Env
from douzero.checkpoint import save_v2_position_weights
from douzero.env.rules import RuleSet
from douzero.evaluation.paired import evaluate_scenario
from douzero.evaluation.p17 import (
    P17MatrixError,
    empty_matrix,
    normalize_matrix,
    result_readiness,
    write_p17_artifacts,
)
from douzero.evaluation.protocol import EVALUATION_PROTOCOL, P17_READINESS_PROTOCOL
from douzero.evaluation.scenario import BundleSpec, EvaluationScenario
from douzero.evaluation.agents import RuleAgent
from douzero.evaluation.deep_agent import DeepAgentV2
from douzero.evaluation.legacy_data_adapter import deal_standard_deck
from douzero.human_data import (
    dedupe_by_game_id,
    load_hmac_project_key,
    read_jsonl,
    rebuild_without_game_ids,
    write_jsonl,
)
from douzero.human_data.synthetic import generate_synthetic_records
from douzero.models_v2 import ModelV2, ModelV2Config
from douzero.observation import build_v2_schema, get_obs_v2
import tools.gpu_validation_probe as gpu_validation_probe
from tools.gpu_validation_probe import probe_environment
from tools.validate_amp_fallback import validate_amp_fallback
from train_v2 import _build_training_metrics
from evaluate_paired import generate_deals


ROOT = Path(__file__).resolve().parents[1]


def _model(
    *, belief_enabled: bool = False, bidding_enabled: bool = False
) -> tuple[ModelV2, RuleSet]:
    torch.manual_seed(1700)
    config = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        role_embedding_dim=8,
        mlp_layers=1,
        belief_enabled=belief_enabled,
        bidding_enabled=bidding_enabled,
        nan_guard=False,
    )
    ruleset = RuleSet.standard() if bidding_enabled else RuleSet.legacy()
    return ModelV2(build_v2_schema(max_history_len=8), config).eval(), ruleset


def _refresh_checksums(package: Path) -> None:
    names = sorted(path.name for path in package.iterdir() if path.name != "SHA256SUMS")
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((package / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )


def test_p17_package_has_identity_summaries_rollback_and_hash_only_training(tmp_path):
    model, ruleset = _model()
    package = tmp_path / "release"
    secret_path = "/outside/repo/private-human-games.jsonl"
    manifest = create_model_package(
        package,
        model,
        ruleset,
        training_config={"seed": 17, "human_data_path": secret_path},
    )
    assert CURRENT_MODEL_FORMAT_VERSION == 2
    assert manifest.format_version == 2
    assert {path.name for path in package.iterdir()} == {
        "weights.pt", "manifest.json", "ruleset.json", "feature_schema.json",
        "model_config.json", "training_config.json", "README.md",
        "model_card.md",
        "evaluation_summary.md", "gpu_validation_summary.md", "rollback.md",
        "THIRD_PARTY_NOTICES", "SHA256SUMS",
    }
    training_identity = json.loads(
        (package / "training_config.json").read_text(encoding="utf-8")
    )
    assert training_identity["payload_policy"] == "hash_only"
    assert training_identity["training_config_hash"] == manifest.training_config_hash
    assert secret_path not in json.dumps(training_identity)
    weights_payload = torch.load(package / "weights.pt", map_location="cpu", weights_only=True)
    assert secret_path not in json.dumps(weights_payload["manifest"], sort_keys=True)
    assert "NOT MEASURED" in (package / "evaluation_summary.md").read_text()
    default_card = (package / "model_card.md").read_text()
    assert "Release candidate: **NONE**" in default_card
    assert "Release status: **NOT READY**" in default_card
    assert verify_model_package(package) == manifest


def test_rollback_restores_known_good_package_and_fixed_state_inference(tmp_path):
    model, ruleset = _model()
    approved = tmp_path / "approved"
    candidate = tmp_path / "candidate"
    create_model_package(approved, model, ruleset)
    create_model_package(candidate, model, ruleset)

    env = Env("adp")
    env.reset()
    observation = get_obs_v2(
        env.infoset, ruleset=ruleset, schema=model.schema
    )

    def approved_action() -> list[int]:
        loaded = load_model_package(
            approved,
            schema=model.schema,
            ruleset=ruleset,
            config=model.config,
            device="cpu",
        )
        return DeepAgentV2(
            "landlord", loaded, ruleset, device="cpu"
        ).act_v2(observation)

    before = approved_action()
    with (candidate / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered candidate\n")
    with pytest.raises(ModelPackageError, match="checksum mismatch"):
        verify_model_package(candidate)

    # Roll back by reselecting the immutable known-good package, never by
    # modifying the failed candidate in place.
    assert approved_action() == before


def test_p17_package_rejects_config_drift_and_unexpected_dataset(tmp_path):
    model, ruleset = _model()
    package = tmp_path / "release"
    create_model_package(package, model, ruleset)

    config_path = package / "model_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["config"]["score_clamp"] = 31.0
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_checksums(package)
    with pytest.raises(ModelPackageError, match="model_config.json identity"):
        verify_model_package(package)

    package = tmp_path / "release-with-data"
    create_model_package(package, model, ruleset)
    (package / "canonical-human-data.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelPackageError, match="unexpected files"):
        verify_model_package(package)


def test_belief_package_is_self_contained_and_runs_public_inference(tmp_path):
    model, ruleset = _model(belief_enabled=True)
    missing = tmp_path / "missing"
    with pytest.raises(ModelPackageError, match="manifest-bearing belief_checkpoint"):
        create_model_package(missing, model, ruleset)
    assert not missing.exists()

    belief = BeliefModel(
        BeliefConfig(
            hidden_size=24,
            num_layers=1,
            style_enabled=True,
            style_embedding_dim=8,
        )
    ).eval()
    belief_checkpoint = tmp_path / "belief.pt"
    save_belief_checkpoint(
        str(belief_checkpoint),
        belief,
        ruleset=ruleset,
        feature_version=model.schema.feature_version,
    )
    package = tmp_path / "belief-release"
    manifest = create_model_package(
        package,
        model,
        ruleset,
        belief_checkpoint=belief_checkpoint,
    )
    assert manifest.belief_config_hash == belief.config.stable_hash()
    assert (package / "belief_config.json").is_file()
    assert (package / "belief_weights.pt").is_file()
    assert "  belief_weights.pt\n" in (package / "SHA256SUMS").read_text()
    assert verify_model_package(package) == manifest

    payload = json.loads((package / "belief_config.json").read_text())
    assert payload["config"] == asdict(belief.config)
    assert payload["compatibility"] == belief.config.compatibility_dict()

    loaded = load_model_package(
        package,
        schema=model.schema,
        ruleset=ruleset,
        config=model.config,
        device="cpu",
    )
    assert isinstance(loaded.belief_model, BeliefModel)
    assert loaded.belief_model.training is False
    env = Env("adp")
    env.reset()
    observation = get_obs_v2(env.infoset, ruleset=ruleset, schema=model.schema)
    assert len(observation.actions.legal_actions) > 1
    agent = DeepAgentV2("landlord", loaded, ruleset, device="cpu")
    action, explanation = agent.act_v2(observation, return_explanation=True)
    assert action in observation.actions.legal_actions
    assert explanation["source"] == "model"


def test_belief_package_rejects_tamper_and_wrong_checkpoint_identity(tmp_path):
    model, ruleset = _model(belief_enabled=True)
    belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1)).eval()
    checkpoint = tmp_path / "belief.pt"
    save_belief_checkpoint(
        str(checkpoint), belief, ruleset=ruleset,
        feature_version=model.schema.feature_version,
    )
    package = tmp_path / "belief-release"
    create_model_package(package, model, ruleset, belief_checkpoint=checkpoint)

    wrong_config = BeliefModel(BeliefConfig(hidden_size=32, num_layers=1)).eval()
    save_belief_checkpoint(
        str(package / "belief_weights.pt"), wrong_config, ruleset=ruleset,
        feature_version=model.schema.feature_version,
    )
    _refresh_checksums(package)
    with pytest.raises(ModelPackageError, match="belief_weights.pt checkpoint identity"):
        verify_model_package(package)

    arbitrary_config_package = tmp_path / "arbitrary-config"
    create_model_package(
        arbitrary_config_package,
        model,
        ruleset,
        belief_checkpoint=checkpoint,
    )
    config_path = arbitrary_config_package / "belief_config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["config"]["decoder"] = "arbitrary-dict-is-not-a-config"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    _refresh_checksums(arbitrary_config_package)
    with pytest.raises(ModelPackageError, match="exact BeliefConfig fields"):
        verify_model_package(arbitrary_config_package)

    wrong_ruleset = tmp_path / "wrong-ruleset.pt"
    save_belief_checkpoint(
        str(wrong_ruleset), belief, ruleset=RuleSet.standard(),
        feature_version=model.schema.feature_version,
    )
    with pytest.raises(ModelPackageError, match="belief checkpoint identity"):
        create_model_package(
            tmp_path / "wrong-ruleset-package",
            model,
            ruleset,
            belief_checkpoint=wrong_ruleset,
        )

    wrong_feature = tmp_path / "wrong-feature.pt"
    save_belief_checkpoint(
        str(wrong_feature), belief, ruleset=ruleset,
        feature_version="not-runtime-v2",
    )
    with pytest.raises(ModelPackageError, match="belief checkpoint identity"):
        create_model_package(
            tmp_path / "wrong-feature-package",
            model,
            ruleset,
            belief_checkpoint=wrong_feature,
        )


def test_belief_package_cli_embeds_manifest_checkpoint(tmp_path):
    model, ruleset = _model(belief_enabled=True)
    value_checkpoint = tmp_path / "value.pt"
    save_v2_position_weights(
        str(value_checkpoint), model, ruleset=ruleset
    )
    belief = BeliefModel(BeliefConfig(hidden_size=20, num_layers=1)).eval()
    belief_checkpoint = tmp_path / "belief.pt"
    save_belief_checkpoint(
        str(belief_checkpoint),
        belief,
        ruleset=ruleset,
        feature_version=model.schema.feature_version,
    )
    model_config = tmp_path / "model_config.json"
    model_config.write_text(json.dumps(asdict(model.config)), encoding="utf-8")
    package = tmp_path / "cli-package"

    result = subprocess.run(
        [
            sys.executable,
            "tools/package_model.py",
            "--checkpoint", str(value_checkpoint),
            "--belief-checkpoint", str(belief_checkpoint),
            "--output", str(package),
            "--ruleset", "legacy",
            "--model-config", str(model_config),
            "--max-history-len", "8",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    manifest = verify_model_package(package)
    assert manifest.belief_config_hash == belief.config.stable_hash()
    assert (package / "belief_weights.pt").is_file()


def test_learned_bidding_package_requires_identity_bound_schema(tmp_path):
    model, ruleset = _model(bidding_enabled=True)
    package = tmp_path / "bidding-release"
    manifest = create_model_package(package, model, ruleset)
    assert manifest.bidding_enabled is True
    assert manifest.bidding_head_version
    assert manifest.bidding_action_schema
    assert len(manifest.bidding_feature_schema_hash) == 64
    payload = json.loads((package / "bidding_schema.json").read_text())
    assert payload["bidding_actions"] == [0, 1, 2, 3]
    assert verify_model_package(package) == manifest
    loaded = load_model_package(
        package,
        schema=model.schema,
        ruleset=ruleset,
        config=model.config,
        device="cpu",
    )
    deck = list(range(3, 15)) * 4 + [17] * 4 + [20, 30]
    env = GameEnv(
        {role: RuleAgent() for role in ("landlord", "landlord_up", "landlord_down")},
        ruleset=ruleset,
    )
    env.card_play_init_standard(deal_standard_deck(deck), bidding_order=["0", "1", "2"])
    from douzero.observation.bidding import get_bidding_obs_v2

    observation = get_bidding_obs_v2(env.get_bidding_obs(), ruleset=ruleset)
    assert loaded.forward_bidding(observation).argmax_bid() in env.get_legal_bids()

    payload["bidding_actions"] = [0, 1, 2]
    (package / "bidding_schema.json").write_text(json.dumps(payload), encoding="utf-8")
    _refresh_checksums(package)
    with pytest.raises(ModelPackageError, match="bidding_schema.json identity"):
        verify_model_package(package)


def test_gpu_probe_is_sanitized_and_probe_script_handles_hidden_cuda(
    monkeypatch, tmp_path
):
    report = probe_environment()
    assert report["schema_version"] == "p17-gpu-environment-v1"
    assert report["privacy"] == "sanitized_no_host_or_device_identifiers"
    serialized = json.dumps(report).lower()
    for forbidden in ("hostname", "username", "gpu_uuid", "serial_number"):
        assert forbidden not in serialized

    monkeypatch.setattr(gpu_validation_probe.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        gpu_validation_probe,
        "_run",
        lambda *_args, **_kwargs: (1, "private-host /secret/path GPU-UUID"),
    )
    failed_probe = gpu_validation_probe._nvidia_environment()
    assert failed_probe["probe_error_class"] == "NonZeroExit"
    assert "private-host" not in json.dumps(failed_probe)

    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "DOUZERO_PYTHON": sys.executable})
    output = tmp_path / "gpu"
    result = subprocess.run(
        ["bash", "scripts/validate_gpu_training.sh", "--output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 3
    environment = json.loads((output / "environment.json").read_text())
    assert environment["status"] == "blocked_no_cuda_device"
    for name in (
        "single_gpu_fp32", "single_gpu_fp16", "single_gpu_bf16",
        "amp_nonfinite_fallback", "belief_frozen", "belief_joint",
        "checkpoint_resume",
    ):
        assert json.loads((output / f"{name}.json").read_text())["status"] == "not_run"
    ddp = json.loads((output / "ddp_2gpu.json").read_text())
    assert ddp["status"] == "blocked_implementation"
    assert "fails closed" in ddp["reason"]
    assert "NOT RUN" in (output / "summary.md").read_text()


def test_training_metrics_are_measured_and_amp_fallback_is_exercised():
    class Stats:
        episodes_completed = 3
        transitions_collected = 10
        bidding_transitions_collected = 2
        optimizer_steps = 4
        redeals = 1
        belief_supervised_steps = 0
        amp_fallbacks = 1

    report = _build_training_metrics(
        Stats(),
        training_wall_seconds=2.0,
        device_type="cuda",
        peak_memory_bytes=2 * 1024 * 1024,
        peak_reserved_memory_bytes=3 * 1024 * 1024,
        amp_enabled=True,
        amp_dtype="float16",
        amp_fallback_on_nonfinite=True,
        compile_enabled=False,
        ddp_enabled=False,
        world_size=1,
        parameters_changed=True,
    )
    assert report["metrics"] == {
        "peak_memory_mib": 2.0,
        "peak_reserved_memory_mib": 3.0,
        "cardplay_transitions_per_second": 5.0,
        "bidding_decisions_per_second": 1.0,
        "samples_per_second": 6.0,
        "decisions_per_second": 6.0,
        "learner_steps_per_second": 2.0,
    }
    assert report["amp"]["fallback_exercised"] is True
    fallback = validate_amp_fallback(device="cpu", dtype="bfloat16")
    assert fallback["status"] == "passed"
    assert fallback["fallback_count"] == 1
    assert fallback["parameter_finite"] is True


def test_hmac_key_env_and_ingest_data_path_env(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    monkeypatch.setenv("DOUZERO_HUMAN_DATA_HMAC_KEY_FILE", str(key))
    assert load_hmac_project_key() == b"k" * 32

    raw = tmp_path / "authorized.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DOUZERO_HUMAN_DATA_PATH", str(raw))
    args = ingest_human_games._parse_args(["--output", str(tmp_path / "out.jsonl")])
    assert args.input == str(raw)

    key.write_bytes(b"short")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        load_hmac_project_key()


def test_game_id_deletion_rebuild_removes_complete_record_without_logging_ids(tmp_path):
    records = list(generate_synthetic_records(num_games=3, base_seed=17))
    source = tmp_path / "canonical.jsonl"
    output = tmp_path / "rebuilt.jsonl"
    write_jsonl(records, str(source))
    removed = records[1].game_id

    report = rebuild_without_game_ids(source, output, [removed])
    assert report.to_dict() == {
        "input_records": 3,
        "output_records": 2,
        "excluded_records": 1,
        "requested_ids": 1,
    }
    rebuilt = list(read_jsonl(str(output)))
    assert removed not in {record.game_id for record in rebuilt}
    assert removed not in json.dumps(report.to_dict())
    assert {record.game_id for record in rebuilt} == {
        records[0].game_id, records[2].game_id
    }


def test_duplicate_ingest_warning_redacts_canonical_identifier(caplog):
    record = next(iter(generate_synthetic_records(num_games=1, base_seed=18)))
    assert list(dedupe_by_game_id([record, record])) == [record]
    assert "identifier redacted" in caplog.text
    assert record.game_id not in caplog.text


def test_p17_empty_matrix_writes_explicit_not_run_artifact_set(tmp_path):
    matrix = empty_matrix()
    normalized = normalize_matrix(matrix)
    assert all(
        normalized["models"][name]["full_game"]["status"] == "unavailable"
        for name in normalized["models"]
    )
    paths = write_p17_artifacts(tmp_path, matrix=matrix)
    assert set(paths) == {
        "model_matrix.json", "cardplay_results.json", "full_game_results.json",
        "ablations.json", "calibration.json", "latency.json", "report.md",
    }
    assert json.loads((tmp_path / "full_game_results.json").read_text())["status"] == "not_run"
    assert all(
        row["status"] == "not_run"
        for row in json.loads((tmp_path / "ablations.json").read_text())["results"].values()
    )


def test_p17_readiness_policy_is_versioned_separately_from_p15_promotion(
    monkeypatch,
):
    assert EVALUATION_PROTOCOL == "p15_paired_v1"
    assert P17_READINESS_PROTOCOL == "p17_empirical_readiness_v1"
    assert P17_READINESS_PROTOCOL != EVALUATION_PROTOCOL
    result = {
        "scenario": {"bootstrap_samples": 1999},
    }
    monkeypatch.setattr(
        p17_evaluation,
        "_recompute_result_evidence",
        lambda _result, *, mode: ({"paired_deals": 999}, []),
    )
    insufficient = result_readiness(result, mode="cardplay_only")
    assert insufficient == {
        "protocol": P17_READINESS_PROTOCOL,
        "requirements": {"bootstrap_samples": 2000, "paired_deals": 1000},
        "status": "insufficient",
        "issues": [
            "requires >= 2000 bootstrap samples",
            "requires >= 1000 paired deals",
        ],
        "evidence": {"paired_deals": 999},
    }

    result["scenario"]["bootstrap_samples"] = 2000
    monkeypatch.setattr(
        p17_evaluation,
        "_recompute_result_evidence",
        lambda _result, *, mode: ({"paired_deals": 1000}, []),
    )
    assert result_readiness(result, mode="cardplay_only")["status"] == "eligible"


def test_p17_matrix_rejects_missing_checkpoints_and_pretend_full_game(tmp_path):
    matrix = empty_matrix()
    row = matrix["models"]["v2_full_stack"]["cardplay_only"]
    row.update({
        "status": "available",
        "reason": "",
        "bundle": {
            "backend": "v2",
            "checkpoints": {
                "landlord": str(tmp_path / "landlord.pt"),
                "landlord_up": str(tmp_path / "up.pt"),
                "landlord_down": str(tmp_path / "down.pt"),
            },
        },
    })
    with pytest.raises(P17MatrixError, match="checkpoint files are missing"):
        normalize_matrix(matrix)

    for name in ("landlord.pt", "up.pt", "down.pt"):
        (tmp_path / name).write_bytes(b"checkpoint-placeholder")
    with pytest.raises(P17MatrixError, match="identity validation failed"):
        normalize_matrix(matrix)

    matrix["models"]["v2_full_stack"]["full_game"] = dict(row)
    matrix["models"]["v2_full_stack"]["cardplay_only"] = {
        "status": "unavailable", "reason": "not supplied", "bundle": None
    }
    with pytest.raises(P17MatrixError, match="manifest-validated learned bidding"):
        normalize_matrix(matrix)


def test_full_game_evaluation_uses_manifest_validated_learned_bidding(tmp_path):
    torch.manual_seed(1717)
    ruleset = RuleSet.standard()
    schema = build_v2_schema()
    config = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        role_embedding_dim=8,
        mlp_layers=1,
        bidding_enabled=True,
        bidding_hidden_size=16,
        nan_guard=False,
    )
    model = ModelV2(schema, config).eval()
    with torch.no_grad():
        model.bidding_heads.policy.weight.zero_()
        model.bidding_heads.policy.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    checkpoint = tmp_path / "learned-bidding.pt"
    save_v2_position_weights(str(checkpoint), model, ruleset=ruleset)
    checkpoints = {role: str(checkpoint) for role in (
        "landlord", "landlord_up", "landlord_down"
    )}
    candidate = BundleSpec(
        name="v2_full_stack",
        backend="v2",
        checkpoints=checkpoints,
        bidding_policy="learned",
        bidding_checkpoint=str(checkpoint),
        model_config=asdict(config),
    )
    baseline = BundleSpec(
        name="v2_base",
        backend="v2",
        checkpoints=checkpoints,
        bidding_policy="learned",
        bidding_checkpoint=str(checkpoint),
        model_config=asdict(config),
    )
    matrix = empty_matrix()
    for name in ("v2_full_stack", "v2_base"):
        matrix["models"][name]["full_game"] = {
            "status": "available",
            "reason": "",
            "bundle": {
                "backend": "v2",
                "checkpoints": checkpoints,
                "bidding_policy": "learned",
                "bidding_checkpoint": str(checkpoint),
                "model_config": asdict(config),
            },
        }
    normalized = normalize_matrix(matrix)
    assert normalized["models"]["v2_full_stack"]["full_game"]["status"] == "available"
    scenario = EvaluationScenario(
        mode="full_game",
        ruleset=ruleset,
        candidate=candidate,
        baseline=baseline,
        deals=generate_deals("full_game", 1, 1717, ruleset),
        deterministic_seed=1717,
        bootstrap_samples=20,
    )
    result = evaluate_scenario(scenario)
    assert result.scenario["candidate"]["bidding_policy"] == "learned"
    assert result.scenario["candidate"]["bidding_checkpoint"] is True
    assert result.metrics["sample_counts"]["bidding_inference_calls"] >= 1
    assert all(game.bid_value == 3 for game in result.games)
    assert result.metrics["redeals"]["total"] == 0
    artifacts = tmp_path / "p17"
    result_payload = result.to_dict()
    write_p17_artifacts(
        artifacts,
        matrix=matrix,
        full_game_result=result_payload,
    )
    full_report = json.loads((artifacts / "full_game_results.json").read_text())
    assert full_report["status"] == "completed"
    assert full_report["readiness"]["status"] == "insufficient"
    assert any("1000 paired deals" in issue for issue in full_report["readiness"]["issues"])

    wrong_identity = copy.deepcopy(result_payload)
    wrong_identity["scenario"]["candidate"]["checkpoint_identities"][
        "bidding"
    ] = "0" * 64
    with pytest.raises(P17MatrixError, match="checkpoint/manifest identity"):
        write_p17_artifacts(
            tmp_path / "wrong-identity",
            matrix=matrix,
            full_game_result=wrong_identity,
        )

    inflated = copy.deepcopy(result_payload)
    inflated["metrics"]["paired_estimate_ci"]["paired_deals"] = 1000
    readiness = result_readiness(inflated, mode="full_game")
    assert readiness["status"] == "insufficient"
    assert readiness["evidence"]["paired_deals"] == 1
    assert any("does not match game rows" in issue for issue in readiness["issues"])
