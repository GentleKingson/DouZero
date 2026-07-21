"""H1 role-residual public policy correctness and compatibility tests."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import torch

from douzero.checkpoint.io import CheckpointCompatibilityError
from douzero.checkpoint.v2 import save_v2_position_weights
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2 import ModelV2, ModelV2Config, observation_to_model_inputs
from douzero.observation import build_v2_schema, get_obs_v2
from douzero.v3_hybrid import (
    CHANNEL_GATE_SE,
    V3_HYBRID_MODEL_VERSION,
    V3HybridModel,
    V3HybridModelConfig,
    export_v3_hybrid_padded,
    load_v3_hybrid_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
)

ROLES = ("landlord", "landlord_up", "landlord_down")


def _observation(seed: int, role: str):
    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    for _ in range(60):
        if env._acting_player_position == role:
            return get_obs_v2(env.infoset)
        env.step(env.infoset.legal_actions[0])
    raise AssertionError(f"could not reach role {role}")


def _config(**overrides) -> V3HybridModelConfig:
    values = {
        "hidden_size": 32,
        "history_layers": 1,
        "history_heads": 4,
        "shared_fusion_layers": 1,
        "landlord_adapter_layers": 2,
        "farmer_adapter_layers": 4,
    }
    values.update(overrides)
    return V3HybridModelConfig(**values)


def _model(**config_overrides) -> V3HybridModel:
    torch.manual_seed(1234)
    return V3HybridModel(build_v2_schema(), _config(**config_overrides))


def _raw_forward(model: V3HybridModel, observation, **changes):
    bundle = observation_to_model_inputs(observation)
    values = {
        "state_card_vectors": bundle.state_card_vectors,
        "state_context_flat": bundle.state_context_flat,
        "context_card_vectors": bundle.context_card_vectors,
        "context_flat": bundle.context_flat,
        "history_tokens": bundle.history_tokens,
        "history_key_padding_mask": bundle.history_key_padding_mask,
        "action_features": bundle.action_features,
        "action_mask": bundle.action_mask,
        "acting_role": bundle.acting_role,
    }
    values.update(changes)
    return model(**values)


def _output_tensors(output):
    return (
        output.dmc_q,
        output.win_logit,
        output.score_if_win,
        output.score_if_loss,
        output.p_win,
        output.score_mean,
    )


def test_config_identity_binds_every_architecture_field():
    baseline = _config()
    assert baseline.history_encoder == "lstm"
    assert baseline.attention_type == "none"
    assert baseline.stable_hash() == _config().stable_hash()
    probes = {
        "hidden_size": 48,
        "history_encoder": "transformer",
        "history_layers": 2,
        "history_heads": 8,
        "history_dropout": 0.1,
        "shared_fusion_layers": 2,
        "landlord_adapter_layers": 1,
        "farmer_adapter_layers": 3,
        "farmer_channel_gate": CHANNEL_GATE_SE,
        "farmer_channel_gate_reduction": 8,
        "adapter_dropout": 0.1,
        "score_clamp": 16.0,
        "dmc_target_transform": "signed_log",
        "dmc_target_clamp": 16.0,
        "nan_guard": False,
    }
    for field, value in probes.items():
        changed = dataclasses.replace(baseline, **{field: value})
        assert changed.stable_hash() != baseline.stable_hash(), field


def test_config_rejects_unimplemented_attention_and_unknown_serialized_fields():
    with pytest.raises(ValueError, match="attention_type='none'"):
        _config(attention_type="cross_attention")
    payload = dataclasses.asdict(_config())
    with pytest.raises(ValueError, match="unknown"):
        V3HybridModelConfig.from_dict({**payload, "oracle_enabled": False})


def test_role_graph_has_required_depth_and_independent_parameters():
    model = _model(farmer_channel_gate=CHANNEL_GATE_SE)
    assert len(model.role_adapters["landlord"].blocks) == 2
    assert len(model.role_adapters["landlord_up"].blocks) == 4
    assert len(model.role_adapters["landlord_down"].blocks) == 4
    assert model.role_adapters["landlord"].gate is None
    assert model.role_adapters["landlord_up"].gate is not None
    for left, right in (("landlord_up", "landlord_down"), ("landlord", "landlord_up")):
        left_ids = {id(parameter) for parameter in model.role_adapters[left].parameters()}
        right_ids = {id(parameter) for parameter in model.role_adapters[right].parameters()}
        assert left_ids.isdisjoint(right_ids)
        head_left = {id(parameter) for parameter in model.role_heads[left].parameters()}
        head_right = {id(parameter) for parameter in model.role_heads[right].parameters()}
        assert head_left.isdisjoint(head_right)


@pytest.mark.parametrize("role", ROLES)
def test_public_forward_has_independent_finite_q_win_and_score_outputs(role):
    model = _model().eval()
    observation = _observation(10 + ROLES.index(role), role)
    output = model.forward_observation(observation)
    count = len(observation.actions.legal_actions)
    assert output.dmc_q.shape == (count, 1)
    for tensor in _output_tensors(output):
        assert tensor.shape == (count, 1)
        assert torch.isfinite(tensor).all()
    assert torch.allclose(output.p_win, torch.sigmoid(output.win_logit))
    assert model.role_heads[role].dmc_head is not model.role_heads[role].win_head


def test_variable_action_counts_and_single_action_are_supported():
    model = _model().eval()
    observation = _observation(20, "landlord")
    bundle = observation_to_model_inputs(observation)
    for count in (1, min(3, bundle.action_features.shape[0])):
        output = _raw_forward(
            model,
            observation,
            action_features=bundle.action_features[:count],
            action_mask=bundle.action_mask[:count],
        )
        assert output.dmc_q.shape == (count, 1)


def test_action_permutation_equivariance_and_sensitivity():
    model = _model().eval()
    observation = _observation(30, "landlord")
    bundle = observation_to_model_inputs(observation)
    count = bundle.action_features.shape[0]
    if count < 2:
        pytest.skip("fixture produced one legal action")
    baseline = _raw_forward(model, observation)
    permutation = torch.arange(count - 1, -1, -1)
    permuted = _raw_forward(
        model,
        observation,
        action_features=bundle.action_features[permutation],
        action_mask=bundle.action_mask[permutation],
    )
    for original, changed in zip(_output_tensors(baseline), _output_tensors(permuted)):
        assert torch.allclose(changed, original[permutation], atol=1e-6, rtol=1e-6)
    altered_actions = bundle.action_features.clone()
    altered_actions[0] = altered_actions[0] + 1.0
    altered = _raw_forward(model, observation, action_features=altered_actions)
    assert not torch.allclose(altered.dmc_q[0], baseline.dmc_q[0])


def test_same_public_tensors_take_different_role_paths():
    model = _model().eval()
    observation = _observation(40, "landlord")
    outputs = [
        _raw_forward(model, observation, acting_role=role).dmc_q
        for role in ROLES
    ]
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])


def test_backward_updates_shared_and_selected_role_only():
    model = _model().train()
    observation = _observation(50, "landlord")
    output = model.forward_observation(observation)
    output.dmc_q[output.action_mask].mean().backward()
    assert any(parameter.grad is not None for parameter in model.state_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.role_adapters["landlord"].parameters())
    assert any(parameter.grad is not None for parameter in model.role_heads["landlord"].parameters())
    for role in ("landlord_up", "landlord_down"):
        assert all(parameter.grad is None for parameter in model.role_adapters[role].parameters())
        assert all(parameter.grad is None for parameter in model.role_heads[role].parameters())


def test_optimizer_step_changes_selected_q_head():
    model = _model().train()
    observation = _observation(60, "landlord")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.role_heads["landlord"].dmc_head.weight.detach().clone()
    loss = model.forward_observation(observation).dmc_q.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.role_heads["landlord"].dmc_head.weight)


def test_padding_is_never_selected_or_gathered():
    model = _model().eval()
    observations = [_observation(70, "landlord"), _observation(71, "landlord_up")]
    counts = [len(observation.actions.legal_actions) for observation in observations]
    padded = max(counts) + 3
    output = model.forward_observation_batch(observations, pad_to_actions=padded)
    for index, count in enumerate(counts):
        assert not output.action_mask[index, count:].any()
        selected = output.select(index).argmax("dmc_q")
        assert selected < count
    invalid = torch.tensor([counts[0], counts[1]], dtype=torch.long)
    with pytest.raises(ValueError, match="padded"):
        output.gather_chosen(invalid)


def test_scalar_and_padded_batch_outputs_match_for_all_roles():
    model = _model().eval()
    observations = [
        _observation(80 + index, role) for index, role in enumerate(ROLES)
    ]
    batched = model.forward_observation_batch(observations)
    for index, observation in enumerate(observations):
        scalar = model.forward_observation(observation)
        selected = batched.select(index)
        count = scalar.num_actions
        for expected, actual in zip(_output_tensors(scalar), _output_tensors(selected)):
            assert torch.allclose(expected, actual[:count], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_valid_action_fails(nonfinite):
    model = _model().eval()
    observation = _observation(90, "landlord")
    bundle = observation_to_model_inputs(observation)
    invalid = bundle.action_features.clone()
    invalid[0, 0] = nonfinite
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        _raw_forward(model, observation, action_features=invalid)


def test_act_returns_only_an_environment_legal_action_and_rejects_privileged():
    model = _model().eval()
    observation = _observation(100, "landlord")
    assert model.act(observation) in observation.actions.legal_actions
    with pytest.raises(TypeError, match="privileged"):
        model.act({"kind": "privileged", "all_handcards": {}})
    with pytest.raises(RuntimeError, match="card-play only"):
        model.forward_bidding(object())


def test_import_graph_is_public_only():
    probe = (
        "import sys; import douzero.v3_hybrid; "
        "assert 'douzero.observation.privileged' not in sys.modules; "
        "assert 'douzero.distillation.teacher_model' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_strict_checkpoint_round_trip_and_identity_drift(tmp_path):
    model = _model().eval()
    ruleset = RuleSet.legacy()
    path = tmp_path / "v3.ckpt"
    manifest = save_v3_hybrid_public_checkpoint(path, model, ruleset=ruleset)
    assert manifest.model_version == V3_HYBRID_MODEL_VERSION
    loaded = load_v3_hybrid_public_checkpoint(
        path,
        schema=model.schema,
        ruleset=ruleset,
        config=model.config,
    ).eval()
    observation = _observation(110, "landlord_down")
    for expected, actual in zip(
        _output_tensors(model.forward_observation(observation)),
        _output_tensors(loaded.forward_observation(observation)),
    ):
        assert torch.equal(expected, actual)
    with pytest.raises(CheckpointCompatibilityError, match="model config"):
        load_v3_hybrid_public_checkpoint(
            path,
            schema=model.schema,
            ruleset=ruleset,
            config=dataclasses.replace(model.config, dmc_target_transform="signed_log"),
        )
    with pytest.raises(CheckpointCompatibilityError, match="ruleset"):
        load_v3_hybrid_public_checkpoint(
            path,
            schema=model.schema,
            ruleset=RuleSet.standard(),
            config=model.config,
        )


def test_v2_legacy_and_privileged_payloads_fail_closed(tmp_path):
    schema = build_v2_schema()
    config = _config()
    ruleset = RuleSet.legacy()
    v2 = ModelV2(
        schema,
        ModelV2Config(
            hidden_size=32,
            history_layers=1,
            history_heads=4,
            mlp_layers=1,
        ),
    )
    v2_path = tmp_path / "v2.ckpt"
    save_v2_position_weights(v2_path, v2, ruleset=ruleset)
    with pytest.raises(CheckpointCompatibilityError):
        load_v3_hybrid_public_checkpoint(
            v2_path, schema=schema, ruleset=ruleset, config=config
        )
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save(v2.state_dict(), legacy_path)
    with pytest.raises(CheckpointCompatibilityError):
        load_v3_hybrid_public_checkpoint(
            legacy_path, schema=schema, ruleset=ruleset, config=config
        )

    model = V3HybridModel(schema, config)
    public_path = tmp_path / "public.ckpt"
    save_v3_hybrid_public_checkpoint(public_path, model, ruleset=ruleset)
    bundle = torch.load(public_path, weights_only=True)
    bundle["state_dict"]["oracle.teacher.weight"] = torch.zeros(1)
    poisoned = tmp_path / "poisoned.ckpt"
    torch.save(bundle, poisoned)
    with pytest.raises(CheckpointCompatibilityError, match="forbidden"):
        load_v3_hybrid_public_checkpoint(
            poisoned, schema=schema, ruleset=ruleset, config=config
        )


def test_public_export_reloads_and_aligns(tmp_path):
    model = _model().eval()
    observation = _observation(120, "landlord_up")
    bundle = observation_to_model_inputs(observation)
    output = tmp_path / "v3.pt2"
    error = export_v3_hybrid_padded(
        model,
        bundle,
        output,
        acting_role="landlord_up",
        max_actions=bundle.action_features.shape[0] + 2,
    )
    assert output.is_file()
    assert error <= 1e-5
    probe = (
        "import sys, torch; torch.export.load(sys.argv[1]); "
        "assert 'douzero.observation.privileged' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe, str(output)], check=True)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_forward_backward_and_scalar_batch_alignment():
    model = _model().cuda().train()
    observations = [_observation(130, "landlord"), _observation(131, "landlord_up")]
    scalar = [model.forward_observation(observation) for observation in observations]
    batched = model.forward_observation_batch(observations)
    for index, expected in enumerate(scalar):
        actual = batched.select(index)
        for left, right in zip(_output_tensors(expected), _output_tensors(actual)):
            assert torch.allclose(left, right[: expected.num_actions], atol=2e-5, rtol=2e-5)
    loss = sum(output.dmc_q.mean() for output in scalar)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
