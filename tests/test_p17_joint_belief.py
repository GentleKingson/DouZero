"""P17 differentiable constrained belief and joint-training tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from douzero.belief import (
    BeliefConfig,
    BeliefModel,
    belief_features_from_torch_probs,
    build_belief_input,
    constrained_marginals,
    constrained_marginals_torch,
    load_joint_checkpoint,
    save_joint_checkpoint,
)
from douzero.belief.constraints import legal_mask
from douzero.env.rules import RuleSet
from douzero.models_v2 import ModelV2, ModelV2Config
from douzero.observation.schema import build_v2_schema
from douzero.training import LossConfig, TrainerConfig, V2Trainer


def _public_observation(seed: int = 17):
    from douzero.env.env import Env
    from douzero.observation.encode_v2 import get_obs_v2

    np.random.seed(seed)
    env = Env("adp")
    env.reset()
    return get_obs_v2(env.infoset)


def _models():
    value = ModelV2(
        build_v2_schema(),
        ModelV2Config(
            belief_enabled=True,
            hidden_size=32,
            history_layers=1,
            history_heads=4,
            nan_guard=False,
        ),
    )
    belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
    return value, belief


def _trainer_config(mode: str, *, amp: bool = False) -> TrainerConfig:
    return TrainerConfig(
        seed=1701,
        rng_seed=1701,
        max_episodes=2,
        max_steps_per_episode=400,
        batch_size=1,
        buffer_capacity=128,
        optimizer_steps=1,
        learning_rate=1e-3,
        belief_training_mode=mode,
        amp_enabled=amp,
        amp_dtype="bfloat16" if amp else "float16",
    )


def _parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module) -> bool:
    return any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, module.parameters())
    )


def test_torch_constrained_marginals_conserve_counts_jokers_and_gradients():
    torch.manual_seed(17)
    logits = torch.randn(2, 15, 5, requires_grad=True)
    unseen = np.full((2, 15), 4, dtype=np.int64)
    unseen[:, 13:] = 1
    legal = torch.from_numpy(np.stack([legal_mask(row) for row in unseen])).bool()
    totals = torch.tensor([17, 20], dtype=torch.long)

    marginals = constrained_marginals_torch(logits, totals, legal)
    expected = (marginals * torch.arange(5, dtype=torch.float32)).sum(dim=-1)
    torch.testing.assert_close(expected.sum(dim=-1), totals.float(), atol=2e-5, rtol=0)
    assert bool((marginals[:, 13:, 2:] == 0).all())
    torch.testing.assert_close(
        marginals.sum(dim=-1), torch.ones(2, 15), atol=2e-6, rtol=0
    )

    # The torch recurrence is the same posterior as the exact evaluation DP.
    exact = constrained_marginals(
        np.where(legal[0].numpy(), logits.detach()[0].numpy(), -np.inf), 17
    )
    np.testing.assert_allclose(marginals.detach()[0].numpy(), exact, atol=2e-6)

    loss = expected[:, :5].square().mean() + marginals[:, :, 1].mean()
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0


def test_value_only_loss_reaches_belief_encoder_through_torch_dp():
    torch.manual_seed(18)
    obs = _public_observation(18)
    binput = build_belief_input(obs.public)
    value, belief = _models()
    belief_output = belief.forward_differentiable([binput])
    features = belief_features_from_torch_probs(
        belief_output.require_differentiable_probs(),
        belief_output.opponent_a_total,
        np.stack([binput.unseen_counts]),
    )[0]

    from douzero.models_v2.batch import observation_to_model_inputs

    bundle = observation_to_model_inputs(obs)
    output = value(
        bundle.state_card_vectors,
        bundle.state_context_flat,
        bundle.context_card_vectors,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features,
        bundle.action_mask,
        bundle.acting_role,
        belief_features=features,
        belief_stop_gradient=False,
    )
    output.win_logit.square().mean().backward()
    belief_grads = [p.grad for p in belief.parameters() if p.grad is not None]
    assert belief_grads
    assert all(bool(torch.isfinite(grad).all()) for grad in belief_grads)
    assert sum(float(grad.abs().sum()) for grad in belief_grads) > 0


def test_differentiable_dp_stays_float32_and_finite_under_cpu_amp():
    torch.manual_seed(19)
    binput = build_belief_input(_public_observation(19).public)
    belief = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = belief.forward_differentiable([binput])
        probs = output.require_differentiable_probs()
        features = belief_features_from_torch_probs(
            probs,
            output.opponent_a_total,
            np.stack([binput.unseen_counts]),
        )
        loss = features.square().mean()
    assert probs.dtype == torch.float32
    assert features.dtype == torch.float32
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in belief.parameters()
    )


def test_joint_checkpoint_roundtrip_binds_both_models_and_optimizer(tmp_path):
    torch.manual_seed(20)
    value, belief = _models()
    optimizer = torch.optim.RMSprop(
        list(value.parameters()) + list(belief.parameters()), lr=1e-3
    )
    path = str(tmp_path / "joint.pt")
    value_before = {name: tensor.detach().clone() for name, tensor in value.state_dict().items()}
    belief_before = {name: tensor.detach().clone() for name, tensor in belief.state_dict().items()}
    manifest = save_joint_checkpoint(
        path,
        value,
        belief,
        ruleset=RuleSet.legacy(),
        belief_training_mode="joint",
        optimizer=optimizer,
        optimizer_steps=3,
    )
    assert manifest.optimizer_included
    assert manifest.public_input_contract == "belief_input_public_v1"

    with torch.no_grad():
        next(value.parameters()).add_(1)
        next(belief.parameters()).sub_(1)
    loaded = load_joint_checkpoint(
        path,
        value,
        belief,
        expected_ruleset=RuleSet.legacy(),
        expected_belief_training_mode="joint",
        optimizer=optimizer,
    )
    assert loaded.optimizer_steps == 3
    for name, tensor in value.state_dict().items():
        assert torch.equal(tensor, value_before[name])
    for name, tensor in belief.state_dict().items():
        assert torch.equal(tensor, belief_before[name])

    bundle = torch.load(path, map_location="cpu", weights_only=True)
    assert set(bundle) == {
        "manifest", "value_state_dict", "belief_state_dict", "optimizer_state_dict"
    }
    assert not any("label" in key or "privileged" in key for key in bundle)


def test_joint_checkpoint_rejects_cross_mode_and_deploys_public_only(tmp_path):
    value, belief = _models()
    path = str(tmp_path / "joint.pt")
    save_joint_checkpoint(
        path,
        value,
        belief,
        ruleset=RuleSet.legacy(),
        belief_training_mode="joint",
    )
    with pytest.raises(ValueError, match="mode mismatch"):
        load_joint_checkpoint(
            path,
            value,
            belief,
            expected_ruleset=RuleSet.legacy(),
            expected_belief_training_mode="alternating",
        )

    # The restored pair can only deploy through PublicObservation ->
    # BeliefInput; neither model accepts a privileged label/input argument.
    from douzero.evaluation.deep_agent import DeepAgentV2

    agent = DeepAgentV2("landlord", value, RuleSet.legacy(), belief_model=belief)
    obs = _public_observation(21)
    assert obs.is_privileged is False
    assert agent.act_v2(obs) in obs.public.legal_actions


def test_trainer_frozen_mode_keeps_belief_out_of_optimizer_and_unchanged():
    torch.manual_seed(22)
    value, belief = _models()
    before = _parameters(belief)
    trainer = V2Trainer(
        value,
        belief_model=belief,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=_trainer_config("frozen"),
    )
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids.isdisjoint({id(parameter) for parameter in belief.parameters()})
    assert all(not parameter.requires_grad for parameter in belief.parameters())
    trainer.collect_episodes()
    assert trainer.step() is not None
    assert not _changed(before, belief)
    assert trainer.stats.belief_phase == "frozen"


def test_trainer_joint_value_only_loss_updates_belief_encoder():
    torch.manual_seed(23)
    value, belief = _models()
    before = _parameters(belief)
    trainer = V2Trainer(
        value,
        belief_model=belief,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=_trainer_config("joint"),
    )
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert {id(parameter) for parameter in belief.parameters()} <= optimizer_ids
    trainer.collect_episodes()
    assert trainer.step() is not None
    assert _changed(before, belief)
    assert trainer.stats.belief_phase == "joint"
    assert np.isfinite(trainer.stats.last_loss["loss_total"])


def test_trainer_alternates_value_only_then_supervised_belief_parameters():
    from douzero.belief.data import collect_random_dataset

    torch.manual_seed(24)
    value, belief = _models()
    samples = collect_random_dataset(2, seed=24).samples
    config = _trainer_config("alternating")
    config.belief_supervised_weight = 0.5
    config.belief_supervised_batch_size = 2
    trainer = V2Trainer(
        value,
        belief_model=belief,
        belief_supervised_samples=samples,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=config,
    )
    trainer.collect_episodes()

    value_before = _parameters(value)
    belief_before = _parameters(belief)
    assert trainer.step() is not None
    assert trainer.stats.belief_phase == "value"
    assert _changed(value_before, value)
    assert not _changed(belief_before, belief)

    value_before = _parameters(value)
    belief_before = _parameters(belief)
    assert trainer.step() is not None
    assert trainer.stats.belief_phase == "belief"
    assert not _changed(value_before, value)
    assert _changed(belief_before, belief)
    assert trainer.stats.belief_supervised_steps == 1
    assert np.isfinite(trainer.stats.last_loss["belief_loss_total"])


def test_joint_trainer_cpu_bfloat16_amp_is_finite_and_updates_belief():
    torch.manual_seed(25)
    value, belief = _models()
    before = _parameters(belief)
    trainer = V2Trainer(
        value,
        belief_model=belief,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=_trainer_config("joint", amp=True),
    )
    trainer.collect_episodes()
    assert trainer.step() is not None
    assert _changed(before, belief)
    assert trainer.stats.amp_fallbacks == 0
    assert np.isfinite(trainer.stats.grad_norm_last_step)
    assert np.isfinite(trainer.stats.last_loss["loss_total"])


def test_joint_trainer_checkpoint_resume_restores_both_and_continues(tmp_path):
    torch.manual_seed(26)
    value, belief = _models()
    trainer = V2Trainer(
        value,
        belief_model=belief,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=_trainer_config("joint"),
    )
    trainer.collect_episodes()
    assert trainer.step() is not None
    saved_value = {name: tensor.detach().clone() for name, tensor in value.state_dict().items()}
    saved_belief = {name: tensor.detach().clone() for name, tensor in belief.state_dict().items()}
    path = str(tmp_path / "trainer_joint.pt")
    identity = trainer.save_training_checkpoint(path)
    assert identity["checkpoint_version"] == 2
    assert identity["belief_training_mode"] == "joint"

    torch.manual_seed(27)
    restored_value, restored_belief = _models()
    restored = V2Trainer(
        restored_value,
        belief_model=restored_belief,
        loss_config=LossConfig(lambda_win=1.0, lambda_score=0.5),
        config=_trainer_config("joint"),
    )
    restored.load_training_checkpoint(path)
    for name, tensor in restored_value.state_dict().items():
        assert torch.equal(tensor, saved_value[name])
    for name, tensor in restored_belief.state_dict().items():
        assert torch.equal(tensor, saved_belief[name])
    previous_steps = restored.stats.optimizer_steps
    restored.collect_episodes()
    assert restored.step() is not None
    assert restored.stats.optimizer_steps == previous_steps + 1


def test_non_frozen_belief_ddp_fails_closed():
    from douzero.runtime import DistributedContext

    value, belief = _models()
    with pytest.raises(NotImplementedError, match="not supported under DDP"):
        V2Trainer(
            value,
            belief_model=belief,
            loss_config=LossConfig(lambda_win=1.0),
            config=_trainer_config("joint"),
            distributed_context=DistributedContext(
                enabled=True,
                rank=0,
                world_size=2,
                local_rank=0,
                backend="gloo",
                device=torch.device("cpu"),
            ),
        )
