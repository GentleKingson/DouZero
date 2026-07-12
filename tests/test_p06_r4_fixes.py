"""P06 r4 regression tests: replay transition integrity, no-op optimization
prevention, and failure-path model-state recovery.

Covers the three blockers from the r3 review:

- Replay buffer accepts corrupted labels (target_win=2.0, target_score=inf)
  and out-of-range action_index.
- A valid config with all-zero loss weights "trains" silently (zero
  gradient, no parameter change, optimizer_steps still increments).
- A non-finite gradient exception leaves the model in training mode (no
  try/finally around the train→eval transition).
"""

from __future__ import annotations

import math

import pytest
import torch

from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training import (
    LossConfig,
    TrainerConfig,
    V2Trainer,
)
from douzero.training.v2_buffer import Episode, Minibatch, Transition


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_model(**kwargs):
    torch.manual_seed(42)
    return ModelV2(build_v2_schema(), ModelV2Config(**kwargs))


def _valid_transition(**overrides):
    """Build a Transition with valid defaults; override any field."""
    from douzero.env.env import Env
    from douzero.observation.encode_v2 import get_obs_v2

    import numpy as np

    np.random.seed(999)
    env = Env("adp")
    env.reset()
    # Drive to landlord and ensure >= 1 legal action.
    while env._acting_player_position != "landlord":
        env.step(env.infoset.legal_actions[0])
    obs = get_obs_v2(env.infoset)
    defaults = dict(
        obs=obs,
        action_index=0,
        position="landlord",
        target_win=1.0,
        target_score=2.0,
        target_log_score=0.7,
    )
    defaults.update(overrides)
    return Transition(**defaults)


# --------------------------------------------------------------------------- #
# Blocker 1: Transition.validate() rejects corrupted data
# --------------------------------------------------------------------------- #
def test_transition_validate_rejects_target_win_not_binary():
    tr = _valid_transition(target_win=2.0)
    with pytest.raises(ValueError, match="target_win must be 0.0 or 1.0"):
        tr.validate()


def test_transition_validate_rejects_target_score_inf():
    tr = _valid_transition(target_score=float("inf"))
    with pytest.raises(ValueError, match="target_score must be finite"):
        tr.validate()


def test_transition_validate_rejects_target_log_score_neg_inf():
    tr = _valid_transition(target_log_score=float("-inf"))
    with pytest.raises(ValueError, match="target_log_score must be finite"):
        tr.validate()


def test_transition_validate_rejects_action_index_out_of_range():
    tr = _valid_transition(action_index=9999)
    with pytest.raises(ValueError, match="outside the observation's legal-action range"):
        tr.validate()


def test_transition_validate_rejects_invalid_position():
    tr = _valid_transition(position="bystander")
    with pytest.raises(ValueError, match="position must be one of"):
        tr.validate()


def test_transition_validate_rejects_bool_action_index():
    tr = _valid_transition(action_index=True)
    with pytest.raises(TypeError, match="action_index must be int"):
        tr.validate()


def test_add_episode_rejects_corrupted_transition():
    """add_episode() must call validate() and reject a corrupted transition."""
    from douzero.training.v2_buffer import V2ReplayBuffer

    tr = _valid_transition(target_win=2.0)  # corrupted
    ep = Episode(transitions=[tr], terminal_result={})
    buf = V2ReplayBuffer(capacity_transitions=10)
    with pytest.raises(ValueError, match="target_win must be 0.0 or 1.0"):
        buf.add_episode(ep)


def test_add_episode_accepts_valid_transition():
    """A valid transition passes validation and enters the buffer."""
    from douzero.training.v2_buffer import V2ReplayBuffer

    tr = _valid_transition()
    ep = Episode(transitions=[tr], terminal_result={})
    buf = V2ReplayBuffer(capacity_transitions=10)
    buf.add_episode(ep)
    assert len(buf) == 1


# --------------------------------------------------------------------------- #
# Blocker 1: Minibatch.validate() batch-length consistency
# --------------------------------------------------------------------------- #
def test_minibatch_validate_rejects_inconsistent_lengths():
    tr = _valid_transition()
    mb = Minibatch(
        observations=[tr.obs, tr.obs],  # 2
        action_indices=torch.tensor([0, 0], dtype=torch.long),  # 2
        target_win=torch.tensor([1.0], dtype=torch.float32),  # 1 — mismatch
        target_score=torch.tensor([2.0, 2.0], dtype=torch.float32),
        target_log_score=torch.tensor([0.7, 0.7], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="batch lengths disagree"):
        mb.validate()


# --------------------------------------------------------------------------- #
# Blocker 2: No-op optimization prevention
# --------------------------------------------------------------------------- #
def test_trainer_rejects_all_zero_loss_with_optimizer_steps():
    """optimizer_steps > 0 with all λ=0 must raise (would silently no-op)."""
    model = _build_model()
    with pytest.raises(ValueError, match="at least one non-zero loss weight"):
        V2Trainer(
            model,
            loss_config=LossConfig(lambda_win=0.0, lambda_score=0.0, lambda_uncertainty=0.0),
            config=TrainerConfig(optimizer_steps=1, max_episodes=0),
        )


def test_trainer_accepts_all_zero_loss_with_zero_optimizer_steps():
    """optimizer_steps=0 with all λ=0 is valid (collect-only mode)."""
    model = _build_model()
    trainer = V2Trainer(
        model,
        loss_config=LossConfig(lambda_win=0.0, lambda_score=0.0),
        config=TrainerConfig(optimizer_steps=0, max_episodes=0),
    )
    assert trainer is not None


def test_trainer_rejects_buffer_smaller_than_batch():
    model = _build_model()
    with pytest.raises(ValueError, match="buffer_capacity.*must be >= batch_size"):
        V2Trainer(
            model,
            loss_config=LossConfig(lambda_win=1.0),
            config=TrainerConfig(
                optimizer_steps=1, batch_size=32, buffer_capacity=16, max_episodes=0
            ),
        )


def test_train_raises_when_insufficient_transitions():
    """train() must raise RuntimeError when optimizer steps were skipped."""
    model = _build_model()
    trainer = V2Trainer(
        model,
        loss_config=LossConfig(lambda_win=1.0),
        config=TrainerConfig(
            max_episodes=1,
            optimizer_steps=1,
            batch_size=256,  # larger than what 1 episode produces
            buffer_capacity=512,
        ),
    )
    with pytest.raises(RuntimeError, match="optimizer steps"):
        trainer.train()


# --------------------------------------------------------------------------- #
# Blocker 2c: λ=0 skips computation (no 0*NaN propagation)
# --------------------------------------------------------------------------- #
def test_loss_rejects_nan_target_win_at_boundary():
    """P06 r5: the public loss API must reject target_win=NaN at its
    boundary. The r4 version of this test asserted the loss was finite —
    which codified the bug where NaN silently routed to score_if_loss via
    ``NaN >= 0.5 → False`` in _select_per_sample. Now the loss module
    validates labels before any computation."""
    from douzero.training import MultiObjectiveLoss

    win_logit = torch.tensor([[0.5]], requires_grad=True)
    score_if_win = torch.tensor([[1.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([float("nan")]),  # invalid outcome label
        "target_score": torch.tensor([1.0]),
    }
    fn = MultiObjectiveLoss(
        LossConfig(lambda_win=0.0, lambda_score=1.0)  # score active
    )
    with pytest.raises(ValueError, match="target_win contains non-finite"):
        fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)


def test_loss_rejects_non_binary_target_win():
    """target_win=2.0 is not in {0, 1} and must be rejected."""
    from douzero.training import MultiObjectiveLoss

    win_logit = torch.tensor([[0.5]], requires_grad=True)
    score_if_win = torch.tensor([[1.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([2.0]),
        "target_score": torch.tensor([1.0]),
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=0.5))
    with pytest.raises(ValueError, match="target_win must be binary"):
        fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)


def test_loss_rejects_non_finite_target_score_when_score_active():
    """target_score=Inf must be rejected when lambda_score > 0."""
    from douzero.training import MultiObjectiveLoss

    win_logit = torch.tensor([[0.5]], requires_grad=True)
    score_if_win = torch.tensor([[1.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([float("inf")]),
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=0.5))
    with pytest.raises(ValueError, match="target_score contains non-finite"):
        fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)


def test_loss_accepts_nan_target_score_when_score_inactive():
    """When lambda_score=0 AND lambda_uncertainty=0, a non-finite
    target_score is acceptable because no score loss is computed."""
    from douzero.training import MultiObjectiveLoss

    win_logit = torch.tensor([[0.5]], requires_grad=True)
    score_if_win = torch.tensor([[1.0]], requires_grad=True)
    score_if_loss = torch.tensor([[-1.0]], requires_grad=True)
    labels = {
        "target_win": torch.tensor([1.0]),
        "target_score": torch.tensor([float("inf")]),  # OK: score disabled
    }
    fn = MultiObjectiveLoss(LossConfig(lambda_win=1.0, lambda_score=0.0))
    comps = fn.forward_gathered(win_logit, score_if_win, score_if_loss, labels)
    assert torch.isfinite(comps.total)


# --------------------------------------------------------------------------- #
# Blocker 3: try/finally restores eval mode on exception
# --------------------------------------------------------------------------- #
def test_step_restores_eval_mode_on_nonfinite_gradient(monkeypatch):
    """When clip_grad_norm_ raises (non-finite gradient), the model must be
    returned to eval mode via try/finally (P06 r4)."""
    model = _build_model()
    trainer = V2Trainer(
        model,
        loss_config=LossConfig(lambda_win=1.0),
        config=TrainerConfig(
            max_episodes=0,
            optimizer_steps=0,
            batch_size=4,
            exp_epsilon=0.5,
        ),
    )
    # Collect enough transitions for a minibatch.
    trainer.collect_episodes(6)
    assert trainer.buffer._size >= 4

    before = next(model.parameters()).detach().clone()
    steps_before = trainer.stats.optimizer_steps

    # Inject a NaN gradient by monkeypatching backward to produce NaN grads.
    def nan_backward(*args, **kwargs):
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, float("nan"))

    monkeypatch.setattr(
        "douzero.training.v2_trainer.nn.utils.clip_grad_norm_",
        lambda params, max_norm, error_if_nonfinite=True: (_ for _ in ()).throw(
            FloatingPointError("injected non-finite gradient")
        ),
    )
    # Patch backward on the total tensor — we do this by patching the
    # loss to return a tensor whose backward produces NaN. Simpler: just
    # patch clip_grad_norm_ to raise (which is the real failure path).
    with pytest.raises(FloatingPointError):
        trainer.step()

    # Model must be back in eval mode.
    assert not model.training, (
        "model left in training mode after a failed optimizer step; "
        "try/finally did not restore eval."
    )
    # Parameters unchanged (optimizer.step() was never reached).
    after = next(model.parameters()).detach().clone()
    assert torch.equal(before, after)
    # optimizer_steps did not increment.
    assert trainer.stats.optimizer_steps == steps_before


def test_step_restores_eval_mode_on_nonfinite_loss(monkeypatch):
    """When the non-finite-loss guard raises, the model must also be in eval
    mode (the try/finally covers this path too)."""
    model = _build_model()
    trainer = V2Trainer(
        model,
        loss_config=LossConfig(lambda_win=1.0),
        config=TrainerConfig(
            max_episodes=0,
            optimizer_steps=0,
            batch_size=4,
            exp_epsilon=0.5,
        ),
    )
    trainer.collect_episodes(6)

    from douzero.training.losses import LossComponents

    def nan_loss(self, win_logit, score_if_win, score_if_loss, labels):
        return LossComponents(
            total=win_logit.new_full((), float("nan")),
            win=float("nan"),
            score=0.0,
            uncertainty=0.0,
            num_win=1,
            num_loss=0,
        )

    from douzero.training import MultiObjectiveLoss

    monkeypatch.setattr(MultiObjectiveLoss, "forward_gathered", nan_loss)
    with pytest.raises(FloatingPointError):
        trainer.step()

    assert not model.training


def test_step_leaves_no_stale_gradients_on_exception(monkeypatch):
    """After a failed step, gradients should be cleared (set_to_none=True in
    the finally block) so they don't accumulate on the next step."""
    model = _build_model()
    trainer = V2Trainer(
        model,
        loss_config=LossConfig(lambda_win=1.0),
        config=TrainerConfig(
            max_episodes=0, optimizer_steps=0, batch_size=4, exp_epsilon=0.5,
        ),
    )
    trainer.collect_episodes(6)

    from douzero.training.losses import LossComponents

    def nan_loss(self, win_logit, score_if_win, score_if_loss, labels):
        return LossComponents(
            total=win_logit.new_full((), float("nan")),
            win=float("nan"), score=0.0, uncertainty=0.0,
            num_win=1, num_loss=0,
        )

    from douzero.training import MultiObjectiveLoss

    monkeypatch.setattr(MultiObjectiveLoss, "forward_gathered", nan_loss)
    with pytest.raises(FloatingPointError):
        trainer.step()

    # No parameter should have a stale .grad.
    for p in model.parameters():
        assert p.grad is None, (
            f"parameter {p.shape} has a stale .grad after a failed step; "
            f"zero_grad(set_to_none=True) was not called in the finally block."
        )
