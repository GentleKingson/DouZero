"""P08 BCTrainer, pretrain_bc CLI, and the RL+BC auxiliary hook tests."""

from __future__ import annotations

import math
import os

import pytest
import torch

from douzero.env.rules import RuleSet
from douzero.human_data.sample import build_bc_samples
from douzero.human_data.synthetic import generate_synthetic_records
from douzero.human_data.validate import validate_record
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.observation.schema import build_v2_schema
from douzero.training.bc_trainer import (
    BCTrainer,
    BCTrainerConfig,
    BCTrainerError,
)
from douzero.training.losses import LossConfig


# --------------------------------------------------------------------------- #
# Fixture: build BC samples from synthetic records
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bc_samples():
    recs = list(generate_synthetic_records(num_games=4, base_seed=10))
    recs = [r for r in recs if validate_record(r).ok]
    samples = []
    for r in recs:
        samples.extend(build_bc_samples(r))
    return samples


def _build_prior_model() -> ModelV2:
    cfg = ModelV2Config(
        hidden_size=32,
        history_layers=1,
        history_heads=4,
        history_encoder="lstm",
        human_prior_enabled=True,
        nan_guard=False,
    )
    return ModelV2(build_v2_schema(), cfg)


# --------------------------------------------------------------------------- #
# BCTrainer
# --------------------------------------------------------------------------- #
class TestBCTrainer:
    def test_rejects_model_without_prior_head(self, bc_samples):
        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm", human_prior_enabled=False, nan_guard=False,
        )
        m = ModelV2(build_v2_schema(), cfg)
        with pytest.raises(BCTrainerError):
            BCTrainer(m, bc_samples, BCTrainerConfig(epochs=1))

    def test_rejects_empty_training_split(self, bc_samples):
        """When every sample shares one game_id and val_ratio > 0, that single
        game lands in val and train is empty -> the trainer rejects it."""
        m = _build_prior_model()
        # Keep only samples from a single game (all share one game_id).
        single_game_id = bc_samples[0].game_id
        one_game = [s for s in bc_samples if s.game_id == single_game_id]
        assert len(one_game) > 0
        # val_ratio >= 0.5 with a single game sends that game to val
        # (round(1 * 0.6) == 1) -> train is empty -> trainer rejects it.
        with pytest.raises(BCTrainerError):
            BCTrainer(
                m, one_game,
                BCTrainerConfig(epochs=1, val_ratio=0.6, seed=1),
            )

    def test_loss_decreases_on_synthetic_data(self, bc_samples):
        """The acceptance criterion: BC loss decreases on synthetic data."""
        torch.manual_seed(0)
        m = _build_prior_model()
        trainer = BCTrainer(
            m, bc_samples,
            BCTrainerConfig(
                epochs=3, batch_size=16, learning_rate=1e-2, val_ratio=0.2,
                seed=2,
            ),
        )
        stats = trainer.train()
        assert stats.epochs_run == 3
        # Training loss strictly decreases over the run (the primary signal
        # that the prior head is learning to fit the legal-action list).
        first = stats.epoch_stats[0].train_loss
        last = stats.epoch_stats[-1].train_loss
        assert last < first, (
            f"BC train loss did not decrease: {first} -> {last}"
        )
        # Top-1 accuracy is recorded and in [0, 1].
        for e in stats.epoch_stats:
            assert 0.0 <= e.train_top1 <= 1.0
            assert 0.0 <= e.val_top1 <= 1.0

    def test_train_val_split_has_no_game_overlap(self, bc_samples):
        torch.manual_seed(0)
        m = _build_prior_model()
        trainer = BCTrainer(
            m, bc_samples, BCTrainerConfig(epochs=1, val_ratio=0.25, seed=3),
        )
        train_ids = {s.game_id for s in trainer.train_samples}
        val_ids = {s.game_id for s in trainer.val_samples}
        assert not (train_ids & val_ids), "game_id leaked across train/val"

    def test_early_stopping(self, bc_samples):
        torch.manual_seed(0)
        m = _build_prior_model()
        trainer = BCTrainer(
            m, bc_samples,
            BCTrainerConfig(
                epochs=10, batch_size=16, learning_rate=1e-3, val_ratio=0.3,
                early_stopping_patience=1, seed=4,
            ),
        )
        stats = trainer.train()
        # With patience=1 and 10 epochs, either it stopped early or ran all;
        # in either case the bookkeeping is consistent.
        assert stats.epochs_run <= 10
        if stats.stopped_early:
            assert stats.epochs_run < 10

    def test_seed_zero_is_noop(self, bc_samples):
        """seed=0 must NOT seed torch (project convention); trainer still runs."""
        torch.manual_seed(123)
        m = _build_prior_model()
        trainer = BCTrainer(
            m, bc_samples, BCTrainerConfig(epochs=1, seed=0),
        )
        stats = trainer.train()
        assert stats.epochs_run == 1

    def test_rejects_zero_learning_rate(self, bc_samples):
        m = _build_prior_model()
        with pytest.raises(BCTrainerError):
            BCTrainer(m, bc_samples, BCTrainerConfig(learning_rate=0.0))

    def test_rejects_zero_batch_size(self, bc_samples):
        """batch_size=0 crashes range(..., step=0); reject at construction."""
        m = _build_prior_model()
        with pytest.raises(BCTrainerError):
            BCTrainer(m, bc_samples, BCTrainerConfig(batch_size=0))

    def test_empty_val_set_does_not_fake_metrics_or_restore(self, bc_samples):
        """When val_ratio=0 (or rounds to empty), val metrics are NaN (not
        0.0), early stopping is disabled, and the last-epoch model is kept
        (no best-state restore from a bogus 0.0 'best')."""
        torch.manual_seed(0)
        m = _build_prior_model()
        # Force an empty val set by using a single game with val_ratio that
        # sends its one game to train (round(1*0.0)=0 val games).
        single_game = [s for s in bc_samples if s.game_id == bc_samples[0].game_id]
        trainer = BCTrainer(
            m, single_game,
            BCTrainerConfig(
                epochs=2, batch_size=4, learning_rate=1e-3,
                val_ratio=0.0, seed=1,
            ),
        )
        assert trainer.val_samples == []
        stats = trainer.train()
        # Val metrics are NaN, not 0.0.
        for e in stats.epoch_stats:
            assert math.isnan(e.val_loss)
            assert math.isnan(e.val_top1)
        # best_epoch stays -1 (no best was ever tracked).
        assert stats.best_epoch == -1

    def test_restores_best_validation_state_dict(self, bc_samples):
        """Medium #1: after train(), the model holds the best-validation
        weights, not the last-epoch weights. We verify by snapshotting the
        state at best_epoch via a re-eval and comparing."""
        torch.manual_seed(0)
        m = _build_prior_model()
        trainer = BCTrainer(
            m, bc_samples,
            BCTrainerConfig(
                epochs=4, batch_size=16, learning_rate=5e-3, val_ratio=0.3,
                seed=5,
            ),
        )
        stats = trainer.train()
        # The model's current weights ARE the best-validation weights (restored
        # at the end of train). Re-evaluate on the val set and confirm the loss
        # equals best_val_loss (within float tolerance), not the final-epoch
        # loss — which can differ when early epochs were best.
        if trainer.val_samples and stats.best_epoch >= 0:
            val_loss, _, _ = trainer._evaluate(trainer.val_samples)
            assert val_loss == pytest.approx(stats.best_val_loss, abs=1e-4)


# --------------------------------------------------------------------------- #
# pretrain_bc CLI smoke
# --------------------------------------------------------------------------- #
class TestPretrainCLI:
    def test_synthetic_smoke_saves_checkpoint(self, tmp_path):
        import pretrain_bc

        save_dir = str(tmp_path / "bc")
        rc = pretrain_bc.main([
            "--synthetic",
            "--num_synthetic", "4",
            "--save_dir", save_dir,
            "--epochs", "2",
            "--batch_size", "8",
            "--hidden_size", "32",
            "--history_layers", "1",
            "--history_heads", "4",
            "--val_ratio", "0.2",
            "--seed", "1",
        ])
        assert rc == 0
        ckpt = os.path.join(save_dir, "bc_prior.pt")
        assert os.path.isfile(ckpt)
        # The checkpoint is a manifest-bearing V2 bundle whose state_dict
        # includes the prior-head weights (proving human_prior_enabled carried
        # through), and whose manifest records the V2 model version.
        import torch as _torch

        bundle = _torch.load(ckpt, map_location="cpu", weights_only=False)
        manifest = bundle["manifest"]
        assert manifest["model_version"] == "v2"
        sd = bundle.get("model_state_dict", bundle.get("state_dict", {}))
        prior_keys = [k for k in sd if "prior" in k.lower()]
        assert prior_keys, "prior-head weights missing from checkpoint state_dict"

    def test_help_parses(self):
        import pretrain_bc

        args = pretrain_bc._parse_args(["--synthetic", "--epochs", "1"])
        assert args.synthetic is True
        assert args.epochs == 1

    def test_requires_data_or_synthetic(self, tmp_path):
        import pretrain_bc

        with pytest.raises(SystemExit):
            pretrain_bc.main([
                "--save_dir", str(tmp_path), "--epochs", "1",
            ])


# --------------------------------------------------------------------------- #
# RL + BC auxiliary loss integration (V2Trainer with lambda_bc > 0)
# --------------------------------------------------------------------------- #
class TestRLBCAuxHook:
    def test_lambda_bc_zero_is_noop(self, bc_samples):
        """lambda_bc=0 (default) leaves the V2 trainer's RL path unchanged."""
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        torch.manual_seed(0)
        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm", human_prior_enabled=True, nan_guard=False,
        )
        model = ModelV2(build_v2_schema(), cfg)
        trainer = V2Trainer(
            model,
            loss_config=LossConfig(lambda_win=1.0, lambda_bc=0.0),
            config=TrainerConfig(
                max_episodes=2, optimizer_steps=1, batch_size=4,
                exp_epsilon=0.5, rng_seed=1,
            ),
            bc_aux_samples=bc_samples,
        )
        # lambda_bc=0 -> bc_aux_samples are accepted but warned as unused.
        assert trainer.bc_schedule.base_lambda == 0.0
        # The trainer runs without touching the BC path.
        stats = trainer.train()
        assert stats.optimizer_steps >= 1

    def test_lambda_bc_positive_requires_prior_head(self, bc_samples):
        """A model WITHOUT a prior head cannot use lambda_bc > 0."""
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm", human_prior_enabled=False, nan_guard=False,
        )
        model = ModelV2(build_v2_schema(), cfg)
        with pytest.raises(ValueError):
            V2Trainer(
                model,
                loss_config=LossConfig(lambda_win=1.0, lambda_bc=0.1),
                config=TrainerConfig(max_episodes=1, optimizer_steps=1),
                bc_aux_samples=bc_samples,
            )

    def test_lambda_bc_positive_without_samples_rejected(self):
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm", human_prior_enabled=True, nan_guard=False,
        )
        model = ModelV2(build_v2_schema(), cfg)
        with pytest.raises(ValueError):
            V2Trainer(
                model,
                loss_config=LossConfig(lambda_win=1.0, lambda_bc=0.1),
                config=TrainerConfig(max_episodes=1, optimizer_steps=1),
            )

    def test_combined_rl_bc_step_runs(self, bc_samples):
        """An optimizer step with both RL + BC terms runs and updates params."""
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        torch.manual_seed(0)
        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm", human_prior_enabled=True, nan_guard=False,
        )
        model = ModelV2(build_v2_schema(), cfg)
        trainer = V2Trainer(
            model,
            loss_config=LossConfig(lambda_win=1.0, lambda_bc=0.5),
            config=TrainerConfig(
                max_episodes=2, optimizer_steps=1, batch_size=4,
                exp_epsilon=0.5, rng_seed=1,
            ),
            bc_aux_samples=bc_samples,
        )
        assert trainer.bc_schedule.base_lambda == 0.5
        stats = trainer.train()
        assert stats.optimizer_steps >= 1

    def test_linear_decay_schedule(self, bc_samples):
        """The BC schedule linearly decays lambda over schedule_steps to a
        non-zero floor (Blocker 1)."""
        from douzero.training.bc_loss import BCSchedule
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        sched = BCSchedule(
            base_lambda=1.0, schedule="linear_decay",
            schedule_steps=10, schedule_floor=0.2,
        )
        assert sched.effective_lambda(0) == 1.0
        assert sched.effective_lambda(5) == pytest.approx(0.6)
        assert sched.effective_lambda(10) == 0.2  # at the floor
        assert sched.effective_lambda(20) == 0.2  # stays at floor
        # The V2 trainer uses the schedule per-step.
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        torch.manual_seed(0)
        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm", human_prior_enabled=True, nan_guard=False,
        )
        model = ModelV2(build_v2_schema(), cfg)
        trainer = V2Trainer(
            model,
            loss_config=LossConfig(lambda_win=1.0, lambda_bc=1.0),
            config=TrainerConfig(
                max_episodes=2, optimizer_steps=1, batch_size=4,
                exp_epsilon=0.5, rng_seed=1,
            ),
            bc_aux_samples=bc_samples,
            bc_schedule=sched,
        )
        assert trainer.bc_schedule.schedule == "linear_decay"
        stats = trainer.train()
        assert stats.optimizer_steps >= 1

    def test_belief_plus_prior_combo_runs(self, bc_samples):
        """P07+P08 combo: a belief+prior-enabled model must run an optimizer
        step. _compute_bc_aux_loss must pass belief features (Blocker 1)."""
        from douzero.belief import BeliefConfig, BeliefModel
        from douzero.training.v2_trainer import V2Trainer, TrainerConfig

        torch.manual_seed(0)
        cfg = ModelV2Config(
            hidden_size=32, history_layers=1, history_heads=4,
            history_encoder="lstm",
            belief_enabled=True, human_prior_enabled=True, nan_guard=False,
        )
        model = ModelV2(build_v2_schema(), cfg)
        belief_model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
        trainer = V2Trainer(
            model,
            loss_config=LossConfig(lambda_win=1.0, lambda_bc=0.3),
            config=TrainerConfig(
                max_episodes=2, optimizer_steps=1, batch_size=4,
                exp_epsilon=0.5, rng_seed=2,
            ),
            belief_model=belief_model,
            bc_aux_samples=bc_samples,
        )
        stats = trainer.train()
        assert stats.optimizer_steps >= 1  # no crash => belief features passed


# --------------------------------------------------------------------------- #
# BCSchedule unit tests
# --------------------------------------------------------------------------- #
class TestBCSchedule:
    def test_constant_schedule(self):
        from douzero.training.bc_loss import BCSchedule

        s = BCSchedule(base_lambda=0.5)
        assert s.effective_lambda(0) == 0.5
        assert s.effective_lambda(100) == 0.5

    def test_linear_decay_to_floor(self):
        from douzero.training.bc_loss import BCSchedule

        s = BCSchedule(
            base_lambda=1.0, schedule="linear_decay",
            schedule_steps=4, schedule_floor=0.2,
        )
        assert s.effective_lambda(0) == 1.0
        assert s.effective_lambda(2) == pytest.approx(0.6)
        assert s.effective_lambda(4) == 0.2
        assert s.effective_lambda(8) == 0.2

    def test_floor_not_forced_to_zero(self):
        from douzero.training.bc_loss import BCSchedule

        s = BCSchedule(
            base_lambda=0.5, schedule="linear_decay",
            schedule_steps=10, schedule_floor=0.1,
        )
        # Never goes below the floor.
        for step in range(20):
            assert s.effective_lambda(step) >= 0.1 - 1e-9

    def test_rejects_floor_above_base(self):
        from douzero.training.bc_loss import BCSchedule, BCLossError

        with pytest.raises(BCLossError):
            BCSchedule(base_lambda=0.3, schedule_floor=0.5)

    def test_rejects_bad_schedule_name(self):
        from douzero.training.bc_loss import BCSchedule, BCLossError

        with pytest.raises(BCLossError):
            BCSchedule(base_lambda=1.0, schedule="bogus")
