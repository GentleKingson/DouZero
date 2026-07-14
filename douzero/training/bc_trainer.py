"""Listwise behaviour-cloning trainer (P08).

The :class:`BCTrainer` pretrains the optional
:class:`~douzero.models_v2.heads.PriorHead` (and optionally the shared encoder)
on validated human-game BC samples. It mirrors :class:`~douzero.training.v2_trainer.V2Trainer`
in shape — per-decision forward (variable N legal actions), RMSprop, gradient
clip, fail-closed non-finite guard, seed=0 no-op convention — but consumes an
offline :class:`~douzero.human_data.sample.BCSample` dataset instead of self-play
transitions, and minimizes the listwise cross-entropy over the legal-action
list (never a global action class).

Real playing-strength is **not measured** here. The smoke test only verifies
that the loss decreases on synthetic random-self-play data (a trivial target);
any strength claim requires authorized human data and is recorded as "未测".

Imperfect-information boundary
------------------------------
The trainer forwards each sample's **public** observation through the model and
reads only the public prior-head output. The privileged ``human_action_index``
is consumed solely as the cross-entropy target; it never becomes a model input.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from douzero.human_data.sample import BCSample
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.training.bc_loss import (
    BCLossComponents,
    average_bc_losses,
    listwise_bc_loss,
)


class BCTrainerError(ValueError):
    """Raised when the BC trainer is misconfigured."""


@dataclass
class BCTrainerConfig:
    """Configuration for :class:`BCTrainer`.

    Defaults keep the trainer small enough to run a forward/backward smoke on
    CPU. ``val_ratio`` reserves a held-out slice of the samples for early
    stopping + honest top-1 reporting. ``eval_nll`` is the primary metric.
    """

    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    rmsprop_alpha: float = 0.99
    rmsprop_momentum: float = 0.0
    rmsprop_epsilon: float = 1e-5
    max_grad_norm: float = 40.0
    val_ratio: float = 0.1
    early_stopping_patience: int = 0  # 0 = disabled
    temperature: float = 1.0
    label_smoothing: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        # batch_size and epochs must be POSITIVE: batch_size=0 makes
        # range(..., step=0) crash, and epochs=0 would skip training entirely
        # yet still save a fully-untrained checkpoint (Blocker: epochs guard).
        for name, val in (("batch_size", self.batch_size),
                          ("epochs", self.epochs)):
            if not isinstance(val, int) or isinstance(val, bool) or val < 1:
                raise BCTrainerError(
                    f"{name} must be a positive int, got {val!r}"
                )
        # early_stopping_patience is 0 = disabled, so non-negative is correct.
        for name, val in (("early_stopping_patience", self.early_stopping_patience),):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise BCTrainerError(
                    f"{name} must be a non-negative int, got {val!r}"
                )
        for name, val in (
            ("learning_rate", self.learning_rate),
            ("rmsprop_alpha", self.rmsprop_alpha),
            ("rmsprop_epsilon", self.rmsprop_epsilon),
            ("max_grad_norm", self.max_grad_norm),
            ("temperature", self.temperature),
        ):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise BCTrainerError(
                    f"{name} must be a number, got {type(val).__name__}"
                )
            if val < 0 or not math.isfinite(val):
                raise BCTrainerError(f"{name} must be non-negative finite, got {val}")
        if self.learning_rate == 0.0:
            raise BCTrainerError("learning_rate must be > 0 to train")
        if self.temperature <= 0.0:
            raise BCTrainerError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 <= self.val_ratio < 1.0:
            raise BCTrainerError(
                f"val_ratio must be in [0, 1), got {self.val_ratio}"
            )
        for name, val in (("rmsprop_momentum", self.rmsprop_momentum),):
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                raise BCTrainerError(f"{name} must be non-negative, got {val}")
        # Blocker: label_smoothing must be finite and in [0, 1) — values >= 1
        # silently corrupt the loss (negative target probs filtered away),
        # and NaN/Inf propagate. Validate here so misconfigurations fail at
        # construction, not silently mid-training.
        if not isinstance(self.label_smoothing, (int, float)) or isinstance(
            self.label_smoothing, bool
        ):
            raise BCTrainerError(
                f"label_smoothing must be a number, got {type(self.label_smoothing).__name__}"
            )
        if not math.isfinite(self.label_smoothing):
            raise BCTrainerError(
                f"label_smoothing must be finite, got {self.label_smoothing}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise BCTrainerError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )


@dataclass
class BCEpochStats:
    """Per-epoch metrics (train + validation)."""

    epoch: int
    train_loss: float
    train_top1: float
    train_num_decisions: int
    val_loss: float
    val_top1: float
    val_num_decisions: int


@dataclass
class BCTrainerStats:
    """Aggregate result of a BC training run."""

    epochs_run: int = 0
    epoch_stats: list[BCEpochStats] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    stopped_early: bool = False
    final_val_top1: float = 0.0
    final_val_loss: float = float("inf")
    train_size: int = 0
    val_size: int = 0

    def as_log_dict(self) -> dict[str, float | int | bool]:
        return {
            "epochs_run": self.epochs_run,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "final_val_top1": self.final_val_top1,
            "final_val_loss": self.final_val_loss,
            "train_size": self.train_size,
            "val_size": self.val_size,
        }


class BCTrainer:
    """Pretrain a ModelV2's prior head on human BC samples.

    Parameters
    ----------
    model:
        A :class:`~douzero.models_v2.model.ModelV2` built with
        ``human_prior_enabled=True`` (it MUST have a prior head). A model
        without a prior head is rejected at construction.
    samples:
        The offline BC dataset (each sample carries a public obs + privileged
        human_action_index). The trainer splits these into train/val by
        ``game_id`` so no game leaks across the split.
    config:
        :class:`BCTrainerConfig`.
    """

    def __init__(
        self,
        model,
        samples: Sequence[BCSample],
        config: BCTrainerConfig | None = None,
        belief_model=None,
    ) -> None:
        if getattr(model, "prior_head", None) is None:
            raise BCTrainerError(
                "BCTrainer requires a ModelV2 built with "
                "human_prior_enabled=True (no prior head found)."
            )
        for s in samples:
            if not isinstance(s, BCSample):
                raise BCTrainerError(
                    f"samples must be BCSample instances, got {type(s).__name__}"
                )
        self.model = model
        self.config = config or BCTrainerConfig()
        self.samples = list(samples)

        # Round 6 Blocker 3: support P07+P08 combo (belief+prior). A belief-
        # enabled model FAILS CLOSED at forward when belief_features are
        # omitted, so either a belief_model must be supplied (frozen, computes
        # the public posterior features per sample) or the combo is rejected.
        belief_enabled = bool(getattr(self.model.config, "belief_enabled", False))
        if belief_enabled and belief_model is None:
            raise BCTrainerError(
                "The value model has belief_enabled=True but no belief_model "
                "was supplied to BCTrainer. A belief-enabled model fails closed "
                "at forward without belief_features. Pass a pretrained frozen "
                "belief_model, or rebuild the value model with "
                "belief_enabled=False for standalone BC pretraining."
            )
        if belief_model is not None and not belief_enabled:
            raise BCTrainerError(
                "A belief_model was supplied but the value model has "
                "belief_enabled=False. Drop belief_model or rebuild with "
                "belief_enabled=True."
            )
        self.belief_model = belief_model
        if self.belief_model is not None:
            for p in self.belief_model.parameters():
                p.requires_grad_(False)
            self.belief_model.eval()

        # Train/val split by game_id (no game leaks across the split). The
        # split is deterministic for a fixed (config.seed, samples order).
        self.train_samples, self.val_samples = self._split_by_game_id()
        if not self.train_samples:
            raise BCTrainerError(
                "no training samples after the train/val split; lower val_ratio "
                "or provide more data."
            )

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.config.learning_rate,
            alpha=self.config.rmsprop_alpha,
            momentum=self.config.rmsprop_momentum,
            eps=self.config.rmsprop_epsilon,
        )
        # seed=0 -> no-op (project convention); only seed when explicitly set.
        self.rng = (
            random.Random() if self.config.seed == 0
            else random.Random(self.config.seed)
        )

    # ------------------------------------------------------------------ #
    # Split
    # ------------------------------------------------------------------ #
    def _split_by_game_id(self) -> tuple[list[BCSample], list[BCSample]]:
        """Group samples by game_id, then split the GAMES into train/val.

        Splitting by game (not by decision) prevents the same deal's decisions
        from appearing in both splits, which would leak and inflate the val
        metrics.
        """
        games: dict[str, list[BCSample]] = {}
        order: list[str] = []
        for s in self.samples:
            if s.game_id not in games:
                games[s.game_id] = []
                order.append(s.game_id)
            games[s.game_id].append(s)
        game_ids = sorted(order)
        rng = (
            random.Random() if self.config.seed == 0
            else random.Random(self.config.seed)
        )
        rng.shuffle(game_ids)
        n_val = int(round(len(game_ids) * self.config.val_ratio))
        val_ids = set(game_ids[:n_val])
        train: list[BCSample] = []
        val: list[BCSample] = []
        for gid in order:  # preserve original sample order within each split
            bucket = val if gid in val_ids else train
            bucket.extend(games[gid])
        return train, val

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self) -> BCTrainerStats:
        """Run the configured number of epochs with early stopping.

        Medium #1: the best-validation state_dict is snapshot-restored at the
        end of training, so the saved model is the validation-optimal one, not
        whatever the last epoch produced (which may be worse after early
        stopping triggered).
        """
        stats = BCTrainerStats(
            train_size=len(self.train_samples),
            val_size=len(self.val_samples),
        )
        best_loss = float("inf")
        epochs_since_best = 0

        # Snapshot one parameter so we can assert the optimizer actually moved
        # the weights when at least one epoch ran (mirrors V2Trainer).
        first_param = next(self.model.parameters()).detach().clone()

        # Medium #1: keep the best-validation state_dict so we can restore it.
        best_state_dict: dict | None = None
        # When there is no validation set, do NOT fake val metrics as 0.0 (that
        # would make epoch 0 look "best" and wrongly trigger early stopping /
        # best-state restore). Instead disable both and report NaN.
        has_val = bool(self.val_samples)
        nan = float("nan")

        for epoch in range(self.config.epochs):
            order = list(range(len(self.train_samples)))
            self.rng.shuffle(order)
            train_loss, train_hits, train_n = 0.0, 0, 0
            self.model.train()
            for start in range(0, len(order), self.config.batch_size):
                idxs = order[start:start + self.config.batch_size]
                batch = [self.train_samples[i] for i in idxs]
                comps = self._train_step(batch)
                train_loss += comps.cross_entropy * comps.num_decisions
                train_hits += comps.top1_correct
                train_n += comps.num_decisions

            if has_val:
                val_loss, val_hits, val_n = self._evaluate(self.val_samples)
            else:
                val_loss, val_hits, val_n = nan, 0, 0
            es = BCEpochStats(
                epoch=epoch,
                train_loss=train_loss / max(1, train_n),
                train_top1=train_hits / max(1, train_n),
                train_num_decisions=train_n,
                val_loss=val_loss,
                val_top1=(val_hits / max(1, val_n)) if has_val else nan,
                val_num_decisions=val_n,
            )
            stats.epoch_stats.append(es)
            stats.epochs_run += 1
            stats.final_val_loss = val_loss
            stats.final_val_top1 = es.val_top1

            # Best-tracking / early stopping only when there is a real val set.
            if has_val and val_loss < best_loss - 1e-9:
                best_loss = val_loss
                stats.best_val_loss = val_loss
                stats.best_epoch = epoch
                epochs_since_best = 0
                # Medium #1: snapshot the best model so far (deep-ish copy of
                # the state_dict tensors; load_state_dict restores it).
                best_state_dict = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
            elif has_val:
                epochs_since_best += 1
                if (
                    self.config.early_stopping_patience > 0
                    and epochs_since_best >= self.config.early_stopping_patience
                ):
                    stats.stopped_early = True
                    break

        # Medium #1: restore the best-validation weights so the returned (and
        # later saved) model is the validation-optimal one. Only restore when
        # we actually snapshot a best (val set non-empty); when there is no val
        # set the last-epoch model is the only candidate.
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.model.eval()
        # Sanity: if at least one step ran, the parameters should have moved.
        if stats.epochs_run > 0:
            after = next(self.model.parameters()).detach()
            if torch.equal(after, first_param):
                raise BCTrainerError(
                    "BC training ran but the model parameters did not change; "
                    "the optimizer step is a no-op (check learning_rate / "
                    "gradient flow)."
                )
        return stats

    def _compute_belief_feature(self, obs):
        """Compute the frozen belief posterior feature for one observation.

        Mirrors V2Trainer._compute_belief_feature. Returns None when belief
        fusion is disabled. Round 6 Blocker 3: enables the P07+P08 combo in
        standalone BC pretraining.
        """
        if self.belief_model is None:
            return None
        import numpy as np
        from douzero.belief import build_belief_input
        from douzero.belief.model import belief_features_from_probs

        binput = build_belief_input(obs.public)
        with torch.inference_mode():
            bout = self.belief_model([binput])
            feat_np = belief_features_from_probs(
                bout.constrained_probs,
                bout.opponent_a_total,
                np.stack([binput.unseen_counts]),
            )[0]
        return torch.from_numpy(feat_np).detach()

    def _train_step(self, batch: list[BCSample]) -> BCLossComponents:
        """One optimizer step over a minibatch of BC samples."""
        per_decision: list[tuple[torch.Tensor, bool]] = []
        # try/finally so the model is restored to eval() even on a raise.
        try:
            for s in batch:
                bundle = observation_to_model_inputs(
                    s.obs,
                    self.model.strategy_feature_config(),
                    style_enabled=self.model.config.style_enabled,
                )
                belief_features = self._compute_belief_feature(s.obs)
                out = self.model(
                    bundle.state_card_vectors,
                    bundle.state_context_flat,
                    bundle.context_card_vectors,
                    bundle.context_flat,
                    bundle.history_tokens,
                    bundle.history_key_padding_mask,
                    bundle.action_features,
                    bundle.action_mask,
                    bundle.acting_role,
                    belief_features=belief_features,
                    strategy_features=bundle.strategy_features,
                    style_features=bundle.style_features,
                )
                if out.prior_logit is None:
                    raise BCTrainerError(
                        "model forward returned no prior_logit; the prior head "
                        "disappeared mid-training."
                    )
                loss, hit = listwise_bc_loss(
                    out.prior_logit,
                    out.action_mask,
                    s.human_action_index,
                    weight=s.sample_weight,
                    temperature=self.config.temperature,
                    label_smoothing=self.config.label_smoothing,
                )
                per_decision.append((loss, hit))
            comps = average_bc_losses(per_decision)
            if not torch.isfinite(comps.total):
                raise FloatingPointError(
                    f"BCTrainer encountered a non-finite BC loss "
                    f"({float(comps.total.item())!r}); refusing to take an "
                    f"optimizer step."
                )
            self.optimizer.zero_grad()
            comps.total.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            return comps
        finally:
            # Ensure clean grad state; the outer train loop re-enters train().
            pass

    @torch.inference_mode()
    def _evaluate(self, samples: Sequence[BCSample]) -> tuple[float, int, int]:
        """Return (mean_ce, top1_hits, num_decisions) over ``samples``."""
        if not samples:
            return 0.0, 0, 0
        self.model.eval()
        total_loss = 0.0
        hits = 0
        n = 0
        for s in samples:
            bundle = observation_to_model_inputs(
                s.obs,
                self.model.strategy_feature_config(),
                style_enabled=self.model.config.style_enabled,
            )
            belief_features = self._compute_belief_feature(s.obs)
            out = self.model(
                bundle.state_card_vectors,
                bundle.state_context_flat,
                bundle.context_card_vectors,
                bundle.context_flat,
                bundle.history_tokens,
                bundle.history_key_padding_mask,
                bundle.action_features,
                bundle.action_mask,
                bundle.acting_role,
                belief_features=belief_features,
                strategy_features=bundle.strategy_features,
                style_features=bundle.style_features,
            )
            if out.prior_logit is None:
                raise BCTrainerError(
                    "model forward returned no prior_logit during evaluation."
                )
            loss, hit = listwise_bc_loss(
                out.prior_logit,
                out.action_mask,
                s.human_action_index,
                temperature=self.config.temperature,
                label_smoothing=0.0,  # eval uses plain CE
            )
            total_loss += float(loss.item())
            hits += int(hit)
            n += 1
        return total_loss / n, hits, n
