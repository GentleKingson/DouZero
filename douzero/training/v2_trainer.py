"""Minimal single-process V2 multi-objective trainer (P06).

A complete, CPU-friendly training loop that demonstrates the P06 acceptance
criterion: "极短训练能完成一次优化且参数变化" (a very short training run
completes one optimizer step and changes the parameters).

Design
------
- Single process by default, with explicit device and mixed-precision support.
  The legacy multiprocessing path (:mod:`douzero.dmc`) is untouched.
- Self-play: a single :class:`~douzero.env.env.Env` plays
  games to terminal. Each decision is made by an epsilon-greedy policy
  over the current :class:`~douzero.models_v2.model.ModelV2` outputs
  (random for the first few episodes when the buffer is empty).
- Transition recording: every decision's :class:`ObservationV2` + chosen
  action index + acting position is appended to the current
  :class:`~douzero.training.v2_buffer.Episode`.
- Labelling: at terminal, the env's ``info['team_targets']`` (populated by
  :meth:`~douzero.env.env.Env._attach_team_perspective_labels`) is read and
  the episode's transitions receive team-perspective Monte-Carlo labels.
- Optimizer step: the trainer samples a minibatch, forwards each decision,
  gathers the chosen action's head values, concatenates them, and calls
  :meth:`MultiObjectiveLoss.forward_gathered`.

Hardening
---------
- Legacy card-play mode remains available with ``ruleset=None``. Standard
  mode runs the bidding/redeal/reveal/play state machine with a separate
  learned bidding head. It currently fails closed under DDP until the mixed
  bidding/card-play graph has been validated.
- The per-decision heads are concatenated with :func:`torch.cat` (not
  :func:`torch.stack`), so the gathered heads have shape ``(B, 1)`` as the
  loss module requires.
- The optimizer step is fail-closed: a non-finite loss or gradient raises
  :class:`FloatingPointError` BEFORE the optimizer is allowed to mutate
  parameters (PyTorch's ``clip_grad_norm_`` defaults to
  ``error_if_nonfinite=False`` which silently lets NaN/Inf through).
- The unsupported ``checkpoint_dir`` / ``save_every_steps`` options that
  were advertised in r0 are removed; they will be wired up alongside the
  P14 high-throughput trainer where save/resume actually matters.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from douzero._version import git_sha
from douzero.runtime import DistributedContext, SafeMixedPrecision

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.models_v2.model import ModelV2
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.bidding import get_bidding_obs_v2

from douzero.training.decision_policy import DecisionConfig, select_action
from douzero.training.losses import LossComponents, LossConfig, MultiObjectiveLoss
from douzero.training.v2_buffer import Episode, Transition, V2ReplayBuffer
from douzero.training.bidding import (
    BiddingPolicyConfig,
    BiddingReplayBuffer,
    BiddingTransition,
    bidding_loss,
    select_bidding_action,
)


@dataclass
class TrainerConfig:
    """Knobs for :class:`V2Trainer`.

    Defaults are tuned for a CPU smoke test (a handful of short episodes
    and a single optimizer step). Production-scale tuning belongs in
    ``configs/enhanced.yaml``, not here.

    P06 r1 removed ``checkpoint_dir`` and ``save_every_steps``: they were
    advertised but never implemented. They will be reintroduced alongside
    the P14 high-throughput trainer where checkpoint/resume is exercised.
    """

    seed: int = 0
    # Episode collection.
    max_episodes: int = 8
    max_steps_per_episode: int = 600
    # Exploration (training-time). Evaluation MUST be deterministic
    # (``epsilon=0``); the trainer's exploration is off at evaluation.
    exp_epsilon: float = 0.3
    # Minibatch / optimization.
    batch_size: int = 16
    learning_rate: float = 1e-4
    rmsprop_alpha: float = 0.99
    rmsprop_momentum: float = 0.0
    rmsprop_epsilon: float = 1e-5
    max_grad_norm: float = 40.0
    optimizer_steps: int = 1
    # Replay buffer.
    buffer_capacity: int = 4096
    # RNG for action sampling / minibatch sampling.
    rng_seed: int = 0
    # P14 learner placement. ``cpu`` preserves the P06 path; torchrun assigns
    # one CUDA device per process and passes it here for DDP.
    device: str = "cpu"
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    amp_fallback_on_nonfinite: bool = True
    # P17 belief/value optimization. Frozen preserves the P07 path exactly.
    belief_training_mode: str = "frozen"
    belief_supervised_weight: float = 0.0
    belief_alternating_interval: int = 1
    belief_supervised_batch_size: int = 16

    def __post_init__(self) -> None:
        """Validate ranges so a malformed config fails fast (P06 r2).

        Without these checks a ``batch_size=0`` would silently produce zero
        training, ``optimizer_steps=-1`` would skip the loop, and
        ``exp_epsilon=2.0`` would explore outside [0, 1]. Each guard raises
        a precise ValueError naming the field and the offending value.
        """
        import math as _math

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.optimizer_steps < 0:
            raise ValueError(f"optimizer_steps must be >= 0, got {self.optimizer_steps}")
        if not (0.0 <= self.exp_epsilon <= 1.0):
            raise ValueError(
                f"exp_epsilon must be in [0, 1], got {self.exp_epsilon}"
            )
        if not (self.learning_rate > 0.0 and _math.isfinite(self.learning_rate)):
            raise ValueError(
                f"learning_rate must be positive and finite, got {self.learning_rate}"
            )
        if not (self.max_grad_norm > 0.0 and _math.isfinite(self.max_grad_norm)):
            raise ValueError(
                f"max_grad_norm must be positive and finite, got {self.max_grad_norm}"
            )
        if self.buffer_capacity < 1:
            raise ValueError(
                f"buffer_capacity must be >= 1, got {self.buffer_capacity}"
            )
        if self.max_steps_per_episode < 1:
            raise ValueError(
                f"max_steps_per_episode must be >= 1, got {self.max_steps_per_episode}"
            )
        if self.max_episodes < 0:
            raise ValueError(f"max_episodes must be >= 0, got {self.max_episodes}")
        # P06 r3: RMSprop parameter ranges. alpha is the squared-gradient
        # running-average decay (0 <= alpha < 1); momentum is non-negative;
        # epsilon is the denominator stability term (must be positive).
        if not (_math.isfinite(self.rmsprop_alpha) and 0.0 <= self.rmsprop_alpha < 1.0):
            raise ValueError(
                f"rmsprop_alpha must be finite and in [0, 1), got {self.rmsprop_alpha}"
            )
        if not (_math.isfinite(self.rmsprop_momentum) and self.rmsprop_momentum >= 0.0):
            raise ValueError(
                f"rmsprop_momentum must be finite and >= 0, got {self.rmsprop_momentum}"
            )
        if not (_math.isfinite(self.rmsprop_epsilon) and self.rmsprop_epsilon > 0.0):
            raise ValueError(
                f"rmsprop_epsilon must be finite and > 0, got {self.rmsprop_epsilon}"
            )
        if self.belief_training_mode not in {"frozen", "joint", "alternating"}:
            raise ValueError(
                "belief_training_mode must be 'frozen', 'joint', or 'alternating'"
            )
        if (
            not _math.isfinite(self.belief_supervised_weight)
            or self.belief_supervised_weight < 0
        ):
            raise ValueError("belief_supervised_weight must be non-negative finite")
        if self.belief_alternating_interval < 1:
            raise ValueError("belief_alternating_interval must be >= 1")
        if self.belief_supervised_batch_size < 1:
            raise ValueError("belief_supervised_batch_size must be >= 1")


@dataclass
class TrainerStats:
    """Per-step / per-episode statistics surfaced by the trainer."""

    episodes_completed: int = 0
    episodes_per_team: dict[str, int] = field(default_factory=dict)
    transitions_collected: int = 0
    optimizer_steps: int = 0
    last_loss: dict[str, float] = field(default_factory=dict)
    grad_norm_last_step: float = 0.0
    p_win_mean: float = float("nan")
    p_win_std: float = float("nan")
    score_mean_avg: float = float("nan")
    opening_strategy_counts: dict[str, int] = field(default_factory=dict)
    opening_predicted_win_mean: float = float("nan")
    amp_fallbacks: int = 0
    bidding_transitions_collected: int = 0
    redeals: int = 0
    belief_phase: str = "frozen"
    belief_supervised_steps: int = 0


def _validate_training_ruleset(ruleset: RuleSet | None) -> None:
    """Allow legacy card-play mode or the explicit standard state machine."""
    if ruleset is None:
        return
    if not isinstance(ruleset, RuleSet):
        raise TypeError("ruleset must be a RuleSet or None")
    if ruleset.ruleset_id != "standard":
        raise ValueError(
            "a non-None V2Trainer ruleset must be standard; legacy card-play "
            "mode is represented by ruleset=None"
        )


class V2Trainer:
    """Single-process V2 multi-objective trainer.

    Parameters
    ----------
    model:
        The :class:`ModelV2` to optimize. The trainer takes ownership of
        gradient updates. The model is moved to ``config.device``.
    ruleset:
        :class:`RuleSet` for the env. ``None`` (default) keeps the legacy
        card-play-only env (no bidding); ``RuleSet.standard()`` enables the
        complete learned-bidding game path.
    loss_config:
        :class:`LossConfig` for the multi-objective loss.
    decision_config:
        :class:`DecisionConfig` for action selection during self-play.
    config:
        :class:`TrainerConfig` for episode/optimization knobs.
    """

    def __init__(
        self,
        model: ModelV2,
        *,
        ruleset: RuleSet | None = None,
        loss_config: LossConfig | None = None,
        decision_config: DecisionConfig | None = None,
        config: TrainerConfig | None = None,
        belief_model=None,
        belief_supervised_samples=None,
        bc_aux_samples=None,
        bc_schedule=None,
        bc_temperature: float = 1.0,
        bc_label_smoothing: float = 0.0,
        policy_pool=None,
        opponent_selectors=None,
        matchup_logger=None,
        opening_sampler=None,
        coach_label_store=None,
        policy_version: str = "current",
        policy_step: int = 0,
        bidding_policy_config: BiddingPolicyConfig | None = None,
        distributed_context: DistributedContext | None = None,
    ) -> None:
        _validate_training_ruleset(ruleset)
        self.config = config or TrainerConfig()
        loss_cfg = loss_config or LossConfig()
        # P06 r2/r7: the loss target is clamped to loss_cfg.score_clamp in
        # BOTH raw and signed_log modes (r5 made the clamp universal). If
        # that clamp does not match the model's head clamp, the target range
        # and the representable output range disagree (a head clamped to ±8
        # cannot fit a target clamped to ±32), producing a systematic fit
        # gap. Reject this at construction rather than letting it poison
        # every gradient step.
        if abs(loss_cfg.score_clamp - model.config.score_clamp) > 1e-9:
            raise ValueError(
                f"LossConfig.score_clamp ({loss_cfg.score_clamp}) does not match "
                f"model.config.score_clamp ({model.config.score_clamp}). The "
                f"loss target is clamped to score_clamp in both raw and "
                f"signed_log modes, so it must equal the model's head clamp "
                f"or the target range and the representable output range "
                f"disagree. Either align both values, or construct the model "
                f"with ModelV2Config(score_clamp=loss_cfg.score_clamp)."
            )
        # P06 r5: the score_target_transform must also agree between the
        # loss config and the model config, so a model trained with "raw"
        # is not accidentally paired with a "signed_log" loss (the model's
        # checkpoint identity records which transform its outputs were
        # trained against).
        if loss_cfg.score_target_transform != model.config.score_target_transform:
            raise ValueError(
                f"LossConfig.score_target_transform ({loss_cfg.score_target_transform!r}) "
                f"does not match model.config.score_target_transform "
                f"({model.config.score_target_transform!r}). A model trained "
                f"under one score semantics must not be optimized under another."
            )
        self.model = model
        self.ruleset = ruleset
        self.standard_mode = ruleset is not None
        if self.standard_mode and not bool(
            getattr(self.model.config, "bidding_enabled", False)
        ):
            raise ValueError(
                "standard V2 training requires model.config.bidding_enabled=True"
            )
        if not self.standard_mode and bidding_policy_config is not None:
            raise ValueError("bidding_policy_config is only valid in standard mode")
        self.bidding_policy_config = (
            bidding_policy_config or BiddingPolicyConfig()
            if self.standard_mode
            else None
        )
        # P17 belief/value training modes. Frozen keeps the established exact
        # NumPy posterior + detached fusion path. Joint uses differentiable
        # constrained PyTorch marginals so value loss reaches BeliefModel.
        # Alternating separates value-only and supervised-belief phases.
        belief_enabled = bool(getattr(self.model.config, "belief_enabled", False))
        self.belief_training_mode = self.config.belief_training_mode
        self.belief_supervised_weight = self.config.belief_supervised_weight
        self.belief_supervised_samples = list(belief_supervised_samples or [])
        if belief_enabled and belief_model is None:
            raise ValueError(
                "The value model has belief_enabled=True but no belief_model "
                "was supplied to V2Trainer. Pass a pretrained belief_model= "
                "(load it with load_belief_checkpoint)."
            )
        if belief_model is not None and not belief_enabled:
            raise ValueError(
                "A belief_model was supplied but the value model has "
                "belief_enabled=False. Drop belief_model or rebuild the value "
                "model with belief_enabled=True."
            )
        if not belief_enabled and self.belief_training_mode != "frozen":
            raise ValueError(
                f"belief_training_mode={self.belief_training_mode!r} requires "
                "model.config.belief_enabled=True"
            )
        if self.belief_training_mode == "frozen" and self.belief_supervised_weight > 0:
            raise ValueError(
                "belief_supervised_weight must be zero in frozen mode"
            )
        if self.belief_supervised_weight > 0 and not self.belief_supervised_samples:
            raise ValueError(
                "belief_supervised_weight > 0 requires belief_supervised_samples"
            )
        if self.belief_training_mode == "alternating":
            if self.belief_supervised_weight <= 0:
                raise ValueError(
                    "alternating belief training requires a positive "
                    "belief_supervised_weight"
                )
            if not self.belief_supervised_samples:
                raise ValueError(
                    "alternating belief training requires supervised samples"
                )
        if belief_model is not None:
            trainable = self.belief_training_mode != "frozen"
            for p in belief_model.parameters():
                p.requires_grad_(trainable)
            belief_model.eval()
        self.belief_model = belief_model
        # P08: optional listwise BC auxiliary loss. When the BC schedule's base
        # lambda is > 0 the trainer adds ``effective_lambda(t) * L_BC`` to the
        # multi-objective RL loss at each optimizer step, where L_BC is the
        # listwise cross-entropy over the legal-action list on a minibatch of
        # human BC samples. This is the combined RL+BC path (task 11). A model
        # without a prior head, or missing BC samples, is rejected when the
        # base lambda is > 0.
        from douzero.training.bc_loss import BCSchedule

        base_lambda = float(getattr(loss_cfg, "lambda_bc", 0.0))
        if bc_schedule is not None:
            self.bc_schedule = bc_schedule
        else:
            # Default: a constant schedule at the loss config's lambda_bc.
            self.bc_schedule = BCSchedule(base_lambda=base_lambda)
        self.bc_aux_samples = list(bc_aux_samples) if bc_aux_samples is not None else []
        # Blocker 3: temperature + label_smoothing actually reach listwise_bc_loss
        # in the RL+BC path (previously ignored, so bc.temperature/label_smoothing
        # in the YAML had no effect on the auxiliary term).
        self.bc_temperature = float(bc_temperature)
        self.bc_label_smoothing = float(bc_label_smoothing)
        if self.bc_schedule.base_lambda > 0:
            if getattr(self.model.config, "human_prior_enabled", False) is not True:
                raise ValueError(
                    "BC lambda_bc > 0 requires a value model built with "
                    "human_prior_enabled=True (no prior head found). The "
                    "BC auxiliary loss trains the prior head."
                )
            if not self.bc_aux_samples:
                raise ValueError(
                    "BC lambda_bc > 0 requires bc_aux_samples (the validated "
                    "human BC dataset). Pass bc_aux_samples= to V2Trainer, or "
                    "set lambda_bc=0 to disable the BC aux term."
                )
        elif self.bc_aux_samples:
            # BC samples supplied but lambda_bc=0 -> they are unused. Warn
            # rather than silently ignoring, so a misconfigured run is visible.
            import warnings

            warnings.warn(
                "bc_aux_samples were supplied but lambda_bc == 0; "
                "the BC auxiliary loss is disabled and the samples are unused.",
                stacklevel=2,
            )
        self.loss_fn = MultiObjectiveLoss(loss_cfg)
        self.strategy_aux_weight = sum(
            float(getattr(loss_cfg, name, 0.0))
            for name in (
                "lambda_min_turns", "lambda_regain_initiative",
                "lambda_teammate_finish", "lambda_spring", "lambda_structure",
            )
        )
        if self.strategy_aux_weight > 0 and not self.model.config.strategy_aux_enabled:
            raise ValueError(
                "non-zero strategy auxiliary weights require "
                "model.strategy_aux_enabled=true"
            )
        self.decision_config = decision_config or DecisionConfig()
        self.device = torch.device(self.config.device)
        self.distributed = distributed_context or DistributedContext(enabled=False)
        if self.distributed.enabled and self.distributed.device != self.device:
            raise ValueError(
                "distributed context device does not match TrainerConfig.device: "
                f"{self.distributed.device} != {self.device}"
            )
        if self.standard_mode and self.distributed.enabled:
            raise NotImplementedError(
                "bidding-enabled standard training is not supported under DDP: "
                "the current P14 wrapper uses a static graph while bid and play "
                "paths have different parameter usage. Run single-process until "
                "a stable combined DDP graph is validated."
            )
        if self.distributed.enabled and self.belief_training_mode != "frozen":
            raise NotImplementedError(
                "joint/alternating belief training is not supported under DDP: "
                "P14 wraps only ModelV2, so belief gradients would not be "
                "synchronized across ranks. Use frozen mode or single-process "
                "training until the belief model has its own DDP reducer."
            )
        if (
            self.distributed.enabled
            and self.model.config.human_prior_enabled
            and self.bc_schedule.base_lambda == 0
        ):
            raise ValueError(
                "DDP cannot train an enabled prior head when lambda_bc=0; "
                "disable human_prior_enabled or enable its loss"
            )
        if (
            self.distributed.enabled
            and self.model.config.strategy_aux_enabled
            and self.strategy_aux_weight == 0
        ):
            raise ValueError(
                "DDP cannot train enabled strategy auxiliary heads when all "
                "strategy auxiliary loss weights are zero"
            )
        self.model.to(self.device)
        if self.belief_model is not None:
            self.belief_model.to(self.device)
        # DDP forward is a synchronization point. Self-play control flow is
        # intentionally rank-local, so inference must bypass the wrapper and
        # call the local module directly. Optimizer closures continue to use
        # ``self.model`` so training forward/backward remains synchronized.
        self.inference_model = (
            self.model.module if self.distributed.enabled else self.model
        )
        self.mixed_precision = SafeMixedPrecision(
            self.device,
            enabled=self.config.amp_enabled,
            dtype=self.config.amp_dtype,
            fallback_on_nonfinite=self.config.amp_fallback_on_nonfinite,
        )
        # P06 r4: reject a "valid but trains nothing" configuration.
        # (a) optimizer_steps > 0 with all loss weights at 0 produces a
        #     zero-gradient step that silently changes nothing.
        # (b) optimizer_steps > 0 with buffer_capacity < batch_size means
        #     step() can never sample a minibatch and silently skips.
        active_loss = (
            loss_cfg.lambda_win + loss_cfg.lambda_score + loss_cfg.lambda_uncertainty
            + self.strategy_aux_weight
            + loss_cfg.lambda_bid_policy + loss_cfg.lambda_bid_win
            + loss_cfg.lambda_bid_score + loss_cfg.lambda_bid_regret
        )
        self.bidding_loss_weight = (
            loss_cfg.lambda_bid_policy + loss_cfg.lambda_bid_win
            + loss_cfg.lambda_bid_score + loss_cfg.lambda_bid_regret
        )
        if not self.standard_mode and self.bidding_loss_weight > 0:
            raise ValueError("bidding loss weights require standard rules")
        if (
            self.standard_mode
            and self.config.optimizer_steps > 0
            and self.bidding_loss_weight == 0
        ):
            raise ValueError(
                "standard learned-bidding training requires at least one non-zero "
                "lambda_bid_policy/lambda_bid_win/lambda_bid_score weight"
            )
        if self.config.optimizer_steps > 0 and active_loss == 0:
            raise ValueError(
                f"optimizer_steps > 0 requires at least one non-zero loss "
                f"weight (lambda_win/lambda_score/lambda_uncertainty); got "
                f"all zeros. A zero-loss training run would silently produce "
                f"no parameter change."
            )
        if self.config.optimizer_steps > 0 and self.config.buffer_capacity < self.config.batch_size:
            raise ValueError(
                f"buffer_capacity ({self.config.buffer_capacity}) must be >= "
                f"batch_size ({self.config.batch_size}) when optimizer_steps > 0; "
                f"otherwise step() can never sample a minibatch."
            )
        self._value_parameters = list(self.model.parameters())
        self._belief_parameters = (
            list(self.belief_model.parameters())
            if self.belief_model is not None else []
        )
        self._optimizer_parameters = list(self._value_parameters)
        if self.belief_training_mode != "frozen":
            self._optimizer_parameters.extend(self._belief_parameters)
        self.optimizer = torch.optim.RMSprop(
            self._optimizer_parameters,
            lr=self.config.learning_rate,
            alpha=self.config.rmsprop_alpha,
            momentum=self.config.rmsprop_momentum,
            eps=self.config.rmsprop_epsilon,
        )
        self.buffer = V2ReplayBuffer(capacity_transitions=self.config.buffer_capacity)
        self.bidding_buffer = BiddingReplayBuffer(self.config.buffer_capacity)
        # P06 r3: respect the project's seed=0 → no-op contract. When
        # rng_seed is 0, use system entropy (random.Random() with no arg)
        # so the trainer's action sampling is unseeded — matching the
        # unseeded deal shuffle and model init. Only seed the local RNG
        # when the user explicitly requested a non-zero seed.
        if self.config.rng_seed == 0:
            self.rng = random.Random()
        else:
            self.rng = random.Random(self.config.rng_seed)
        self.stats = TrainerStats(
            episodes_per_team={"landlord": 0, "farmer": 0},
            opening_strategy_counts={},
            belief_phase=self.belief_training_mode,
        )
        if coach_label_store is not None and opening_sampler is None:
            raise ValueError("coach_label_store requires an opening_sampler")
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("policy_version must be a non-empty string")
        if isinstance(policy_step, bool) or not isinstance(policy_step, int) or policy_step < 0:
            raise ValueError("policy_step must be a non-negative int")
        self.opening_sampler = opening_sampler
        self.coach_label_store = coach_label_store
        self.policy_version = policy_version
        self.policy_step = policy_step
        if opening_sampler is not None:
            sampler_policy_version = getattr(opening_sampler, "policy_version", None)
            if sampler_policy_version != policy_version:
                raise ValueError(
                    "opening_sampler.policy_version must match V2Trainer "
                    f"policy_version: sampler={sampler_policy_version!r}, "
                    f"trainer={policy_version!r}"
                )
        self._curriculum_game_index = 0
        self._opening_prediction_sum = 0.0
        self._opening_prediction_count = 0
        # P06 r3: put the model in eval mode for self-play collection.
        # ``inference_mode`` in _choose_action_index only disables autograd;
        # it does NOT switch Dropout / BatchNorm behaviour, so without
        # eval() a model with non-zero history_dropout or mlp_dropout would
        # produce non-deterministic action selection even with exp_epsilon=0.
        # step() toggles to train() for the optimizer step, then back to
        # eval() after.
        self.model.eval()
        if self.belief_model is not None:
            self.belief_model.eval()
        self.population_runner = None
        self._league_game_index = 0
        if policy_pool is not None:
            from douzero.league.policy_pool import PolicyLoaderContract
            from douzero.league.self_play import PopulationEpisodeRunner

            belief_config_hash = ""
            if self.belief_model is not None:
                belief_config_hash = self.belief_model.config.stable_hash()
            actual_runtime = PolicyLoaderContract.for_v2_runtime(
                self.model.schema,
                self.model.config,
                checkpoint_kind=policy_pool.runtime_loader.checkpoint_kind,
                loader_name=policy_pool.runtime_loader.loader_name,
                belief_config_hash=belief_config_hash,
            )
            if actual_runtime != policy_pool.runtime_loader:
                raise ValueError(
                    "policy pool runtime loader identity does not match the "
                    "V2Trainer model/schema identity"
                )
            if self.policy_version != policy_pool.current.policy_id:
                raise ValueError(
                    "V2Trainer policy_version must match the policy pool's "
                    "current policy_id so bidding and card-play transitions "
                    "carry one snapshot identity: "
                    f"trainer={self.policy_version!r}, "
                    f"pool={policy_pool.current.policy_id!r}"
                )

            self.population_runner = PopulationEpisodeRunner(
                policy_pool,
                self._choose_action_index,
                opponent_selectors=opponent_selectors,
                current_bidding_selector=(
                    self._choose_bidding_action if self.standard_mode else None
                ),
                bidding_policy_config=self.bidding_policy_config,
                ruleset=self.ruleset,
                max_steps=self.config.max_steps_per_episode,
                logger=matchup_logger,
            )

    # ------------------------------------------------------------------ #
    # Self-play episode collection
    # ------------------------------------------------------------------ #
    def collect_episodes(self, num_episodes: int | None = None) -> None:
        """Run ``num_episodes`` self-play games and add them to the buffer."""
        target = num_episodes if num_episodes is not None else self.config.max_episodes
        for _ in range(target):
            episode = self._run_one_episode()
            if episode.transitions:
                self.buffer.add_episode(episode)
            if episode.bidding_transitions:
                self.bidding_buffer.add_terminal_deal(
                    episode.bidding_transitions, episode.terminal_result
                )
            if episode.transitions or episode.bidding_transitions:
                self.stats.episodes_completed += 1
                team = episode.terminal_result.get("winner_team", "landlord")
                self.stats.episodes_per_team[team] = (
                    self.stats.episodes_per_team.get(team, 0) + 1
                )
                # Lifetime counters remain monotonic across replay eviction and
                # checkpoint resume. Buffer occupancy is available separately
                # via len(buffer); it must not overwrite restored counters.
                self.stats.transitions_collected += len(episode.transitions)
                self.stats.bidding_transitions_collected += len(
                    episode.bidding_transitions
                )
                self.stats.redeals += episode.redeal_count

    def _run_one_episode(self) -> Episode:
        """Play one game to terminal, recording decisions and labels."""
        # Freeze identity before sampling the opening. The single-process V2
        # trainer cannot optimize during a game, so this exactly identifies
        # the weights that generated the full trajectory. Keeping it on the
        # Episode also remains correct if collection and optimization are
        # interleaved between games.
        policy_version_at_start = self.policy_version
        policy_step_at_start = self.policy_step + self.stats.optimizer_steps
        opening = None
        sampling_record = None
        if self.opening_sampler is not None:
            denominator = max(1, self.config.max_episodes - 1)
            progress = min(1.0, self._curriculum_game_index / denominator)
            opening, sampling_record = self.opening_sampler.sample(
                progress=progress,
                current_policy_step=policy_step_at_start,
            )
            self._curriculum_game_index += 1
            strategy = sampling_record.selected_strategy
            self.stats.opening_strategy_counts[strategy] = (
                self.stats.opening_strategy_counts.get(strategy, 0) + 1
            )
            if sampling_record.predicted_landlord_win is not None:
                self._opening_prediction_sum += sampling_record.predicted_landlord_win
                self._opening_prediction_count += 1
                self.stats.opening_predicted_win_mean = (
                    self._opening_prediction_sum / self._opening_prediction_count
                )
        if self.population_runner is not None:
            episode, _record = self.population_runner.run(
                self._league_game_index,
                opening=opening,
                policy_version_at_start=policy_version_at_start,
                policy_step_at_start=policy_step_at_start,
            )
            self._league_game_index += 1
            if self.strategy_aux_weight > 0:
                episode.label_strategy_auxiliary(
                    node_budget=self.model.config.strategy_node_budget,
                    time_budget_ms=self.model.config.strategy_time_budget_ms,
                )
            self._record_coach_label(opening, episode)
            return episode
        env = Env(objective="adp", ruleset=self.ruleset)
        env.reset(opening=opening)
        episode = Episode(
            policy_version_at_start=policy_version_at_start,
            policy_step_at_start=policy_step_at_start,
        )
        steps = 0
        while True:
            # P06 r2: use an explicit raise, not assert — ``python -O`` strips
            # asserts, which would disable this infinite-loop guard.
            if steps >= self.config.max_steps_per_episode:
                raise RuntimeError(
                    f"episode exceeded max_steps_per_episode "
                    f"({self.config.max_steps_per_episode}); possible infinite "
                    f"loop in the env or the decision policy."
                )
            steps += 1
            if self.standard_mode and env.bidding_obs is not None:
                bid_obs = get_bidding_obs_v2(
                    env.bidding_obs,
                    ruleset=self.ruleset,
                    redeal_count=env._redeal_count,
                )
                bid, source_policy = self._choose_bidding_action(bid_obs)
                episode.bidding_transitions.append(BiddingTransition(
                    obs=bid_obs,
                    bid_action=bid,
                    policy_version=policy_version_at_start,
                    source_policy=source_policy,
                ))
                _obs_out, _reward, done, info = env.step(None, bid_value=bid)
                if done and info.get("redeal"):
                    # The abandoned deal has no landlord outcome. Never attach
                    # the later deal's terminal result to these bid decisions.
                    episode.abandoned_bidding_transitions += len(
                        episode.bidding_transitions
                    )
                    episode.bidding_transitions.clear()
                    episode.redeal_count = int(info["redeal_count"])
                    env.redeal()
                    continue
                if done:
                    raise RuntimeError(
                        "bidding ended the episode without a terminal card-play result"
                    )
                if env.bidding_obs is not None:
                    continue
                # Landlord assignment and bottom reveal completed; roles now
                # exist and the next iteration enters ordinary card play.
                continue
            position = env._acting_player_position
            infoset = env.infoset
            legal_actions = infoset.legal_actions
            if len(legal_actions) == 1:
                action = legal_actions[0]
                action_index = 0
            else:
                obs = get_obs_v2(infoset, ruleset=self.ruleset or RuleSet.legacy())
                action_index = self._choose_action_index(obs)
                action = legal_actions[action_index]
                # Record the decision (single-legal-action steps are not
                # trained on — there is nothing to learn).
                episode.transitions.append(
                    Transition(
                        obs=obs,
                        action_index=action_index,
                        position=position,
                        trace_index=len(episode.action_trace),
                        policy_id=policy_version_at_start,
                    )
                )
            episode.action_trace.append((position, tuple(sorted(action))))
            _obs_out, _reward, done, info = env.step(action)
            if done:
                episode.terminal_result = info or {}
                break
        if self.strategy_aux_weight > 0:
            episode.label_strategy_auxiliary(
                node_budget=self.model.config.strategy_node_budget,
                time_budget_ms=self.model.config.strategy_time_budget_ms,
            )
        self._record_coach_label(opening, episode)
        return episode

    def _record_coach_label(self, opening, episode: Episode) -> None:
        """Append a policy-versioned label after a sampled game completes."""

        if opening is None or self.coach_label_store is None:
            return
        if episode.redeal_count:
            # A redeal replaces the coach-selected deck. Labelling the original
            # opening with the replacement game's result would corrupt the
            # curriculum dataset.
            return
        from douzero.coach import CoachLabel

        if not episode.policy_version_at_start or episode.policy_step_at_start < 0:
            raise RuntimeError(
                "sampled episode is missing its policy identity at start"
            )

        self.coach_label_store.append(CoachLabel.from_terminal(
            opening,
            episode.terminal_result,
            policy_version=episode.policy_version_at_start,
            policy_step=episode.policy_step_at_start,
        ))

    def _compute_belief_feature(
        self, obs, *, differentiable: bool = False
    ) -> "torch.Tensor | None":
        """Compute a public-only exact or differentiable posterior feature."""
        if self.belief_model is None:
            return None
        from douzero.belief import build_belief_input
        from douzero.belief.model import (
            belief_features_from_probs,
            belief_features_from_torch_probs,
        )

        binput = build_belief_input(obs.public)
        if differentiable:
            if self.belief_training_mode == "frozen":
                raise RuntimeError(
                    "differentiable belief features were requested in frozen mode"
                )
            bout = self.belief_model([binput], differentiable=True)
            return belief_features_from_torch_probs(
                bout.require_differentiable_probs(),
                bout.opponent_a_total,
                np.stack([binput.unseen_counts]),
            )[0].to(self.device)
        with torch.inference_mode():
            bout = self.belief_model([binput])
            feat_np = belief_features_from_probs(
                bout.constrained_probs,
                bout.opponent_a_total,
                np.stack([binput.unseen_counts]),
            )[0]
        # Detached leaf tensor: the value model casts it to its trunk
        # device/dtype and the value loss updates only belief_proj.
        return torch.from_numpy(feat_np).detach().to(self.device)

    def _forward_bundle(self, bundle, belief_features=None):
        """Move one variable-action decision to the learner rank and forward."""
        return self._forward_bundle_with(
            self.model,
            bundle,
            belief_features=belief_features,
            belief_stop_gradient=self.belief_training_mode != "joint",
        )

    def _inference_forward_bundle(self, bundle, belief_features=None):
        """Forward rank-local self-play without entering a DDP sync point."""
        return self._forward_bundle_with(
            self.inference_model,
            bundle,
            belief_features=belief_features,
            belief_stop_gradient=True,
        )

    def _forward_bundle_with(
        self,
        model,
        bundle,
        belief_features=None,
        *,
        belief_stop_gradient: bool = True,
    ):
        """Move one variable-action decision to the learner device and forward."""
        bundle.to(self.device)
        if belief_features is not None:
            belief_features = belief_features.to(self.device)
        with self.mixed_precision.autocast():
            return model(
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
                belief_stop_gradient=belief_stop_gradient,
                strategy_features=bundle.strategy_features,
                style_features=bundle.style_features,
            )

    def _compute_bc_aux_loss(self, *, differentiable_belief: bool = False):
        """Sample a BC minibatch and return the averaged listwise BC loss.

        P08 task 11 (RL + BC combination). Forwards each sampled BC sample's
        PUBLIC observation through the model's prior head and computes the
        listwise cross-entropy over its legal-action list against the recorded
        human action index. The returned :class:`~douzero.training.bc_loss.BCLossComponents`
        carries the gradient; the caller scales it by ``lambda_bc`` and adds
        it to the RL loss before ``backward()``.
        """
        from douzero.training.bc_loss import average_bc_losses, listwise_bc_loss

        bs = self.config.batch_size
        # Sample with replacement when the BC dataset is smaller than the
        # batch (common in smoke tests); the loss is still an unbiased estimate.
        idxs = [self.rng.randrange(len(self.bc_aux_samples)) for _ in range(bs)]
        per_decision = []
        for i in idxs:
            s = self.bc_aux_samples[i]
            bundle = observation_to_model_inputs(
                s.obs,
                self.model.strategy_feature_config(),
                style_enabled=self.model.config.style_enabled,
            )
            # P08 Blocker 1: compute the frozen belief features for this BC
            # sample when the value model is belief-enabled. A belief-enabled
            # model FAILS CLOSED at forward when belief_features are omitted,
            # so without this the P07+P08 combo would crash at every optimizer
            # step. The features come from the PUBLIC observation only.
            belief_features = self._compute_belief_feature(
                s.obs, differentiable=differentiable_belief
            )
            out = self._forward_bundle(bundle, belief_features)
            if out.prior_logit is None:
                raise RuntimeError(
                    "BC auxiliary loss requested but the model produced no "
                    "prior_logit (the prior head disappeared mid-training)."
                )
            loss, hit = listwise_bc_loss(
                out.prior_logit.float(),
                out.action_mask,
                s.human_action_index,
                weight=s.sample_weight,
                temperature=self.bc_temperature,
                label_smoothing=self.bc_label_smoothing,
            )
            per_decision.append((loss, hit))
        return average_bc_losses(per_decision)

    def _belief_phase_for_step(self) -> str:
        """Return ``frozen``, ``joint``, ``value``, or ``belief``."""
        if self.belief_training_mode != "alternating":
            return self.belief_training_mode
        block = self.stats.optimizer_steps // self.config.belief_alternating_interval
        return "value" if block % 2 == 0 else "belief"

    def _configure_belief_phase(self, phase: str) -> None:
        """Select exactly the parameters owned by this optimizer phase."""
        value_trainable = phase != "belief"
        belief_trainable = phase in {"joint", "belief"}
        for parameter in self._value_parameters:
            parameter.requires_grad_(value_trainable)
        for parameter in self._belief_parameters:
            parameter.requires_grad_(belief_trainable)
        self.stats.belief_phase = phase
        if self.belief_model is not None:
            self.belief_model.train(belief_trainable)

    def _restore_belief_trainability(self) -> None:
        """Restore stable between-step flags after an alternating phase."""
        for parameter in self._value_parameters:
            parameter.requires_grad_(True)
        belief_trainable = self.belief_training_mode != "frozen"
        for parameter in self._belief_parameters:
            parameter.requires_grad_(belief_trainable)
        if self.belief_model is not None:
            self.belief_model.eval()

    def _compute_belief_supervised_loss(self):
        """Compute masked CE from public inputs and privileged targets.

        Privileged allocations are consumed only as loss targets. They are
        never passed to ``BeliefModel.forward`` or to the public value model.
        """
        from douzero.belief.losses import belief_loss

        batch_size = self.config.belief_supervised_batch_size
        indices = [
            self.rng.randrange(len(self.belief_supervised_samples))
            for _ in range(batch_size)
        ]
        samples = [self.belief_supervised_samples[index] for index in indices]
        output = self.belief_model(
            [sample.binput for sample in samples], differentiable=True
        )
        targets = torch.as_tensor(
            np.stack([sample.label.count_onehot for sample in samples]),
            device=self.device,
            dtype=torch.float32,
        )
        return belief_loss(output.logits, targets, output.legal)

    def _choose_action_index(self, obs) -> int:
        """Epsilon-greedy action selection over the model's valid actions."""
        if self.config.exp_epsilon > 0.0 and self.rng.random() < self.config.exp_epsilon:
            mask = obs.actions.action_mask
            valid = [i for i, m in enumerate(mask) if m]
            return self.rng.choice(valid)
        belief_features = self._compute_belief_feature(obs)
        with torch.inference_mode():
            bundle = observation_to_model_inputs(
                obs,
                self.model.strategy_feature_config(),
                style_enabled=self.model.config.style_enabled,
            )
            out = self._inference_forward_bundle(bundle, belief_features)
        return select_action(out, self.decision_config)

    def _choose_bidding_action(self, obs) -> tuple[int, str]:
        if self.bidding_policy_config is None:
            raise RuntimeError("bidding decision requested outside standard mode")

        def learned_selector(bidding_obs) -> int:
            if (
                self.config.exp_epsilon > 0.0
                and self.rng.random() < self.config.exp_epsilon
            ):
                return int(self.rng.choice(bidding_obs.legal_bids))
            with torch.inference_mode(), self.mixed_precision.autocast():
                out = self.inference_model.forward_bidding(bidding_obs)
            return out.argmax_bid()

        return select_bidding_action(
            obs,
            self.bidding_policy_config,
            self.rng,
            learned_selector,
        )

    # ------------------------------------------------------------------ #
    # Optimization (fail-closed on non-finite loss / gradient)
    # ------------------------------------------------------------------ #
    def _capture_retry_rng_state(self):
        """Capture every RNG source used by the optimizer closure."""
        return {
            "trainer": self.rng.getstate(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda" else None
            ),
        }

    def _restore_retry_rng_state(self, state) -> None:
        """Replay an AMP fallback with identical BC samples and dropout."""
        self.rng.setstate(state["trainer"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"], self.device)

    @staticmethod
    def _encode_numpy_rng_state(state) -> dict:
        return {
            "algorithm": state[0],
            "keys": state[1].tolist(),
            "position": int(state[2]),
            "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        }

    @staticmethod
    def _decode_numpy_rng_state(state: dict):
        return (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )

    def save_training_checkpoint(self, path: str) -> dict:
        """Atomically save resumable optimizer/counter/RNG and identity state."""
        from douzero.observation.bidding import (
            BIDDING_ACTION_SCHEMA_VERSION,
            BIDDING_HEAD_VERSION,
        )

        active_ruleset = self.ruleset or RuleSet.legacy()
        bidding_schema_hash = ""
        if self.model.config.bidding_enabled:
            bidding_schema_hash = self.model.bidding_schema.stable_hash()
        belief_config_hash = ""
        if self.belief_model is not None:
            belief_config_hash = self.belief_model.config.stable_hash()
        coupled_belief = self.belief_training_mode != "frozen"
        source_sha = git_sha()
        if (
            source_sha == "unknown"
            or len(source_sha) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in source_sha)
        ):
            raise RuntimeError(
                "resumable trainer checkpoints require a full source Git SHA; "
                "build from a Git checkout or set DOUZERO_GIT_SHA"
            )
        bundle = {
            # Version 1 is intentionally retained byte-for-field compatible
            # for frozen mode. Version 2 is an atomic belief+value bundle.
            "checkpoint_version": 2 if coupled_belief else 1,
            "source_git_sha": source_sha,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": asdict(self.stats),
            "policy_version": self.policy_version,
            "policy_step": self.policy_step,
            "feature_schema_hash": self.model.schema.stable_hash(),
            "model_config_hash": self.model.config.stable_hash(),
            "model_config_identity_version": self.model.config.IDENTITY_VERSION,
            "ruleset_id": active_ruleset.ruleset_id,
            "ruleset_version": active_ruleset.ruleset_version,
            "ruleset_hash": active_ruleset.stable_hash(),
            "bidding_head_version": (
                BIDDING_HEAD_VERSION if self.model.config.bidding_enabled else ""
            ),
            "bidding_action_schema": (
                BIDDING_ACTION_SCHEMA_VERSION
                if self.model.config.bidding_enabled else ""
            ),
            "bidding_feature_schema_hash": bidding_schema_hash,
            "belief_config_hash": belief_config_hash,
            "loss_config": self.loss_fn.config.to_dict(),
            "bidding_policy_config": (
                asdict(self.bidding_policy_config)
                if self.bidding_policy_config is not None else None
            ),
            "counters": {
                "curriculum_game_index": self._curriculum_game_index,
                "league_game_index": self._league_game_index,
                "opening_prediction_sum": self._opening_prediction_sum,
                "opening_prediction_count": self._opening_prediction_count,
            },
            "rng": {
                "trainer": self.rng.getstate(),
                "python": random.getstate(),
                "numpy": self._encode_numpy_rng_state(np.random.get_state()),
                "torch": torch.random.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda" else None
                ),
            },
        }
        if coupled_belief:
            bundle.update({
                "belief_training_mode": self.belief_training_mode,
                "belief_state_dict": self.belief_model.state_dict(),
                "belief_public_input_contract": "belief_input_public_v1",
                "belief_supervised_weight": self.belief_supervised_weight,
                "belief_alternating_interval": self.config.belief_alternating_interval,
                "belief_supervised_batch_size": self.config.belief_supervised_batch_size,
            })
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(bundle, temporary)
        os.replace(temporary, destination)
        identity_keys = [
            "checkpoint_version", "source_git_sha", "policy_version",
            "feature_schema_hash", "model_config_hash",
            "ruleset_id", "ruleset_version", "ruleset_hash",
            "bidding_head_version", "bidding_action_schema",
            "bidding_feature_schema_hash", "belief_config_hash",
            "loss_config", "bidding_policy_config",
        ]
        if coupled_belief:
            identity_keys.extend([
                "belief_training_mode", "belief_public_input_contract",
                "belief_supervised_weight", "belief_alternating_interval",
                "belief_supervised_batch_size",
            ])
        return {
            key: bundle[key]
            for key in identity_keys
        }

    def load_training_checkpoint(self, path: str) -> dict:
        """Strictly restore a checkpoint saved by :meth:`save_training_checkpoint`."""
        from douzero.checkpoint.io import CheckpointCompatibilityError
        from douzero.observation.bidding import (
            BIDDING_ACTION_SCHEMA_VERSION,
            BIDDING_HEAD_VERSION,
        )

        bundle = torch.load(path, map_location=self.device, weights_only=True)
        expected_version = 1 if self.belief_training_mode == "frozen" else 2
        if (
            not isinstance(bundle, dict)
            or bundle.get("checkpoint_version") != expected_version
        ):
            raise CheckpointCompatibilityError("unsupported V2 trainer checkpoint")
        active_ruleset = self.ruleset or RuleSet.legacy()
        source_sha = git_sha()
        if (
            source_sha == "unknown"
            or len(source_sha) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in source_sha)
        ):
            raise CheckpointCompatibilityError(
                "resumable trainer checkpoints require a full runtime Git SHA"
            )
        expected = {
            "source_git_sha": source_sha,
            "policy_version": self.policy_version,
            "feature_schema_hash": self.model.schema.stable_hash(),
            "model_config_hash": self.model.config.stable_hash(),
            "model_config_identity_version": self.model.config.IDENTITY_VERSION,
            "ruleset_id": active_ruleset.ruleset_id,
            "ruleset_version": active_ruleset.ruleset_version,
            "ruleset_hash": active_ruleset.stable_hash(),
            "bidding_head_version": (
                BIDDING_HEAD_VERSION if self.model.config.bidding_enabled else ""
            ),
            "bidding_action_schema": (
                BIDDING_ACTION_SCHEMA_VERSION
                if self.model.config.bidding_enabled else ""
            ),
            "bidding_feature_schema_hash": (
                self.model.bidding_schema.stable_hash()
                if self.model.config.bidding_enabled else ""
            ),
            "belief_config_hash": (
                self.belief_model.config.stable_hash()
                if self.belief_model is not None else ""
            ),
            "loss_config": self.loss_fn.config.to_dict(),
            "bidding_policy_config": (
                asdict(self.bidding_policy_config)
                if self.bidding_policy_config is not None else None
            ),
        }
        if self.belief_training_mode != "frozen":
            expected.update({
                "belief_training_mode": self.belief_training_mode,
                "belief_public_input_contract": "belief_input_public_v1",
                "belief_supervised_weight": self.belief_supervised_weight,
                "belief_alternating_interval": self.config.belief_alternating_interval,
                "belief_supervised_batch_size": self.config.belief_supervised_batch_size,
            })
        for name, value in expected.items():
            if bundle.get(name) != value:
                raise CheckpointCompatibilityError(
                    f"V2 trainer checkpoint {name} mismatch: checkpoint has "
                    f"{bundle.get(name)!r}, runtime expects {value!r}"
                )
        counters = bundle.get("counters")
        required_counters = {
            "curriculum_game_index", "league_game_index",
            "opening_prediction_sum", "opening_prediction_count",
        }
        if not isinstance(counters, dict) or set(counters) != required_counters:
            raise CheckpointCompatibilityError(
                "V2 trainer checkpoint has invalid resumable counters"
            )
        for name in (
            "curriculum_game_index", "league_game_index",
            "opening_prediction_count",
        ):
            value = counters[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointCompatibilityError(
                    f"V2 trainer checkpoint counter {name} must be non-negative int"
                )
        opening_prediction_sum = counters["opening_prediction_sum"]
        if (
            isinstance(opening_prediction_sum, bool)
            or not isinstance(opening_prediction_sum, (int, float))
            or not math.isfinite(opening_prediction_sum)
        ):
            raise CheckpointCompatibilityError(
                "V2 trainer checkpoint opening_prediction_sum must be finite"
            )
        self.model.load_state_dict(bundle["model_state_dict"], strict=True)
        if self.belief_training_mode != "frozen":
            belief_state = bundle.get("belief_state_dict")
            if not isinstance(belief_state, dict):
                raise CheckpointCompatibilityError(
                    "coupled trainer checkpoint is missing belief_state_dict"
                )
            self.belief_model.load_state_dict(belief_state, strict=True)
        self.optimizer.load_state_dict(bundle["optimizer_state_dict"])
        self.stats = TrainerStats(**bundle["stats"])
        self.policy_version = str(bundle["policy_version"])
        self.policy_step = int(bundle["policy_step"])
        self._curriculum_game_index = int(counters["curriculum_game_index"])
        self._league_game_index = int(counters["league_game_index"])
        self._opening_prediction_sum = float(counters["opening_prediction_sum"])
        self._opening_prediction_count = int(counters["opening_prediction_count"])
        rng = bundle["rng"]
        self.rng.setstate(rng["trainer"])
        random.setstate(rng["python"])
        np.random.set_state(self._decode_numpy_rng_state(rng["numpy"]))
        torch.random.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"], self.device)
        self.model.eval()
        if self.belief_model is not None:
            self.belief_model.eval()
        self._restore_belief_trainability()
        return {name: bundle[name] for name in expected}

    def step(self) -> LossComponents | None:
        """Run one optimizer step on a sampled minibatch.

        Returns the :class:`LossComponents` if a step was taken, or ``None``
        if the buffer did not have enough labelled transitions yet.

        Non-finite loss or gradients are detected before optimizer mutation.
        An AMP step retries once in float32; a float32 anomaly raises
        :class:`FloatingPointError`.
        """
        local_batch_ready = len(self.buffer) >= self.config.batch_size
        if self.bidding_loss_weight > 0:
            local_batch_ready = (
                local_batch_ready
                and len(self.bidding_buffer) >= self.config.batch_size
            )
        if not self.distributed.all_true(local_batch_ready):
            return None
        batch = self.buffer.sample_minibatch(self.config.batch_size, rng=self.rng)
        if batch is None:
            raise RuntimeError("replay buffer reported ready but returned no minibatch")
        bidding_batch = None
        if self.bidding_loss_weight > 0:
            bidding_batch = self.bidding_buffer.sample(
                self.config.batch_size, self.rng
            )
            if bidding_batch is None:
                raise RuntimeError(
                    "bidding replay reported ready but returned no minibatch"
                )

        # P06 r4: use try/finally so model.eval() + gradient cleanup are
        # guaranteed even when clip_grad_norm_(error_if_nonfinite=True) or
        # the non-finite-loss guard raises. Without this, an exception
        # leaves the model in training mode, and subsequent self-play
        # collection would run with dropout active.
        belief_phase = self._belief_phase_for_step()
        self._configure_belief_phase(belief_phase)
        differentiable_belief = self.belief_training_mode != "frozen"
        self.model.train()
        try:
            components = None
            aux_diag: dict[str, float] = {}
            bc_diag: dict[str, float] = {}
            bid_diag: dict[str, float] = {}
            belief_diag: dict[str, float] = {}
            win_logit = score_if_win = score_if_loss = None

            def loss_closure():
                nonlocal components, aux_diag, bc_diag, bid_diag, belief_diag
                nonlocal win_logit, score_if_win, score_if_loss
                gathered_win: list[torch.Tensor] = []
                gathered_siw: list[torch.Tensor] = []
                gathered_sil: list[torch.Tensor] = []
                gathered_aux: dict[str, list[torch.Tensor]] = {
                    "min_turns_after": [], "regain_initiative_logit": [],
                    "teammate_finish_logit": [], "spring_probability_logit": [],
                    "structure_cost": [],
                }
                for i, obs in enumerate(batch.observations):
                    bundle = observation_to_model_inputs(
                        obs, self.model.strategy_feature_config(),
                        style_enabled=self.model.config.style_enabled,
                    )
                    out = self._forward_bundle(
                        bundle,
                        self._compute_belief_feature(
                            obs, differentiable=differentiable_belief
                        ),
                    )
                    idx = int(batch.action_indices[i].item())
                    gathered_win.append(out.win_logit[idx : idx + 1])
                    gathered_siw.append(out.score_if_win[idx : idx + 1])
                    gathered_sil.append(out.score_if_loss[idx : idx + 1])
                    if self.model.config.strategy_aux_enabled:
                        for name in gathered_aux:
                            tensor = getattr(out, name)
                            if tensor is None:
                                raise RuntimeError(
                                    f"strategy auxiliary head {name!r} disappeared mid-training"
                                )
                            gathered_aux[name].append(tensor[idx : idx + 1])

                # Keep all numerically sensitive objectives in float32 even
                # when the model forward ran under autocast.
                win_logit = torch.cat(gathered_win, dim=0).float()
                score_if_win = torch.cat(gathered_siw, dim=0).float()
                score_if_loss = torch.cat(gathered_sil, dim=0).float()
                labels = {
                    "target_win": batch.target_win.to(self.device),
                    "target_score": batch.target_score.to(self.device),
                    "target_log_score": batch.target_log_score.to(self.device),
                }
                components = self.loss_fn.forward_gathered(
                    win_logit, score_if_win, score_if_loss, labels
                )
                total = components.total
                aux_diag = {}
                if self.strategy_aux_weight > 0:
                    from douzero.strategy.auxiliary import strategy_auxiliary_loss

                    target_names = (
                        "min_turns_after", "min_turns_exact_mask",
                        "regain_initiative", "teammate_finish",
                        "teammate_finish_mask", "spring_probability", "structure_cost",
                    )
                    targets = {
                        name: getattr(batch, f"target_{name}")
                        for name in target_names
                    }
                    if any(value is None for value in targets.values()):
                        raise RuntimeError(
                            "strategy auxiliary training requires trajectory labels; "
                            "the replay minibatch contains unlabeled transitions"
                        )
                    targets = {
                        name: value.to(self.device) for name, value in targets.items()
                    }
                    predictions = {
                        name: torch.cat(values, dim=0).float()
                        for name, values in gathered_aux.items()
                    }
                    aux_components = strategy_auxiliary_loss(
                        predictions, targets, self.loss_fn.config
                    )
                    total = total + aux_components.total
                    aux_diag = aux_components.as_log_dict()
                bid_diag = {}
                if bidding_batch is not None:
                    bid_outputs = []
                    for transition in bidding_batch.transitions:
                        with self.mixed_precision.autocast():
                            bid_outputs.append(
                                self.model.forward_bidding(transition.obs)
                            )
                    cfg = self.loss_fn.config
                    bid_components = bidding_loss(
                        bid_outputs,
                        bidding_batch,
                        lambda_policy=cfg.lambda_bid_policy,
                        lambda_landlord_win=cfg.lambda_bid_win,
                        lambda_landlord_score=cfg.lambda_bid_score,
                        lambda_regret=cfg.lambda_bid_regret,
                        score_delta=cfg.score_delta,
                        score_target_transform=cfg.score_target_transform,
                        score_clamp=cfg.score_clamp,
                    )
                    total = total + bid_components.total
                    bid_diag = bid_components.as_log_dict()
                eff_lambda = self.bc_schedule.effective_lambda(
                    self.stats.optimizer_steps
                )
                bc_diag = {}
                if eff_lambda > 0.0:
                    bc_term = self._compute_bc_aux_loss(
                        differentiable_belief=differentiable_belief
                    )
                    total = total + eff_lambda * bc_term.total
                    bc_diag = {
                        "bc_cross_entropy": bc_term.cross_entropy,
                        "bc_top1_accuracy": (
                            bc_term.top1_correct / bc_term.num_decisions
                            if bc_term.num_decisions > 0 else 0.0
                        ),
                        "bc_effective_lambda": eff_lambda,
                        "bc_num_decisions": bc_term.num_decisions,
                    }
                belief_diag = {}
                if (
                    self.belief_supervised_weight > 0
                    and belief_phase in {"joint", "belief"}
                ):
                    belief_term = self._compute_belief_supervised_loss()
                    belief_diag = belief_term.as_log_dict()
                    weighted_belief = self.belief_supervised_weight * belief_term.total
                    if belief_phase == "belief":
                        # Strict alternating phase: only the supervised belief
                        # target updates parameters; every value parameter is
                        # frozen by _configure_belief_phase.
                        total = weighted_belief
                    else:
                        total = total + weighted_belief
                return total

            step_result = self.mixed_precision.step(
                loss_closure, self.optimizer, self._optimizer_parameters,
                max_grad_norm=self.config.max_grad_norm,
                clip_grad_norm=nn.utils.clip_grad_norm_,
                collective_all_true=self.distributed.all_true,
                synchronize_abandoned_backward=self.distributed.enabled,
                capture_retry_state=self._capture_retry_rng_state,
                restore_retry_state=self._restore_retry_rng_state,
            )
            if components is None or win_logit is None:
                raise RuntimeError("optimizer closure did not produce diagnostics")
            total_loss = step_result.loss
            grad_norm = step_result.grad_norm
            self.stats.amp_fallbacks = self.mixed_precision.fallback_count
            self.stats.optimizer_steps += 1
            if belief_diag:
                self.stats.belief_supervised_steps += 1
            # Merge BC diagnostics into the last_loss log dict when active.
            loss_log = components.as_log_dict()
            loss_log["loss_total"] = float(total_loss.detach().float().item())
            if bc_diag:
                loss_log.update(bc_diag)
            if aux_diag:
                loss_log.update(aux_diag)
                # Round 6 suggestion: loss_total must reflect the ACTUAL total
                # that was back-propagated (RL + BC), not just the RL part.
                loss_log["loss_total"] = float(total_loss.detach().item())
            if bid_diag:
                loss_log.update(bid_diag)
                loss_log["loss_total"] = float(total_loss.detach().float().item())
            if belief_diag:
                loss_log.update(belief_diag)
                loss_log["belief_supervised_weight"] = self.belief_supervised_weight
                loss_log["loss_total"] = float(total_loss.detach().float().item())
            self.stats.last_loss = loss_log
            self.stats.grad_norm_last_step = float(grad_norm.detach().float().item())
            # p_win distribution diagnostics.
            with torch.no_grad():
                p = torch.sigmoid(win_logit).reshape(-1)
                self.stats.p_win_mean = float(p.mean().item())
                self.stats.p_win_std = float(p.std().item()) if p.numel() > 1 else 0.0
                sig = torch.sigmoid(win_logit.detach())
                self.stats.score_mean_avg = float(
                    (sig * score_if_win.detach() + (1 - sig) * score_if_loss.detach())
                    .mean()
                    .item()
                )
            return components
        finally:
            # Guarantee the model returns to eval mode and gradients are
            # cleared even on exception, so subsequent self-play collection
            # runs without dropout and without stale .grad accumulations.
            self.model.eval()
            self.optimizer.zero_grad(set_to_none=True)
            self._restore_belief_trainability()

    def train(self) -> TrainerStats:
        """Run the configured number of episodes + optimizer steps.

        P06 r4: raises :class:`RuntimeError` if fewer optimizer steps were
        taken than requested (e.g. not enough transitions collected to fill
        a minibatch). The caller must either collect more episodes, reduce
        ``batch_size``, or explicitly set ``optimizer_steps=0`` for a
        collect-only run.
        """
        # Watch each independently-trainable path. In particular, a bid-only
        # run must not be reported unchanged merely because the first card
        # encoder parameter correctly received no gradient.
        watched = [next(self.model.parameters())]
        if self.model.bidding_heads is not None:
            watched.append(next(self.model.bidding_heads.parameters()))
        if self.belief_model is not None and self.belief_training_mode != "frozen":
            watched.append(next(self.belief_model.parameters()))
        before = [parameter.detach().clone() for parameter in watched]
        self.collect_episodes()
        steps_taken = 0
        for _ in range(self.config.optimizer_steps):
            result = self.step()
            if result is not None:
                steps_taken += 1
        self.stats_last_run_changed = any(
            not torch.equal(snapshot, parameter.detach())
            for snapshot, parameter in zip(before, watched)
        )
        if self.config.optimizer_steps > 0 and steps_taken < self.config.optimizer_steps:
            raise RuntimeError(
                f"requested {self.config.optimizer_steps} optimizer steps but "
                f"only {steps_taken} were taken "
                f"(collected {self.stats.transitions_collected} transitions, "
                f"batch_size={self.config.batch_size}). "
                f"Collect more episodes or reduce batch_size."
            )
        return self.stats
