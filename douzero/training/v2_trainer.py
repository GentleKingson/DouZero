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
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from douzero._version import git_sha
from douzero.runtime import DistributedContext, SafeMixedPrecision

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import (
    bidding_observations_to_model_input,
    observation_batch_to_model_inputs,
    observation_to_model_inputs,
)
from douzero.models_v2.model import ModelV2
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.bidding import get_bidding_obs_v2

from douzero.training.decision_policy import DecisionConfig, select_action
from douzero.training.losses import LossComponents, LossConfig, MultiObjectiveLoss
from douzero.training.v2_buffer import (
    Episode,
    Transition,
    V2ReplayBuffer,
    action_count_bucket,
)
from douzero.training.bidding import (
    BiddingPolicyConfig,
    BiddingReplayBuffer,
    BiddingTransition,
    bidding_loss,
    select_bidding_action,
)
from douzero.training.standard_v2_contract import (
    BASE_ASYNC_PROTOCOL_VERSION,
    BASE_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION,
    BASE_EPISODE_COMMIT_SEMANTICS,
    BASE_EPISODE_TASK_SEMANTICS,
)


# Format 3 is the single-process topology contract, format 4 introduced the
# base async topology, and format 5 binds async protocol/task/commit identities.
# Formats 1/2 predated explicit topology and remain intentionally rejected.
_TRAINER_CHECKPOINT_VERSION = 5
_COMPATIBLE_TRAINER_CHECKPOINT_VERSIONS = frozenset({3, 4, 5})
_SINGLE_PROCESS_TOPOLOGY = "single_process"
_ASYNC_SINGLE_GPU_TOPOLOGY = "async_single_gpu"
_ASYNC_UNSUPPORTED_DECISION_MODES = frozenset({
    "pure_prior", "uncertainty_gated_prior",
})
_POLICY_STEP_SEMANTICS = "absolute_v1"
_SNAPSHOT_PUBLICATION_SEMANTICS = "cycle_quiescent_atomic_copy_v1"
_REQUEST_ORDERING_SEMANTICS = "policy_inference_bucket_interleaved_games_v3"
_ACTOR_RNG_RESUME_SEMANTICS = "restart_from_configured_seeds_v1"


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
    # ``None`` inherits ``batch_size`` so existing configs retain their exact
    # play/bid sample ratio until they opt into independent tuning.
    bidding_batch_size: int | None = None
    bidding_update_interval: int = 1
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
    first_bidder_mode: str = "rotate"
    v2_training_mode: str = _SINGLE_PROCESS_TOPOLOGY
    num_actors: int = 1
    games_per_actor: int = 4
    replay_schema_version: int = 1
    snapshot_publication_semantics: str = _SNAPSHOT_PUBLICATION_SEMANTICS
    request_ordering_semantics: str = _REQUEST_ORDERING_SEMANTICS
    async_protocol_version: int = BASE_ASYNC_PROTOCOL_VERSION
    compact_bidding_replay_schema_version: int = (
        BASE_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION
    )
    episode_task_semantics: str = BASE_EPISODE_TASK_SEMANTICS
    episode_commit_semantics: str = BASE_EPISODE_COMMIT_SEMANTICS

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
        if self.bidding_batch_size is None:
            self.bidding_batch_size = self.batch_size
        if (
            isinstance(self.bidding_batch_size, bool)
            or not isinstance(self.bidding_batch_size, int)
            or self.bidding_batch_size < 1
        ):
            raise ValueError(
                "bidding_batch_size must be >= 1, got "
                f"{self.bidding_batch_size}"
            )
        if (
            isinstance(self.bidding_update_interval, bool)
            or not isinstance(self.bidding_update_interval, int)
            or self.bidding_update_interval < 1
        ):
            raise ValueError("bidding_update_interval must be >= 1")
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
        if self.first_bidder_mode not in {"rotate", "seeded_random"}:
            raise ValueError("first_bidder_mode must be 'rotate' or 'seeded_random'")
        if self.v2_training_mode not in {
            _SINGLE_PROCESS_TOPOLOGY, _ASYNC_SINGLE_GPU_TOPOLOGY
        }:
            raise ValueError("unknown v2_training_mode")
        if self.num_actors < 1:
            raise ValueError("num_actors must be >= 1")
        if self.games_per_actor < 1:
            raise ValueError("games_per_actor must be >= 1")
        if self.replay_schema_version != 1:
            raise ValueError("unknown compact replay schema version")
        if self.snapshot_publication_semantics != _SNAPSHOT_PUBLICATION_SEMANTICS:
            raise ValueError("unknown snapshot publication semantics")
        if self.request_ordering_semantics != _REQUEST_ORDERING_SEMANTICS:
            raise ValueError("unknown request ordering semantics")
        if self.async_protocol_version != BASE_ASYNC_PROTOCOL_VERSION:
            raise ValueError("unknown async protocol version")
        if (
            self.compact_bidding_replay_schema_version
            != BASE_COMPACT_BIDDING_REPLAY_SCHEMA_VERSION
        ):
            raise ValueError("unknown compact bidding replay schema version")
        if self.episode_task_semantics != BASE_EPISODE_TASK_SEMANTICS:
            raise ValueError("unknown episode task semantics")
        if self.episode_commit_semantics != BASE_EPISODE_COMMIT_SEMANTICS:
            raise ValueError("unknown episode commit semantics")


@dataclass
class TrainerStats:
    """Per-step / per-episode statistics surfaced by the trainer."""

    episodes_completed: int = 0
    games_collected: int = 0
    episodes_per_team: dict[str, int] = field(default_factory=dict)
    transitions_collected: int = 0
    decisions_collected: int = 0
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
    bidding_decisions_collected: int = 0
    abandoned_bidding_transitions: int = 0
    learner_cardplay_samples: int = 0
    learner_bidding_samples: int = 0
    metrics_history_complete: bool = True
    metrics_history_source: str = "native"
    redeals: int = 0
    max_redeals_exceeded: int = 0
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
        self.async_mode = self.config.v2_training_mode == _ASYNC_SINGLE_GPU_TOPOLOGY
        if self.async_mode:
            if self.decision_config.mode in _ASYNC_UNSUPPORTED_DECISION_MODES:
                raise NotImplementedError(
                    "async_single_gpu does not publish prior_logit and rejects "
                    f"decision mode {self.decision_config.mode!r}; use a value-based "
                    "decision mode or single_process"
                )
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError(
                    "async_single_gpu requires an available CUDA device and never falls back"
                )
            if self.distributed.enabled:
                raise ValueError("async_single_gpu rejects DDP")
            unsupported = (
                self.standard_mode
                or self.belief_model is not None
                or self.bc_schedule.base_lambda > 0
                or self.model.config.style_enabled
                or self.model.config.strategy_features_enabled
                or self.model.config.strategy_aux_enabled
                or policy_pool is not None
                or opening_sampler is not None
            )
            if unsupported:
                raise NotImplementedError(
                    "async_single_gpu supports only base legacy-ruleset V2"
                )
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
            copy.deepcopy(self.model).to(self.device).eval()
            if self.async_mode
            else (self.model.module if self.distributed.enabled else self.model)
        )
        self.mixed_precision = SafeMixedPrecision(
            self.device,
            enabled=self.config.amp_enabled,
            dtype=self.config.amp_dtype,
            fallback_on_nonfinite=self.config.amp_fallback_on_nonfinite,
        )
        self._learner_gpu_seconds = 0.0
        self._last_collection_seconds = 0.0
        self._last_optimization_seconds = 0.0
        # P06 r4: reject a "valid but trains nothing" configuration.
        # (a) optimizer_steps > 0 with all loss weights at 0 produces a
        #     zero-gradient step that silently changes nothing.
        # (b) optimizer_steps > 0 with buffer_capacity < batch_size means
        #     step() can never sample a minibatch and silently skips.
        if loss_cfg.lambda_bid_regret != 0:
            raise ValueError(
                "lambda_bid_regret is unsupported by bid-policy-value-v2; "
                "per-bid regret requires a separate action-value head"
            )
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
        if (
            self.config.optimizer_steps > 0
            and self.bidding_loss_weight > 0
            and self.config.buffer_capacity < self.config.bidding_batch_size
        ):
            raise ValueError(
                f"buffer_capacity ({self.config.buffer_capacity}) must be >= "
                f"bidding_batch_size ({self.config.bidding_batch_size}) when "
                "bidding optimization is enabled"
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
        if self.async_mode:
            from douzero.training.v2_buffer import (
                CompactTensorReplayBuffer,
                compact_model_input_shapes,
            )

            self.buffer = CompactTensorReplayBuffer(
                self.config.buffer_capacity,
                expected_schema_hash=self.model.schema.stable_hash(),
                expected_tensor_shapes=compact_model_input_shapes(self.model.schema),
            )
        else:
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
        self._policy_step_origin = policy_step
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
        self._first_bidder_game_index = 0
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
                ruleset=self.ruleset,
                max_steps=self.config.max_steps_per_episode,
                logger=matchup_logger,
            )
        self._async_runtime_started = False
        if self.async_mode:
            self._async_snapshot = self.policy_step
            self._async_pending_scheduler = None
            self._async_stagers = {}
            self._reset_async_interval_metrics()

    # ------------------------------------------------------------------ #
    # Self-play episode collection
    # ------------------------------------------------------------------ #
    def _reset_async_interval_metrics(self) -> None:
        self._async_request_count = 0
        self._async_action_count = 0
        self._async_microbatch_count = 0
        self._async_claimed_count = 0
        self._async_queue_latencies_ms = []
        self._async_inference_seconds = 0.0
        self._async_segment_seconds = {
            "claim_wait": 0.0,
            "slot_read": 0.0,
            "collate": 0.0,
            "h2d": 0.0,
            "forward": 0.0,
            "d2h": 0.0,
            "publish": 0.0,
            "replay_drain": 0.0,
        }
        self._async_claim_size_histogram = {}
        self._async_bucket_histogram = {}
        self._async_batch_size_histogram = {}

    @staticmethod
    def _increment_histogram(histogram: dict[str, int], value) -> None:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1

    def _async_actor_runtime_kwargs(self) -> dict:
        """Bind the two actor RNG domains to their declared config fields."""
        return {
            "environment_seed": self.config.seed,
            "action_rng_seed": self.config.rng_seed,
            "epsilon": self.config.exp_epsilon,
            "max_steps": self.config.max_steps_per_episode,
            "decision_config": self.decision_config,
            "ruleset": None,
            "feature_schema_hash": self.model.schema.stable_hash(),
            "policy_version": self.policy_version,
            "policy_step": self._async_policy_step,
            "games_per_actor": self.config.games_per_actor,
        }

    def _start_async_runtime(self) -> None:
        from douzero.training.async_single_gpu import (
            AsyncRequestCoordinator,
            PendingRequestScheduler,
            SharedReplaySlots,
            async_actor_main,
        )

        context = __import__("multiprocessing").get_context("spawn")
        self._async_tasks = context.Queue()
        self._async_events = context.Queue()
        self._async_policy_step = context.Value("q", self.policy_step, lock=True)
        self._async_max_inference_batch = (
            self.config.num_actors * self.config.games_per_actor
        )
        self._async_coordinator = AsyncRequestCoordinator(
            self.model.schema,
            num_slots=max(self._async_max_inference_batch, 2),
            max_actions=4096,
            request_timeout_seconds=30.0,
        )
        self._async_replay_slots = SharedReplaySlots(
            self.model.schema,
            num_slots=max(
                self.config.num_actors * self.config.games_per_actor * 2,
                min(self.config.batch_size * 2, 64),
            ),
            max_actions=4096,
        )
        self._async_pending_scheduler = PendingRequestScheduler(
            max_batch_size=self._async_max_inference_batch,
            target_batch_size=4,
            max_delay_seconds=0.002,
        )
        self._async_stagers = {}
        self._async_workers = []
        for actor_id in range(self.config.num_actors):
            process = context.Process(
                target=async_actor_main,
                args=(
                    actor_id, self._async_tasks, self._async_events,
                    self._async_coordinator, self._async_replay_slots,
                ),
                kwargs=self._async_actor_runtime_kwargs(),
                name=f"douzero-v2-actor-{actor_id}",
            )
            process.start()
            self._async_workers.append(process)
        self._async_runtime_started = True
        self._async_snapshot = self.policy_step
        self._reset_async_interval_metrics()

    def _drain_async_replay(self) -> int:
        started = time.perf_counter()
        try:
            records = self._async_replay_slots.read_ready(
                self.model.schema.stable_hash(), self.policy_version
            )
            self.buffer.add_many(records)
        except BaseException as exc:
            self._async_coordinator.fail(
                f"shared replay validation failed: {type(exc).__name__}: {exc}"
            )
            raise
        finally:
            segments = getattr(self, "_async_segment_seconds", None)
            if segments is not None:
                segments["replay_drain"] += time.perf_counter() - started
        return len(records)

    def _service_async_requests(self, wait_seconds: float = 0.001) -> int:
        """Service requests and publish any main-process failure globally."""
        try:
            return self._service_async_requests_impl(wait_seconds)
        except BaseException as exc:
            self._async_coordinator.fail(
                f"main inference service failed: {type(exc).__name__}: {exc}"
            )
            raise

    def _service_async_requests_impl(self, wait_seconds: float = 0.001) -> int:
        from douzero.training.async_single_gpu import PinnedObservationBatchStager

        scheduler = self._async_pending_scheduler
        scheduled = None

        def claim(wait: float) -> int:
            claim_started = time.perf_counter()
            requests = self._async_coordinator.claim_ready(
                max_items=self._async_max_inference_batch,
                wait_seconds=wait,
            )
            self._async_segment_seconds["claim_wait"] += (
                time.perf_counter() - claim_started
            )
            if requests:
                for request in requests:
                    if request.policy_snapshot != self._async_snapshot:
                        self._async_coordinator.fail(
                            "request references an unpublished snapshot"
                        )
                        raise RuntimeError("async request policy snapshot mismatch")
                self._async_claimed_count += len(requests)
                self._increment_histogram(
                    self._async_claim_size_histogram, len(requests)
                )
                scheduler.add(requests)
            return len(requests)

        # Merge requests already visible in the coordinator before selecting
        # a retained group. This prevents an aged singleton from launching
        # just as compatible peers arrive.
        claim(0.0 if scheduler.pending_count else wait_seconds)
        scheduled = scheduler.pop_ready()
        if scheduled is None:
            claim(wait_seconds)
            scheduled = scheduler.pop_ready()
        if scheduled is None:
            return 0

        grouping_key, group = scheduled
        _snapshot, bucket = grouping_key
        capacity = (
            int(bucket)
            if isinstance(bucket, int)
            else min(
                self._async_coordinator.slots.max_actions,
                1 << (max(item.action_count for item in group) - 1).bit_length(),
            )
        )
        stager = self._async_stagers.get(capacity)
        if stager is None:
            stager = PinnedObservationBatchStager(
                self._async_coordinator.slots,
                max_batch_size=self._async_max_inference_batch,
                action_capacity=capacity,
            )
            self._async_stagers[capacity] = stager

        slot_read_started = time.perf_counter()
        batch_size = stager.gather_slots(
            [request.slot_id for request in group]
        )
        self._async_segment_seconds["slot_read"] += (
            time.perf_counter() - slot_read_started
        )
        collate_started = time.perf_counter()
        batched = stager.batch_view(
            batch_size, self.model.schema.stable_hash()
        )
        self._async_segment_seconds["collate"] += (
            time.perf_counter() - collate_started
        )

        h2d_started = torch.cuda.Event(enable_timing=True)
        h2d_finished = torch.cuda.Event(enable_timing=True)
        forward_finished = torch.cuda.Event(enable_timing=True)
        d2h_finished = torch.cuda.Event(enable_timing=True)
        h2d_started.record()
        batched.to(self.device, non_blocking=True)
        h2d_finished.record()
        with torch.inference_mode(), self.mixed_precision.autocast():
            output = self.inference_model.forward_batched(
                batched.state_card_vectors,
                batched.state_context_flat,
                batched.context_card_vectors,
                batched.context_flat,
                batched.history_tokens,
                batched.history_key_padding_mask,
                batched.action_features,
                batched.action_mask,
                batched.acting_role,
            )
            packed_output = torch.stack(
                (
                    output.win_logit.squeeze(-1),
                    output.score_if_win.squeeze(-1),
                    output.score_if_loss.squeeze(-1),
                    output.p_win.squeeze(-1),
                    output.score_mean.squeeze(-1),
                ),
                dim=-1,
            ).float().contiguous()
        forward_finished.record()
        packed_cpu = stager.stage_outputs(packed_output)
        d2h_finished.record()
        d2h_finished.synchronize()

        h2d_seconds = h2d_started.elapsed_time(h2d_finished) / 1000.0
        forward_seconds = h2d_finished.elapsed_time(forward_finished) / 1000.0
        d2h_seconds = forward_finished.elapsed_time(d2h_finished) / 1000.0
        self._async_segment_seconds["h2d"] += h2d_seconds
        self._async_segment_seconds["forward"] += forward_seconds
        self._async_segment_seconds["d2h"] += d2h_seconds
        self._async_inference_seconds += forward_seconds

        publish_started = time.perf_counter()
        slots = self._async_coordinator.slots
        for row, request in enumerate(group):
            count = request.action_count
            slots.output_values[request.slot_id, :count].copy_(
                packed_cpu[row, :count]
            )
            self._async_coordinator.complete(request.slot_id)
            if request.submitted_ns > 0:
                self._async_queue_latencies_ms.append(
                    (time.monotonic_ns() - request.submitted_ns) / 1_000_000.0
                )
        self._async_segment_seconds["publish"] += (
            time.perf_counter() - publish_started
        )
        self._async_request_count += len(group)
        self._async_action_count += sum(item.action_count for item in group)
        self._async_microbatch_count += 1
        self._increment_histogram(self._async_bucket_histogram, bucket)
        self._increment_histogram(self._async_batch_size_histogram, len(group))
        return len(group)

    def _publish_async_snapshot(self) -> None:
        """Atomically switch inference weights only at a quiescent boundary."""
        self._async_coordinator.quiesce()
        self.inference_model.load_state_dict(self.model.state_dict(), strict=True)
        self.inference_model.eval()
        torch.cuda.synchronize(self.device)
        with self._async_policy_step.get_lock():
            self._async_policy_step.value = self.policy_step
        self._async_snapshot = self.policy_step

    def _collect_episodes_async(self, target: int) -> None:
        """Collect async episodes while propagating every coordinator failure."""
        try:
            self._collect_episodes_async_impl(target)
        except BaseException as exc:
            if self._async_runtime_started:
                self._async_coordinator.fail(
                    f"main async collection failed: {type(exc).__name__}: {exc}"
                )
            raise

    def _collect_episodes_async_impl(self, target: int) -> None:
        if target < 0:
            raise ValueError("num_episodes must be non-negative")
        if not self._async_runtime_started:
            self._start_async_runtime()
        for episode_id in range(target):
            self._async_tasks.put(episode_id)
        completed = 0
        expected_transitions = 0
        received_transitions = 0
        while completed < target or received_transitions < expected_transitions:
            self._service_async_requests()
            received_transitions += self._drain_async_replay()
            for process in self._async_workers:
                if process.exitcode is not None:
                    self._async_coordinator.fail(
                        f"worker {process.name} exited with code {process.exitcode}"
                    )
                    raise RuntimeError("async V2 actor exited unexpectedly")
            try:
                event = self._async_events.get_nowait()
            except __import__("queue").Empty:
                continue
            if event[0] == "failed":
                raise RuntimeError(f"async actor {event[1]} failed: {event[2]}")
            if event[0] == "started":
                self._async_coordinator.active_games += 1
                continue
            if event[0] == "completed":
                completed += 1
                self._async_coordinator.active_games -= 1
                count = int(event[3])
                decision_count = int(event[6])
                expected_transitions += count
                self.stats.games_collected += 1
                self.stats.episodes_completed += 1
                self.stats.transitions_collected += count
                self.stats.decisions_collected += decision_count
                team = "landlord" if int(event[4]) == 0 else "farmer"
                self.stats.episodes_per_team[team] = (
                    self.stats.episodes_per_team.get(team, 0) + 1
                )
        # Queue implementations may publish the final slot just after the
        # completion event; drain until counters agree, never guess.
        deadline = time.monotonic() + self._async_coordinator.request_timeout_seconds
        while received_transitions < expected_transitions:
            received_transitions += self._drain_async_replay()
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out committing completed async episodes")
            time.sleep(0.001)

    def collect_episodes(self, num_episodes: int | None = None) -> None:
        """Run ``num_episodes`` self-play games and add them to the buffer."""
        target = num_episodes if num_episodes is not None else self.config.max_episodes
        if self.async_mode:
            self._collect_episodes_async(target)
            return
        for _ in range(target):
            episode = self._run_one_episode()
            self.stats.games_collected += 1
            self.stats.decisions_collected += len(episode.action_trace)
            self.stats.bidding_decisions_collected += (
                len(episode.bidding_transitions)
                + episode.abandoned_bidding_transitions
            )
            self.stats.abandoned_bidding_transitions += (
                episode.abandoned_bidding_transitions
            )
            self.stats.redeals += episode.redeal_count
            self.stats.max_redeals_exceeded += int(
                episode.max_redeals_exceeded
            )
            if episode.max_redeals_exceeded:
                episode.excluded_from_training = True
                episode.exclusion_reason = "redeal_cap_guard"
                # The forced landlord is a liveness fallback, not a sample from
                # the declared rules. Exclude the whole game from every target.
                episode.transitions.clear()
                episode.bidding_transitions.clear()
                continue
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

    def _reset_cycle_interval_metrics(self) -> None:
        self._learner_gpu_seconds = 0.0
        if self.async_mode:
            self._reset_async_interval_metrics()

    def _quiesce_cycle_boundary(
        self, *, consume_interval_metrics: bool
    ) -> dict[str, object]:
        """Quiesce, optionally consuming metrics for one controller cycle."""
        learner_gpu_seconds = self._learner_gpu_seconds
        if self.async_mode:
            if not self._async_runtime_started:
                status = {
                    "active_slots": 0, "in_flight_slots": 0,
                    "ready_requests": 0, "running_requests": 0,
                    "replay_occupancy": len(self.buffer),
                    "bidding_replay_occupancy": 0, "quiesce_seconds": 0.0,
                    "requests_per_microbatch": 0.0,
                    "actions_per_microbatch": 0.0,
                    "inference_queue_p50_ms": 0.0,
                    "inference_queue_p95_ms": 0.0,
                    "inference_gpu_seconds": 0.0,
                    "learner_gpu_seconds": learner_gpu_seconds,
                    "collection_seconds": getattr(
                        self, "_last_collection_seconds", 0.0
                    ),
                    "optimization_seconds": getattr(
                        self, "_last_optimization_seconds", 0.0
                    ),
                    "policy_lag": 0,
                    "games_per_actor": self.config.games_per_actor,
                    "claimed_requests": 0,
                    "pending_requests": 0,
                    "claim_size_histogram": {},
                    "inference_bucket_histogram": {},
                    "microbatch_size_histogram": {},
                }
                for name in (
                    "claim_wait", "slot_read", "collate", "h2d",
                    "forward", "d2h", "publish", "replay_drain",
                ):
                    status[f"{name}_seconds"] = 0.0
                if consume_interval_metrics:
                    self._reset_cycle_interval_metrics()
                return status
            started = time.perf_counter()
            self._drain_async_replay()
            counts = self._async_coordinator.quiesce()
            latencies = sorted(self._async_queue_latencies_ms)
            percentile = lambda q: (
                latencies[min(len(latencies) - 1, int((len(latencies) - 1) * q))]
                if latencies else 0.0
            )
            status = {
                "active_slots": counts["writing"] + counts["ready"] + counts["running"],
                "in_flight_slots": counts["ready"] + counts["running"],
                "ready_requests": counts["ready"],
                "running_requests": counts["running"],
                "replay_occupancy": len(self.buffer),
                "bidding_replay_occupancy": 0,
                "quiesce_seconds": time.perf_counter() - started,
                "requests_per_microbatch": (
                    self._async_request_count / max(1, self._async_microbatch_count)
                ),
                "actions_per_microbatch": (
                    self._async_action_count / max(1, self._async_microbatch_count)
                ),
                "inference_queue_p50_ms": percentile(0.50),
                "inference_queue_p95_ms": percentile(0.95),
                "inference_gpu_seconds": self._async_inference_seconds,
                "learner_gpu_seconds": learner_gpu_seconds,
                "collection_seconds": getattr(
                    self, "_last_collection_seconds", 0.0
                ),
                "optimization_seconds": getattr(
                    self, "_last_optimization_seconds", 0.0
                ),
                "policy_lag": self.policy_step - self._async_snapshot,
                "games_per_actor": getattr(
                    getattr(self, "config", None), "games_per_actor", 4
                ),
                "claimed_requests": getattr(self, "_async_claimed_count", 0),
                "pending_requests": getattr(
                    getattr(self, "_async_pending_scheduler", None),
                    "pending_count", 0,
                ),
                "claim_size_histogram": dict(
                    getattr(self, "_async_claim_size_histogram", {})
                ),
                "inference_bucket_histogram": dict(
                    getattr(self, "_async_bucket_histogram", {})
                ),
                "microbatch_size_histogram": dict(
                    getattr(self, "_async_batch_size_histogram", {})
                ),
            }
            segments = getattr(self, "_async_segment_seconds", {})
            for name in (
                "claim_wait", "slot_read", "collate", "h2d",
                "forward", "d2h", "publish", "replay_drain",
            ):
                status[f"{name}_seconds"] = float(segments.get(name, 0.0))
            if consume_interval_metrics:
                self._reset_cycle_interval_metrics()
            return status
        status = {
            "active_slots": 0,
            "in_flight_slots": 0,
            "ready_requests": 0,
            "running_requests": 0,
            "replay_occupancy": len(self.buffer),
            "bidding_replay_occupancy": len(self.bidding_buffer),
            "quiesce_seconds": 0.0,
            "learner_gpu_seconds": learner_gpu_seconds,
            "collection_seconds": getattr(
                self, "_last_collection_seconds", 0.0
            ),
            "optimization_seconds": getattr(
                self, "_last_optimization_seconds", 0.0
            ),
        }
        if consume_interval_metrics:
            self._reset_cycle_interval_metrics()
        return status

    def quiesce_cycle_boundary(self) -> dict[str, object]:
        """Establish a checkpoint-safe boundary and consume cycle metrics."""
        return self._quiesce_cycle_boundary(consume_interval_metrics=True)

    def runtime_metrics_snapshot(self) -> dict[str, object]:
        """Return quiescent interval diagnostics without resetting counters."""
        return self._quiesce_cycle_boundary(consume_interval_metrics=False)

    def clear_replay(self) -> None:
        """Clear card-play and bidding replay at an explicit cycle boundary."""
        self.buffer.clear()
        self.bidding_buffer.clear()

    def shutdown(self) -> None:
        """Release trainer background resources (none in single-process mode)."""
        if self.async_mode and self._async_runtime_started:
            active_exception = sys.exc_info()[0] is not None
            cleanup_error: BaseException | None = None
            alive = []
            try:
                # Wake actors blocked in acquire/wait_done/replay handoff before
                # waiting for process joins. Queue sentinels alone cannot wake
                # an actor that is already waiting on an inference request.
                self._async_coordinator.request_shutdown()
                for _ in self._async_workers:
                    self._async_tasks.put(None)
                deadline = time.monotonic() + 5.0
                for process in self._async_workers:
                    process.join(max(0.0, deadline - time.monotonic()))
                alive = [process for process in self._async_workers if process.is_alive()]
                if alive:
                    cleanup_error = RuntimeError("async actor shutdown timed out")
            except BaseException as exc:
                cleanup_error = exc
            finally:
                for process in self._async_workers:
                    if process.is_alive():
                        process.terminate()
                        process.join(1.0)
                for close in (
                    self._async_coordinator.shutdown,
                    self._async_replay_slots.close,
                    self._async_tasks.close,
                    self._async_events.close,
                ):
                    try:
                        close()
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                getattr(self, "_async_stagers", {}).clear()
                self._async_pending_scheduler = None
                self._async_runtime_started = False
            if cleanup_error is not None and not active_exception:
                raise cleanup_error
        return None

    def _run_one_episode(self) -> Episode:
        """Play one game to terminal, recording decisions and labels."""
        # Freeze identity before sampling the opening. The single-process V2
        # trainer cannot optimize during a game, so this exactly identifies
        # the weights that generated the full trajectory. Keeping it on the
        # Episode also remains correct if collection and optimization are
        # interleaved between games.
        policy_version_at_start = self.policy_version
        policy_step_at_start = max(
            self.policy_step,
            self._policy_step_origin + self.stats.optimizer_steps,
        )
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
            if episode.max_redeals_exceeded:
                episode.excluded_from_training = True
                episode.exclusion_reason = "redeal_cap_guard"
                episode.transitions.clear()
                episode.bidding_transitions.clear()
            if self.strategy_aux_weight > 0:
                episode.label_strategy_auxiliary(
                    node_budget=self.model.config.strategy_node_budget,
                    time_budget_ms=self.model.config.strategy_time_budget_ms,
                )
            self._record_coach_label(opening, episode)
            return episode
        env = Env(objective="adp", ruleset=self.ruleset)
        bidding_order = None
        if self.standard_mode and opening is None:
            seats = ["0", "1", "2"]
            if self.config.first_bidder_mode == "rotate":
                offset = self._first_bidder_game_index % len(seats)
            else:
                offset = self.rng.randrange(len(seats))
            bidding_order = seats[offset:] + seats[:offset]
            self._first_bidder_game_index += 1
        env.reset(opening=opening, bidding_order=bidding_order)
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
                if info.get("max_redeals_exceeded"):
                    # The cap fallback creates a playable guard state, not a
                    # real auction outcome. Never attach its later terminal
                    # result to the all-pass bidding decisions.
                    episode.abandoned_bidding_transitions += len(
                        episode.bidding_transitions
                    )
                    episode.bidding_transitions.clear()
                    episode.max_redeals_exceeded = True
                if done:
                    raise RuntimeError(
                        "bidding ended the episode without a terminal card-play result"
                    )
                if env.bidding_obs is not None:
                    continue
                # Landlord assignment and bottom reveal completed; roles now
                # exist and the next iteration enters ordinary card play.
                for transition in episode.bidding_transitions:
                    transition.assign_actor_role(env._env._seat_to_role)
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
                        policy_version=policy_version_at_start,
                        policy_step=policy_step_at_start,
                    )
                )
            episode.action_trace.append((position, tuple(sorted(action))))
            _obs_out, _reward, done, info = env.step(action)
            if done:
                episode.terminal_result = info or {}
                break
        if episode.max_redeals_exceeded:
            episode.excluded_from_training = True
            episode.exclusion_reason = "redeal_cap_guard"
            episode.transitions.clear()
            episode.bidding_transitions.clear()
        if self.strategy_aux_weight > 0 and not episode.excluded_from_training:
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
        if episode.redeal_count or episode.max_redeals_exceeded:
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
            # Keep the complete graph-bearing belief path in float32 even
            # under an outer CPU/CUDA autocast context. In particular, CPU
            # BF16 can quantize a small value-only update to zero on some
            # PyTorch/oneDNN combinations. The value model remains autocast;
            # only the numerically sensitive belief encoder, constrained DP,
            # and feature projection stay float32.
            with torch.autocast(device_type=self.device.type, enabled=False):
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

    def _forward_batched_bundle(self, bundle, belief_features=None):
        """Move a padded decision batch once and execute one model forward."""
        if self.distributed.enabled:
            raise RuntimeError("forward_batched is not routed through the DDP wrapper")
        bundle.to(self.device)
        if belief_features is not None:
            belief_features = belief_features.to(self.device)
        with self.mixed_precision.autocast():
            return self.model.forward_batched(
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
                belief_stop_gradient=self.belief_training_mode != "joint",
                strategy_features=bundle.strategy_features,
                style_features=bundle.style_features,
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

        if (
            self.bidding_policy_config.policy == "learned"
            and self.config.exp_epsilon > 0.0
            and self.rng.random() < self.config.exp_epsilon
        ):
            return int(self.rng.choice(obs.legal_bids)), "epsilon_random"

        def learned_selector(bidding_obs) -> int:
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

    @staticmethod
    def _state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
        """Hash tensor names, shapes, dtypes, and exact CPU bytes."""
        digest = hashlib.sha256()
        for name, tensor in sorted(state_dict.items()):
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("state_dict must map strings to tensors")
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _trainer_config_identity(self) -> tuple[dict, str]:
        payload = asdict(self.config)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return payload, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _v4_trainer_config_identity(self) -> tuple[dict, str]:
        payload = asdict(self.config)
        for name in (
            "bidding_batch_size",
            "bidding_update_interval",
            "async_protocol_version",
            "compact_bidding_replay_schema_version",
            "episode_task_semantics",
            "episode_commit_semantics",
        ):
            payload.pop(name, None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return payload, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _pre_m1_v5_trainer_config_identity(self) -> tuple[dict, str]:
        """Identity used by format-5 checkpoints before M1 batch controls."""

        payload = asdict(self.config)
        payload.pop("bidding_batch_size", None)
        payload.pop("bidding_update_interval", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return payload, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _v3_trainer_config_identity(self) -> tuple[dict, str]:
        payload, _ = self._v4_trainer_config_identity()
        for name in (
            "v2_training_mode", "num_actors", "games_per_actor",
            "replay_schema_version",
            "snapshot_publication_semantics", "request_ordering_semantics",
        ):
            payload.pop(name, None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return payload, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _restore_checkpoint_stats(
        self, payload: object, *, checkpoint_version: int
    ) -> TrainerStats:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint stats must be an object")
        field_names = {item.name for item in fields(TrainerStats)}
        unknown = set(payload) - field_names
        if unknown:
            raise ValueError(
                "checkpoint stats contain unknown fields: "
                + ", ".join(sorted(unknown))
            )

        new_counters = {
            "games_collected",
            "bidding_decisions_collected",
            "abandoned_bidding_transitions",
            "learner_cardplay_samples",
            "learner_bidding_samples",
        }
        history_fields = {
            "metrics_history_complete",
            "metrics_history_source",
        }
        present_new = new_counters & set(payload)
        if present_new and present_new != new_counters:
            raise ValueError(
                "checkpoint stats contain a partial benchmark counter set"
            )
        if checkpoint_version in {4, 5}:
            required = field_names - new_counters - history_fields
            missing = required - set(payload)
            if missing:
                raise ValueError(
                    "checkpoint stats are missing required fields: "
                    + ", ".join(sorted(missing))
                )
        if checkpoint_version == 5 and not present_new:
            raise ValueError("format 5 checkpoint is missing benchmark counters")

        migrated = asdict(TrainerStats())
        migrated.update(payload)
        if not present_new:
            optimizer_steps = int(migrated["optimizer_steps"])
            batch_samples = optimizer_steps * int(self.config.batch_size)
            migrated.update({
                "games_collected": int(migrated["episodes_completed"]),
                "bidding_decisions_collected": int(
                    migrated["bidding_transitions_collected"]
                ),
                "abandoned_bidding_transitions": 0,
                "learner_cardplay_samples": batch_samples,
                "learner_bidding_samples": (
                    batch_samples if self.bidding_loss_weight > 0 else 0
                ),
            })
            if checkpoint_version == 4:
                if migrated["bidding_transitions_collected"] != 0:
                    raise ValueError(
                        "format 4 async checkpoint cannot contain bidding samples"
                    )
                migrated["metrics_history_complete"] = True
                migrated["metrics_history_source"] = "migrated_v4_exact"
            else:
                migrated["metrics_history_complete"] = False
                migrated["metrics_history_source"] = "migrated_v3_partial"
        else:
            migrated.setdefault("metrics_history_complete", True)
            migrated.setdefault("metrics_history_source", "native")

        if not isinstance(migrated["metrics_history_complete"], bool):
            raise ValueError("metrics_history_complete must be bool")
        source = migrated["metrics_history_source"]
        if not isinstance(source, str) or not source:
            raise ValueError("metrics_history_source must be non-empty text")
        for name in (
            "episodes_completed",
            "games_collected",
            "transitions_collected",
            "decisions_collected",
            "optimizer_steps",
            "bidding_transitions_collected",
            "bidding_decisions_collected",
            "abandoned_bidding_transitions",
            "learner_cardplay_samples",
            "learner_bidding_samples",
        ):
            value = migrated[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"checkpoint stat {name} must be non-negative int")
        if migrated["games_collected"] < migrated["episodes_completed"]:
            raise ValueError("games_collected cannot be below episodes_completed")
        if (
            migrated["bidding_decisions_collected"]
            != migrated["bidding_transitions_collected"]
            + migrated["abandoned_bidding_transitions"]
        ):
            raise ValueError(
                "bidding decision counters do not reconcile with transitions"
            )
        return TrainerStats(**migrated)

    def _require_single_process_checkpoint_io(self, operation: str) -> None:
        """Reject distributed trainer save/resume before touching the path."""
        if self.distributed.enabled or self.distributed.world_size != 1:
            raise NotImplementedError(
                f"trainer checkpoint {operation} is single-process only; "
                "distributed resume/checkpoint publication is not implemented"
            )

    def save_training_checkpoint(
        self, path: str, *, long_running_state: dict | None = None
    ) -> dict:
        """Atomically save resumable optimizer/counter/RNG and identity state."""
        self._require_single_process_checkpoint_io("save")
        if self.async_mode:
            self._quiesce_cycle_boundary(consume_interval_metrics=False)
        from douzero.observation.bidding import (
            BIDDING_ACTION_SCHEMA_VERSION,
            BIDDING_HEAD_VERSION,
        )

        active_ruleset = self.ruleset or RuleSet.legacy()
        bidding_schema_hash = ""
        if self.model.config.bidding_enabled:
            bidding_schema_hash = self.model.bidding_schema.stable_hash()
        belief_config_hash = ""
        belief_state_dict = None
        belief_state_hash = ""
        if self.belief_model is not None:
            belief_config_hash = self.belief_model.config.stable_hash()
            belief_state_dict = self.belief_model.state_dict()
            belief_state_hash = self._state_dict_hash(belief_state_dict)
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
        checkpoint_version = (
            3 if self.config.v2_training_mode == _SINGLE_PROCESS_TOPOLOGY
            else _TRAINER_CHECKPOINT_VERSION
        )
        if checkpoint_version == 3:
            trainer_config, trainer_config_hash = self._v3_trainer_config_identity()
        else:
            trainer_config, trainer_config_hash = self._trainer_config_identity()
        bundle = {
            "checkpoint_version": checkpoint_version,
            "training_topology": self.config.v2_training_mode,
            "training_world_size": 1,
            "source_git_sha": source_sha,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": asdict(self.stats),
            "policy_version": self.policy_version,
            "policy_step": self.policy_step,
            "policy_step_semantics": _POLICY_STEP_SEMANTICS,
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
            "belief_state_dict": belief_state_dict,
            "belief_state_dict_hash": belief_state_hash,
            "trainer_config": trainer_config,
            "trainer_config_hash": trainer_config_hash,
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
                "first_bidder_game_index": self._first_bidder_game_index,
            },
            "mixed_precision": self.mixed_precision.state_dict(),
            "replay_resume_policy": "flushed_checkpoint_boundary_v1",
            "long_running_state": long_running_state,
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
                "belief_public_input_contract": "belief_input_public_v1",
                "belief_supervised_weight": self.belief_supervised_weight,
                "belief_alternating_interval": self.config.belief_alternating_interval,
                "belief_supervised_batch_size": self.config.belief_supervised_batch_size,
            })
        if checkpoint_version >= 4:
            bundle.update({
                "num_actors": self.config.num_actors,
                "games_per_actor": self.config.games_per_actor,
                "replay_schema_version": self.config.replay_schema_version,
                "snapshot_publication_semantics": self.config.snapshot_publication_semantics,
                "request_ordering_semantics": self.config.request_ordering_semantics,
                "actor_rng_resume_semantics": _ACTOR_RNG_RESUME_SEMANTICS,
            })
        if checkpoint_version >= 5:
            bundle.update({
                "async_protocol_version": self.config.async_protocol_version,
                "compact_bidding_replay_schema_version": (
                    self.config.compact_bidding_replay_schema_version
                ),
                "episode_task_semantics": self.config.episode_task_semantics,
                "episode_commit_semantics": self.config.episode_commit_semantics,
            })
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{id(bundle)}.tmp"
        )
        try:
            torch.save(bundle, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        # Replay observations contain rich Python objects that are deliberately
        # excluded from the weights-only checkpoint. A successful save defines
        # an explicit empty-replay boundary for both the continuing trainer and
        # a fresh resumed process, preserving deterministic N+M semantics.
        self.buffer.clear()
        self.bidding_buffer.clear()
        identity_keys = [
            "checkpoint_version", "training_topology", "training_world_size",
            "source_git_sha", "policy_version",
            "policy_step_semantics",
            "feature_schema_hash", "model_config_hash",
            "ruleset_id", "ruleset_version", "ruleset_hash",
            "bidding_head_version", "bidding_action_schema",
            "bidding_feature_schema_hash", "belief_config_hash",
            "belief_state_dict_hash", "trainer_config", "trainer_config_hash",
            "replay_resume_policy",
            "loss_config", "bidding_policy_config",
        ]
        if coupled_belief:
            identity_keys.extend([
                "belief_training_mode", "belief_public_input_contract",
                "belief_supervised_weight", "belief_alternating_interval",
                "belief_supervised_batch_size",
            ])
        if checkpoint_version >= 4:
            identity_keys.extend([
                "num_actors", "games_per_actor", "replay_schema_version",
                "snapshot_publication_semantics", "request_ordering_semantics",
                "actor_rng_resume_semantics",
            ])
        if checkpoint_version >= 5:
            identity_keys.extend([
                "async_protocol_version",
                "compact_bidding_replay_schema_version",
                "episode_task_semantics",
                "episode_commit_semantics",
            ])
        identity = {
            key: bundle[key]
            for key in identity_keys
        }
        identity["long_running_state"] = long_running_state
        return identity

    def load_training_checkpoint(self, path: str) -> dict:
        """Strictly restore a checkpoint saved by :meth:`save_training_checkpoint`."""
        self._require_single_process_checkpoint_io("resume")
        if self.async_mode and self._async_runtime_started:
            raise RuntimeError(
                "async checkpoint resume requires a fresh V2Trainer before "
                "Actor startup; live Actors retain process-local RNG state"
            )
        from douzero.checkpoint.io import CheckpointCompatibilityError
        from douzero.observation.bidding import (
            BIDDING_ACTION_SCHEMA_VERSION,
            BIDDING_HEAD_VERSION,
        )

        bundle = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(bundle, dict):
            raise CheckpointCompatibilityError("unsupported V2 trainer checkpoint")
        checkpoint_version = bundle.get("checkpoint_version")
        if checkpoint_version not in _COMPATIBLE_TRAINER_CHECKPOINT_VERSIONS:
            raise CheckpointCompatibilityError("unsupported V2 trainer checkpoint")
        if checkpoint_version == 3 and self.config.v2_training_mode != _SINGLE_PROCESS_TOPOLOGY:
            raise CheckpointCompatibilityError(
                "v3 checkpoints are single_process only and cannot resume async_single_gpu"
            )
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
            "training_topology": self.config.v2_training_mode,
            "training_world_size": 1,
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
            "replay_resume_policy": "flushed_checkpoint_boundary_v1",
        }
        trainer_config, trainer_config_hash = self._trainer_config_identity()
        if checkpoint_version == 3:
            trainer_config, trainer_config_hash = self._v3_trainer_config_identity()
        elif checkpoint_version == 4:
            trainer_config, trainer_config_hash = self._v4_trainer_config_identity()
        elif (
            checkpoint_version == 5
            and isinstance(bundle.get("trainer_config"), dict)
            and "bidding_batch_size" not in bundle["trainer_config"]
            and "bidding_update_interval" not in bundle["trainer_config"]
        ):
            if (
                self.config.bidding_batch_size != self.config.batch_size
                or self.config.bidding_update_interval != 1
            ):
                raise CheckpointCompatibilityError(
                    "pre-M1 format 5 checkpoint requires bidding_batch_size "
                    "to inherit batch_size and bidding_update_interval=1"
                )
            trainer_config, trainer_config_hash = (
                self._pre_m1_v5_trainer_config_identity()
            )
        expected["trainer_config"] = trainer_config
        expected["trainer_config_hash"] = trainer_config_hash
        if checkpoint_version >= 4:
            expected.update({
                "num_actors": self.config.num_actors,
                "games_per_actor": self.config.games_per_actor,
                "replay_schema_version": self.config.replay_schema_version,
                "snapshot_publication_semantics": self.config.snapshot_publication_semantics,
                "request_ordering_semantics": self.config.request_ordering_semantics,
                "actor_rng_resume_semantics": _ACTOR_RNG_RESUME_SEMANTICS,
            })
        if checkpoint_version >= 5:
            expected.update({
                "async_protocol_version": self.config.async_protocol_version,
                "compact_bidding_replay_schema_version": (
                    self.config.compact_bidding_replay_schema_version
                ),
                "episode_task_semantics": self.config.episode_task_semantics,
                "episode_commit_semantics": self.config.episode_commit_semantics,
            })
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
        policy_step_semantics = bundle.get(
            "policy_step_semantics", "base_plus_optimizer_steps_v1"
        )
        if policy_step_semantics not in {
            _POLICY_STEP_SEMANTICS, "base_plus_optimizer_steps_v1"
        }:
            raise CheckpointCompatibilityError(
                "V2 trainer checkpoint has unknown policy_step semantics"
            )
        counters = bundle.get("counters")
        required_counters = {
            "curriculum_game_index", "league_game_index",
            "opening_prediction_sum", "opening_prediction_count",
            "first_bidder_game_index",
        }
        if not isinstance(counters, dict) or set(counters) != required_counters:
            raise CheckpointCompatibilityError(
                "V2 trainer checkpoint has invalid resumable counters"
            )
        for name in (
            "curriculum_game_index", "league_game_index",
            "opening_prediction_count",
            "first_bidder_game_index",
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
        belief_state = bundle.get("belief_state_dict")
        belief_state_hash = bundle.get("belief_state_dict_hash", "")
        if self.belief_model is not None:
            if not isinstance(belief_state, dict) or (
                self._state_dict_hash(belief_state) != belief_state_hash
            ):
                raise CheckpointCompatibilityError(
                    "trainer checkpoint belief weights or provenance are invalid"
                )
            if (
                self.belief_training_mode == "frozen"
                and self._state_dict_hash(self.belief_model.state_dict())
                != belief_state_hash
            ):
                raise CheckpointCompatibilityError(
                    "frozen belief weights do not match the checkpoint"
                )
        elif belief_state is not None or belief_state_hash:
            raise CheckpointCompatibilityError(
                "belief-disabled trainer checkpoint carries belief state"
            )
        try:
            restored_stats = self._restore_checkpoint_stats(
                bundle["stats"], checkpoint_version=checkpoint_version
            )
            rng = bundle["rng"]
            if not isinstance(rng, dict) or set(rng) != {
                "trainer", "python", "numpy", "torch", "cuda"
            }:
                raise ValueError("checkpoint RNG state has an invalid field set")
            probe_rng = random.Random()
            probe_rng.setstate(rng["trainer"])
            probe_python_rng = random.Random()
            probe_python_rng.setstate(rng["python"])
            probe_numpy = self._decode_numpy_rng_state(rng["numpy"])
            np.random.RandomState().set_state(probe_numpy)
            probe_torch_rng = torch.Generator(device="cpu")
            probe_torch_rng.set_state(rng["torch"].cpu())
            cuda_rng_state = rng["cuda"]
            if self.device.type == "cuda":
                if not isinstance(cuda_rng_state, torch.Tensor):
                    raise ValueError("CUDA trainer resume requires CUDA RNG state")
                # ``map_location=self.device`` also moves serialized RNG state.
                # PyTorch generators require their opaque state as a CPU
                # ByteTensor even when the generator itself targets CUDA.
                cuda_rng_state = cuda_rng_state.cpu()
                probe_cuda_rng = torch.Generator(device=self.device)
                probe_cuda_rng.set_state(cuda_rng_state)
            elif cuda_rng_state is not None:
                raise ValueError("CPU trainer checkpoint must not carry CUDA RNG state")
            policy_step = bundle["policy_step"]
            if (
                isinstance(policy_step, bool)
                or not isinstance(policy_step, int)
                or policy_step < 0
            ):
                raise ValueError("checkpoint policy_step must be a non-negative int")
            if policy_step_semantics == "base_plus_optimizer_steps_v1":
                policy_step += restored_stats.optimizer_steps
            temp_model, temp_belief, temp_optimizer, temp_mixed = copy.deepcopy((
                self.model, self.belief_model, self.optimizer, self.mixed_precision
            ))
            temp_model.load_state_dict(bundle["model_state_dict"], strict=True)
            if temp_belief is not None and self.belief_training_mode != "frozen":
                temp_belief.load_state_dict(belief_state, strict=True)
            temp_optimizer.load_state_dict(bundle["optimizer_state_dict"])
            temp_mixed.load_state_dict(bundle["mixed_precision"])
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise CheckpointCompatibilityError(
                f"V2 trainer checkpoint state cannot be restored atomically: {exc}"
            ) from exc

        # All compatibility and load operations above target temporary objects.
        # The live trainer is mutated only after that complete dry run succeeds.
        original = copy.deepcopy((
            self.model.state_dict(),
            self.belief_model.state_dict() if self.belief_model is not None else None,
            self.optimizer.state_dict(),
            self.mixed_precision.state_dict(),
        ))
        try:
            self.model.load_state_dict(bundle["model_state_dict"], strict=True)
            if self.belief_model is not None and self.belief_training_mode != "frozen":
                self.belief_model.load_state_dict(belief_state, strict=True)
            self.optimizer.load_state_dict(bundle["optimizer_state_dict"])
            self.mixed_precision.load_state_dict(bundle["mixed_precision"])
        except Exception:
            self.model.load_state_dict(original[0], strict=True)
            if self.belief_model is not None and original[1] is not None:
                self.belief_model.load_state_dict(original[1], strict=True)
            self.optimizer.load_state_dict(original[2])
            self.mixed_precision.load_state_dict(original[3])
            raise
        self.stats = restored_stats
        self.buffer.clear()
        self.bidding_buffer.clear()
        self.policy_version = str(bundle["policy_version"])
        self.policy_step = policy_step
        self._policy_step_origin = policy_step - restored_stats.optimizer_steps
        self._curriculum_game_index = int(counters["curriculum_game_index"])
        self._league_game_index = int(counters["league_game_index"])
        self._opening_prediction_sum = float(counters["opening_prediction_sum"])
        self._opening_prediction_count = int(counters["opening_prediction_count"])
        self._first_bidder_game_index = int(counters["first_bidder_game_index"])
        if self.async_mode:
            self.inference_model.load_state_dict(self.model.state_dict(), strict=True)
            self.inference_model.eval()
            self._async_snapshot = self.policy_step
            if self._async_runtime_started:
                with self._async_policy_step.get_lock():
                    self._async_policy_step.value = self.policy_step
        self.rng.setstate(rng["trainer"])
        random.setstate(rng["python"])
        np.random.set_state(self._decode_numpy_rng_state(rng["numpy"]))
        torch.random.set_rng_state(rng["torch"].cpu())
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
        self.model.eval()
        if self.belief_model is not None:
            self.belief_model.eval()
        self._restore_belief_trainability()
        identity = {name: bundle[name] for name in expected}
        identity["checkpoint_version"] = checkpoint_version
        identity["policy_step_semantics"] = policy_step_semantics
        identity["long_running_state"] = bundle.get("long_running_state")
        return identity

    def step(self) -> LossComponents | None:
        """Run one optimizer step on a sampled minibatch.

        Returns the :class:`LossComponents` if a step was taken, or ``None``
        if the buffer did not have enough labelled transitions yet.

        Non-finite loss or gradients are detected before optimizer mutation.
        An AMP step retries once in float32; a float32 anomaly raises
        :class:`FloatingPointError`.
        """
        local_batch_ready = len(self.buffer) >= self.config.batch_size
        bidding_update_due = (
            self.bidding_loss_weight > 0
            and self.stats.optimizer_steps % self.config.bidding_update_interval == 0
        )
        if bidding_update_due:
            local_batch_ready = (
                local_batch_ready
                and len(self.bidding_buffer) >= self.config.bidding_batch_size
            )
        if not self.distributed.all_true(local_batch_ready):
            return None
        batch = self.buffer.sample_minibatch(self.config.batch_size, rng=self.rng)
        if batch is None:
            raise RuntimeError("replay buffer reported ready but returned no minibatch")
        bidding_batch = None
        if bidding_update_due:
            bidding_batch = self.bidding_buffer.sample(
                self.config.bidding_batch_size, self.rng
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
                gathered_aux: dict[str, torch.Tensor] = {}
                if not self.distributed.enabled:
                    from douzero.models_v2.batch import model_input_bundles_to_batch

                    output_names = (
                        "win_logit", "score_if_win", "score_if_loss",
                        "min_turns_after", "regain_initiative_logit",
                        "teammate_finish_logit", "spring_probability_logit",
                        "structure_cost",
                    )
                    rows: dict[str, list[torch.Tensor | None]] = {
                        name: [None] * len(batch.action_indices)
                        for name in output_names
                    }
                    groups: dict[int | str, list[int]] = {}
                    for index in range(len(batch.action_indices)):
                        action_count = (
                            len(batch.observations[index].actions.legal_actions)
                            if batch.model_inputs is None
                            else int(batch.model_inputs[index].action_features.shape[0])
                        )
                        groups.setdefault(
                            action_count_bucket(action_count), []
                        ).append(index)

                    for indices in groups.values():
                        chosen = batch.action_indices[indices]
                        if batch.model_inputs is None:
                            observations = [batch.observations[i] for i in indices]
                            bundle = observation_batch_to_model_inputs(
                                observations,
                                chosen,
                                strategy_config=self.model.strategy_feature_config(),
                                style_enabled=self.model.config.style_enabled,
                            )
                            belief_values = [
                                self._compute_belief_feature(
                                    obs, differentiable=differentiable_belief
                                )
                                for obs in observations
                            ]
                            belief_features = (
                                torch.stack(belief_values)
                                if belief_values and belief_values[0] is not None
                                else None
                            )
                        else:
                            bundle = model_input_bundles_to_batch(
                                [batch.model_inputs[i] for i in indices], chosen
                            )
                            # Compact replay is currently async-only, and async
                            # startup rejects every belief mode. It intentionally
                            # stores model tensors rather than ObservationV2.
                            belief_features = None
                        out = self._forward_batched_bundle(bundle, belief_features)
                        gathered = out.gather_chosen(bundle.chosen_action_index)
                        for name in output_names:
                            value = gathered[name]
                            if value is None:
                                continue
                            for row, original_index in enumerate(indices):
                                rows[name][original_index] = value[row : row + 1].float()

                    def combine(name: str) -> torch.Tensor:
                        values = rows[name]
                        if any(value is None for value in values):
                            raise RuntimeError(
                                f"batched learner output {name!r} is partially populated"
                            )
                        return torch.cat(values, dim=0)  # type: ignore[arg-type]

                    win_logit = combine("win_logit")
                    score_if_win = combine("score_if_win")
                    score_if_loss = combine("score_if_loss")
                    for name in output_names[3:]:
                        if all(value is not None for value in rows[name]):
                            gathered_aux[name] = combine(name)
                else:
                    gathered_win: list[torch.Tensor] = []
                    gathered_siw: list[torch.Tensor] = []
                    gathered_sil: list[torch.Tensor] = []
                    gathered_aux_lists: dict[str, list[torch.Tensor]] = {
                        "min_turns_after": [], "regain_initiative_logit": [],
                        "teammate_finish_logit": [], "spring_probability_logit": [],
                        "structure_cost": [],
                    }
                    for i, obs in enumerate(batch.observations):
                        scalar_bundle = observation_to_model_inputs(
                            obs, self.model.strategy_feature_config(),
                            style_enabled=self.model.config.style_enabled,
                        )
                        scalar_out = self._forward_bundle(
                            scalar_bundle,
                            self._compute_belief_feature(
                                obs, differentiable=differentiable_belief
                            ),
                        )
                        idx = int(batch.action_indices[i].item())
                        gathered_win.append(scalar_out.win_logit[idx : idx + 1])
                        gathered_siw.append(scalar_out.score_if_win[idx : idx + 1])
                        gathered_sil.append(scalar_out.score_if_loss[idx : idx + 1])
                        if self.model.config.strategy_aux_enabled:
                            for name in gathered_aux_lists:
                                tensor = getattr(scalar_out, name)
                                if tensor is None:
                                    raise RuntimeError(
                                        f"strategy auxiliary head {name!r} disappeared mid-training"
                                    )
                                gathered_aux_lists[name].append(tensor[idx : idx + 1])
                    win_logit = torch.cat(gathered_win, dim=0).float()
                    score_if_win = torch.cat(gathered_siw, dim=0).float()
                    score_if_loss = torch.cat(gathered_sil, dim=0).float()
                    gathered_aux = {
                        name: torch.cat(values, dim=0).float()
                        for name, values in gathered_aux_lists.items()
                        if values
                    }
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
                    predictions = gathered_aux
                    aux_components = strategy_auxiliary_loss(
                        predictions, targets, self.loss_fn.config
                    )
                    total = total + aux_components.total
                    aux_diag = aux_components.as_log_dict()
                bid_diag = {}
                if bidding_batch is not None:
                    bid_inputs = bidding_observations_to_model_input(
                        [transition.obs for transition in bidding_batch.transitions]
                    )
                    with self.mixed_precision.autocast():
                        bid_output = self.model.forward_bidding_batched(bid_inputs)
                    bid_targets = bidding_batch.to_targets(
                        bid_output.bid_logits.device,
                        dtype=bid_output.bid_logits.dtype,
                    )
                    cfg = self.loss_fn.config
                    bid_components = bidding_loss(
                        bid_output,
                        bid_targets,
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

            gpu_start = gpu_end = None
            if self.device.type == "cuda":
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            step_result = self.mixed_precision.step(
                loss_closure, self.optimizer, self._optimizer_parameters,
                max_grad_norm=self.config.max_grad_norm,
                clip_grad_norm=nn.utils.clip_grad_norm_,
                collective_all_true=self.distributed.all_true,
                synchronize_abandoned_backward=self.distributed.enabled,
                capture_retry_state=self._capture_retry_rng_state,
                restore_retry_state=self._restore_retry_rng_state,
            )
            if gpu_end is not None:
                gpu_end.record()
                gpu_end.synchronize()
                self._learner_gpu_seconds += gpu_start.elapsed_time(gpu_end) / 1000.0
            if components is None or win_logit is None:
                raise RuntimeError("optimizer closure did not produce diagnostics")
            total_loss = step_result.loss
            grad_norm = step_result.grad_norm
            self.stats.amp_fallbacks = self.mixed_precision.fallback_count
            self.stats.optimizer_steps += 1
            self.stats.learner_cardplay_samples += batch.batch_size
            if bidding_batch is not None:
                self.stats.learner_bidding_samples += bidding_batch.batch_size
            self.policy_step += 1
            if self.async_mode:
                self._publish_async_snapshot()
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

    def _parameter_update_watch(self) -> list[torch.nn.Parameter]:
        watched = [next(self.model.parameters())]
        if self.model.bidding_heads is not None:
            watched.append(next(self.model.bidding_heads.parameters()))
        if self.belief_model is not None and self.belief_training_mode != "frozen":
            watched.append(next(self.belief_model.parameters()))
        return watched

    def _parameter_update_snapshot(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            parameter.detach().clone()
            for parameter in self._parameter_update_watch()
        )

    def _parameters_changed_since(
        self, snapshots: tuple[torch.Tensor, ...]
    ) -> bool:
        watched = self._parameter_update_watch()
        if len(snapshots) != len(watched):
            raise ValueError("parameter update snapshot shape changed")
        return any(
            not torch.equal(snapshot, parameter.detach())
            for snapshot, parameter in zip(snapshots, watched)
        )

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
        before = self._parameter_update_snapshot()
        collection_started = time.perf_counter()
        self.collect_episodes()
        self._last_collection_seconds = time.perf_counter() - collection_started
        optimization_started = time.perf_counter()
        steps_taken = 0
        for _ in range(self.config.optimizer_steps):
            result = self.step()
            if result is not None:
                steps_taken += 1
        self._last_optimization_seconds = (
            time.perf_counter() - optimization_started
        )
        self.stats_last_run_changed = self._parameters_changed_since(before)
        if self.config.optimizer_steps > 0 and steps_taken < self.config.optimizer_steps:
            raise RuntimeError(
                f"requested {self.config.optimizer_steps} optimizer steps but "
                f"only {steps_taken} were taken "
                f"(collected {self.stats.transitions_collected} transitions, "
                f"batch_size={self.config.batch_size}). "
                f"Collect more episodes or reduce batch_size."
            )
        return self.stats
