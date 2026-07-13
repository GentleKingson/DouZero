"""Minimal single-process V2 multi-objective trainer (P06).

A complete, CPU-friendly training loop that demonstrates the P06 acceptance
criterion: "极短训练能完成一次优化且参数变化" (a very short training run
completes one optimizer step and changes the parameters).

Design
------
- Single process, CPU-only. No shared memory, no actor subprocesses, no
  GPU handling. The legacy multiprocessing path (:mod:`douzero.dmc`) is
  untouched. GPU + multi-process is P14.
- Self-play: a single :class:`~douzero.env.env.Env` (legacy mode) plays
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

P06 r1 hardening
----------------
- The trainer rejects a non-legacy ruleset at construction (standard mode
  requires a bidding driver that is not part of P06).
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
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import observation_to_model_inputs
from douzero.models_v2.model import ModelV2
from douzero.observation.encode_v2 import get_obs_v2

from douzero.training.decision_policy import DecisionConfig, select_action
from douzero.training.losses import LossComponents, LossConfig, MultiObjectiveLoss
from douzero.training.v2_buffer import Episode, Transition, V2ReplayBuffer


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


def _legacy_only(ruleset: RuleSet | None) -> None:
    """Reject any non-None ruleset (P06 has no bidding driver).

    ``Env`` treats ANY non-None ``RuleSet`` as standard mode (it enters the
    bidding phase), even ``RuleSet.legacy()``. The P06 trainer has no
    bidding driver, so the only valid value is ``ruleset=None`` (which
    gives the legacy card-play-only env). This is a P06 limitation; P11's
    league work adds the bidding driver.
    """
    if ruleset is None:
        return
    raise NotImplementedError(
        f"V2Trainer only supports ruleset=None (the legacy card-play-only "
        f"env) in P06. Env treats any non-None RuleSet as standard mode and "
        f"enters the bidding phase, which requires a bidding driver that is "
        f"part of P11's league work. Got ruleset={ruleset!r}."
    )


class V2Trainer:
    """Single-process V2 multi-objective trainer.

    Parameters
    ----------
    model:
        The :class:`ModelV2` to optimize. The trainer takes ownership of
        gradient updates. CPU-only — the model is NOT moved to a device
        (GPU is P14).
    ruleset:
        :class:`RuleSet` for the env. ``None`` (default) keeps the legacy
        card-play-only env (no bidding). Non-legacy rulesets are rejected
        at construction (P06 has no bidding driver).
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
        bc_aux_samples=None,
        bc_schedule=None,
        bc_temperature: float = 1.0,
        bc_label_smoothing: float = 0.0,
    ) -> None:
        _legacy_only(ruleset)
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
        # P07 belief training integration (review round 3). A belief-enabled
        # value model can only be trained via the "pretrain belief, freeze its
        # features, train belief_proj" path: the trainer holds the frozen
        # BeliefModel, computes the constrained posterior features from each
        # obs.public, and passes them into ModelV2.forward at both the
        # collection and optimizer call sites. Without this, a belief_enabled
        # value model fails closed at the first forward.
        belief_enabled = bool(getattr(self.model.config, "belief_enabled", False))
        if belief_enabled and belief_model is None:
            raise ValueError(
                "The value model has belief_enabled=True but no belief_model "
                "was supplied to V2Trainer. A belief-enabled value model can "
                "only be trained with a frozen BeliefModel that computes its "
                "posterior features. Pass a pretrained belief_model= (load it "
                "with load_belief_checkpoint)."
            )
        if belief_model is not None and not belief_enabled:
            raise ValueError(
                "A belief_model was supplied but the value model has "
                "belief_enabled=False. Drop belief_model or rebuild the value "
                "model with belief_enabled=True."
            )
        if belief_model is not None:
            # Freeze the belief model: only belief_proj (a value-model param)
            # is trained; the belief posterior is a frozen feature source.
            for p in belief_model.parameters():
                p.requires_grad_(False)
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
        self.decision_config = decision_config or DecisionConfig()
        self.config = config or TrainerConfig()
        # P06 r4: reject a "valid but trains nothing" configuration.
        # (a) optimizer_steps > 0 with all loss weights at 0 produces a
        #     zero-gradient step that silently changes nothing.
        # (b) optimizer_steps > 0 with buffer_capacity < batch_size means
        #     step() can never sample a minibatch and silently skips.
        active_loss = (
            loss_cfg.lambda_win + loss_cfg.lambda_score + loss_cfg.lambda_uncertainty
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
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.config.learning_rate,
            alpha=self.config.rmsprop_alpha,
            momentum=self.config.rmsprop_momentum,
            eps=self.config.rmsprop_epsilon,
        )
        self.buffer = V2ReplayBuffer(capacity_transitions=self.config.buffer_capacity)
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
            episodes_per_team={"landlord": 0, "farmer": 0}
        )
        # P06 r3: put the model in eval mode for self-play collection.
        # ``inference_mode`` in _choose_action_index only disables autograd;
        # it does NOT switch Dropout / BatchNorm behaviour, so without
        # eval() a model with non-zero history_dropout or mlp_dropout would
        # produce non-deterministic action selection even with exp_epsilon=0.
        # step() toggles to train() for the optimizer step, then back to
        # eval() after.
        self.model.eval()

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
                self.stats.episodes_completed += 1
                team = episode.terminal_result.get("winner_team", "landlord")
                self.stats.episodes_per_team[team] = (
                    self.stats.episodes_per_team.get(team, 0) + 1
                )
                self.stats.transitions_collected = len(self.buffer)

    def _run_one_episode(self) -> Episode:
        """Play one game to terminal, recording decisions and labels."""
        env = Env(objective="adp", ruleset=self.ruleset)
        env.reset()
        episode = Episode()
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
                    Transition(obs=obs, action_index=action_index, position=position)
                )
            _obs_out, _reward, done, info = env.step(action)
            if done:
                episode.terminal_result = info or {}
                break
        return episode

    def _compute_belief_feature(self, obs) -> "torch.Tensor | None":
        """Compute the (frozen) belief posterior feature for one observation.

        Returns the 48-dim feature vector (detached) built from ``obs.public``
        via the frozen :class:`BeliefModel`, or ``None`` when belief fusion is
        disabled (so the value model's fail-closed path is not triggered and
        ``belief_features`` is simply omitted). Always runs under
        ``inference_mode`` and never builds a graph — the belief model is a
        frozen feature source, not an optimization target.
        """
        if self.belief_model is None:
            return None
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
        # Detached leaf tensor: the value model casts it to its trunk
        # device/dtype and the value loss updates only belief_proj.
        return torch.from_numpy(feat_np).detach()

    def _compute_bc_aux_loss(self):
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
            bundle = observation_to_model_inputs(s.obs)
            # P08 Blocker 1: compute the frozen belief features for this BC
            # sample when the value model is belief-enabled. A belief-enabled
            # model FAILS CLOSED at forward when belief_features are omitted,
            # so without this the P07+P08 combo would crash at every optimizer
            # step. The features come from the PUBLIC observation only.
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
            )
            if out.prior_logit is None:
                raise RuntimeError(
                    "BC auxiliary loss requested but the model produced no "
                    "prior_logit (the prior head disappeared mid-training)."
                )
            loss, hit = listwise_bc_loss(
                out.prior_logit,
                out.action_mask,
                s.human_action_index,
                weight=s.sample_weight,
                temperature=self.bc_temperature,
                label_smoothing=self.bc_label_smoothing,
            )
            per_decision.append((loss, hit))
        return average_bc_losses(per_decision)

    def _choose_action_index(self, obs) -> int:
        """Epsilon-greedy action selection over the model's valid actions."""
        if self.config.exp_epsilon > 0.0 and self.rng.random() < self.config.exp_epsilon:
            mask = obs.actions.action_mask
            valid = [i for i, m in enumerate(mask) if m]
            return self.rng.choice(valid)
        belief_features = self._compute_belief_feature(obs)
        with torch.inference_mode():
            bundle = observation_to_model_inputs(obs)
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
            )
        return select_action(out, self.decision_config)

    # ------------------------------------------------------------------ #
    # Optimization (fail-closed on non-finite loss / gradient)
    # ------------------------------------------------------------------ #
    def step(self) -> LossComponents | None:
        """Run one optimizer step on a sampled minibatch.

        Returns the :class:`LossComponents` if a step was taken, or ``None``
        if the buffer did not have enough labelled transitions yet.

        Fail-closed (P06 r1): if the total loss is non-finite OR the
        gradient norm is non-finite, raise :class:`FloatingPointError`
        BEFORE the optimizer mutates parameters. PyTorch's
        :func:`torch.nn.utils.clip_grad_norm_` defaults to
        ``error_if_nonfinite=False`` which silently lets NaN/Inf through;
        we pass ``error_if_nonfinite=True`` to make the failure loud.
        """
        batch = self.buffer.sample_minibatch(self.config.batch_size, rng=self.rng)
        if batch is None:
            return None

        # P06 r4: use try/finally so model.eval() + gradient cleanup are
        # guaranteed even when clip_grad_norm_(error_if_nonfinite=True) or
        # the non-finite-loss guard raises. Without this, an exception
        # leaves the model in training mode, and subsequent self-play
        # collection would run with dropout active.
        self.model.train()
        try:
            # Per-decision forward; gather the chosen action's heads and
            # CONCATENATE (not stack) so the resulting tensors are (B, 1).
            gathered_win: list[torch.Tensor] = []
            gathered_siw: list[torch.Tensor] = []
            gathered_sil: list[torch.Tensor] = []
            for i, obs in enumerate(batch.observations):
                bundle = observation_to_model_inputs(obs)
                belief_features = self._compute_belief_feature(obs)
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
                )
                idx = int(batch.action_indices[i].item())
                gathered_win.append(out.win_logit[idx : idx + 1])
                gathered_siw.append(out.score_if_win[idx : idx + 1])
                gathered_sil.append(out.score_if_loss[idx : idx + 1])

            win_logit = torch.cat(gathered_win, dim=0)        # (B, 1)
            score_if_win = torch.cat(gathered_siw, dim=0)    # (B, 1)
            score_if_loss = torch.cat(gathered_sil, dim=0)   # (B, 1)
            batch_labels = {
                "target_win": batch.target_win,
                "target_score": batch.target_score,
                "target_log_score": batch.target_log_score,
            }
            components = self.loss_fn.forward_gathered(
                win_logit, score_if_win, score_if_loss, batch_labels
            )

            # P08: optional listwise BC auxiliary term. When lambda_bc > 0 the
            # trainer samples a minibatch of human BC samples, forwards each
            # through the prior head, and adds the (weighted, averaged) listwise
            # cross-entropy scaled by lambda_bc to the RL loss. Both terms
            # contribute gradients in a single backward pass.
            total_loss = components.total
            # P08 Blocker 1: the BC auxiliary weight follows the configured
            # schedule (constant or linear_decay with a floor), evaluated at
            # the CURRENT optimizer step so the weight evolves as configured.
            eff_lambda = self.bc_schedule.effective_lambda(
                self.stats.optimizer_steps
            )
            bc_diag: dict[str, float] = {}
            if eff_lambda > 0.0:
                bc_term = self._compute_bc_aux_loss()
                total_loss = total_loss + eff_lambda * bc_term.total
                # Non-blocking (round 4): record BC diagnostics for logging so
                # schedule effects / BC collapse / weight imbalance are visible.
                bc_diag = {
                    "bc_cross_entropy": bc_term.cross_entropy,
                    "bc_top1_accuracy": (
                        bc_term.top1_correct / bc_term.num_decisions
                        if bc_term.num_decisions > 0 else 0.0
                    ),
                    "bc_effective_lambda": eff_lambda,
                    "bc_num_decisions": bc_term.num_decisions,
                }

            # Fail-closed: a non-finite loss means something is wrong.
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"V2Trainer encountered a non-finite loss "
                    f"({float(total_loss.item())!r}); refusing to take an "
                    f"optimizer step. Check the head clamp, the target clamp, "
                    f"and the input encoding."
                )

            self.optimizer.zero_grad()
            total_loss.backward()
            # error_if_nonfinite=True so a NaN/Inf gradient raises loudly
            # here instead of silently corrupting the optimizer state.
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.stats.optimizer_steps += 1
            # Merge BC diagnostics into the last_loss log dict when active.
            loss_log = components.as_log_dict()
            if bc_diag:
                loss_log.update(bc_diag)
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

    def train(self) -> TrainerStats:
        """Run the configured number of episodes + optimizer steps.

        P06 r4: raises :class:`RuntimeError` if fewer optimizer steps were
        taken than requested (e.g. not enough transitions collected to fill
        a minibatch). The caller must either collect more episodes, reduce
        ``batch_size``, or explicitly set ``optimizer_steps=0`` for a
        collect-only run.
        """
        # Snapshot one parameter for the "parameters changed" smoke check.
        before = next(self.model.parameters()).detach().clone()
        self.collect_episodes()
        steps_taken = 0
        for _ in range(self.config.optimizer_steps):
            result = self.step()
            if result is not None:
                steps_taken += 1
        after = next(self.model.parameters()).detach().clone()
        self.stats_last_run_changed = not torch.equal(before, after)
        if self.config.optimizer_steps > 0 and steps_taken < self.config.optimizer_steps:
            raise RuntimeError(
                f"requested {self.config.optimizer_steps} optimizer steps but "
                f"only {steps_taken} were taken "
                f"(collected {self.stats.transitions_collected} transitions, "
                f"batch_size={self.config.batch_size}). "
                f"Collect more episodes or reduce batch_size."
            )
        return self.stats
