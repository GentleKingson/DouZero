"""Minimal single-process V2 multi-objective trainer (P06).

A complete, CPU-friendly training loop that demonstrates the P06 acceptance
criterion: "极短训练能完成一次优化且参数变化" (a very short training run
completes one optimizer step and changes the parameters).

Design
------
- Single process. No shared memory, no actor subprocesses. The legacy
  multiprocessing path (:mod:`douzero.dmc`) is untouched.
- Self-play: a single :class:`~douzero.env.env.Env` (legacy mode by default)
  plays games to terminal. Each decision is made by an epsilon-greedy policy
  over the current :class:`~douzero.models_v2.model.ModelV2` outputs (random
  for the first few episodes when the buffer is empty).
- Transition recording: every decision's :class:`ObservationV2` + chosen
  action index + acting position is appended to the current
  :class:`~douzero.training.v2_buffer.Episode`.
- Labelling: at terminal, the env's ``info['team_targets']`` (populated by
  :meth:`~douzero.env.env.Env._attach_team_perspective_labels`) is read and
  the episode's transitions receive team-perspective Monte-Carlo labels.
- Optimizer step: the trainer samples a minibatch, forwards each decision,
  gathers the chosen action's head values, and calls
  :meth:`MultiObjectiveLoss.forward_gathered`.

This trainer is intentionally NOT high-throughput. It is the bounded test
the P06 acceptance criteria require; P14 introduces the multiprocessing
actor/learner.
"""

from __future__ import annotations

import math
import os
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
from douzero.training.labels import ALL_POSITIONS
from douzero.training.losses import LossComponents, LossConfig, MultiObjectiveLoss
from douzero.training.v2_buffer import Episode, V2ReplayBuffer


@dataclass
class TrainerConfig:
    """Knobs for :class:`V2Trainer`.

    Defaults are tuned for a CPU smoke test (a handful of short episodes and
    a single optimizer step). Production-scale tuning belongs in
    ``configs/enhanced.yaml``, not here.
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
    max_grad_norm: float = 40.0
    optimizer_steps: int = 1
    # Replay buffer.
    buffer_capacity: int = 4096
    # Where to save V2 checkpoints (empty string = no saving).
    checkpoint_dir: str = ""
    save_every_steps: int = 0
    # RNG for action sampling / minibatch sampling.
    rng_seed: int = 0


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


class V2Trainer:
    """Single-process V2 multi-objective trainer.

    Parameters
    ----------
    model:
        The :class:`ModelV2` to optimize. The trainer takes ownership of
        gradient updates but does NOT move the model to a device; the
        caller constructs the model on the desired device.
    ruleset:
        :class:`RuleSet` for the env. ``None`` (default) keeps the legacy
        card-play-only env (no bidding). Standard mode requires the trainer
        to drive bidding, which is deferred to P11's league work; the P06
        smoke uses legacy mode.
    loss_config:
        :class:`LossConfig` for the multi-objective loss.
    decision_config:
        :class:`DecisionConfig` for action selection during self-play. The
        default ``pure_win`` matches the deployment default.
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
    ) -> None:
        self.model = model
        self.ruleset = ruleset
        self.loss_fn = MultiObjectiveLoss(loss_config or LossConfig())
        self.decision_config = decision_config or DecisionConfig()
        self.config = config or TrainerConfig()
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.buffer = V2ReplayBuffer(capacity_transitions=self.config.buffer_capacity)
        self.rng = random.Random(self.config.rng_seed)
        self.stats = TrainerStats(
            episodes_per_team={"landlord": 0, "farmer": 0}
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
            assert steps < self.config.max_steps_per_episode, (
                "episode exceeded max_steps_per_episode; possible infinite loop"
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
                    _make_transition(obs, action_index, position)
                )
            obs_out, reward, done, info = env.step(action)
            if done:
                episode.terminal_result = info or {}
                break
        return episode

    def _choose_action_index(self, obs) -> int:
        """Epsilon-greedy action selection over the model's valid actions."""
        if self.config.exp_epsilon > 0.0 and self.rng.random() < self.config.exp_epsilon:
            mask = obs.actions.action_mask
            valid = [i for i, m in enumerate(mask) if m]
            return self.rng.choice(valid)
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
            )
        return select_action(out, self.decision_config)

    # ------------------------------------------------------------------ #
    # Optimization
    # ------------------------------------------------------------------ #
    def step(self) -> LossComponents | None:
        """Run one optimizer step on a sampled minibatch.

        Returns the :class:`LossComponents` if a step was taken, or ``None``
        if the buffer did not have enough labelled transitions yet.
        """
        batch = self.buffer.sample_minibatch(
            self.config.batch_size, rng=self.rng
        )
        if batch is None:
            return None

        self.model.train()
        gathered_win: list[torch.Tensor] = []
        gathered_siw: list[torch.Tensor] = []
        gathered_sil: list[torch.Tensor] = []
        for i, obs in enumerate(batch.observations):
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
            )
            idx = int(batch.action_indices[i].item())
            gathered_win.append(out.win_logit[idx : idx + 1])
            gathered_siw.append(out.score_if_win[idx : idx + 1])
            gathered_sil.append(out.score_if_loss[idx : idx + 1])

        win_logit = torch.stack(gathered_win)  # (B, 1)
        score_if_win = torch.stack(gathered_siw)
        score_if_loss = torch.stack(gathered_sil)
        batch_labels = {
            "target_win": batch.target_win,
            "target_score": batch.target_score,
            "target_log_score": batch.target_log_score,
        }
        components = self.loss_fn.forward_gathered(
            win_logit, score_if_win, score_if_loss, batch_labels
        )

        self.optimizer.zero_grad()
        components.total.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.stats.optimizer_steps += 1
        self.stats.last_loss = components.as_log_dict()
        self.stats.grad_norm_last_step = float(grad_norm.detach().float().item())
        # p_win distribution diagnostics.
        with torch.no_grad():
            p = torch.sigmoid(win_logit).reshape(-1)
            self.stats.p_win_mean = float(p.mean().item())
            self.stats.p_win_std = float(p.std().item()) if p.numel() > 1 else 0.0
            self.stats.score_mean_avg = float(
                (torch.sigmoid(win_logit.detach()) * score_if_win.detach()
                 + (1 - torch.sigmoid(win_logit.detach())) * score_if_loss.detach())
                .mean()
                .item()
            )
        self.model.eval()
        return components

    def train(self) -> TrainerStats:
        """Run the configured number of episodes + optimizer steps."""
        # Snapshot one parameter for the "parameters changed" smoke check.
        before = next(self.model.parameters()).detach().clone()
        self.collect_episodes()
        for _ in range(self.config.optimizer_steps):
            self.step()
        after = next(self.model.parameters()).detach().clone()
        self.stats_last_run_changed = not torch.equal(before, after)
        return self.stats


def _make_transition(obs, action_index: int, position: str):
    """Build a Transition; imported lazily to keep the top-level import light."""
    from douzero.training.v2_buffer import Transition

    return Transition(obs=obs, action_index=action_index, position=position)
