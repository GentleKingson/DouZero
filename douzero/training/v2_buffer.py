"""Per-episode V2 transition store for the single-process trainer (P06).

The legacy multiprocessing buffer (``douzero/dmc/utils.py``) is keyed by six
fixed-shape tensors per decision, which cannot represent the V2 model's
variable-N-action decisions. Rewriting it would overlap with P14's
high-throughput work and risk regressing the legacy path.

This module provides a minimal, single-process transition store that:

- Records one :class:`~douzero.observation.encode_v2.ObservationV2` per
  decision along with the action index the actor chose and the acting
  position.
- At terminal, looks up the team-perspective labels from the env's
  ``info['team_targets']`` (populated by
  :meth:`~douzero.env.env.Env._attach_team_perspective_labels`) and attaches
  them to every transition of the same position in that episode. This is
  the Monte-Carlo return: every decision the landlord made in a game the
  landlord ultimately won gets ``target_win = 1`` and the same final
  ``target_score``; ditto for farmers.
- Exposes :meth:`V2ReplayBuffer.sample_minibatch` which returns ``B``
  transitions (with their labels) for the trainer's forward+backward pass.

This is intentionally NOT a high-throughput data structure. It exists to
satisfy the P06 acceptance criterion ("极短训练能完成一次优化且参数变化")
on CPU while preserving the legacy path. P14 will introduce the
multiprocessing/shmem equivalent.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import torch

from douzero.observation.encode_v2 import ObservationV2


@dataclass
class Transition:
    """One decision in a V2 self-play episode.

    ``obs`` is the public :class:`ObservationV2` captured at the decision
    point. ``action_index`` is the row index (into
    ``obs.actions.features``) the actor took. ``position`` is the acting
    role. The four ``target_*`` fields are filled in at episode terminal
    from the team-perspective label (Monte-Carlo return).
    """

    obs: ObservationV2
    action_index: int
    position: str
    target_win: float = float("nan")
    target_score: float = float("nan")
    target_log_score: float = float("nan")

    def has_labels(self) -> bool:
        return not (
            isinstance(self.target_win, float) and math_isnan(self.target_win)
        )


def math_isnan(x: float) -> bool:
    import math

    return isinstance(x, float) and math.isnan(x)


@dataclass
class Episode:
    """All transitions of one game, plus the terminal result dict."""

    transitions: list[Transition] = field(default_factory=list)
    terminal_result: dict = field(default_factory=dict)

    def label_from_terminal(self) -> None:
        """Apply per-position team-perspective labels to every transition.

        Looks up ``self.terminal_result['team_targets']`` (populated by the
        env at terminal) and stamps each transition's labels based on its
        ``position``. Idempotent.
        """
        team_targets = self.terminal_result.get("team_targets")
        if team_targets is None:
            raise ValueError(
                "Episode.terminal_result is missing 'team_targets'; the env "
                "must populate it at terminal (see Env._attach_team_perspective_labels)."
            )
        for tr in self.transitions:
            labels = team_targets[tr.position]
            tr.target_win = float(labels["target_win"])
            tr.target_score = float(labels["target_score"])
            tr.target_log_score = float(labels["target_log_score"])


@dataclass
class Minibatch:
    """A minibatch of ``B`` labelled transitions.

    Tensors are built by the trainer from ``observations`` (which are run
    through the model one decision at a time, since :meth:`ModelV2.forward`
    handles a single decision). The four label tensors are stacked here so
    the loss module can consume them directly.
    """

    observations: list[ObservationV2]
    action_indices: torch.Tensor  # (B,) long
    target_win: torch.Tensor  # (B,) float
    target_score: torch.Tensor  # (B,) float
    target_log_score: torch.Tensor  # (B,) float

    @property
    def batch_size(self) -> int:
        return int(self.action_indices.shape[0])


class V2ReplayBuffer:
    """A bounded, label-checked replay buffer for the V2 trainer.

    Stores completed, labelled :class:`Episode` objects and samples
    minibatches uniformly across all labelled transitions. The buffer is
    intentionally simple (``collections.deque`` of episodes); it is not
    shared-memory and not thread-safe. The single-process trainer owns it.
    """

    def __init__(self, capacity_transitions: int = 4096) -> None:
        if capacity_transitions <= 0:
            raise ValueError(
                f"capacity_transitions must be positive, got {capacity_transitions}"
            )
        self._capacity = int(capacity_transitions)
        self._episodes: deque[Episode] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def add_episode(self, episode: Episode) -> None:
        """Append a labelled episode, evicting oldest episodes as needed."""
        if not episode.transitions:
            return
        # Ensure labels are stamped.
        if any(not tr.has_labels() for tr in episode.transitions):
            episode.label_from_terminal()
        self._episodes.append(episode)
        self._size += len(episode.transitions)
        while self._size > self._capacity and self._episodes:
            evicted = self._episodes.popleft()
            self._size -= len(evicted.transitions)

    def extend(self, episodes: Iterable[Episode]) -> None:
        for ep in episodes:
            self.add_episode(ep)

    def sample_minibatch(
        self,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> Minibatch | None:
        """Sample ``batch_size`` labelled transitions uniformly at random.

        Returns ``None`` if the buffer has fewer than ``batch_size``
        transitions. The trainer treats ``None`` as "not enough data yet".
        """
        if self._size < batch_size:
            return None
        rng = rng or random
        # Flatten transitions across episodes (preserving episode identity
        # is not needed for the off-policy MC value update).
        flat: list[Transition] = []
        for ep in self._episodes:
            flat.extend(ep.transitions)
        picks = rng.sample(flat, batch_size)
        return Minibatch(
            observations=[p.obs for p in picks],
            action_indices=torch.tensor(
                [p.action_index for p in picks], dtype=torch.long
            ),
            target_win=torch.tensor([p.target_win for p in picks], dtype=torch.float32),
            target_score=torch.tensor(
                [p.target_score for p in picks], dtype=torch.float32
            ),
            target_log_score=torch.tensor(
                [p.target_log_score for p in picks], dtype=torch.float32
            ),
        )
