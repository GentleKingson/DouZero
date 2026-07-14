"""Synthetic self-play data collection for belief training (P07).

A small, single-process collector that plays random self-play games on the
legacy card-play env and, at every non-trivial decision, records:

- the public :class:`~douzero.belief.features.BeliefInput` (built from the
  public observation), and
- the privileged :class:`~douzero.belief.labels.BeliefLabel` (built from
  ``infoset.all_handcards``).

The privileged label is carried in this training-only structure and NEVER on
the public observation. The CLI (``train_belief.py``) is the sole consumer.

This is intentionally a tiny, deterministic collector — enough to exercise the
loss/optimizer/checkpoint path on CPU. High-throughput collection belongs to
P14.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .features import BeliefInput, build_belief_input
from .labels import BeliefLabel, build_belief_label


@dataclass
class BeliefSample:
    """One labelled belief decision (public input + privileged label)."""

    binput: BeliefInput
    label: BeliefLabel


@dataclass
class BeliefDataset:
    """An in-memory collection of labelled belief samples."""

    samples: list[BeliefSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def feature_matrix(self) -> np.ndarray:
        """Return ``(N, BELIEF_INPUT_DIM)`` float32 stacked features."""
        from .features import BELIEF_INPUT_DIM

        if not self.samples:
            return np.zeros((0, BELIEF_INPUT_DIM), dtype=np.float32)
        return np.stack(
            [s.binput.feature_vector for s in self.samples], axis=0
        ).astype(np.float32)

    def target_tensor(self) -> torch.Tensor:
        """Return ``(N, 15, 5)`` float32 one-hot targets."""
        from .labels import target_allocation_tensor

        if not self.samples:
            raise ValueError("empty dataset has no targets")
        return torch.from_numpy(
            target_allocation_tensor([s.label.allocation for s in self.samples])
        )

    def style_feature_matrix(self) -> np.ndarray:
        """Return public P11 style statistics for every belief sample."""

        from douzero.style.features import STYLE_FEATURE_WIDTH

        if not self.samples:
            return np.zeros((0, STYLE_FEATURE_WIDTH), dtype=np.float32)
        return np.stack(
            [sample.binput.style_features for sample in self.samples], axis=0
        ).astype(np.float32)

    def legal_mask_tensor(self) -> torch.Tensor:
        """Return ``(N, 15, 5)`` bool legal-count mask."""
        from .constraints import legal_mask

        if not self.samples:
            raise ValueError("empty dataset has no legal mask")
        return torch.from_numpy(
            np.stack(
                [legal_mask(s.binput.unseen_counts) for s in self.samples],
                axis=0,
            )
        ).bool()

    def unseen_counts_matrix(self) -> np.ndarray:
        """Return ``(N, 15)`` int64 per-rank unknown-pool counts."""
        if not self.samples:
            raise ValueError("empty dataset has no unseen counts")
        return np.stack(
            [s.binput.unseen_counts for s in self.samples], axis=0
        ).astype(np.int64)


def collect_random_dataset(
    num_episodes: int,
    *,
    seed: int = 0,
    max_steps_per_episode: int = 600,
) -> BeliefDataset:
    """Play random self-play games and collect labelled belief samples.

    Random play is a cheap, deterministic source of diverse public states; it
    is sufficient to exercise the belief loss/optimizer on CPU. Real training
    would use model-guided self-play (out of scope for P07's smoke path).
    """
    from douzero.env.env import Env
    from douzero.observation.encode_v2 import get_obs_v2

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 1)
    np.random.seed(seed)  # Env.reset uses np.random.shuffle
    dataset = BeliefDataset()
    for _ in range(num_episodes):
        env = Env("adp")
        env.reset()
        steps = 0
        while True:
            if steps >= max_steps_per_episode:
                raise RuntimeError(
                    f"random episode exceeded {max_steps_per_episode} steps; "
                    "possible infinite loop."
                )
            steps += 1
            infoset = env.infoset
            legal = list(infoset.legal_actions)
            if not legal:
                break
            nonempty = [a for a in legal if len(a) > 0]
            pool = nonempty if nonempty else legal
            action = list(pool[int(np_rng.integers(len(pool)))])
            if len(legal) > 1:
                obs = get_obs_v2(infoset)
                binput = build_belief_input(obs.public)
                # Fail-fast: a conservation inconsistency here indicates an
                # env/feature bug, NOT a recoverable data point. Earlier this
                # was silently swallowed (Medium #5), which hid real
                # inconsistencies and biased the dataset. Let it propagate with
                # a state summary so the source is diagnosable.
                label = build_belief_label(
                    acting_role=infoset.player_position,
                    all_handcards=infoset.all_handcards,
                    unseen_counts=binput.unseen_counts,
                    num_cards_left=infoset.num_cards_left_dict,
                    bottom_unplayed=infoset.three_landlord_cards,
                )
                dataset.samples.append(BeliefSample(binput, label))
            _obs, _r, done, _info = env.step(action)
            if done:
                break
    return dataset


def iterate_minibatches(
    dataset: BeliefDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    rng: random.Random | None = None,
    include_style: bool = False,
) -> "Sequence[tuple[torch.Tensor, ...]]":
    """Yield belief minibatches, optionally with public P11 style features."""
    n = len(dataset)
    if n == 0:
        return []
    idx = list(range(n))
    if shuffle:
        (rng or random).shuffle(idx)
    out = []
    feats_all = dataset.feature_matrix()
    targets_all = dataset.target_tensor()
    legal_all = dataset.legal_mask_tensor()
    style_all = dataset.style_feature_matrix() if include_style else None
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        feats = torch.from_numpy(feats_all[batch_idx].astype(np.float32))
        targets = targets_all[batch_idx]
        legal = legal_all[batch_idx]
        if style_all is None:
            out.append((feats, targets, legal))
        else:
            styles = torch.from_numpy(style_all[batch_idx].astype(np.float32))
            out.append((feats, targets, legal, styles))
    return out
