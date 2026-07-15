"""Learned-bidding policy initialization, replay, and objectives."""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from douzero.models_v2.output import BiddingModelOutput
from douzero.observation.bidding import BIDDING_ACTIONS, BiddingObservationV2

_POLICIES = frozenset({"random", "rule", "max", "pass", "learned"})


@dataclass(frozen=True)
class BiddingPolicyConfig:
    """How bidding actions are initialized before/while learning.

    ``policy='learned'`` mixes the learned head with ``warm_start_policy`` at
    ``learned_probability``.  Setting the probability gradually from 0 to 1
    provides the requested rule-to-learned handoff without ever mislabelling a
    rule bid as a learned one in transition provenance.
    """

    policy: str = "rule"
    warm_start_policy: str = "rule"
    learned_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise ValueError(f"unsupported bidding policy {self.policy!r}")
        if self.warm_start_policy not in _POLICIES - {"learned"}:
            raise ValueError("warm_start_policy must be random/rule/max/pass")
        if not math.isfinite(self.learned_probability) or not 0 <= self.learned_probability <= 1:
            raise ValueError("learned_probability must be finite and in [0, 1]")


def _rule_bid(obs: BiddingObservationV2) -> int:
    counts = Counter(obs.my_handcards)
    strength = 0.0
    strength += 2.0 if counts[30] else 0.0
    strength += 1.4 if counts[20] else 0.0
    strength += counts[17] * 0.45
    strength += counts[14] * 0.25
    strength += sum(1.25 for count in counts.values() if count == 4)
    if counts[20] and counts[30]:
        strength += 1.0
    desired = 3 if strength >= 5.0 else 2 if strength >= 3.2 else 1 if strength >= 1.6 else 0
    affordable = [bid for bid in obs.legal_bids if bid <= desired]
    return max(affordable) if affordable else 0


def select_bidding_action(
    obs: BiddingObservationV2,
    config: BiddingPolicyConfig,
    rng: random.Random,
    learned_selector: Callable[[BiddingObservationV2], int] | None = None,
) -> tuple[int, str]:
    """Return ``(bid, source_policy)`` and enforce environment legality."""

    policy = config.policy
    if policy == "learned":
        use_learned = rng.random() < config.learned_probability
        policy = "learned" if use_learned else config.warm_start_policy
    if policy == "learned":
        if learned_selector is None:
            raise ValueError("learned bidding policy requires a learned_selector")
        bid = int(learned_selector(obs))
    elif policy == "random":
        bid = int(rng.choice(obs.legal_bids))
    elif policy == "rule":
        bid = _rule_bid(obs)
    elif policy == "max":
        bid = max(obs.legal_bids)
    else:
        bid = 0
    if bid not in obs.legal_bids:
        raise ValueError(
            f"{policy} bidding policy selected illegal bid {bid}; legal={obs.legal_bids}"
        )
    return bid, policy


@dataclass
class BiddingTransition:
    obs: BiddingObservationV2
    bid_action: int
    policy_version: str
    source_policy: str
    target_landlord_win: float = float("nan")
    target_landlord_score: float = float("nan")
    target_regret: float = float("nan")

    def label_from_terminal(self, terminal: dict) -> None:
        # These are deliberately landlord-side labels for every neutral bidder.
        targets = terminal.get("team_targets", {}).get("landlord")
        if targets is None:
            raise ValueError("terminal result is missing landlord team_targets")
        self.target_landlord_win = float(targets["target_win"])
        self.target_landlord_score = float(targets["target_score"])

    def validate(self) -> None:
        if self.bid_action not in BIDDING_ACTIONS:
            raise ValueError("bid_action is outside the bidding action schema")
        if self.bid_action not in self.obs.legal_bids:
            raise ValueError("bid_action was not legal in its observation")
        if not self.policy_version:
            raise ValueError("bidding transition must record policy_version")
        if self.source_policy not in _POLICIES:
            raise ValueError("bidding transition source_policy is invalid")
        if self.target_landlord_win not in (0.0, 1.0):
            raise ValueError("target_landlord_win must be binary")
        if not math.isfinite(self.target_landlord_score):
            raise ValueError("target_landlord_score must be finite")


@dataclass
class BiddingMinibatch:
    transitions: list[BiddingTransition]

    @property
    def batch_size(self) -> int:
        return len(self.transitions)


class BiddingReplayBuffer:
    """Bounded transition buffer; abandoned all-pass deals are never added."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("bidding replay capacity must be positive")
        self.capacity = int(capacity)
        self._transitions: deque[BiddingTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._transitions)

    def add_terminal_deal(
        self, transitions: Iterable[BiddingTransition], terminal: dict
    ) -> None:
        if terminal.get("redeal"):
            raise ValueError("all-pass redeal transitions cannot receive terminal labels")
        for transition in transitions:
            transition.label_from_terminal(terminal)
            transition.validate()
            self._transitions.append(transition)

    def sample(
        self, batch_size: int, rng: random.Random
    ) -> BiddingMinibatch | None:
        if len(self) < batch_size:
            return None
        return BiddingMinibatch(rng.sample(list(self._transitions), batch_size))


@dataclass
class BiddingLossComponents:
    total: torch.Tensor
    policy: float
    landlord_win: float
    landlord_score: float
    regret: float

    def as_log_dict(self) -> dict[str, float]:
        return {
            "loss_bid_total": float(self.total.detach().float().item()),
            "loss_bid_policy": self.policy,
            "loss_bid_landlord_win": self.landlord_win,
            "loss_bid_landlord_score": self.landlord_score,
            "loss_bid_regret": self.regret,
        }


def bidding_loss(
    outputs: list[BiddingModelOutput],
    batch: BiddingMinibatch,
    *,
    lambda_policy: float,
    lambda_landlord_win: float,
    lambda_landlord_score: float,
    lambda_regret: float = 0.0,
    score_delta: float = 1.0,
    score_target_transform: str = "raw",
    score_clamp: float = 32.0,
) -> BiddingLossComponents:
    """Masked listwise policy CE plus landlord-perspective outcome losses."""

    if len(outputs) != batch.batch_size or not outputs:
        raise ValueError("bidding outputs and minibatch must have equal non-zero size")
    logits = torch.stack([out.bid_logits.float() for out in outputs])
    masks = torch.stack([out.bid_action_mask for out in outputs])
    actions = torch.tensor(
        [transition.bid_action for transition in batch.transitions],
        device=logits.device,
        dtype=torch.long,
    )
    if not bool(masks[torch.arange(len(outputs), device=logits.device), actions].all()):
        raise ValueError("bidding policy label points to an illegal action")
    masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    policy = F.cross_entropy(masked_logits, actions)
    win_logits = torch.stack([out.landlord_win_logit.float() for out in outputs])
    target_win = torch.tensor(
        [transition.target_landlord_win for transition in batch.transitions],
        device=logits.device,
        dtype=torch.float32,
    )
    win = F.binary_cross_entropy_with_logits(win_logits.reshape(-1), target_win)
    score_predictions = torch.stack(
        [out.expected_landlord_score.float() for out in outputs]
    ).reshape(-1)
    raw_target_score = torch.tensor(
        [transition.target_landlord_score for transition in batch.transitions],
        device=logits.device,
        dtype=torch.float32,
    )
    from douzero.training.losses import resolve_score_target

    target_score = resolve_score_target(
        raw_target_score,
        score_target_transform=score_target_transform,
        score_clamp=score_clamp,
    )
    score = F.huber_loss(score_predictions, target_score, delta=score_delta)
    regret = logits.new_zeros(())
    if lambda_regret > 0:
        regrets = [transition.target_regret for transition in batch.transitions]
        if not all(math.isfinite(value) for value in regrets):
            raise ValueError(
                "bid regret loss is enabled but rollout regret labels are missing"
            )
        # A conservative scalar auxiliary: lower regret should correspond to a
        # larger selected logit. It is opt-in until candidate rollouts exist.
        chosen = logits.gather(1, actions[:, None]).squeeze(1)
        regret_target = -torch.tensor(regrets, device=logits.device)
        regret = F.huber_loss(chosen, regret_target, delta=score_delta)
    total = (
        float(lambda_policy) * policy
        + float(lambda_landlord_win) * win
        + float(lambda_landlord_score) * score
        + float(lambda_regret) * regret
    )
    return BiddingLossComponents(
        total=total,
        policy=float(policy.detach().item()),
        landlord_win=float(win.detach().item()),
        landlord_score=float(score.detach().item()),
        regret=float(regret.detach().item()),
    )
