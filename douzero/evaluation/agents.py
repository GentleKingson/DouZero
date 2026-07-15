"""Built-in agents, model loaders, and inference instrumentation for P15."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from douzero.env.rules import RuleSet

from .scenario import BundleSpec


class RuleAgent:
    """Deterministic public heuristic: shed more cards, then higher ranks."""

    def act(self, infoset):
        return max(
            infoset.legal_actions,
            key=lambda action: (len(action), sum(action), tuple(action)),
        )


class SeededRandomAgent:
    """Random legal-action agent with an isolated deterministic RNG."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, infoset):
        return self.rng.choice(infoset.legal_actions)


@dataclass
class TimedAgent:
    """Measure inference latency and retain optional selected-action p_win."""

    inner: Any
    bundle_label: str
    role: str
    latencies_ms: list[float] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)

    def act(self, infoset):
        started = time.perf_counter_ns()
        action = self.inner.act(infoset)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.latencies_ms.append(elapsed_ms)
        prediction = getattr(self.inner, "last_p_win", None)
        if prediction is not None:
            self.predictions.append(float(prediction))
        return action


class BundleFactory:
    """Build role-specific agents while loading checkpoint weights only once."""

    def __init__(self, ruleset: RuleSet) -> None:
        self.ruleset = ruleset
        self._model_agents: dict[tuple[int, str], Any] = {}

    def build(
        self,
        bundle: BundleSpec,
        role: str,
        *,
        seed: int,
        bundle_label: str,
    ) -> TimedAgent:
        if bundle.backend == "random":
            inner = SeededRandomAgent(seed)
        elif bundle.backend == "rule":
            inner = RuleAgent()
        else:
            key = (id(bundle), role)
            if key not in self._model_agents:
                self._model_agents[key] = self._load_model_agent(bundle, role)
            inner = self._model_agents[key]
        return TimedAgent(inner=inner, bundle_label=bundle_label, role=role)

    def _load_model_agent(self, bundle: BundleSpec, role: str):
        checkpoint = bundle.checkpoints[role]
        if bundle.backend in ("legacy", "legacy_factorized"):
            from douzero.evaluation.deep_agent import DeepAgent

            return DeepAgent(
                role,
                checkpoint,
                backend=(
                    "legacy_factorized"
                    if bundle.backend == "legacy_factorized"
                    else "legacy"
                ),
            )
        if bundle.backend in ("v2", "bc"):
            from douzero.belief.checkpoint import load_belief_checkpoint
            from douzero.evaluation.deep_agent import DeepAgentV2, load_v2_model
            from douzero.models_v2.config import ModelV2Config
            from douzero.observation.schema import build_v2_schema
            from douzero.search.budget import SearchConfig

            config = ModelV2Config(**dict(bundle.model_config))
            schema = build_v2_schema()
            model = load_v2_model(checkpoint, schema, self.ruleset, config=config)
            belief_model = None
            if bundle.belief_checkpoint:
                belief_model = load_belief_checkpoint(
                    bundle.belief_checkpoint,
                    expected_ruleset=self.ruleset,
                )
            search_config = SearchConfig(**dict(bundle.search_config))
            return DeepAgentV2(
                position=role,
                model=model,
                ruleset=self.ruleset,
                decision_mode=bundle.decision_mode,
                belief_model=belief_model,
                search_config=search_config,
            )
        raise ValueError(f"unsupported bundle backend {bundle.backend!r}")


def choose_bid(
    bundle: BundleSpec,
    bidding_observation: dict[str, Any],
    legal_bids: list[int],
    rng: random.Random,
) -> int:
    """Apply the bundle's explicit bidding policy to a public observation."""
    if bundle.bidding_policy == "pass":
        return 0
    if bundle.bidding_policy == "max":
        return max(legal_bids)
    if bundle.bidding_policy == "random":
        return rng.choice(legal_bids)

    hand = bidding_observation["my_handcards"]
    # Fixed public hand-strength policy. It exists to make full-game smoke
    # evaluation meaningful until a learned bidding checkpoint interface is
    # introduced; it is never presented as learned-model bidding.
    high_cards = sum(card in (17, 20, 30) for card in hand)
    bombs = sum(hand.count(rank) == 4 for rank in set(hand))
    strength = high_cards + 2 * bombs
    target = 3 if strength >= 7 else 2 if strength >= 5 else 1 if strength >= 3 else 0
    allowed = [bid for bid in legal_bids if bid <= target]
    return max(allowed) if allowed else 0
