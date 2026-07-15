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
    search_calls: int = 0
    search_timeouts: int = 0
    search_fallbacks: int = 0

    def act(self, infoset):
        started = time.perf_counter_ns()
        action = self.inner.act(infoset)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.latencies_ms.append(elapsed_ms)
        prediction = getattr(self.inner, "last_p_win", None)
        if prediction is not None:
            self.predictions.append(float(prediction))
        search_log = getattr(self.inner, "last_search_log", None)
        if search_log is not None:
            self.search_calls += 1
            self.search_timeouts += int(bool(search_log.timed_out))
            self.search_fallbacks += int(bool(search_log.fallback_reason))
        return action


class BundleFactory:
    """Build role-specific agents while loading checkpoint weights only once."""

    def __init__(self, ruleset: RuleSet) -> None:
        self.ruleset = ruleset
        self._model_agents: dict[tuple[int, str], Any] = {}
        self._bidding_models: dict[int, Any] = {}

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
        self,
        bundle: BundleSpec,
        bidding_observation: dict[str, Any],
        legal_bids: list[int],
        rng: random.Random,
        *,
        redeal_count: int,
    ) -> int:
        """Select an external or manifest-validated learned bid."""

        if bundle.bidding_policy != "learned":
            return choose_bid(bundle, bidding_observation, legal_bids, rng)
        key = id(bundle)
        if key not in self._bidding_models:
            from douzero.evaluation.deep_agent import load_v2_model
            from douzero.models_v2.config import ModelV2Config
            from douzero.observation.schema import build_v2_schema

            config = ModelV2Config(**dict(bundle.model_config))
            if not config.bidding_enabled:
                raise ValueError(
                    "learned bidding requires model_config.bidding_enabled=true"
                )
            self._bidding_models[key] = load_v2_model(
                bundle.bidding_checkpoint,
                build_v2_schema(),
                self.ruleset,
                config=config,
                device="cpu",
            )
        from douzero.observation.bidding import get_bidding_obs_v2

        observation = get_bidding_obs_v2(
            {**bidding_observation, "legal_bids": list(legal_bids)},
            ruleset=self.ruleset,
            redeal_count=redeal_count,
        )
        import torch

        with torch.inference_mode():
            bid = self._bidding_models[key].forward_bidding(observation).argmax_bid()
        if bid not in legal_bids:
            raise RuntimeError("learned bidding model selected an illegal bid")
        return bid


def choose_bid(
    bundle: BundleSpec,
    bidding_observation: dict[str, Any],
    legal_bids: list[int],
    rng: random.Random,
) -> int:
    """Apply the bundle's explicit bidding policy to a public observation."""
    if bundle.bidding_policy == "learned":
        raise ValueError(
            "learned bidding must use BundleFactory.choose_bid so its checkpoint "
            "identity is validated"
        )
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
