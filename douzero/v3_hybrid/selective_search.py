"""H7 public-only selective-search gate, safe fallback, and metrics."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from douzero.belief import build_belief_input
from douzero.search import BeliefSearch, SearchConfig

V3_H7_SEARCH_GATE_VERSION = "v3-hybrid-h7-public-gate-v1"


@dataclass(frozen=True)
class V3H7SearchGateConfig:
    enabled: bool = False
    max_total_cards: int = 24
    max_own_cards: int = 10
    max_top2_q_gap: float = 0.15
    min_belief_entropy: float = 0.0
    trigger_on_bomb_risk: bool = True
    trigger_on_spring_risk: bool = True
    require_card_control: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("H7 search gate enabled must be bool")
        for name in ("max_total_cards", "max_own_cards"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"H7 search gate {name} must be non-negative")
        for name in ("max_top2_q_gap", "min_belief_entropy"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"H7 search gate {name} must be finite and non-negative")
        for name in (
            "trigger_on_bomb_risk", "trigger_on_spring_risk",
            "require_card_control",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"H7 search gate {name} must be bool")

    def identity(self) -> dict[str, object]:
        return {"version": V3_H7_SEARCH_GATE_VERSION, **asdict(self)}


@dataclass(frozen=True)
class V3H7SearchRecord:
    base_action_index: int
    selected_action_index: int
    triggered: bool
    trigger_reasons: tuple[str, ...]
    fallback_reason: str
    completed_belief_samples: int
    nodes: int
    rollouts: int
    elapsed_milliseconds: float


@dataclass
class V3H7SearchMetrics:
    decisions: int = 0
    triggered: int = 0
    changed_actions: int = 0
    completed_samples: int = 0
    nodes: int = 0
    rollouts: int = 0
    trigger_counts: Counter = field(default_factory=Counter)
    fallback_counts: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)

    def update(self, record: V3H7SearchRecord) -> None:
        self.decisions += 1
        self.triggered += int(record.triggered)
        self.changed_actions += int(
            record.selected_action_index != record.base_action_index
        )
        self.completed_samples += record.completed_belief_samples
        self.nodes += record.nodes
        self.rollouts += record.rollouts
        self.trigger_counts.update(record.trigger_reasons)
        if record.fallback_reason:
            self.fallback_counts[record.fallback_reason] += 1
        self.latencies_ms.append(record.elapsed_milliseconds)

    def snapshot(self) -> dict[str, object]:
        ordered = sorted(self.latencies_ms)

        def percentile(fraction: float) -> float:
            if not ordered:
                return 0.0
            return ordered[int((len(ordered) - 1) * fraction)]

        return {
            "decisions": self.decisions,
            "triggered": self.triggered,
            "trigger_rate": self.triggered / max(1, self.decisions),
            "changed_actions": self.changed_actions,
            "change_rate": self.changed_actions / max(1, self.decisions),
            "completed_belief_samples": self.completed_samples,
            "nodes": self.nodes,
            "rollouts": self.rollouts,
            "trigger_counts": dict(sorted(self.trigger_counts.items())),
            "fallback_counts": dict(sorted(self.fallback_counts.items())),
            "latency_p50_ms": percentile(0.50),
            "latency_p95_ms": percentile(0.95),
            "latency_p99_ms": percentile(0.99),
        }


class V3SelectiveSearch:
    """Run existing belief search only when a public composite gate fires."""

    def __init__(
        self,
        gate_config: V3H7SearchGateConfig,
        search_config: SearchConfig,
        ruleset,
        *,
        search_compatible: bool,
        exception_reporter: Callable[[Exception], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if gate_config.enabled != search_config.enabled:
            raise ValueError("H7 gate and search enabled flags disagree")
        if not isinstance(search_compatible, bool):
            raise TypeError("search_compatible must be bool")
        self.gate_config = gate_config
        self.search_config = search_config
        self.search_compatible = search_compatible
        self.search = BeliefSearch(search_config, ruleset)
        self.exception_reporter = exception_reporter
        self.stop_requested = stop_requested or (lambda: False)
        self.metrics = V3H7SearchMetrics()

    @staticmethod
    def _base_record(
        base_action_index: int,
        *,
        reason: str,
        started: float,
        triggered: bool = False,
        trigger_reasons: tuple[str, ...] = (),
        samples: int = 0,
        nodes: int = 0,
        rollouts: int = 0,
    ) -> V3H7SearchRecord:
        return V3H7SearchRecord(
            base_action_index=base_action_index,
            selected_action_index=base_action_index,
            triggered=triggered,
            trigger_reasons=trigger_reasons,
            fallback_reason=reason,
            completed_belief_samples=samples,
            nodes=nodes,
            rollouts=rollouts,
            elapsed_milliseconds=(time.monotonic() - started) * 1000.0,
        )

    def _finish(self, record: V3H7SearchRecord) -> V3H7SearchRecord:
        self.metrics.update(record)
        return record

    def _gate_reasons(self, observation, model_output, belief_output) -> tuple[str, ...]:
        public = observation.public
        cfg = self.gate_config
        total_cards = sum(int(value) for value in public.num_cards_left.values())
        own_cards = int(public.num_cards_left[public.acting_role])
        reasons: list[str] = []
        if total_cards <= cfg.max_total_cards:
            reasons.append("remaining_cards")
        if own_cards <= cfg.max_own_cards:
            reasons.append("own_cards")
        mask = model_output.action_mask.bool()
        q_values = model_output.dmc_q.squeeze(-1)[mask]
        if q_values.numel() >= 2:
            top = torch.topk(q_values.float(), 2).values
            if float(top[0] - top[1]) <= cfg.max_top2_q_gap:
                reasons.append("top2_q_gap")
        unseen = tuple(public.other_handcards)
        bomb_risk = any(unseen.count(rank) >= 4 for rank in set(unseen)) or (
            20 in unseen and 30 in unseen
        )
        if cfg.trigger_on_bomb_risk and bomb_risk:
            reasons.append("bomb_risk")
        landlord_moves = int(public.non_pass_action_counts.get("landlord", 0))
        farmer_moves = sum(
            int(public.non_pass_action_counts.get(role, 0))
            for role in ("landlord_up", "landlord_down")
        )
        if cfg.trigger_on_spring_risk and (
            (landlord_moves == 0 and public.acting_role != "landlord")
            or (farmer_moves == 0 and public.acting_role == "landlord")
        ):
            reasons.append("spring_risk")
        if belief_output is not None:
            entropy = float(np.asarray(belief_output.entropy).reshape(-1)[0])
            if entropy >= cfg.min_belief_entropy:
                reasons.append("belief_entropy")
        if cfg.require_card_control and public.last_move:
            reasons.append("card_control")
        return tuple(reasons)

    @staticmethod
    def _belief_is_conserved(observation, belief_output) -> bool:
        if belief_output is None:
            return True
        binput = build_belief_input(observation.public)
        expected = np.asarray(belief_output.expected_counts)[0]
        if expected.shape != binput.unseen_counts.shape:
            return False
        if not np.isfinite(expected).all():
            return False
        if np.any(expected < -1e-8) or np.any(expected - binput.unseen_counts > 1e-8):
            return False
        return math.isclose(
            float(expected.sum()),
            float(np.asarray(belief_output.opponent_a_total).reshape(-1)[0]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    def select(
        self,
        *,
        observation,
        model_output,
        base_action_index: int,
        belief_model,
        belief_output=None,
    ) -> V3H7SearchRecord:
        started = time.monotonic()
        if getattr(getattr(observation, "public", None), "kind", None) != "public":
            raise TypeError("H7 selective search requires PublicObservation")
        legal_actions = tuple(observation.actions.legal_actions)
        if not 0 <= base_action_index < len(legal_actions):
            raise ValueError("base action index is outside the legal action list")
        if not self.gate_config.enabled:
            return self._finish(self._base_record(
                base_action_index, reason="disabled", started=started
            ))
        if not self.search_compatible:
            return self._finish(self._base_record(
                base_action_index, reason="package_not_search_compatible", started=started
            ))
        if self.stop_requested():
            return self._finish(self._base_record(
                base_action_index, reason="global_stop", started=started
            ))
        if not self._belief_is_conserved(observation, belief_output):
            return self._finish(self._base_record(
                base_action_index, reason="belief_not_conserved", started=started
            ))
        reasons = self._gate_reasons(observation, model_output, belief_output)
        if not reasons:
            return self._finish(self._base_record(
                base_action_index, reason="gate_not_triggered", started=started
            ))
        try:
            decision = self.search.select(
                observation=observation,
                model_output=model_output,
                base_action_index=base_action_index,
                belief_model=belief_model,
                belief_output=belief_output,
            )
            if tuple(observation.actions.legal_actions) != legal_actions:
                return self._finish(self._base_record(
                    base_action_index,
                    reason="legal_actions_changed",
                    started=started,
                    triggered=True,
                    trigger_reasons=reasons,
                ))
            if not 0 <= decision.action_index < len(legal_actions):
                return self._finish(self._base_record(
                    base_action_index,
                    reason="action_alignment_error",
                    started=started,
                    triggered=True,
                    trigger_reasons=reasons,
                ))
            log = decision.log
            record = V3H7SearchRecord(
                base_action_index=base_action_index,
                selected_action_index=decision.action_index,
                triggered=True,
                trigger_reasons=reasons,
                fallback_reason=log.fallback_reason,
                completed_belief_samples=log.samples,
                nodes=log.nodes,
                rollouts=log.rollouts,
                elapsed_milliseconds=(time.monotonic() - started) * 1000.0,
            )
            return self._finish(record)
        except Exception as exc:
            if self.exception_reporter is not None:
                self.exception_reporter(exc)
            reason = f"search_exception:{type(exc).__name__}:{exc}"
            return self._finish(self._base_record(
                base_action_index,
                reason=reason,
                started=started,
                triggered=True,
                trigger_reasons=reasons,
            ))


__all__ = [
    "V3_H7_SEARCH_GATE_VERSION",
    "V3H7SearchGateConfig",
    "V3H7SearchMetrics",
    "V3H7SearchRecord",
    "V3SelectiveSearch",
]
