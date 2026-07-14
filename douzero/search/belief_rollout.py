"""Budgeted belief sampling, deterministic rollout, and action aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from douzero.belief import BELIEF_RANKS, build_belief_input
from douzero.belief.model import BeliefModel, BeliefOutput
from douzero.env.rules import PLAYER_POSITIONS, RuleSet
from douzero.models_v2.output import ModelOutput

from .budget import BudgetExceeded, SearchBudget, SearchConfig
from .candidate import Candidate, select_top_k
from .endgame_solver import (
    EndgameSolver,
    SearchGameState,
    SolveValue,
    infer_trick_context,
)


@dataclass(frozen=True, slots=True)
class CandidateValue:
    """Aggregated search statistics for one root action."""

    action: tuple[int, ...]
    mean_win_probability: float
    expected_score: float
    lower_confidence: float
    num_samples: int


@dataclass(frozen=True, slots=True)
class SearchLog:
    """Structured audit record for one optional search decision."""

    base_action: tuple[int, ...]
    searched_action: tuple[int, ...]
    candidate_values: tuple[CandidateValue, ...]
    samples: int
    nodes: int
    rollouts: int
    elapsed_milliseconds: float
    timed_out: bool
    fallback_reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True, slots=True)
class SearchDecision:
    """Selected legal-action index and its search audit record."""

    action_index: int
    log: SearchLog


FastPolicy = Callable[[SearchGameState, tuple[tuple[int, ...], ...]], tuple[int, ...]]


def fast_policy(
    state: SearchGameState, actions: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Deterministic cheap rollout policy: shed most cards, conserve bombs."""
    hand_size = len(state.hands[state.acting_role])

    def key(action: tuple[int, ...]) -> tuple[int, int, int, tuple[int, ...]]:
        wins = int(len(action) == hand_size and bool(action))
        is_bomb = int(len(action) == 4 and len(set(action)) == 1)
        is_rocket = int(action == (20, 30))
        return (-wins, is_bomb + is_rocket, -len(action), action)

    return min(actions, key=key)


def _sampled_hands(
    public,
    belief_input,
    allocation_a: np.ndarray,
) -> dict[str, tuple[int, ...]]:
    """Convert one legal rank allocation into three independent hands."""
    counts_a = np.asarray(allocation_a, dtype=np.int64)
    counts_b = belief_input.unseen_counts.astype(np.int64) - counts_a
    if np.any(counts_b < 0):
        raise ValueError("belief allocation exceeds public unseen pool")

    def expand(counts: np.ndarray) -> list[int]:
        cards: list[int] = []
        for rank, count in zip(BELIEF_RANKS, counts.tolist()):
            cards.extend([int(rank)] * int(count))
        return cards

    hands = {
        public.acting_role: list(public.my_handcards),
        belief_input.opponent_a_role: expand(counts_a),
        belief_input.opponent_b_role: expand(counts_b),
    }
    # Unplayed bottom cards are public landlord property and were intentionally
    # excluded from the belief pool. Add them back after sampling.
    if public.acting_role != "landlord":
        hands["landlord"].extend(public.bottom_cards.unplayed)
    return {role: tuple(sorted(hands[role])) for role in PLAYER_POSITIONS}


def state_from_sample(public, ruleset: RuleSet, hands) -> SearchGameState:
    """Build an independent search state using public data plus one sample."""
    last_move, owner, passes = infer_trick_context(
        public.acting_role, public.action_history
    )
    # Older PublicObservation producers may omit action_history while still
    # carrying last_move. Use that public field as a conservative fallback.
    if not public.action_history and public.last_move:
        last_move = tuple(public.last_move)
        owner = previous_role = None
        for role, action in public.last_move_dict.items():
            if tuple(sorted(action)) == tuple(sorted(public.last_move)):
                previous_role = role
        owner = previous_role
    return SearchGameState(
        hands=hands,
        acting_role=public.acting_role,
        last_move=last_move,
        last_non_pass_role=owner,
        consecutive_passes=passes,
        ruleset=ruleset,
        bid_value=public.bid_value,
        bomb_count=public.bomb_count,
        rocket_count=public.rocket_count,
        played_cards=public.played_cards,
        action_counts=public.non_pass_action_counts,
    )


class BeliefSearch:
    """Optional root search that never accepts a true hidden-hand allocation."""

    def __init__(
        self,
        config: SearchConfig,
        ruleset: RuleSet,
        policy: FastPolicy | None = None,
    ) -> None:
        self.config = config
        self.ruleset = ruleset
        self.policy = policy or fast_policy

    def select(
        self,
        *,
        observation,
        model_output: ModelOutput,
        base_action_index: int,
        belief_model: BeliefModel,
        belief_output: BeliefOutput | None = None,
    ) -> SearchDecision:
        """Search top-k actions and fail closed to the base action on limits."""
        if getattr(getattr(observation, "public", None), "kind", None) != "public":
            raise TypeError(
                "BeliefSearch requires an ObservationV2 with a public payload"
            )
        legal_actions = observation.actions.legal_actions
        base_action = tuple(legal_actions[base_action_index])
        budget = SearchBudget(self.config)
        if not self.config.enabled:
            return self._fallback(base_action_index, base_action, budget, "disabled")
        if (
            self.config.max_nodes == 0
            or self.config.max_rollouts == 0
            or self.config.max_milliseconds == 0
        ):
            return self._fallback(
                base_action_index, base_action, budget, "zero budget", True
            )
        candidates = select_top_k(
            legal_actions,
            model_output,
            self.config.top_k,
            mode=self.config.selection_mode,
        )
        if not candidates:
            return self._fallback(base_action_index, base_action, budget, "no candidates")

        binput = build_belief_input(observation.public)
        if belief_output is None:
            with torch.inference_mode():
                belief_output = belief_model([binput])
        rng = np.random.default_rng(self.config.seed)

        try:
            budget.check()
            allocations = belief_model.sample(
                belief_output, rng, num_samples=self.config.belief_samples
            )[0]
            sampled_states = [
                state_from_sample(
                    observation.public,
                    self.ruleset,
                    _sampled_hands(observation.public, binput, allocation),
                )
                for allocation in allocations
            ]
            values: list[CandidateValue] = []
            for candidate in candidates:
                outcomes: list[SolveValue] = []
                for root in sampled_states:
                    budget.start_rollout()
                    child = root.apply(candidate.action)
                    outcomes.append(self._evaluate(child, _team(root.acting_role), budget))
                win = np.asarray([value.win_probability for value in outcomes])
                scores = np.asarray([value.expected_score for value in outcomes])
                stderr = float(win.std(ddof=0) / np.sqrt(max(1, len(win))))
                values.append(CandidateValue(
                    action=candidate.action,
                    mean_win_probability=float(win.mean()),
                    expected_score=float(scores.mean()),
                    lower_confidence=float(win.mean() - self.config.risk_penalty * stderr),
                    num_samples=len(outcomes),
                ))
            selected = self._choose(candidates, values)
            log = SearchLog(
                base_action=base_action,
                searched_action=selected.action,
                candidate_values=tuple(values),
                samples=len(sampled_states),
                nodes=budget.nodes,
                rollouts=budget.rollouts,
                elapsed_milliseconds=budget.elapsed_milliseconds,
                timed_out=False,
            )
            return SearchDecision(selected.index, log)
        except BudgetExceeded as exc:
            return self._fallback(base_action_index, base_action, budget, str(exc), True)

    def _evaluate(
        self, state: SearchGameState, root_team: str, budget: SearchBudget
    ) -> SolveValue:
        if state.terminal:
            return state.terminal_value(root_team)
        if state.total_cards <= self.config.endgame_cards_threshold:
            return EndgameSolver(budget).solve(state, root_team)
        current = state
        for _ in range(self.config.rollout_depth):
            budget.visit_node()
            if current.terminal:
                return current.terminal_value(root_team)
            actions = current.legal_actions(budget)
            current = current.apply(self.policy(current, actions), validate=False)
        if current.terminal:
            return current.terminal_value(root_team)
        return current.heuristic_value(root_team)

    def _choose(
        self, candidates: tuple[Candidate, ...], values: list[CandidateValue]
    ) -> Candidate:
        pairs = list(zip(candidates, values))
        if self.config.selection_mode == "win":
            key = lambda pair: (pair[1].lower_confidence, -pair[0].index)
        elif self.config.selection_mode == "score":
            key = lambda pair: (pair[1].expected_score, -pair[0].index)
        else:
            key = lambda pair: (
                pair[1].lower_confidence,
                pair[1].expected_score,
                -pair[0].index,
            )
        return max(pairs, key=key)[0]

    @staticmethod
    def _fallback(
        index: int,
        action: tuple[int, ...],
        budget: SearchBudget,
        reason: str,
        timed_out: bool = False,
    ) -> SearchDecision:
        return SearchDecision(index, SearchLog(
            base_action=action,
            searched_action=action,
            candidate_values=(),
            samples=0,
            nodes=budget.nodes,
            rollouts=budget.rollouts,
            elapsed_milliseconds=budget.elapsed_milliseconds,
            timed_out=timed_out,
            fallback_reason=reason,
        ))


def _team(role: str) -> str:
    return "landlord" if role == "landlord" else "farmer"
