"""Perfect-information state and bounded team-minimax endgame solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from douzero.env import move_detector as md
from douzero.env.game import GameEnv
from douzero.env.rules import PLAYER_POSITIONS, RuleSet
from douzero.env.scoring import compute_game_result, compute_team_score_magnitude
from douzero.observation.seats import next_seat, previous_seat

from .budget import SearchBudget
from .transposition import TranspositionKey, TranspositionTable


def _team(role: str) -> str:
    return "landlord" if role == "landlord" else "farmer"


@dataclass(frozen=True, slots=True)
class SolveValue:
    """Win probability and expected score from the root team's perspective."""

    win_probability: float
    expected_score: float

    def ordering_key(self) -> tuple[float, float]:
        return self.win_probability, self.expected_score


class _HandInfo:
    def __init__(self, cards: tuple[int, ...]) -> None:
        self.player_hand_cards = list(cards)


class _LegalityView:
    """Minimal read-only shape consumed by GameEnv's canonical legal generator."""

    get_legal_card_play_actions = GameEnv.get_legal_card_play_actions

    def __init__(
        self, hand: tuple[int, ...], acting_role: str, last_move: tuple[int, ...]
    ) -> None:
        self.acting_player_position = acting_role
        self.info_sets = {acting_role: _HandInfo(hand)}
        self.card_play_action_seq = [list(last_move)] if last_move else []


@dataclass(frozen=True, slots=True)
class SearchGameState:
    """Independent perfect-information state produced from one belief sample."""

    hands: Mapping[str, tuple[int, ...]]
    acting_role: str
    last_move: tuple[int, ...]
    last_non_pass_role: str | None
    consecutive_passes: int
    ruleset: RuleSet
    bid_value: int = 0
    bomb_count: int = 0
    rocket_count: int = 0
    played_cards: Mapping[str, tuple[int, ...]] | None = None
    action_counts: Mapping[str, int] | None = None
    winner_position: str | None = None

    def __post_init__(self) -> None:
        if self.acting_role not in PLAYER_POSITIONS:
            raise ValueError(f"unknown acting role {self.acting_role!r}")
        frozen_hands = {
            role: tuple(sorted(int(card) for card in self.hands[role]))
            for role in PLAYER_POSITIONS
        }
        frozen_played = {
            role: tuple((self.played_cards or {}).get(role, ()))
            for role in PLAYER_POSITIONS
        }
        frozen_counts = {
            role: int((self.action_counts or {}).get(role, 0))
            for role in PLAYER_POSITIONS
        }
        object.__setattr__(self, "hands", MappingProxyType(frozen_hands))
        object.__setattr__(self, "played_cards", MappingProxyType(frozen_played))
        object.__setattr__(self, "action_counts", MappingProxyType(frozen_counts))
        object.__setattr__(self, "last_move", tuple(sorted(self.last_move)))
        if self.consecutive_passes not in (0, 1, 2):
            raise ValueError("consecutive_passes must be 0, 1, or 2")

    @property
    def total_cards(self) -> int:
        return sum(len(hand) for hand in self.hands.values())

    @property
    def terminal(self) -> bool:
        return self.winner_position is not None

    def legal_actions(self, budget: SearchBudget | None = None) -> tuple[tuple[int, ...], ...]:
        """Generate exactly the environment's legal action set."""
        if self.terminal:
            return ()
        view = _LegalityView(
            self.hands[self.acting_role], self.acting_role, self.last_move
        )
        check = budget.check if budget is not None else None
        actions = view.get_legal_card_play_actions(budget_check=check)
        return tuple(tuple(sorted(action)) for action in actions)

    def apply(self, action: tuple[int, ...], *, validate: bool = True) -> "SearchGameState":
        """Return an independent child state; this state is never mutated."""
        action = tuple(sorted(int(card) for card in action))
        if validate and action not in self.legal_actions():
            raise ValueError(f"illegal action {action!r} for {self.acting_role}")

        hands = {role: list(cards) for role, cards in self.hands.items()}
        played = {role: list(cards) for role, cards in self.played_cards.items()}
        counts = dict(self.action_counts)
        for card in action:
            try:
                hands[self.acting_role].remove(card)
            except ValueError as exc:
                raise ValueError(
                    f"action {action!r} is not contained in {self.acting_role}'s hand"
                ) from exc

        bomb_count = self.bomb_count
        rocket_count = self.rocket_count
        last_move = self.last_move
        last_non_pass_role = self.last_non_pass_role
        passes = self.consecutive_passes
        if action:
            played[self.acting_role].extend(action)
            counts[self.acting_role] += 1
            move_type = md.get_move_type(list(action))["type"]
            if move_type == md.TYPE_4_BOMB:
                bomb_count += 1
            elif move_type == md.TYPE_5_KING_BOMB:
                rocket_count += 1
            last_move = action
            last_non_pass_role = self.acting_role
            passes = 0
        else:
            passes += 1

        winner = self.acting_role if not hands[self.acting_role] else None
        next_role = next_seat(self.acting_role)
        if winner is None and passes >= 2:
            # Two passes return initiative to the player who made last_move.
            if next_role != last_non_pass_role:
                raise RuntimeError("invalid pass state: initiative did not cycle to leader")
            last_move = ()
            last_non_pass_role = None
            passes = 0

        return SearchGameState(
            hands={role: tuple(cards) for role, cards in hands.items()},
            acting_role=next_role,
            last_move=last_move,
            last_non_pass_role=last_non_pass_role,
            consecutive_passes=passes,
            ruleset=self.ruleset,
            bid_value=self.bid_value,
            bomb_count=bomb_count,
            rocket_count=rocket_count,
            played_cards={role: tuple(cards) for role, cards in played.items()},
            action_counts=counts,
            winner_position=winner,
        )

    def transposition_key(self) -> TranspositionKey:
        return TranspositionKey(
            hands=tuple(self.hands[role] for role in PLAYER_POSITIONS),
            acting_role=self.acting_role,
            last_move=self.last_move,
            last_non_pass_role=self.last_non_pass_role,
            consecutive_passes=self.consecutive_passes,
            bomb_count=self.bomb_count,
            rocket_count=self.rocket_count,
            bid_value=self.bid_value,
            action_counts=tuple(self.action_counts[role] for role in PLAYER_POSITIONS),
            ruleset_hash=self.ruleset.stable_hash(),
        )

    def terminal_value(self, root_team: str) -> SolveValue:
        if self.winner_position is None:
            raise ValueError("terminal_value requires a terminal state")
        result = compute_game_result(
            played_cards={role: list(cards) for role, cards in self.played_cards.items()},
            action_counts=dict(self.action_counts),
            winner_position=self.winner_position,
            bomb_count=self.bomb_count,
            rocket_count=self.rocket_count,
            bid_value=self.bid_value,
            ruleset=self.ruleset,
        )
        won = result.winner_team == root_team
        score = result.landlord_score if root_team == "landlord" else result.farmer_score
        return SolveValue(float(won), float(score))

    def heuristic_value(self, root_team: str) -> SolveValue:
        """Deterministic bounded-rollout estimate based only on sampled state."""
        landlord_cards = len(self.hands["landlord"])
        farmer_cards = min(
            len(self.hands["landlord_up"]), len(self.hands["landlord_down"])
        )
        advantage = farmer_cards - landlord_cards
        if root_team == "farmer":
            advantage = -advantage
        probability = 1.0 / (1.0 + math.exp(-advantage / 2.0))
        signed = 2.0 * probability - 1.0
        magnitude = compute_team_score_magnitude(
            team=root_team,
            bomb_count=self.bomb_count,
            rocket_count=self.rocket_count,
            bid_value=self.bid_value,
            ruleset=self.ruleset,
        )
        return SolveValue(probability, signed * magnitude)


class EndgameSolver:
    """Exact team-minimax solver, cooperatively bounded by SearchBudget."""

    def __init__(self, budget: SearchBudget) -> None:
        self.budget = budget
        self.table: TranspositionTable[
            tuple[str, TranspositionKey], SolveValue
        ] = TranspositionTable(budget.config.max_nodes)

    def solve(self, state: SearchGameState, root_team: str) -> SolveValue:
        if root_team not in ("landlord", "farmer"):
            raise ValueError("root_team must be landlord or farmer")
        return self._solve(state, root_team)

    def _solve(self, state: SearchGameState, root_team: str) -> SolveValue:
        self.budget.visit_node()
        if state.terminal:
            return state.terminal_value(root_team)
        # Values and max/min direction are root-team relative. Scope cache
        # entries by that perspective while retaining reuse within one team.
        key = (root_team, state.transposition_key())
        cached = self.table.get(key)
        if cached is not None:
            return cached

        actions = state.legal_actions(self.budget)
        # Winning moves and larger shedding moves first improve early bests.
        ordered = sorted(actions, key=lambda a: (-len(a), a))
        maximizing = _team(state.acting_role) == root_team
        best: SolveValue | None = None
        for action in ordered:
            self.budget.check()
            child = state.apply(action, validate=False)
            value = self._solve(child, root_team)
            if best is None:
                best = value
            elif maximizing and value.ordering_key() > best.ordering_key():
                best = value
            elif not maximizing and value.ordering_key() < best.ordering_key():
                best = value
        if best is None:  # pragma: no cover - every nonterminal state has a move
            raise RuntimeError("nonterminal state produced no legal action")
        self.table.put(key, best)
        return best


def infer_trick_context(
    acting_role: str, action_history: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], str | None, int]:
    """Infer move-to-beat, its owner, and pass count from public history."""
    role = previous_seat(acting_role)
    passes = 0
    for action in reversed(action_history):
        cards = tuple(sorted(action))
        if cards:
            return cards, role, min(passes, 1)
        passes += 1
        role = previous_seat(role)
        if passes >= 2:
            return (), None, 0
    return (), None, 0
