"""Fixed-layout P09 tactical features derived only from public information."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from douzero.env.move_detector import get_move_type
from douzero.env.utils import TYPE_0_PASS, TYPE_1_SINGLE, TYPE_4_BOMB, TYPE_5_KING_BOMB

from .config import StrategyFeatureConfig
from .hand_decomposition import hand_decomposition
from .structure import action_structure_cost

STRATEGY_FEATURE_NAMES: tuple[str, ...] = (
    "min_turns_before", "min_turns_after", "min_turns_delta", "min_turns_exact",
    "single_delta", "pair_delta", "triple_delta", "straight_delta",
    "serial_pair_delta", "airplane_delta", "bomb_delta", "bomb_break_cost",
    "joker_pair_break", "high_control_card_cost", "structure_cost",
    "takes_initiative", "control_strength", "blocks_one_card", "blocks_two_cards",
    "blocks_three_cards", "teammate_cards_left", "landlord_cards_left",
    "suppresses_teammate", "feeds_teammate", "is_landlord_up", "is_landlord_down",
    "spring_risk", "bomb_opportunity_cost",
)
STRATEGY_FEATURE_WIDTH: int = len(STRATEGY_FEATURE_NAMES)


def _team(role: str) -> str:
    return "landlord" if role == "landlord" else "farmer"


def _teammate(role: str) -> str | None:
    if role == "landlord_up":
        return "landlord_down"
    if role == "landlord_down":
        return "landlord_up"
    return None


@lru_cache(maxsize=131_072)
def _cached_decomposition(
    hand: tuple[int, ...], node_budget: int, time_budget_ms: int
):
    return hand_decomposition(
        hand, node_budget=node_budget, time_budget_ms=time_budget_ms
    )


def _safe_norm(value: float, scale: float) -> float:
    return float(value) / float(scale)


def _control_strength(move: tuple[int, ...]) -> float:
    if not move:
        return 0.0
    info = get_move_type(list(move))
    if info["type"] == TYPE_5_KING_BOMB:
        return 1.0
    if info["type"] == TYPE_4_BOMB:
        return 0.9
    main_rank = float(info.get("rank", max(move)))
    return min(1.0, main_rank / 30.0 + min(len(move), 12) / 60.0)


def _spring_risk(public) -> float:
    landlord_plays = len(public.played_cards.get("landlord", ()))
    farmer_plays = len(public.played_cards.get("landlord_up", ())) + len(
        public.played_cards.get("landlord_down", ())
    )
    if public.acting_role == "landlord":
        return float(farmer_plays == 0 and public.num_cards_left.get("landlord", 0) <= 6)
    return float(landlord_plays <= 1 and public.num_cards_left.get("landlord", 0) <= 6)


def _feature_row(public, move: tuple[int, ...], cfg: StrategyFeatureConfig) -> list[float]:
    hand = tuple(sorted(public.my_handcards))
    remaining = list(hand)
    for card in move:
        remaining.remove(card)
    remaining_key = tuple(remaining)
    info = get_move_type(list(move))
    row = [0.0] * STRATEGY_FEATURE_WIDTH

    if cfg.hand_enabled:
        before = _cached_decomposition(hand, cfg.node_budget, cfg.time_budget_ms)
        after = _cached_decomposition(remaining_key, cfg.node_budget, cfg.time_budget_ms)
        row[0:4] = [
            _safe_norm(before.min_turns, 20.0),
            _safe_norm(after.min_turns, 20.0),
            _safe_norm(after.min_turns - before.min_turns, 20.0),
            float(before.exact and after.exact),
        ]

    structure = action_structure_cost(hand, move)
    if cfg.structure_enabled:
        row[4:15] = [
            _safe_norm(structure.single_delta, 4.0),
            _safe_norm(structure.pair_delta, 4.0),
            _safe_norm(structure.triple_delta, 4.0),
            _safe_norm(structure.straight_delta, 8.0),
            _safe_norm(structure.serial_pair_delta, 6.0),
            _safe_norm(structure.airplane_delta, 4.0),
            _safe_norm(structure.bomb_delta, 2.0),
            structure.bomb_break_cost,
            structure.joker_pair_break,
            _safe_norm(structure.high_control_card_cost, 4.0),
            _safe_norm(structure.total, 12.0),
        ]

    role = public.acting_role
    threat_count = (
        public.num_cards_left.get("landlord", 0)
        if role != "landlord"
        else min(
            public.num_cards_left.get("landlord_up", 20),
            public.num_cards_left.get("landlord_down", 20),
        )
    )
    non_pass = info["type"] != TYPE_0_PASS
    is_single = info["type"] == TYPE_1_SINGLE
    single_rank = move[0] if is_single else 0
    if cfg.control_enabled:
        # Every legal non-pass becomes the current trick's controlling move;
        # whether that control survives later responses is an auxiliary target.
        takes_initiative = float(non_pass)
        row[15:20] = [
            takes_initiative,
            _control_strength(move),
            float(threat_count == 1 and (not is_single or single_rank >= 17)),
            float(threat_count == 2 and info["type"] not in (TYPE_0_PASS, TYPE_1_SINGLE)),
            float(threat_count == 3 and non_pass and len(move) >= 2),
        ]

    teammate = _teammate(role)
    teammate_left = public.num_cards_left.get(teammate, 0) if teammate else 0
    last_from_teammate = bool(
        teammate
        and public.last_move
        and tuple(public.last_move_dict.get(teammate, ())) == tuple(public.last_move)
    )
    if cfg.cooperation_enabled:
        row[20:26] = [
            _safe_norm(teammate_left, 20.0),
            _safe_norm(public.num_cards_left.get("landlord", 0), 20.0),
            float(last_from_teammate and non_pass),
            float(teammate_left == 1 and is_single and single_rank <= 10),
            float(role == "landlord_up"),
            float(role == "landlord_down"),
        ]

    if cfg.risk_enabled:
        row[26:28] = [
            _spring_risk(public),
            float(info["type"] in (TYPE_4_BOMB, TYPE_5_KING_BOMB))
            + 0.5 * structure.bomb_break_cost,
        ]
    return row


def build_strategy_feature_matrix(public, config: StrategyFeatureConfig | None = None) -> np.ndarray:
    """Return ``(N, 28)`` float32 features for the public legal-action list."""

    cfg = config or StrategyFeatureConfig()
    rows = [_feature_row(public, tuple(action), cfg) for action in public.legal_actions]
    out = np.asarray(rows, dtype=np.float32)
    if out.shape != (len(public.legal_actions), STRATEGY_FEATURE_WIDTH):
        raise RuntimeError(
            f"strategy feature shape drift: expected "
            f"({len(public.legal_actions)}, {STRATEGY_FEATURE_WIDTH}), got {out.shape}"
        )
    out.setflags(write=False)
    return out
