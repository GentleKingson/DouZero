"""Deterministic replay validation for formal P17 game evidence.

The replay boundary deliberately accepts an approved deal separately from the
untrusted result row.  Terminal summaries in the row are never consulted: a
successful replay derives every outcome field from :class:`GameEnv`.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from douzero.env.game import GameEnv
from douzero.env.rules import PHASE_BIDDING, RuleSet
from douzero.env.scoring import compute_game_result

from .legacy_data_adapter import _validate_standard_record, deal_standard_deck
from .scenario import ROLES, canonical_deal_hash, canonical_deal_id


EVALUATION_TRACE_SCHEMA_VERSION = "p17-deterministic-game-trace-v1"
REDEAL_CAP_EXCLUSION_REASON = "redeal_cap_exhausted_forced_smoke_fallback"

_CARD_RANKS = frozenset((*range(3, 15), 17, 20, 30))
_DEAL_ID_RE = re.compile(r"^(?P<index>[0-9]{6,12})-(?P<prefix>[0-9a-f]{12})$")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NEUTRAL_SEATS = ("0", "1", "2")


class ReplayValidationError(ValueError):
    """Raised when untrusted trace evidence cannot be replayed exactly."""


@dataclass(frozen=True)
class ReplayOutcome:
    """Trusted terminal facts derived exclusively from deterministic replay."""

    winner_position: str
    winner_team: str
    bid_value: int
    bomb_count: int
    rocket_count: int
    spring: bool
    anti_spring: bool
    game_length: int
    seat_to_role: dict[str, str] | None
    bidding_order: tuple[str, ...]
    bidding_history: tuple[tuple[str, int], ...]
    redeal_count: int
    max_redeals_exceeded: bool
    team_scores: dict[str, float]
    cardplay_legal_action_counts: tuple[int, ...]


def _trace_json_value(value: Any) -> Any:
    """Normalize tuples to JSON arrays without accepting opaque objects."""

    if isinstance(value, (list, tuple)):
        return [_trace_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "evaluation traces may contain only JSON scalar values and arrays"
    )


def evaluation_trace_digest(
    *,
    mode: str,
    deal_hash: str,
    bidding_trace: Sequence[Any],
    cardplay_trace: Sequence[Any],
) -> str:
    """Return the versioned canonical SHA-256 for one complete game trace."""

    payload = {
        "schema_version": EVALUATION_TRACE_SCHEMA_VERSION,
        "mode": mode,
        "deal_hash": deal_hash,
        "bidding_trace": _trace_json_value(bidding_trace),
        "cardplay_trace": _trace_json_value(cardplay_trace),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _seed(base: int, *parts: object) -> int:
    """Mirror paired evaluation's stable, process-independent seed derivation."""

    token = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def _validate_legacy_deal(deal: Mapping[str, Any]) -> None:
    required = {
        "landlord",
        "landlord_up",
        "landlord_down",
        "three_landlord_cards",
    }
    if set(deal) != required:
        raise ReplayValidationError(
            f"approved legacy deal must contain exactly {sorted(required)}"
        )
    for field in required:
        cards = deal[field]
        if not isinstance(cards, list):
            raise ReplayValidationError(f"approved deal {field} must be a list")
        if any(
            isinstance(card, bool)
            or not isinstance(card, int)
            or card not in _CARD_RANKS
            for card in cards
        ):
            raise ReplayValidationError(f"approved deal {field} contains invalid cards")
    if not (
        len(deal["landlord"]) == 20
        and len(deal["landlord_up"]) == 17
        and len(deal["landlord_down"]) == 17
        and len(deal["three_landlord_cards"]) == 3
    ):
        raise ReplayValidationError("approved legacy deal has invalid hand sizes")
    expected = Counter((*range(3, 15),) * 4 + (17,) * 4 + (20, 30))
    dealt = Counter(
        list(deal["landlord"])
        + list(deal["landlord_up"])
        + list(deal["landlord_down"])
    )
    if dealt != expected:
        raise ReplayValidationError("approved legacy deal is not a valid 54-card deal")
    if Counter(deal["three_landlord_cards"]) - Counter(deal["landlord"]):
        raise ReplayValidationError(
            "approved legacy bottom cards are not in the landlord hand"
        )


def _validate_deal(
    approved_deal: Mapping[str, Any], *, mode: str, ruleset: RuleSet
) -> str:
    if not isinstance(approved_deal, Mapping):
        raise ReplayValidationError("approved_deal must be a mapping")
    if mode == "cardplay_only":
        if ruleset.ruleset_id != "legacy":
            raise ReplayValidationError("cardplay_only replay requires a legacy ruleset")
        _validate_legacy_deal(approved_deal)
    elif mode == "full_game":
        if ruleset.ruleset_id != "standard":
            raise ReplayValidationError("full_game replay requires a standard ruleset")
        try:
            _validate_standard_record(dict(approved_deal), 0, ruleset)
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayValidationError(f"approved standard deal is invalid: {exc}") from exc
        if approved_deal.get("bidding_script") is not None:
            raise ReplayValidationError(
                "formal replay does not accept an embedded bidding_script"
            )
    else:
        raise ReplayValidationError(f"unsupported evaluation mode {mode!r}")
    try:
        return canonical_deal_hash(approved_deal)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayValidationError(f"approved deal cannot be hashed: {exc}") from exc


def _validate_bidding_trace(value: Any, *, mode: str) -> list[list[tuple[str, int]]]:
    if not isinstance(value, list):
        raise ReplayValidationError("bidding_trace must be a list of attempts")
    attempts: list[list[tuple[str, int]]] = []
    for attempt_index, raw_attempt in enumerate(value):
        if not isinstance(raw_attempt, list) or not raw_attempt:
            raise ReplayValidationError(
                f"bidding attempt {attempt_index} must be a non-empty list"
            )
        attempt: list[tuple[str, int]] = []
        for action_index, raw_action in enumerate(raw_attempt):
            if not isinstance(raw_action, list) or len(raw_action) != 2:
                raise ReplayValidationError(
                    f"bidding action {attempt_index}:{action_index} must be [seat, bid]"
                )
            seat, bid = raw_action
            if seat not in _NEUTRAL_SEATS:
                raise ReplayValidationError(
                    f"bidding action {attempt_index}:{action_index} has invalid seat"
                )
            if isinstance(bid, bool) or not isinstance(bid, int):
                raise ReplayValidationError(
                    f"bidding action {attempt_index}:{action_index} has invalid bid"
                )
            attempt.append((seat, bid))
        attempts.append(attempt)
    if mode == "cardplay_only" and attempts:
        raise ReplayValidationError("cardplay_only replay must not contain bidding")
    if mode == "full_game" and not attempts:
        raise ReplayValidationError("full_game replay requires a bidding trace")
    return attempts


def _validate_cardplay_trace(value: Any) -> list[tuple[str, list[int]]]:
    if not isinstance(value, list) or not value:
        raise ReplayValidationError("cardplay_trace must be a non-empty list")
    trace: list[tuple[str, list[int]]] = []
    for index, raw_action in enumerate(value):
        if not isinstance(raw_action, list) or len(raw_action) != 2:
            raise ReplayValidationError(
                f"card-play action {index} must be [role, cards]"
            )
        role, cards = raw_action
        if role not in ROLES:
            raise ReplayValidationError(f"card-play action {index} has invalid role")
        if not isinstance(cards, list):
            raise ReplayValidationError(f"card-play action {index} cards must be a list")
        if any(
            isinstance(card, bool)
            or not isinstance(card, int)
            or card not in _CARD_RANKS
            for card in cards
        ):
            raise ReplayValidationError(
                f"card-play action {index} contains an invalid card"
            )
        if cards != sorted(cards):
            raise ReplayValidationError(
                f"card-play action {index} cards must be in canonical sorted order"
            )
        trace.append((role, list(cards)))
    return trace


class _TraceAgent:
    def __init__(self) -> None:
        self.action: list[int] = []

    def act(self, _infoset: Any) -> list[int]:
        return list(self.action)


def _replay_cardplay(
    env: GameEnv, trace: list[tuple[str, list[int]]]
) -> tuple[int, ...]:
    legal_action_counts: list[int] = []
    for index, (role, action) in enumerate(trace):
        if env.game_over:
            raise ReplayValidationError(
                f"card-play trace continues after terminal action {index - 1}"
            )
        if role != env.acting_player_position:
            raise ReplayValidationError(
                f"card-play action {index} is out of turn: expected "
                f"{env.acting_player_position!r}, got {role!r}"
            )
        legal_action_counts.append(len(env.game_infoset.legal_actions))
        if action not in env.game_infoset.legal_actions:
            raise ReplayValidationError(f"card-play action {index} is illegal")
        agent = env.players[role]
        agent.action = list(action)
        try:
            env.step()
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            raise ReplayValidationError(
                f"card-play action {index} could not be replayed"
            ) from exc
    if not env.game_over:
        raise ReplayValidationError("card-play trace ended before the game was terminal")
    return tuple(legal_action_counts)


def _replay_bidding(
    env: GameEnv,
    attempts: list[list[tuple[str, int]]],
    *,
    approved_deal: Mapping[str, Any],
    deal_id: str,
    deterministic_seed: int,
    ruleset: RuleSet,
) -> tuple[int, bool]:
    if len(attempts) > ruleset.max_redeals + 1:
        raise ReplayValidationError("bidding trace exceeds the ruleset redeal cap")
    bidding_order = list(approved_deal["bidding_order"])
    deck = list(approved_deal["deck"])
    redeal_count = 0
    for attempt_index, attempt in enumerate(attempts):
        env.reset()
        env.card_play_init_standard(
            deal_standard_deck(deck), bidding_order=bidding_order
        )
        redeal = False
        for action_index, (seat, bid) in enumerate(attempt):
            if env.phase != PHASE_BIDDING:
                raise ReplayValidationError(
                    f"bidding attempt {attempt_index} continues after resolution"
                )
            if seat != env.acting_player_position:
                raise ReplayValidationError(
                    f"bidding action {attempt_index}:{action_index} is out of turn"
                )
            if bid not in env.get_legal_bids():
                raise ReplayValidationError(
                    f"bidding action {attempt_index}:{action_index} is illegal"
                )
            redeal = env.step_bidding(bid)
            if redeal and action_index != len(attempt) - 1:
                raise ReplayValidationError(
                    f"bidding attempt {attempt_index} continues after all-pass"
                )
        # All-pass redeal deliberately leaves GameEnv in BIDDING; the caller
        # resets it with the deterministically shuffled deck for the retry.
        if env.phase == PHASE_BIDDING and not redeal:
            raise ReplayValidationError(
                f"bidding attempt {attempt_index} ended before resolution"
            )
        if redeal:
            redeal_count += 1
            deck = list(approved_deal["deck"])
            random.Random(
                _seed(
                    deterministic_seed,
                    deal_id,
                    "redeal",
                    attempt_index,
                )
            ).shuffle(deck)
            if attempt_index == len(attempts) - 1:
                if len(attempts) != ruleset.max_redeals + 1:
                    raise ReplayValidationError(
                        "bidding trace has no successful terminal attempt"
                    )
                # Mirror the evaluator's bounded smoke fallback exactly. The
                # last all-pass attempt deterministically prepares one more
                # shuffled deck, then force-assigns the first bidder at bid 1.
                env.reset()
                env.card_play_init_standard(
                    deal_standard_deck(deck), bidding_order=bidding_order
                )
                env.landlord_position = bidding_order[0]
                env.bid_value = 1
                env.bidding_history = [(bidding_order[0], 1)]
                env._reveal_bottom_cards()
                return redeal_count, True
            continue
        if attempt_index != len(attempts) - 1:
            raise ReplayValidationError(
                "bidding trace contains attempts after bidding resolved"
            )
        return redeal_count, False
    raise ReplayValidationError("bidding trace has no playable terminal attempt")


def _trusted_outcome(
    env: GameEnv,
    *,
    mode: str,
    ruleset: RuleSet,
    redeal_count: int,
    max_redeals_exceeded: bool,
    cardplay_legal_action_counts: tuple[int, ...],
) -> ReplayOutcome:
    empty_roles = [
        role for role in ROLES if not env.info_sets[role].player_hand_cards
    ]
    if len(empty_roles) != 1:
        raise ReplayValidationError("replay did not produce a unique terminal winner")

    if mode == "full_game":
        result = env.game_result
        if result is None:
            raise ReplayValidationError("standard replay did not produce GameResult")
        seat_to_role: dict[str, str] | None = dict(env._seat_to_role)
        bidding_order = tuple(str(seat) for seat in env.bidding_order)
        bidding_history = tuple(
            (str(seat), int(bid)) for seat, bid in env.bidding_history
        )
    else:
        result = compute_game_result(
            played_cards=env.played_cards,
            action_counts=dict(env.action_counts),
            winner_position=empty_roles[0],
            bomb_count=env.bomb_count,
            rocket_count=env.rocket_count,
            bid_value=0,
            ruleset=ruleset,
        )
        seat_to_role = None
        bidding_order = ()
        bidding_history = ()
    if result.winner_position != empty_roles[0]:
        raise ReplayValidationError("GameEnv winner does not match terminal hands")
    return ReplayOutcome(
        winner_position=result.winner_position,
        winner_team=result.winner_team,
        bid_value=int(result.bid_value),
        bomb_count=int(result.bomb_count),
        rocket_count=int(result.rocket_count),
        spring=bool(result.spring),
        anti_spring=bool(result.anti_spring),
        game_length=len(env.card_play_action_seq),
        seat_to_role=seat_to_role,
        bidding_order=bidding_order,
        bidding_history=bidding_history,
        redeal_count=redeal_count,
        max_redeals_exceeded=max_redeals_exceeded,
        team_scores={
            "landlord": float(result.landlord_score),
            "farmer": float(result.farmer_score),
        },
        cardplay_legal_action_counts=cardplay_legal_action_counts,
    )


def replay_game_record(
    row: Mapping[str, Any],
    approved_deal: Mapping[str, Any],
    *,
    mode: str,
    ruleset: RuleSet,
    deterministic_seed: int,
) -> ReplayOutcome:
    """Replay one untrusted row against a separately approved deal.

    The function validates the full trace before returning.  Fields such as
    ``winner_position``, score summaries, bomb counts, and spring flags in
    ``row`` are intentionally ignored.
    """

    if not isinstance(row, Mapping):
        raise ReplayValidationError("game row must be a mapping")
    if not isinstance(ruleset, RuleSet):
        raise ReplayValidationError("ruleset must be a RuleSet")
    if isinstance(deterministic_seed, bool) or not isinstance(
        deterministic_seed, int
    ):
        raise ReplayValidationError("deterministic_seed must be an integer")
    if row.get("mode") != mode:
        raise ReplayValidationError("game row mode does not match replay mode")

    approved_hash = _validate_deal(approved_deal, mode=mode, ruleset=ruleset)
    row_hash = row.get("deal_hash")
    if not isinstance(row_hash, str) or not _HEX_SHA256_RE.fullmatch(row_hash):
        raise ReplayValidationError("game row deal_hash must be a full SHA-256")
    if not hmac.compare_digest(row_hash, approved_hash):
        raise ReplayValidationError("game row deal_hash does not match approved deal")

    raw_bidding_trace = row.get("bidding_trace")
    raw_cardplay_trace = row.get("cardplay_trace")
    bidding_trace = _validate_bidding_trace(raw_bidding_trace, mode=mode)
    cardplay_trace = _validate_cardplay_trace(raw_cardplay_trace)
    supplied_digest = row.get("trace_digest")
    if not isinstance(supplied_digest, str) or not _HEX_SHA256_RE.fullmatch(
        supplied_digest
    ):
        raise ReplayValidationError("game row trace_digest must be a full SHA-256")
    expected_digest = evaluation_trace_digest(
        mode=mode,
        deal_hash=row_hash,
        bidding_trace=raw_bidding_trace,
        cardplay_trace=raw_cardplay_trace,
    )
    if not hmac.compare_digest(supplied_digest, expected_digest):
        raise ReplayValidationError("game row trace_digest does not match its traces")

    players = {role: _TraceAgent() for role in ROLES}
    if mode == "cardplay_only":
        env = GameEnv(players)
        env.card_play_init(copy.deepcopy(dict(approved_deal)))
        redeal_count = 0
        max_redeals_exceeded = False
    else:
        deal_id = row.get("deal_id")
        match = _DEAL_ID_RE.fullmatch(deal_id) if isinstance(deal_id, str) else None
        if (
            match is None
            or match.group("prefix") != row_hash[:12]
            or deal_id != canonical_deal_id(int(match.group("index")), row_hash)
        ):
            raise ReplayValidationError(
                "full-game row deal_id is not canonical for its deal_hash"
            )
        env = GameEnv(players, ruleset=ruleset)
        redeal_count, max_redeals_exceeded = _replay_bidding(
            env,
            bidding_trace,
            approved_deal=approved_deal,
            deal_id=deal_id,
            deterministic_seed=deterministic_seed,
            ruleset=ruleset,
        )
        env.players = players

    cardplay_legal_action_counts = _replay_cardplay(env, cardplay_trace)
    return _trusted_outcome(
        env,
        mode=mode,
        ruleset=ruleset,
        redeal_count=redeal_count,
        max_redeals_exceeded=max_redeals_exceeded,
        cardplay_legal_action_counts=cardplay_legal_action_counts,
    )


__all__ = [
    "EVALUATION_TRACE_SCHEMA_VERSION",
    "REDEAL_CAP_EXCLUSION_REASON",
    "ReplayOutcome",
    "ReplayValidationError",
    "evaluation_trace_digest",
    "replay_game_record",
]
