"""Full-replay validation for human-game records (P08).

Every canonical :class:`~douzero.human_data.schema.HumanGameRecord` is validated
by replaying it action-by-action through the project's own rule engine
(:class:`~douzero.env.game.GameEnv`). This is the AGENTS.md rule:

    "Canonicalize and validate every recorded game by replaying it through the
    rule engine. Quarantine invalid games; do not silently repair them."

A record is *valid* iff:

1. The recorded deal replays cleanly through ``GameEnv.card_play_init`` (hand
   conservation: 20+17+17+3 == 54, no overlaps, all canonical cards).
2. Every recorded action is played by the role whose turn it is (turn order:
   ``landlord -> landlord_down -> landlord_up -> landlord``).
3. Every recorded action is in the legal-action set at that decision (the rule
   engine is the source of truth for legality — a model/heuristic may rank
   legal actions but never manufacture one).
4. The game terminates after the recorded actions and exactly one role's hand
   is empty.
5. The terminal winner matches ``final_result['winner_team']``.

On any failure the record is **quarantined** with a precise reason string; the
pipeline never silently repairs or drops it. This module deliberately uses the
legacy card-play env (``ruleset=None``) because P08 records target the legacy
cardplay-only mode (matching the belief collector and the eval data path).
Standard-ruleset bidding replay is a future extension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .schema import HumanGameRecord

logger = logging.getLogger(__name__)

#: The three legacy card-play roles that ``GameEnv`` keys its players by.
_REPLAY_ROLES: tuple[str, ...] = ("landlord", "landlord_up", "landlord_down")


class ReplayValidationError(ValueError):
    """Raised when a record fails replay validation."""


# --------------------------------------------------------------------------- #
# Replay agent (mirrors douzero.env.env.DummyAgent, with an explicit check)
# --------------------------------------------------------------------------- #
class _ReplayAgent:
    """A minimal agent that returns a pre-set action during replay.

    Unlike :class:`~douzero.env.env.DummyAgent` (which asserts legality with a
    bare ``assert`` that ``python -O`` strips), this raises
    :class:`ReplayValidationError` on an illegal action so a recorded game can
    never be accepted on a legality violation.
    """

    def __init__(self, position: str) -> None:
        self.position = position
        self.action: list[int] | None = None

    def set_action(self, action: list[int]) -> None:
        self.action = list(action)

    def act(self, infoset):  # type: ignore[no-untyped-def]
        assert self.action is not None
        action = self.action
        self.action = None
        # Canonicalize for comparison: legal_actions are lists whose internal
        # card order is sorted at generation time, but compare defensively as
        # sorted tuples (the canonical form used throughout the project).
        action_key = tuple(sorted(action))
        legal_keys = {tuple(sorted(a)) for a in infoset.legal_actions}
        if action_key not in legal_keys:
            raise ReplayValidationError(
                f"recorded action {action!r} is not in the legal-action set at "
                f"position {self.position!r} (turn "
                f"{len(infoset.card_play_action_seq)})"
            )
        # Return the canonical legal action (same multiset as the recording).
        for legal in infoset.legal_actions:
            if tuple(sorted(legal)) == action_key:
                return list(legal)
        raise ReplayValidationError("unreachable: legal match lost")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Validation result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationResult:
    """Outcome of replaying one record.

    ``ok`` is True iff the record replayed cleanly and the terminal state
    matched. On failure, ``reason`` carries a short diagnostic code and
    ``error`` the full message; the record itself is always carried so a
    quarantine writer can serialize it.
    """

    record: HumanGameRecord
    ok: bool
    reason: str
    error: str = ""

    @property
    def game_id(self) -> str:
        return self.record.game_id


@dataclass
class ValidationReport:
    """Aggregate report over a batch of records."""

    valid: list[HumanGameRecord] = field(default_factory=list)
    quarantined: list[tuple[HumanGameRecord, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.valid) + len(self.quarantined)

    @property
    def num_valid(self) -> int:
        return len(self.valid)

    @property
    def num_quarantined(self) -> int:
        return len(self.quarantined)


# --------------------------------------------------------------------------- #
# Deal validation (hand conservation)
# --------------------------------------------------------------------------- #
def validate_deal_conservation(record: HumanGameRecord) -> None:
    """Check the recorded deal is a valid partition of the 54-card deck.

    Legacy layout: landlord=20 (incl. 3 bottom), up=17, down=17, bottom=3 with
    bottom ⊆ landlord. Raises :class:`ReplayValidationError` on any violation.
    """
    from collections import Counter

    from douzero.observation.cards import DECK

    hands = record.initial_hands
    for key in ("landlord", "landlord_up", "landlord_down", "three_landlord_cards"):
        if key not in hands:
            raise ReplayValidationError(
                f"initial_hands missing required key {key!r}"
            )
    landlord = hands["landlord"]
    up = hands["landlord_up"]
    down = hands["landlord_down"]
    bottom = hands["three_landlord_cards"]

    if len(landlord) != 20:
        raise ReplayValidationError(
            f"landlord hand must have 20 cards, got {len(landlord)}"
        )
    if len(up) != 17 or len(down) != 17:
        raise ReplayValidationError(
            f"farmers must have 17 cards each, got up={len(up)} down={len(down)}"
        )
    if len(bottom) != 3:
        raise ReplayValidationError(
            f"three_landlord_cards must have 3 cards, got {len(bottom)}"
        )
    # The full deal must be exactly the 54-card deck (multiset).
    full = list(landlord) + list(up) + list(down)
    if len(full) != 54:
        raise ReplayValidationError(
            f"landlord+up+down must total 54 cards, got {len(full)}"
        )
    full_counter = Counter(full)
    if full_counter != Counter(DECK):
        # Locate the first mismatch for a useful error.
        for card in sorted(set(list(DECK) + full)):
            if full_counter.get(card, 0) != Counter(DECK).get(card, 0):
                raise ReplayValidationError(
                    f"deal is not a partition of the 54-card deck: rank {card} "
                    f"has count {full_counter.get(card, 0)} vs expected "
                    f"{Counter(DECK).get(card, 0)}"
                )
    # Bottom cards must be a subset of the landlord's hand (legacy deal slices
    # three_landlord_cards out of the landlord slot).
    landlord_counter = Counter(landlord)
    bottom_counter = Counter(bottom)
    for card, count in bottom_counter.items():
        if landlord_counter.get(card, 0) < count:
            raise ReplayValidationError(
                f"bottom card {card} (x{count}) is not in the landlord's hand"
            )
    # Record.bottom_cards must match three_landlord_cards.
    if tuple(sorted(bottom)) != record.bottom_cards:
        raise ReplayValidationError(
            "record.bottom_cards != initial_hands['three_landlord_cards']"
        )


# --------------------------------------------------------------------------- #
# Core single-record replay
# --------------------------------------------------------------------------- #
def validate_record(record: HumanGameRecord) -> ValidationResult:
    """Replay one record through the rule engine and return the outcome."""
    try:
        _validate_record_or_raise(record)
    except ReplayValidationError as exc:
        return ValidationResult(
            record=record, ok=False, reason=_classify(exc), error=str(exc)
        )
    return ValidationResult(record=record, ok=True, reason="ok")


def _classify(exc: ReplayValidationError) -> str:
    """Map an exception message to a short diagnostic reason code.

    Order matters: the winner-mismatch message contains the word ``position``
    (``"replay winner position ... "``), so the ``winner`` check must run before
    the generic ``position`` check or a winner error is mislabelled as a turn
    error.
    """
    msg = str(exc).lower()
    if "winner" in msg:
        return "winner_mismatch"
    if "did not terminate" in msg or "extra actions after terminal" in msg:
        return "did_not_terminate"
    if "engine expects" in msg or "recorded position" in msg:
        return "turn_order_mismatch"
    if "not in the legal" in msg or "legal-action" in msg:
        return "illegal_action"
    if "deck" in msg or "partition" in msg or "conservation" in msg:
        return "deal_conservation"
    if "initial_hands" in msg or "cards" in msg:
        return "deal_shape"
    return "replay_error"


def _validate_record_or_raise(record: HumanGameRecord) -> None:
    from douzero.env.game import GameEnv

    # 1. Deal conservation first (cheap, no env construction).
    validate_deal_conservation(record)

    # 2. Build the replay env (legacy cardplay).
    players = {pos: _ReplayAgent(pos) for pos in _REPLAY_ROLES}
    genv = GameEnv(players)  # ruleset=None -> legacy
    deal = {
        "landlord": list(record.initial_hands["landlord"]),
        "landlord_up": list(record.initial_hands["landlord_up"]),
        "landlord_down": list(record.initial_hands["landlord_down"]),
        "three_landlord_cards": list(record.initial_hands["three_landlord_cards"]),
    }
    genv.card_play_init(deal)

    # 3. Replay each recorded action, enforcing turn order + legality.
    expected_len = len(record.action_history)
    for i, (pos, cards) in enumerate(record.action_history):
        acting = genv.acting_player_position
        if acting != pos:
            raise ReplayValidationError(
                f"turn_order_mismatch at step {i}: recorded position {pos!r} "
                f"but the engine expects {acting!r} to act"
            )
        if genv.game_over:
            raise ReplayValidationError(
                f"record has {expected_len} actions but the game terminated at "
                f"step {i} (extra actions after terminal)"
            )
        players[pos].set_action(list(cards))
        genv.step()  # _ReplayAgent.act raises ReplayValidationError on illegal
        if genv.game_over and i != expected_len - 1:
            # The game ended early but the record has more actions.
            raise ReplayValidationError(
                f"game terminated at step {i} but the record has "
                f"{expected_len} actions (extra actions after terminal)"
            )

    # 4. The game must have terminated.
    if not genv.game_over:
        raise ReplayValidationError(
            f"record did not terminate: after {expected_len} actions no hand is "
            f"empty (cards left: landlord="
            f"{len(genv.info_sets['landlord'].player_hand_cards)}, up="
            f"{len(genv.info_sets['landlord_up'].player_hand_cards)}, down="
            f"{len(genv.info_sets['landlord_down'].player_hand_cards)})"
        )

    # 5. Exactly one hand empty, and it matches the recorded winner.
    empty = [
        p for p in _REPLAY_ROLES
        if len(genv.info_sets[p].player_hand_cards) == 0
    ]
    if len(empty) != 1:
        raise ReplayValidationError(
            f"terminal state has {len(empty)} empty hands ({empty}); expected "
            f"exactly one"
        )
    winner_position = empty[0]
    expected_team = record.final_result["winner_team"]
    actual_team = "landlord" if winner_position == "landlord" else "farmer"
    if actual_team != expected_team:
        raise ReplayValidationError(
            f"winner_mismatch: replay winner team {actual_team!r} "
            f"(position {winner_position!r}) != recorded {expected_team!r}"
        )
    # Cross-check the recorded winner_position if it was provided.
    recorded_pos = record.final_result.get("winner_position", "")
    if recorded_pos and recorded_pos != winner_position:
        raise ReplayValidationError(
            f"winner_mismatch: replay winner position {winner_position!r} != "
            f"recorded {recorded_pos!r}"
        )


# --------------------------------------------------------------------------- #
# Batch streaming
# --------------------------------------------------------------------------- #
def validate_records(
    records: Iterable[HumanGameRecord],
    *,
    stop_on_error: bool = False,
) -> ValidationReport:
    """Validate a stream of records, partitioning into valid / quarantined.

    Invalid records are NEVER silently dropped: they appear in
    ``report.quarantined`` with their diagnostic reason so a quarantine writer
    can serialize them. When ``stop_on_error`` is True the first invalid record
    raises (useful in unit tests).
    """
    report = ValidationReport()
    for record in records:
        result = validate_record(record)
        if result.ok:
            report.valid.append(record)
        else:
            report.quarantined.append((record, result.reason or result.error))
            if stop_on_error:
                raise ReplayValidationError(
                    f"{record.game_id}: {result.error}"
                )
    return report


def iter_valid(
    records: Iterable[HumanGameRecord],
) -> Iterator[HumanGameRecord]:
    """Yield only the records that pass validation (streaming)."""
    for record in records:
        if validate_record(record).ok:
            yield record
