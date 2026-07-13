"""Canonical human-game record format (P08).

The canonical format is **JSONL** (one self-describing record per line). JSONL
is chosen over Parquet because (a) it adds no new runtime dependency
(``pyarrow``/``pandas`` are not project dependencies), (b) it is
cross-language and streaming-friendly, and (c) it avoids pickle's
arbitrary-code-execution risk. Every record is validated against a fixed JSON
Schema and is independently replayable through the rule engine (see
:mod:`douzero.human_data.validate`).

A record is **privileged training-only data**: it contains ``initial_hands``
(the true deal) and the recorded human actions. It MUST NOT be passed to any
deployment ``act()`` path; the BC student only consumes the public
:class:`~douzero.observation.encode_v2.ObservationV2` produced by replaying the
record. The ``kind`` stamp lets a deployment guard reject it without
introspection, mirroring :data:`~douzero.observation.privileged.PRIVILEGED_KIND`.

AGENTS.md "Human-game data" rules this module enforces:

- Use only lawfully obtained and authorized data (no scraping / automation here).
- Do not store personal identifiers or credentials (``timestamp`` is
  optional/anonymizable; ``source_metadata`` is audited).
- Canonicalize and validate every recorded game by replaying it through the
  rule engine (the replay lives in :mod:`douzero.human_data.validate`).
- Do not train only on won games (the record keeps the full ``final_result``
  so downstream sampling can stratify).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Canonical version stamps
# --------------------------------------------------------------------------- #
#: The on-disk container format version (the JSONL envelope). Bumped if the
#: file-level framing changes. Record-level schema changes bump
#: :data:`HUMAN_RECORD_SCHEMA_VERSION`.
CANONICAL_FORMAT_VERSION: int = 1

#: The per-record schema version. Bumped whenever a field is added, removed, or
#: its semantic changes. The loader rejects a mismatch rather than guessing.
HUMAN_RECORD_SCHEMA_VERSION: int = 1

#: Kind stamp identifying a human-game record as privileged training data. A
#: deployment guard can reject any object/dict carrying this kind without
#: introspecting further fields.
HUMAN_RECORD_KIND: str = "human_game_record"

#: The three legacy card-play roles, in canonical turn order. Matches
#: :data:`douzero.env.rules.PLAYER_POSITIONS` (the legacy / cardplay-only mode
#: the P08 pipeline primarily targets; standard-ruleset bidding records carry
#: their own ``bidding_history`` and the seat-to-role remap is handled at
#: ingest time).
ACTION_ROLES: tuple[str, ...] = ("landlord", "landlord_down", "landlord_up")

#: Required keys of the ``final_result`` block.
FINAL_RESULT_KEYS: tuple[str, ...] = (
    "winner_team",      # "landlord" | "farmer"
    "winner_position",  # one of ACTION_ROLES, or "" if undetermined
)


class RecordValidationError(ValueError):
    """Raised when a record fails canonical schema validation."""


# --------------------------------------------------------------------------- #
# The canonical record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HumanGameRecord:
    """One canonical human DouDizhu game (privileged training-only data).

    All card values use the legacy integer code points
    (3..14, 17, 20=small joker, 30=big joker; see
    :mod:`douzero.observation.cards`). Every action's cards are stored as a
    sorted tuple for canonical comparison; an empty tuple denotes a pass.

    Deep immutability: ``frozen`` + ``slots``; every mapping field is exposed
    as a read-only :class:`types.MappingProxyType`.

    Fields
    ------
    game_id:
        Stable, globally-unique game identifier (used for split integrity and
        de-duplication). Two records sharing a ``game_id`` are treated as
        duplicates.
    ruleset_id / ruleset_version / ruleset_hash:
        The :class:`~douzero.env.rules.RuleSet` identity the record was played
        under. ``ruleset_hash`` is the full SHA-256 of the ruleset parameter
        dict, so two records under subtly different rules never mix silently.
    seats:
        Ordered role tuple (e.g. ``("landlord", "landlord_down", "landlord_up")``).
        Carries the turn order the ``action_history`` positions refer to.
    initial_hands:
        The true deal. For legacy/cardplay mode this is the 4-key dict
        ``{landlord, landlord_up, landlord_down, three_landlord_cards}`` that
        :meth:`~douzero.env.game.GameEnv.card_play_init` consumes. This is
        PRIVILEGED — it is the analog of ``infoset.all_handcards`` and must
        never reach the deployment model.
    bottom_cards:
        The three revealed bottom cards (entity identity). Redundant with
        ``initial_hands['three_landlord_cards']`` for legacy mode but kept
        explicit for readability and for standard-mode records.
    bidding_history:
        Chronological ``((seat, bid_value), ...)``. Empty for legacy
        cardplay-only records (no bidding phase).
    action_history:
        Chronological ``((position, action_cards), ...)`` where ``position`` is
        a role from :data:`ACTION_ROLES` and ``action_cards`` is a sorted tuple
        of card ints (empty tuple = pass). This is the privileged truth the BC
        label is derived from.
    final_result:
        At least :data:`FINAL_RESULT_KEYS` (``winner_team``, ``winner_position``)
        plus optional scoring breakdown (``bid_value``, ``bomb_count``,
        ``rocket_count``, ``landlord_score``, ``farmer_score``, ``multiplier``).
    player_skill_weight:
        Per-role non-negative float weight (default 1.0) used to emphasize
        stronger players' decisions. Clipped/normalized downstream.
    source_metadata:
        Audit-only provenance (source name, license, collection batch). MUST
        NOT contain personal identifiers or credentials.
    timestamp:
        Optional anonymized timestamp (e.g. a coarse day bucket). May be empty.
    """

    game_id: str
    ruleset_id: str
    ruleset_version: str
    ruleset_hash: str
    seats: tuple[str, ...]
    initial_hands: Mapping[str, tuple[int, ...]]
    bottom_cards: tuple[int, ...]
    action_history: tuple[tuple[str, tuple[int, ...]], ...]
    final_result: Mapping[str, Any]
    bidding_history: tuple[tuple[str, int], ...] = ()
    player_skill_weight: Mapping[str, float] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    format_version: int = CANONICAL_FORMAT_VERSION
    schema_version: int = HUMAN_RECORD_SCHEMA_VERSION
    kind: str = field(default=HUMAN_RECORD_KIND, init=False)

    def __post_init__(self) -> None:
        # Version stamps.
        if self.format_version != CANONICAL_FORMAT_VERSION:
            raise RecordValidationError(
                f"format_version {self.format_version!r} != supported "
                f"{CANONICAL_FORMAT_VERSION!r}"
            )
        if self.schema_version != HUMAN_RECORD_SCHEMA_VERSION:
            raise RecordValidationError(
                f"schema_version {self.schema_version!r} != supported "
                f"{HUMAN_RECORD_SCHEMA_VERSION!r}"
            )
        # Identity strings must be non-empty.
        for name, val in (
            ("game_id", self.game_id),
            ("ruleset_id", self.ruleset_id),
            ("ruleset_version", self.ruleset_version),
            ("ruleset_hash", self.ruleset_hash),
        ):
            if not isinstance(val, str) or not val:
                raise RecordValidationError(
                    f"{name} must be a non-empty string, got {val!r}"
                )
        if self.kind != HUMAN_RECORD_KIND:  # defensive; default already sets it
            object.__setattr__(self, "kind", HUMAN_RECORD_KIND)

        # Seats: non-empty tuple of non-empty strings.
        if not isinstance(self.seats, tuple) or not self.seats:
            raise RecordValidationError(
                f"seats must be a non-empty tuple, got {self.seats!r}"
            )
        for s in self.seats:
            if not isinstance(s, str) or not s:
                raise RecordValidationError(
                    f"each seat must be a non-empty string, got {s!r}"
                )

        # Wrap caller mappings read-only and coerce card lists to sorted-int
        # tuples in one pass (deep immutability + canonical card ordering).
        coerced_hands = {
            role: _coerce_sorted_int_tuple(cards, f"initial_hands[{role!r}]")
            for role, cards in self.initial_hands.items()
        }
        for role in coerced_hands:
            if not isinstance(role, str) or not role:
                raise RecordValidationError(
                    f"initial_hands role must be non-empty string, got {role!r}"
                )
        object.__setattr__(
            self, "initial_hands", MappingProxyType(coerced_hands)
        )
        object.__setattr__(
            self, "final_result", MappingProxyType(dict(self.final_result))
        )
        object.__setattr__(
            self,
            "player_skill_weight",
            MappingProxyType(dict(self.player_skill_weight)),
        )
        object.__setattr__(
            self,
            "source_metadata",
            MappingProxyType(dict(self.source_metadata)),
        )

        # bottom_cards: sorted-int tuple.
        object.__setattr__(
            self,
            "bottom_cards",
            _coerce_sorted_int_tuple(self.bottom_cards, "bottom_cards"),
        )

        # action_history: tuple of (position, sorted-int-tuple).
        if not isinstance(self.action_history, tuple):
            raise RecordValidationError(
                f"action_history must be a tuple, got "
                f"{type(self.action_history).__name__}"
            )
        coerced_actions: list[tuple[str, tuple[int, ...]]] = []
        for entry in self.action_history:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise RecordValidationError(
                    f"each action_history entry must be a (position, cards) "
                    f"tuple, got {entry!r}"
                )
            pos, cards = entry
            if not isinstance(pos, str) or not pos:
                raise RecordValidationError(
                    f"action position must be a non-empty string, got {pos!r}"
                )
            coerced_actions.append(
                (pos, _coerce_sorted_int_tuple(cards, f"action[{pos!r}]"))
            )
        object.__setattr__(
            self, "action_history", tuple(coerced_actions)
        )

        # bidding_history: tuple of (seat, int).
        coerced_bids: list[tuple[str, int]] = []
        for entry in self.bidding_history:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise RecordValidationError(
                    f"each bidding_history entry must be a (seat, value) "
                    f"tuple, got {entry!r}"
                )
            seat, val = entry
            if not isinstance(seat, str) or not seat:
                raise RecordValidationError(
                    f"bid seat must be a non-empty string, got {seat!r}"
                )
            if isinstance(val, bool) or not isinstance(val, int):
                raise RecordValidationError(
                    f"bid value must be an int, got {val!r}"
                )
            coerced_bids.append((seat, val))
        object.__setattr__(self, "bidding_history", tuple(coerced_bids))

        # final_result: must contain the required keys.
        for key in FINAL_RESULT_KEYS:
            if key not in self.final_result:
                raise RecordValidationError(
                    f"final_result missing required key {key!r}"
                )
        wt = self.final_result["winner_team"]
        if wt not in ("landlord", "farmer"):
            raise RecordValidationError(
                f"final_result['winner_team'] must be 'landlord' or 'farmer', "
                f"got {wt!r}"
            )

        # player_skill_weight: non-negative floats.
        coerced_w = {}
        for role, w in self.player_skill_weight.items():
            if not isinstance(role, str) or not role:
                raise RecordValidationError(
                    f"player_skill_weight role must be non-empty string, "
                    f"got {role!r}"
                )
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise RecordValidationError(
                    f"player_skill_weight[{role!r}] must be a number, got {w!r}"
                )
            if w < 0:
                raise RecordValidationError(
                    f"player_skill_weight[{role!r}] must be non-negative, "
                    f"got {w}"
                )
            coerced_w[role] = float(w)
        object.__setattr__(
            self, "player_skill_weight", MappingProxyType(coerced_w)
        )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (the JSONL line payload)."""
        return {
            "format_version": self.format_version,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "game_id": self.game_id,
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_hash": self.ruleset_hash,
            "seats": list(self.seats),
            "initial_hands": {
                role: list(cards)
                for role, cards in self.initial_hands.items()
            },
            "bottom_cards": list(self.bottom_cards),
            "bidding_history": [
                [seat, val] for seat, val in self.bidding_history
            ],
            "action_history": [
                [pos, list(cards)] for pos, cards in self.action_history
            ],
            "final_result": dict(self.final_result),
            "player_skill_weight": dict(self.player_skill_weight),
            "source_metadata": dict(self.source_metadata),
            "timestamp": self.timestamp,
        }

    def to_jsonl_line(self) -> str:
        """Return the canonical JSONL encoding (single line, no trailing NL)."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @property
    def winner_team(self) -> str:
        return str(self.final_result["winner_team"])

    @property
    def winner_position(self) -> str:
        return str(self.final_result.get("winner_position", ""))


# --------------------------------------------------------------------------- #
# Construction from raw dicts / JSONL
# --------------------------------------------------------------------------- #
def _coerce_sorted_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    """Coerce ``value`` into a sorted tuple of non-negative ints (a copy)."""
    if value is None:
        return ()
    if isinstance(value, str):
        raise RecordValidationError(f"{label} must not be a string")
    try:
        items = list(value)
    except TypeError as exc:
        raise RecordValidationError(
            f"{label} must be iterable, got {type(value).__name__}"
        ) from exc
    out: list[int] = []
    for c in items:
        if isinstance(c, bool) or not isinstance(c, int):
            raise RecordValidationError(
                f"{label} must contain ints, got {type(c).__name__}: {c!r}"
            )
        if c < 0:
            raise RecordValidationError(
                f"{label} must contain non-negative ints, got {c}"
            )
        out.append(c)
    return tuple(sorted(out))


def record_from_dict(d: Mapping[str, Any]) -> HumanGameRecord:
    """Build a :class:`HumanGameRecord` from a raw mapping.

    Validates the envelope (``kind``, ``format_version``, ``schema_version``)
    BEFORE construction so a malformed or hostile payload is rejected at the
    boundary. Raises :class:`RecordValidationError` on any mismatch.
    """
    if not isinstance(d, Mapping):
        raise RecordValidationError(
            f"record must be a mapping, got {type(d).__name__}"
        )

    kind = d.get("kind")
    if kind != HUMAN_RECORD_KIND:
        raise RecordValidationError(
            f"record kind {kind!r} != expected {HUMAN_RECORD_KIND!r}"
        )
    fv = d.get("format_version")
    if fv != CANONICAL_FORMAT_VERSION:
        raise RecordValidationError(
            f"record format_version {fv!r} != supported "
            f"{CANONICAL_FORMAT_VERSION!r}"
        )
    sv = d.get("schema_version")
    if sv != HUMAN_RECORD_SCHEMA_VERSION:
        raise RecordValidationError(
            f"record schema_version {sv!r} != supported "
            f"{HUMAN_RECORD_SCHEMA_VERSION!r}"
        )

    required = (
        "game_id",
        "ruleset_id",
        "ruleset_version",
        "ruleset_hash",
        "seats",
        "initial_hands",
        "bottom_cards",
        "action_history",
        "final_result",
    )
    for key in required:
        if key not in d:
            raise RecordValidationError(f"record missing required key {key!r}")

    try:
        return HumanGameRecord(
            game_id=d["game_id"],
            ruleset_id=d["ruleset_id"],
            ruleset_version=d["ruleset_version"],
            ruleset_hash=d["ruleset_hash"],
            seats=tuple(d["seats"]),
            initial_hands={
                role: tuple(cards)
                for role, cards in d["initial_hands"].items()
            },
            bottom_cards=tuple(d["bottom_cards"]),
            bidding_history=tuple(
                (seat, val) for seat, val in d.get("bidding_history", [])
            ),
            action_history=tuple(
                (pos, tuple(cards))
                for pos, cards in d["action_history"]
            ),
            final_result=dict(d["final_result"]),
            player_skill_weight=dict(d.get("player_skill_weight", {})),
            source_metadata=dict(d.get("source_metadata", {})),
            timestamp=d.get("timestamp", ""),
        )
    except RecordValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"malformed record: {exc}") from exc


def record_from_jsonl_line(line: str) -> HumanGameRecord:
    """Parse one JSONL line into a :class:`HumanGameRecord`."""
    if not isinstance(line, str):
        raise RecordValidationError("JSONL line must be a string")
    stripped = line.strip()
    if not stripped:
        raise RecordValidationError("empty JSONL line")
    try:
        d = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RecordValidationError(f"invalid JSON: {exc}") from exc
    return record_from_dict(d)


# --------------------------------------------------------------------------- #
# JSONL file I/O (streaming, ordered)
# --------------------------------------------------------------------------- #
def write_jsonl(records: Iterable[HumanGameRecord], path: str) -> int:
    """Write records to ``path`` as JSONL. Returns the number of records written.

    Records are written in iteration order; deterministic ordering is the
    caller's responsibility (ingest sorts by ``game_id`` for reproducibility).
    """
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_jsonl_line())
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[HumanGameRecord]:
    """Stream records from a JSONL file, yielding one :class:`HumanGameRecord`.

    Each line is parsed independently so a single malformed line does not
    invalidate the whole file (the caller can quarantine it).
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                yield record_from_jsonl_line(line)
            except RecordValidationError as exc:
                raise RecordValidationError(
                    f"{path}:{lineno}: {exc}"
                ) from exc
