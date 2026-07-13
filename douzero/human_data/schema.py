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
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .privacy import assert_no_forbidden as _assert_no_forbidden_privacy

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

#: Whitelist of ALL allowed ``final_result`` keys (Blocker 2, round 4: prevents
#: arbitrary extension fields from carrying PII into the serialized record).
#: Known scoring fields from the legacy + standard rulesets are included.
FINAL_RESULT_ALLOWED_KEYS: frozenset[str] = frozenset({
    "winner_team", "winner_position",
    "bid_value", "bomb_count", "rocket_count", "bomb_num",
    "landlord_score", "farmer_score", "multiplier",
    "spring", "anti_spring",
})

#: The EXACT set of keys allowed in ``initial_hands`` for a legacy record.
#: Extra keys are rejected so PII cannot hide in an undeclared hand slot.
LEGACY_HAND_KEYS: frozenset[str] = frozenset({
    "landlord", "landlord_up", "landlord_down", "three_landlord_cards",
})

#: The three legal card-play roles (used for seats, action positions, skill
#: weights). A record carrying a non-canonical role is rejected so PII cannot
#: hide in a free-text role string.
LEGAL_ROLES: frozenset[str] = frozenset({
    "landlord", "landlord_down", "landlord_up",
})

#: The canonical legacy seat order (turn order). A legacy record's ``seats``
#: must equal this exactly.
LEGACY_SEATS: tuple[str, ...] = ("landlord", "landlord_down", "landlord_up")

#: Allowed ``timestamp`` formats (Blocker 2: prevents a free-text timestamp
#: from carrying PII). Empty string or a coarse ``YYYY-MM`` month bucket.
_TIMESTAMP_PATTERN: re.Pattern[str] = re.compile(r"^(\d{4})-(\d{2})$")


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
        # Identity strings must be non-empty. game_id is the only user-
        # controlled free-text identity field, so it gets a PII value scan
        # (round 5 Blocker 1: "alice@example.com" must not hide here). The
        # ruleset_* fields are structural (short enum/hash) and are NOT scanned
        # (a SHA-256 hash legitimately contains long digit runs that would
        # false-positive on a phone pattern).
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
        # PII scan only on game_id (the free-text field).
        try:
            _assert_no_forbidden_privacy(self.game_id, label="game_id")
        except ValueError as exc:
            raise RecordValidationError(str(exc)) from exc
        if self.kind != HUMAN_RECORD_KIND:  # defensive; default already sets it
            object.__setattr__(self, "kind", HUMAN_RECORD_KIND)

        # Seats: must equal the canonical legacy seat order exactly (round 5
        # Blocker 1: prevents a free-text seat like "user@example.com").
        if not isinstance(self.seats, tuple) or self.seats != LEGACY_SEATS:
            raise RecordValidationError(
                f"seats must equal the canonical legacy seat order {LEGACY_SEATS}, "
                f"got {self.seats!r}"
            )

        # Wrap caller mappings read-only and coerce card lists to sorted-int
        # tuples in one pass (deep immutability + canonical card ordering).
        # Round 5 Blocker 1: reject extra keys so PII cannot hide in an
        # undeclared hand slot (e.g. "user_email_alice@example.com": []).
        hand_keys = set(self.initial_hands.keys())
        extra = hand_keys - LEGACY_HAND_KEYS
        if extra:
            raise RecordValidationError(
                f"initial_hands has unknown keys {sorted(extra)!r}; allowed "
                f"keys are {sorted(LEGACY_HAND_KEYS)}. Extra keys are rejected "
                f"to prevent PII from entering the record."
            )
        missing = LEGACY_HAND_KEYS - hand_keys
        if missing:
            raise RecordValidationError(
                f"initial_hands missing required keys {sorted(missing)!r}"
            )
        coerced_hands = {
            role: _coerce_sorted_int_tuple(cards, f"initial_hands[{role!r}]")
            for role, cards in self.initial_hands.items()
        }
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
        # Round 5 Blocker 1.3: normalize -> privacy-scan -> DEEP-FREEZE so the
        # privacy boundary is durable (nested mutation raises TypeError).
        _normalized_meta = _normalize_json_mapping(self.source_metadata)
        object.__setattr__(
            self,
            "source_metadata",
            _deep_freeze(_normalized_meta),
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
            # Round 5 Blocker 1: position must be a canonical legal role.
            if pos not in LEGAL_ROLES:
                raise RecordValidationError(
                    f"action_history position {pos!r} is not a legal role "
                    f"(one of {sorted(LEGAL_ROLES)}); free-text positions are "
                    f"rejected to prevent PII."
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
        # Blocker 2 (round 4): whitelist final_result keys so arbitrary
        # extension fields cannot carry PII into the serialized record.
        unknown = set(self.final_result.keys()) - FINAL_RESULT_ALLOWED_KEYS
        if unknown:
            raise RecordValidationError(
                f"final_result contains unknown keys {sorted(unknown)!r}; "
                f"allowed keys are {sorted(FINAL_RESULT_ALLOWED_KEYS)}. "
                f"Arbitrary extension fields are rejected to prevent PII from "
                f"entering the serialized record."
            )
        # Scan final_result values for sensitive data (PII/credentials).
        try:
            _assert_no_forbidden_privacy(
                dict(self.final_result), label="final_result"
            )
        except ValueError as exc:
            raise RecordValidationError(str(exc)) from exc

        # player_skill_weight: non-negative finite floats, legal roles only.
        coerced_w = {}
        for role, w in self.player_skill_weight.items():
            # Round 5 Blocker 1: role must be a canonical legal role.
            if role not in LEGAL_ROLES:
                raise RecordValidationError(
                    f"player_skill_weight role {role!r} is not a legal role "
                    f"(one of {sorted(LEGAL_ROLES)}); free-text roles are "
                    f"rejected to prevent PII."
                )
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise RecordValidationError(
                    f"player_skill_weight[{role!r}] must be a number, got {w!r}"
                )
            # Round 5 Blocker 2: reject NaN/Inf (non-standard JSON).
            if not math.isfinite(float(w)):
                raise RecordValidationError(
                    f"player_skill_weight[{role!r}] must be finite, got {w}"
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

        # Blocker 2 (round 4): timestamp must be a coarse format (empty or
        # YYYY-MM) so a free-text timestamp cannot carry PII.
        ts = self.timestamp
        if not isinstance(ts, str):
            raise RecordValidationError(
                f"timestamp must be a string, got {type(ts).__name__}"
            )
        if ts and not _TIMESTAMP_PATTERN.match(ts):
            raise RecordValidationError(
                f"timestamp must be empty or 'YYYY-MM', got {ts!r}"
            )
        if ts:
            # Round 5 Blocker 2: validate the month is 01-12.
            m = _TIMESTAMP_PATTERN.match(ts)
            assert m is not None  # guaranteed by the check above
            month = int(m.group(2))
            if not 1 <= month <= 12:
                raise RecordValidationError(
                    f"timestamp month must be 01-12, got {ts!r}"
                )
        # Scan timestamp for sensitive data (an email/IP shaped value must not
        # hide here even if it matched the YYYY-MM regex — defense in depth).
        if ts:
            try:
                _assert_no_forbidden_privacy(ts, label="timestamp")
            except ValueError as exc:
                raise RecordValidationError(str(exc)) from exc

        # Blocker 2 (round 4): fail-closed privacy scan on source_metadata at
        # the canonical record boundary. This catches PII/credentials that
        # bypassed ingest (e.g. a direct record_from_dict / JSONL load) so they
        # can never reach validation, BC sampling, or training.
        try:
            _assert_no_forbidden_privacy(
                dict(self.source_metadata), label="source_metadata"
            )
        except ValueError as exc:
            raise RecordValidationError(str(exc)) from exc

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
            "source_metadata": _deep_to_json(self.source_metadata),
            "timestamp": self.timestamp,
        }

    def to_jsonl_line(self) -> str:
        """Return the canonical JSONL encoding (single line, no trailing NL).

        ``allow_nan=False`` rejects NaN/Infinity (round 5 Blocker 2: the
        canonical JSONL must be strict standard JSON, not Python's extended
        subset that silently writes ``NaN``/``Infinity`` tokens).
        """
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False
        )

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


def _normalize_json_value(value: Any, label: str) -> Any:
    """Recursively normalize ``value`` to a canonical JSON type.

    Round 5 Blocker 2: rejects NaN/Inf floats (non-standard JSON), rejects
    set/frozenset (non-deterministic iteration order breaks reproducible
    canonical output), converts tuple to list, and rejects non-JSON scalar
    types (bytes, custom objects).
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Reject NaN/Infinity — they are not valid standard JSON and
        # json.dumps(allow_nan=False) would raise at serialization.
        if not math.isfinite(value):
            raise RecordValidationError(
                f"{label}: source_metadata contains a non-finite float "
                f"({value}); NaN/Infinity are not valid standard JSON."
            )
        return value
    if isinstance(value, Mapping):
        return _normalize_json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item, label) for item in value]
    if isinstance(value, (set, frozenset)):
        raise RecordValidationError(
            f"{label}: source_metadata contains a set/frozenset, which has "
            f"non-deterministic iteration order and would break reproducible "
            f"canonical JSONL output. Use a list instead."
        )
    raise RecordValidationError(
        f"{label}: source_metadata contains a non-JSON value of type "
        f"{type(value).__name__}; only dict/list/str/int/float/bool/None are "
        f"allowed."
    )


def _normalize_json_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-normalize a metadata mapping to canonical JSON types."""
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        # Blocker 2 (round 4): require string keys (a non-string key would
        # either fail json.dumps or silently coerce, and masks a malformed
        # adapter payload).
        if not isinstance(key, str):
            raise RecordValidationError(
                f"source_metadata key {key!r} must be a string, got "
                f"{type(key).__name__}"
            )
        out[key] = _normalize_json_value(value, f"source_metadata[{key!r}]")
    return out


def _deep_freeze(obj: Any) -> Any:
    """Recursively freeze a normalized JSON value to be deeply immutable.

    Round 5 Blocker 1.3: wraps every nested dict in :class:`MappingProxyType`
    and converts every list to a tuple, so post-construction mutation (e.g.
    ``record.source_metadata["nested"]["contact"] = "alice@example.com"``)
    raises ``TypeError`` rather than silently injecting PII that survives into
    serialization. The privacy scan ran at construction; the deep freeze makes
    that boundary durable.
    """
    if isinstance(obj, Mapping):
        return MappingProxyType(
            {k: _deep_freeze(v) for k, v in obj.items()}
        )
    if isinstance(obj, list):
        return tuple(_deep_freeze(item) for item in obj)
    if isinstance(obj, tuple):
        return tuple(_deep_freeze(item) for item in obj)
    return obj


def _deep_to_json(obj: Any) -> Any:
    """Recursively convert a deep-frozen value back to plain JSON types."""
    if isinstance(obj, Mapping):
        return {k: _deep_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_to_json(item) for item in obj]
    return obj


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

    Fail-fast: the first malformed line (invalid JSON, wrong schema version,
    missing field, bad type) raises :class:`RecordValidationError` with the
    file path and line number. Use this when the input is expected to be clean
    and any corruption should stop the pipeline immediately.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                yield record_from_jsonl_line(line)
            except RecordValidationError as exc:
                raise RecordValidationError(
                    f"{path}:{lineno}: {exc}"
                ) from exc


@dataclass(frozen=True)
class JsonlLineResult:
    """Outcome of parsing one JSONL line (resilient mode).

    Exactly one of ``record`` / ``error`` is set. ``lineno`` is 1-based.
    A blank line yields ``error="empty line"`` so the caller can decide whether
    to skip or quarantine it.
    """

    lineno: int
    record: HumanGameRecord | None = None
    error: str = ""


def iter_jsonl_resilient(path: str) -> Iterator[JsonlLineResult]:
    """Stream JSONL, yielding one :class:`JsonlLineResult` per line.

    Resilient (Blocker 3): a malformed line (invalid JSON, wrong schema
    version, missing field, bad type) NEVER raises — it yields a result with
    ``error`` set so the caller can quarantine it alongside replay failures.
    This is the reader ``validate_human_games.py`` uses so that JSON/schema
    errors are quarantined, not fatal.

    Empty lines are reported as errors (the caller decides whether to skip or
    quarantine). The yield order matches the file order.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                yield JsonlLineResult(lineno=lineno, error="empty line")
                continue
            try:
                record = record_from_jsonl_line(line)
            except RecordValidationError as exc:
                yield JsonlLineResult(lineno=lineno, error=str(exc))
                continue
            yield JsonlLineResult(lineno=lineno, record=record)
