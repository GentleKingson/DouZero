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

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from douzero._version import git_sha

from .identifiers import is_canonical_game_id
from .privacy import assert_no_forbidden as _assert_no_forbidden_privacy
from .privacy import assert_valid_source_metadata

# --------------------------------------------------------------------------- #
# Canonical version stamps
# --------------------------------------------------------------------------- #
#: The on-disk container format version (the JSONL envelope). Bumped if the
#: file-level framing changes. Record-level schema changes bump
#: :data:`HUMAN_RECORD_SCHEMA_VERSION`.
CANONICAL_FORMAT_VERSION: int = 1

#: The per-record schema version. Bumped whenever a field is added, removed, or
#: its semantic changes. The loader rejects a mismatch rather than guessing.
HUMAN_RECORD_SCHEMA_VERSION: int = 2
HUMAN_DATASET_MANIFEST_VERSION: str = "human-dataset-manifest-v1"

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
        # Canonical game IDs are opaque values created internally or by keyed
        # HMAC.  Raw platform IDs are rejected at the record boundary.
        if not is_canonical_game_id(self.game_id):
            raise RecordValidationError(
                "game_id must be an opaque canonical ID in the form "
                "'dzg_' plus 64 lowercase hex characters; map external IDs "
                "with pseudonymize_external_game_id() before construction"
            )

        # Other identity strings must be non-empty. Round 6: PII-scan
        # ruleset identity fields and validate
        # ruleset_hash is a 64-char hex string (structural format constraint
        # that inherently blocks PII). The tightened phone pattern (requires +
        # or separators) avoids false positives on short enum strings.
        for name, val in (
            ("ruleset_id", self.ruleset_id),
            ("ruleset_version", self.ruleset_version),
        ):
            if not isinstance(val, str) or not val:
                raise RecordValidationError(
                    f"{name} must be a non-empty string, got {val!r}"
                )
            try:
                _assert_no_forbidden_privacy(val, label=name)
            except ValueError as exc:
                raise RecordValidationError(str(exc)) from exc
        # ruleset_hash: must be a 64-char lowercase hex string (SHA-256).
        if not isinstance(self.ruleset_hash, str) or not self.ruleset_hash:
            raise RecordValidationError(
                f"ruleset_hash must be a non-empty string, got {self.ruleset_hash!r}"
            )
        if not re.match(r"^[0-9a-f]{64}$", self.ruleset_hash):
            raise RecordValidationError(
                f"ruleset_hash must be a 64-char lowercase hex SHA-256, got "
                f"{self.ruleset_hash!r}"
            )
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
        # final_result is deep-frozen AFTER per-field validation below.
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

        # bidding_history: tuple of (seat, int). Round 6 Blocker 1a: PII-scan
        # the seat string so "alice@example.com" cannot hide here. (Legacy
        # records must have empty bidding_history — enforced by
        # assert_legacy_rulesat at the validation boundary.)
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
            try:
                _assert_no_forbidden_privacy(seat, label="bidding_history seat")
            except ValueError as exc:
                raise RecordValidationError(str(exc)) from exc
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
        wp = self.final_result.get("winner_position", "")
        if wp and wp not in LEGAL_ROLES:
            raise RecordValidationError(
                f"final_result['winner_position'] must be a legal role or '', "
                f"got {wp!r}"
            )
        # Whitelist keys.
        unknown = set(self.final_result.keys()) - FINAL_RESULT_ALLOWED_KEYS
        if unknown:
            raise RecordValidationError(
                f"final_result contains unknown keys {sorted(unknown)!r}; "
                f"allowed keys are {sorted(FINAL_RESULT_ALLOWED_KEYS)}."
            )
        # Round 6 Blocker 1c: per-field type + finite validation for numeric
        # scoring fields. Prevents a nested dict/list from hiding PII and
        # ensures all floats are finite (standard JSON).
        _FR_INT_FIELDS = frozenset({
            "bid_value", "bomb_count", "rocket_count", "bomb_num", "multiplier",
        })
        _FR_FLOAT_FIELDS = frozenset({"landlord_score", "farmer_score"})
        _FR_BOOL_FIELDS = frozenset({"spring", "anti_spring"})
        coerced_fr: dict[str, Any] = {}
        for key, val in self.final_result.items():
            if key in _FR_INT_FIELDS:
                if isinstance(val, bool) or not isinstance(val, int):
                    raise RecordValidationError(
                        f"final_result['{key}'] must be an int, got {val!r}"
                    )
            elif key in _FR_FLOAT_FIELDS:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise RecordValidationError(
                        f"final_result['{key}'] must be a number, got {val!r}"
                    )
                if not math.isfinite(float(val)):
                    raise RecordValidationError(
                        f"final_result['{key}'] must be finite, got {val}"
                    )
                val = float(val)
            elif key in _FR_BOOL_FIELDS:
                if not isinstance(val, bool):
                    raise RecordValidationError(
                        f"final_result['{key}'] must be a bool, got {val!r}"
                    )
            coerced_fr[key] = val
        # Scan final_result values for sensitive data (PII/credentials).
        try:
            _assert_no_forbidden_privacy(coerced_fr, label="final_result")
        except ValueError as exc:
            raise RecordValidationError(str(exc)) from exc
        # DEEP-FREEZE final_result (same as source_metadata) so nested mutation
        # cannot inject PII after construction.
        object.__setattr__(self, "final_result", _deep_freeze(coerced_fr))

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

        # Fail closed with a flat provenance allowlist.  Pattern blacklists are
        # retained only as defense in depth for the allowed scalar values.
        try:
            assert_valid_source_metadata(dict(self.source_metadata))
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
            "final_result": _deep_to_json(self.final_result),
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


#: Allowed top-level keys in a canonical record dict (round 6: reject unknown
#: top-level fields so a canonical fixed-schema record cannot silently carry
#: arbitrary extra data).
_ALLOWED_TOP_KEYS: frozenset[str] = frozenset({
    "format_version", "schema_version", "kind",
    "game_id", "ruleset_id", "ruleset_version", "ruleset_hash",
    "seats", "initial_hands", "bottom_cards",
    "bidding_history", "action_history", "final_result",
    "player_skill_weight", "source_metadata", "timestamp",
})


def _require_mapping(d: Any, field: str) -> Mapping[str, Any]:
    """Return ``d`` if it is a Mapping; raise RecordValidationError otherwise."""
    if not isinstance(d, Mapping):
        raise RecordValidationError(
            f"record field {field!r} must be a JSON object/mapping, got "
            f"{type(d).__name__}"
        )
    return d


def _require_sequence(d: Any, field: str) -> Sequence:
    """Return ``d`` if it is a non-string Sequence; raise otherwise."""
    if isinstance(d, str) or not isinstance(d, (list, tuple)):
        raise RecordValidationError(
            f"record field {field!r} must be a JSON array, got "
            f"{type(d).__name__}"
        )
    return d


def record_from_dict(d: Mapping[str, Any]) -> HumanGameRecord:
    """Build a :class:`HumanGameRecord` from a raw mapping.

    Validates the envelope (``kind``, ``format_version``, ``schema_version``)
    BEFORE construction so a malformed or hostile payload is rejected at the
    boundary. Raises :class:`RecordValidationError` on any mismatch.

    Round 6 Blocker 2: ALL malformed nested shapes (null, wrong type, missing
    ``.items()``) are converted to :class:`RecordValidationError` — never
    ``AttributeError`` or bare ``TypeError`` — so the resilient JSONL reader
    can quarantine the line instead of crashing the whole validation process.
    Also rejects unknown top-level keys (canonical fixed-schema enforcement).
    """
    if not isinstance(d, Mapping):
        raise RecordValidationError(
            f"record must be a mapping, got {type(d).__name__}"
        )

    # Reject unknown top-level keys (canonical fixed schema).
    unknown_top = set(d.keys()) - _ALLOWED_TOP_KEYS
    if unknown_top:
        raise RecordValidationError(
            f"record has unknown top-level keys {sorted(unknown_top)!r}; "
            f"canonical schema does not allow arbitrary top-level fields."
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
        "game_id", "ruleset_id", "ruleset_version", "ruleset_hash",
        "seats", "initial_hands", "bottom_cards",
        "action_history", "final_result",
    )
    for key in required:
        if key not in d:
            raise RecordValidationError(f"record missing required key {key!r}")

    # Type-check all nested fields BEFORE accessing their methods (.items(),
    # etc.) so a null/list value converts to RecordValidationError, not
    # AttributeError (round 6 Blocker 2).
    _require_sequence(d["seats"], "seats")
    _require_mapping(d["initial_hands"], "initial_hands")
    _require_sequence(d["bottom_cards"], "bottom_cards")
    _require_sequence(d["action_history"], "action_history")
    _require_mapping(d["final_result"], "final_result")
    if "bidding_history" in d and d["bidding_history"] is not None:
        _require_sequence(d["bidding_history"], "bidding_history")
    if "player_skill_weight" in d and d["player_skill_weight"] is not None:
        _require_mapping(d["player_skill_weight"], "player_skill_weight")
    if "source_metadata" in d and d["source_metadata"] is not None:
        _require_mapping(d["source_metadata"], "source_metadata")

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
                (seat, val) for seat, val in d.get("bidding_history") or []
            ),
            action_history=tuple(
                (pos, tuple(cards))
                for pos, cards in d["action_history"]
            ),
            final_result=dict(d["final_result"]),
            player_skill_weight=dict(d.get("player_skill_weight") or {}),
            source_metadata=dict(d.get("source_metadata") or {}),
            timestamp=d.get("timestamp", ""),
        )
    except RecordValidationError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        # AttributeError: a null/wrong-type nested field whose .items() was
        # called. Convert to RecordValidationError so iter_jsonl_resilient
        # quarantines it instead of crashing.
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
def dataset_manifest_path(path: str | Path) -> Path:
    """Return the non-secret provenance sidecar path for canonical JSONL."""

    return Path(f"{path}.manifest.json")


def _canonical_config_hash(config: Mapping[str, Any] | None) -> str:
    try:
        payload = json.dumps(
            dict(config or {"operation": "write_jsonl"}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(
            "dataset config identity must be strict JSON"
        ) from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(
    records: Iterable[HumanGameRecord],
    path: str,
    *,
    config_identity: Mapping[str, Any] | None = None,
    lineage_verified: bool = True,
) -> int:
    """Write records to ``path`` as JSONL. Returns the number of records written.

    Records are written in iteration order; deterministic ordering is the
    caller's responsibility (ingest sorts by ``game_id`` for reproducibility).
    """
    source_sha = git_sha()
    if not isinstance(lineage_verified, bool):
        raise RecordValidationError("lineage_verified must be a bool")
    if (
        len(source_sha) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in source_sha)
    ):
        raise RecordValidationError(
            "canonical dataset writes require a full source Git SHA; set "
            "DOUZERO_GIT_SHA in source-less runtimes"
        )
    n = 0
    digest = hashlib.sha256()
    rulesets: set[tuple[str, str, str]] = set()
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            line = rec.to_jsonl_line() + "\n"
            fh.write(line)
            digest.update(line.encode("utf-8"))
            rulesets.add(
                (rec.ruleset_id, rec.ruleset_version, rec.ruleset_hash)
            )
            n += 1
    manifest = {
        "schema_version": HUMAN_DATASET_MANIFEST_VERSION,
        "canonical_format_version": CANONICAL_FORMAT_VERSION,
        "record_schema_version": HUMAN_RECORD_SCHEMA_VERSION,
        "source_git_sha": source_sha,
        "config_identity_hash": _canonical_config_hash(config_identity),
        "rulesets": [
            {
                "ruleset_id": identity[0],
                "ruleset_version": identity[1],
                "ruleset_hash": identity[2],
            }
            for identity in sorted(rulesets)
        ],
        "record_count": n,
        "dataset_sha256": digest.hexdigest(),
        "access_class": "privileged_training_data",
        "lineage_verified": lineage_verified,
    }
    dataset_manifest_path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return n


def verify_jsonl_manifest(
    path: str | Path, *, allow_unverified_lineage: bool = False
) -> dict[str, Any]:
    """Verify canonical dataset provenance and content checksum."""

    source = Path(path)
    sidecar = dataset_manifest_path(source)
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordValidationError("canonical dataset manifest is missing or invalid") from exc
    expected_keys = {
        "schema_version", "canonical_format_version", "record_schema_version",
        "source_git_sha", "config_identity_hash", "rulesets", "record_count",
        "dataset_sha256", "access_class",
        "lineage_verified",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RecordValidationError("canonical dataset manifest fields are invalid")
    if manifest["schema_version"] != HUMAN_DATASET_MANIFEST_VERSION:
        raise RecordValidationError("canonical dataset manifest schema mismatch")
    if manifest["canonical_format_version"] != CANONICAL_FORMAT_VERSION:
        raise RecordValidationError("canonical dataset format identity mismatch")
    if manifest["record_schema_version"] != HUMAN_RECORD_SCHEMA_VERSION:
        raise RecordValidationError("canonical dataset record schema mismatch")
    for name in ("source_git_sha", "config_identity_hash", "dataset_sha256"):
        value = manifest[name]
        lengths = (40, 64) if name == "source_git_sha" else (64,)
        if (
            not isinstance(value, str)
            or len(value) not in lengths
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise RecordValidationError(f"canonical dataset {name} is invalid")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest["dataset_sha256"]:
        raise RecordValidationError("canonical dataset checksum mismatch")
    if manifest["access_class"] != "privileged_training_data":
        raise RecordValidationError("canonical dataset access class mismatch")
    if not isinstance(manifest["lineage_verified"], bool):
        raise RecordValidationError("canonical dataset lineage identity is invalid")
    if not manifest["lineage_verified"] and not allow_unverified_lineage:
        raise RecordValidationError(
            "canonical dataset has unverified migration lineage and cannot be "
            "used for training or release"
        )
    if (
        isinstance(manifest["record_count"], bool)
        or not isinstance(manifest["record_count"], int)
        or manifest["record_count"] < 0
    ):
        raise RecordValidationError("canonical dataset record_count is invalid")
    actual_count = 0
    actual_rulesets: set[tuple[str, str, str]] = set()
    with source.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                record = record_from_jsonl_line(line)
            except RecordValidationError as exc:
                raise RecordValidationError(
                    f"{source}:{lineno}: {exc}"
                ) from exc
            actual_count += 1
            actual_rulesets.add(
                (record.ruleset_id, record.ruleset_version, record.ruleset_hash)
            )
    if manifest["record_count"] != actual_count:
        raise RecordValidationError(
            "canonical dataset manifest record_count does not match content"
        )
    raw_rulesets = manifest["rulesets"]
    if not isinstance(raw_rulesets, list):
        raise RecordValidationError("canonical dataset rulesets are invalid")
    manifest_rulesets: set[tuple[str, str, str]] = set()
    for identity in raw_rulesets:
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {"ruleset_id", "ruleset_version", "ruleset_hash"}
            or not all(isinstance(value, str) for value in identity.values())
            or not identity["ruleset_id"]
            or not identity["ruleset_version"]
            or len(identity["ruleset_hash"]) != 64
            or any(
                char not in "0123456789abcdef"
                for char in identity["ruleset_hash"]
            )
        ):
            raise RecordValidationError("canonical dataset rulesets are invalid")
        manifest_rulesets.add(
            (
                identity["ruleset_id"],
                identity["ruleset_version"],
                identity["ruleset_hash"],
            )
        )
    if len(manifest_rulesets) != len(raw_rulesets):
        raise RecordValidationError("canonical dataset rulesets contain duplicates")
    if manifest_rulesets != actual_rulesets:
        raise RecordValidationError(
            "canonical dataset manifest rulesets do not match content"
        )
    return manifest


def read_verified_jsonl(path: str) -> Iterator[HumanGameRecord]:
    """Verify the provenance sidecar, then stream canonical records."""

    verify_jsonl_manifest(path)
    yield from read_jsonl(path)


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
