"""Versioned, replayable opening records for coach-guided training."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from douzero.env.rules import RuleSet


CARD_RANKS: tuple[int, ...] = tuple(range(3, 15)) + (17, 20, 30)
CANONICAL_DECK: tuple[int, ...] = tuple(
    card
    for rank in CARD_RANKS
    for card in ([rank] if rank in (20, 30) else [rank] * 4)
)
_LEGACY_ORDER = ("landlord", "landlord_up", "landlord_down")
_STANDARD_ORDER = ("0", "1", "2")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class OpeningRecord:
    """One complete deal plus its bidding and public-opening provenance.

    ``deck`` is training-only privileged data. It is used to initialize the
    environment and by the coach, but is never attached to an observation or
    passed to a deployment policy.
    """

    deck: tuple[int, ...]
    bidding_order: tuple[str, ...]
    ruleset: dict[str, Any]
    bids: tuple[int, ...] = ()
    landlord_candidate: str = "landlord"
    public_features: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported opening schema_version {self.schema_version}")
        if len(self.deck) != 54 or Counter(self.deck) != Counter(CANONICAL_DECK):
            raise ValueError("OpeningRecord.deck must be one complete 54-card deck")
        ruleset = RuleSet.from_dict(self.ruleset)
        expected_order = _LEGACY_ORDER if ruleset.ruleset_id == "legacy" else _STANDARD_ORDER
        if len(self.bidding_order) != 3 or set(self.bidding_order) != set(expected_order):
            raise ValueError(
                f"bidding_order must be a permutation of {expected_order}, "
                f"got {self.bidding_order}"
            )
        if any(isinstance(bid, bool) or not isinstance(bid, int) for bid in self.bids):
            raise TypeError("OpeningRecord.bids must contain integers")
        if ruleset.ruleset_id == "legacy":
            if self.bids:
                raise ValueError("legacy openings cannot contain bids")
            if self.landlord_candidate != "landlord":
                raise ValueError("legacy openings use landlord_candidate='landlord'")
        else:
            if len(self.bids) > 3:
                raise ValueError("standard openings can contain at most three bids")
            if any(bid not in ruleset.bid_values for bid in self.bids):
                raise ValueError("opening contains a bid outside the RuleSet")
            if self.landlord_candidate not in _STANDARD_ORDER:
                raise ValueError("standard landlord_candidate must be a neutral seat")
        if not isinstance(self.public_features, dict):
            raise TypeError("OpeningRecord.public_features must be a dict")

    @property
    def ruleset_obj(self) -> RuleSet:
        """Return the validated :class:`RuleSet` represented by this record."""

        return RuleSet.from_dict(self.ruleset)

    @property
    def opening_id(self) -> str:
        """Content-address the opening for labels, logs, and deduplication."""

        payload = self.to_dict(include_id=False)
        return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        """Serialize to a strict JSON-compatible mapping."""

        payload = asdict(self)
        payload["deck"] = list(self.deck)
        payload["bidding_order"] = list(self.bidding_order)
        payload["bids"] = list(self.bids)
        if include_id:
            payload["opening_id"] = self.opening_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpeningRecord":
        """Load a record and verify its optional content-addressed ID."""

        values = dict(raw)
        supplied_id = values.pop("opening_id", None)
        required = {
            "deck", "bidding_order", "ruleset", "bids", "landlord_candidate",
            "public_features", "schema_version",
        }
        if set(values) != required:
            raise ValueError(
                f"opening fields must be {sorted(required)}, got {sorted(values)}"
            )
        record = cls(
            deck=tuple(values["deck"]),
            bidding_order=tuple(values["bidding_order"]),
            ruleset=dict(values["ruleset"]),
            bids=tuple(values["bids"]),
            landlord_candidate=str(values["landlord_candidate"]),
            public_features=dict(values["public_features"]),
            schema_version=int(values["schema_version"]),
        )
        if supplied_id is not None and supplied_id != record.opening_id:
            raise ValueError("opening_id does not match the record contents")
        return record

    def to_card_play_data(self) -> dict[str, list[int]]:
        """Convert the stored deck to the environment's validated deal shape."""

        if self.ruleset_obj.ruleset_id == "legacy":
            data = {
                "landlord": list(self.deck[:20]),
                "landlord_up": list(self.deck[20:37]),
                "landlord_down": list(self.deck[37:54]),
                "three_landlord_cards": list(self.deck[17:20]),
            }
        else:
            data = {
                "landlord": list(self.deck[:17]),
                "landlord_up": list(self.deck[17:34]),
                "landlord_down": list(self.deck[34:51]),
                "three_landlord_cards": list(self.deck[51:54]),
            }
        return {name: sorted(cards) for name, cards in data.items()}


def random_opening(rng: random.Random, ruleset: RuleSet | None = None) -> OpeningRecord:
    """Generate a complete legal opening using only the supplied local RNG."""

    active_ruleset = ruleset or RuleSet.legacy()
    cards = list(CANONICAL_DECK)
    rng.shuffle(cards)
    if active_ruleset.ruleset_id == "legacy":
        order = _LEGACY_ORDER
        candidate = "landlord"
        bids: tuple[int, ...] = ()
        public = {
            "ruleset_hash": active_ruleset.stable_hash(),
            "bottom_cards_public": True,
            "bottom_card_count": 3,
        }
    else:
        first = rng.randrange(3)
        order = tuple(str((first + offset) % 3) for offset in range(3))
        candidate = order[0]
        bids = ()
        public = {
            "ruleset_hash": active_ruleset.stable_hash(),
            "first_bidder": order[0],
            "bottom_cards_public": False,
            "bottom_card_count": 3,
        }
    return OpeningRecord(
        deck=tuple(cards),
        bidding_order=order,
        ruleset=active_ruleset.to_dict(),
        bids=bids,
        landlord_candidate=candidate,
        public_features=public,
    )
