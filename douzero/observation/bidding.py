"""Public, versioned observation for score bidding.

The card-play :class:`ObservationV2` is deliberately role based and requires a
legal card-action list.  Neither assumption is true before the landlord has
been selected.  This module therefore owns a small, separate bidding schema:
neutral seats stay neutral, actions are the fixed score values ``0/1/2/3``,
and every input is public (the bidder's hand plus public auction state).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .cards import CARD_VECTOR_DIM, cards_to_vector
from .public import BIDDING_TOKEN_WIDTH, encode_bidding_history
from .schema import BIDDING_ENCODING_VERSION

BIDDING_FEATURE_VERSION = "v2-bidding"
BIDDING_SCHEMA_VERSION = "v2-bidding-2"
BIDDING_ACTION_SCHEMA_VERSION = "score-0-1-2-3-v1"
BIDDING_HEAD_VERSION = "bid-policy-value-v2"
BIDDING_ACTIONS: tuple[int, ...] = (0, 1, 2, 3)
NUM_NEUTRAL_SEATS = 3
NEUTRAL_SEATS: tuple[str, ...] = ("0", "1", "2")
MAX_BIDDING_HISTORY = 3
RULESET_ONEHOT_WIDTH = 2
PHASE_ONEHOT_WIDTH = 5
PUBLIC_STYLE_WIDTH = 8


@dataclass(frozen=True)
class BiddingFieldSpec:
    name: str
    width: int
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "width": self.width, "dtype": self.dtype}


@dataclass(frozen=True)
class BiddingFeatureSchema:
    """Exact tensor contract for :class:`BiddingObservationV2`."""

    feature_version: str = BIDDING_FEATURE_VERSION
    schema_version: str = BIDDING_SCHEMA_VERSION
    action_schema_version: str = BIDDING_ACTION_SCHEMA_VERSION
    max_history: int = MAX_BIDDING_HISTORY
    public_style_width: int = PUBLIC_STYLE_WIDTH
    bidding_token_encoding_version: str = BIDDING_ENCODING_VERSION

    @property
    def fields(self) -> tuple[BiddingFieldSpec, ...]:
        return (
            BiddingFieldSpec("my_handcards", CARD_VECTOR_DIM, "float32"),
            BiddingFieldSpec("current_seat", NUM_NEUTRAL_SEATS, "float32"),
            BiddingFieldSpec("first_bidder", NUM_NEUTRAL_SEATS, "float32"),
            BiddingFieldSpec("current_highest_bid", len(BIDDING_ACTIONS), "float32"),
            BiddingFieldSpec(
                "bidding_history",
                self.max_history * BIDDING_TOKEN_WIDTH,
                "float32",
            ),
            BiddingFieldSpec("legal_bid_mask", len(BIDDING_ACTIONS), "float32"),
            BiddingFieldSpec("redeal_count", 1, "float32"),
            BiddingFieldSpec("ruleset_id", RULESET_ONEHOT_WIDTH, "float32"),
            BiddingFieldSpec("phase", PHASE_ONEHOT_WIDTH, "float32"),
            BiddingFieldSpec("public_style", self.public_style_width, "float32"),
        )

    @property
    def input_width(self) -> int:
        return sum(field.width for field in self.fields)

    def compatibility_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "action_schema_version": self.action_schema_version,
            "max_history": self.max_history,
            "public_style_width": self.public_style_width,
            "bidding_token_encoding_version": self.bidding_token_encoding_version,
            "fields": [field.to_dict() for field in self.fields],
        }

    def stable_hash(self) -> str:
        payload = json.dumps(self.compatibility_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_bidding_schema(
    *,
    max_history: int = MAX_BIDDING_HISTORY,
    public_style_width: int = PUBLIC_STYLE_WIDTH,
) -> BiddingFeatureSchema:
    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ValueError("max_history must be a positive int")
    if (
        isinstance(public_style_width, bool)
        or not isinstance(public_style_width, int)
        or public_style_width < 0
    ):
        raise ValueError("public_style_width must be a non-negative int")
    return BiddingFeatureSchema(
        max_history=max_history,
        public_style_width=public_style_width,
    )


@dataclass(frozen=True)
class BiddingObservationV2:
    """One public bidding decision using neutral seats only."""

    schema: BiddingFeatureSchema
    current_seat: str
    my_handcards: tuple[int, ...]
    current_highest_bid: int
    bidding_history: tuple[tuple[str, int], ...]
    first_bidder: str
    bidding_order: tuple[str, ...]
    legal_bids: tuple[int, ...]
    redeal_count: int
    ruleset_id: str
    ruleset_version: str
    ruleset_hash: str
    phase: str
    public_style: tuple[float, ...]
    features: np.ndarray
    bid_action_mask: np.ndarray
    feature_schema_hash: str
    kind: str = "public_bidding"

    def __post_init__(self) -> None:
        if self.phase != "bidding":
            raise ValueError(f"bidding observation phase must be 'bidding', got {self.phase!r}")
        if len(self.bidding_order) != NUM_NEUTRAL_SEATS or len(set(self.bidding_order)) != NUM_NEUTRAL_SEATS:
            raise ValueError("bidding_order must contain three unique neutral seats")
        if self.current_seat not in self.bidding_order or self.first_bidder not in self.bidding_order:
            raise ValueError("current_seat and first_bidder must be in bidding_order")
        if self.current_seat in {"landlord", "landlord_up", "landlord_down"}:
            raise ValueError("bidding observation must use neutral seats, not roles")
        if self.feature_schema_hash != self.schema.stable_hash():
            raise ValueError("bidding observation schema hash does not match its schema")
        if self.features.shape != (self.schema.input_width,):
            raise ValueError(
                f"bidding features must have shape ({self.schema.input_width},), "
                f"got {self.features.shape}"
            )
        if self.bid_action_mask.shape != (len(BIDDING_ACTIONS),):
            raise ValueError("bid_action_mask has the wrong shape")
        if self.bid_action_mask.dtype != np.bool_:
            raise ValueError("bid_action_mask must be bool")
        if not self.bid_action_mask.any():
            raise ValueError("a bidding decision must have at least one legal bid")
        expected = tuple(action for action, allowed in zip(BIDDING_ACTIONS, self.bid_action_mask) if allowed)
        if expected != self.legal_bids:
            raise ValueError("legal_bids and bid_action_mask disagree")
        for name in ("features", "bid_action_mask"):
            value = getattr(self, name)
            if value.flags.writeable:
                copied = value.copy()
                copied.setflags(write=False)
                object.__setattr__(self, name, copied)

    @property
    def is_privileged(self) -> bool:
        return False

    def to_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(self.features).copy()).float()
        return tensor.to(device) if device is not None else tensor


def _onehot(index: int, width: int) -> np.ndarray:
    result = np.zeros(width, dtype=np.float32)
    if 0 <= index < width:
        result[index] = 1.0
    return result


def _read(raw: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def get_bidding_obs_v2(
    raw: Mapping[str, Any],
    *,
    ruleset,
    redeal_count: int | None = None,
    public_style: Sequence[float] | None = None,
    schema: BiddingFeatureSchema | None = None,
) -> BiddingObservationV2:
    """Encode the public dictionary returned by ``Env`` during bidding.

    Unknown keys are intentionally ignored.  In particular, callers cannot
    smuggle opponent hands or bottom cards into the model by attaching them to
    the raw mapping: the encoder reads an explicit public allow-list only.
    """

    from douzero.env.rules import PHASE_BIDDING, RuleSet

    if not isinstance(raw, Mapping):
        raise TypeError("raw bidding observation must be a mapping")
    if not isinstance(ruleset, RuleSet):
        raise TypeError("ruleset must be a RuleSet")
    if ruleset.ruleset_id != "standard":
        raise ValueError("learned score bidding requires a standard ruleset")
    unsupported = set(ruleset.bid_values) - set(BIDDING_ACTIONS)
    if unsupported:
        raise ValueError(f"bidding action schema does not support bid values {sorted(unsupported)}")
    phase = str(_read(raw, "phase", default=""))
    if phase != PHASE_BIDDING:
        raise ValueError(f"raw observation is not in bidding phase: {phase!r}")

    active_schema = schema or build_bidding_schema()
    order = tuple(str(seat) for seat in _read(raw, "bidding_order", default=()))
    current_seat = str(_read(raw, "position", "current_seat", default=""))
    first_bidder = str(_read(raw, "first_bidder", default=(order[0] if order else "")))
    if len(order) != NUM_NEUTRAL_SEATS or set(order) != set(NEUTRAL_SEATS):
        raise ValueError(
            "bidding_order must be a permutation of neutral seats '0'/'1'/'2'"
        )
    if first_bidder != order[0]:
        raise ValueError("first_bidder must equal bidding_order[0]")
    hand = tuple(sorted(int(card) for card in _read(raw, "my_handcards", default=())))
    if len(hand) != 17:
        raise ValueError(
            f"a bidding observation must contain exactly 17 private cards, got {len(hand)}"
        )
    # Validate rank membership and deck multiplicities at the public boundary.
    cards_to_vector(hand)
    history = tuple(
        (str(seat), int(value))
        for seat, value in _read(raw, "bidding_history", default=())
    )
    if len(history) > active_schema.max_history:
        raise ValueError(
            f"bidding history has {len(history)} entries, schema cap is {active_schema.max_history}"
        )
    if tuple(seat for seat, _ in history) != order[: len(history)]:
        raise ValueError(
            "bidding_history seats must follow bidding_order without hidden turns"
        )
    if any(value not in ruleset.bid_values for _, value in history):
        raise ValueError("bidding_history contains a bid outside the ruleset")
    positive_bids = [value for _, value in history if value > 0]
    if any(
        later <= earlier
        for earlier, later in zip(positive_bids, positive_bids[1:])
    ):
        raise ValueError("non-pass bidding_history values must strictly increase")
    if 3 in positive_bids:
        raise ValueError("a bid of 3 is terminal and cannot appear in an active observation")
    if len(history) >= len(order) or current_seat != order[len(history)]:
        raise ValueError(
            "current_seat must be the next neutral seat in bidding_order"
        )
    observed_highest = max((value for _, value in history), default=0)
    highest = int(
        _read(
            raw,
            "current_highest_bid",
            default=observed_highest,
        )
    )
    if highest not in BIDDING_ACTIONS:
        raise ValueError(f"current_highest_bid is outside 0/1/2/3: {highest}")
    if highest != observed_highest:
        raise ValueError(
            "current_highest_bid does not match the public bidding_history"
        )
    legal_source = _read(raw, "legal_bids", default=None)
    if legal_source is None:
        legal_source = [value for value in ruleset.bid_values if value == 0 or value > highest]
    legal_set = {int(value) for value in legal_source}
    legal_bids = tuple(action for action in BIDDING_ACTIONS if action in legal_set)
    if legal_set - set(ruleset.bid_values):
        raise ValueError("raw observation contains bids outside the ruleset")
    expected_legal = {
        value for value in ruleset.bid_values if value == 0 or value > highest
    }
    if legal_set != expected_legal:
        raise ValueError(
            f"legal bidding actions disagree with public highest bid {highest}: "
            f"got {sorted(legal_set)}, expected {sorted(expected_legal)}"
        )
    bid_mask = np.asarray(
        [action in legal_set for action in BIDDING_ACTIONS], dtype=np.bool_
    )

    resolved_redeals = int(
        _read(raw, "redeal_count", default=0)
        if redeal_count is None
        else redeal_count
    )
    if resolved_redeals < 0:
        raise ValueError("redeal_count must be non-negative")
    style_source = () if public_style is None else public_style
    style = tuple(float(value) for value in style_source)
    if not all(math.isfinite(value) for value in style):
        raise ValueError("public_style values must be finite")
    if len(style) > active_schema.public_style_width:
        raise ValueError("public_style is wider than the bidding schema")
    style_vec = np.zeros(active_schema.public_style_width, dtype=np.float32)
    if style:
        style_vec[: len(style)] = np.asarray(style, dtype=np.float32)

    # Neutral-seat identity is canonical rather than relative to the rotated
    # bidding order. Otherwise all three first-bidder rotations collapse to the
    # same model input and neither current_seat nor first_bidder is observable.
    seat_to_index = {seat: index for index, seat in enumerate(NEUTRAL_SEATS)}
    history_batch = encode_bidding_history(history, NEUTRAL_SEATS)
    history_vec = np.zeros(
        (active_schema.max_history, BIDDING_TOKEN_WIDTH), dtype=np.float32
    )
    if history_batch.num_bids:
        history_vec[: history_batch.num_bids] = history_batch.tokens
    identity = ruleset.identity()
    ruleset_vec = _onehot(1 if ruleset.ruleset_id == "standard" else 0, RULESET_ONEHOT_WIDTH)
    phase_vec = _onehot(1, PHASE_ONEHOT_WIDTH)  # deal=0, bidding=1, reveal=2, play=3, terminal=4
    features = np.concatenate(
        (
            cards_to_vector(hand).astype(np.float32),
            _onehot(seat_to_index.get(current_seat, -1), NUM_NEUTRAL_SEATS),
            _onehot(seat_to_index.get(first_bidder, -1), NUM_NEUTRAL_SEATS),
            _onehot(highest, len(BIDDING_ACTIONS)),
            history_vec.reshape(-1),
            bid_mask.astype(np.float32),
            np.asarray([float(resolved_redeals)], dtype=np.float32),
            ruleset_vec,
            phase_vec,
            style_vec,
        )
    ).astype(np.float32)

    return BiddingObservationV2(
        schema=active_schema,
        current_seat=current_seat,
        my_handcards=hand,
        current_highest_bid=highest,
        bidding_history=history,
        first_bidder=first_bidder,
        bidding_order=order,
        legal_bids=legal_bids,
        redeal_count=resolved_redeals,
        ruleset_id=identity["ruleset_id"],
        ruleset_version=identity["ruleset_version"],
        ruleset_hash=identity["ruleset_hash"],
        phase=phase,
        public_style=style,
        features=features,
        bid_action_mask=bid_mask,
        feature_schema_hash=active_schema.stable_hash(),
    )
