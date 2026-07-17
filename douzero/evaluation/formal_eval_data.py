"""Safe, strict deal-set format for formal evaluation.

Formal evaluation data is JSON, never pickle.  The decoder rejects ambiguous
JSON constructs and validates the complete payload before returning any deals.
The legacy pickle adapter remains available only for non-formal compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from douzero.env.rules import RuleSet

from .legacy_data_adapter import (
    _validate_legacy_record,
    _validate_standard_record,
)
from .scenario import canonical_deal_hash


FORMAL_EVAL_DATA_SCHEMA_VERSION = "douzero-formal-eval-data-v1"
_TOP_LEVEL_KEYS = {"schema_version", "mode", "ruleset", "deals"}
_RULESET_IDENTITY_KEYS = {"ruleset_id", "ruleset_version", "ruleset_hash"}
_LEGACY_DEAL_KEYS = {
    "landlord",
    "landlord_up",
    "landlord_down",
    "three_landlord_cards",
}
_STANDARD_DEAL_KEYS = {
    "format_version",
    "schema_version",
    "ruleset_id",
    "ruleset_version",
    "ruleset_hash",
    "deck",
    "first_bidder",
    "bidding_order",
    "bidding_script",
}
_MAX_FORMAL_DATA_BYTES = 64 * 1024 * 1024


class FormalEvalDataError(ValueError):
    """Raised when a formal deal-set file is unsafe or non-canonical."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalEvalDataError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise FormalEvalDataError(
        f"non-finite JSON number {token!r} is not permitted"
    )


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FormalEvalDataError(
            f"{label} must contain exactly {sorted(expected)}; "
            f"missing={missing}, extra={extra}"
        )


def _require_int_list(value: Any, *, label: str) -> list[int]:
    if type(value) is not list:
        raise FormalEvalDataError(f"{label} must be a JSON array")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise FormalEvalDataError(
                f"{label}[{index}] must be a JSON integer, got "
                f"{type(item).__name__}"
            )
    return value


def _validate_ruleset_identity(
    raw: Any, *, expected_ruleset: RuleSet
) -> None:
    if type(raw) is not dict:
        raise FormalEvalDataError("formal eval data ruleset must be a JSON object")
    _require_exact_keys(raw, _RULESET_IDENTITY_KEYS, label="ruleset")
    for name in ("ruleset_id", "ruleset_version", "ruleset_hash"):
        if type(raw[name]) is not str:
            raise FormalEvalDataError(f"ruleset.{name} must be a JSON string")
    digest = raw["ruleset_hash"]
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise FormalEvalDataError(
            "ruleset.ruleset_hash must be a full lowercase SHA-256"
        )
    expected_identity = expected_ruleset.identity()
    if raw != expected_identity:
        raise FormalEvalDataError(
            "formal eval data ruleset identity does not match the active RuleSet"
        )


def _validate_legacy_deal(deal: Any, index: int) -> None:
    if type(deal) is not dict:
        raise FormalEvalDataError(f"deal {index} must be a JSON object")
    _require_exact_keys(deal, _LEGACY_DEAL_KEYS, label=f"legacy deal {index}")
    for name in sorted(_LEGACY_DEAL_KEYS):
        cards = _require_int_list(deal[name], label=f"deal {index}.{name}")
        if cards != sorted(cards):
            raise FormalEvalDataError(
                f"deal {index}.{name} must be in canonical sorted order"
            )
    try:
        _validate_legacy_record(deal, index)
    except (TypeError, ValueError) as exc:
        raise FormalEvalDataError(str(exc)) from exc


def _validate_standard_deal(
    deal: Any, index: int, *, expected_ruleset: RuleSet
) -> None:
    if type(deal) is not dict:
        raise FormalEvalDataError(f"deal {index} must be a JSON object")
    _require_exact_keys(deal, _STANDARD_DEAL_KEYS, label=f"standard deal {index}")
    if type(deal["format_version"]) is not int or deal["format_version"] != 2:
        raise FormalEvalDataError(
            f"standard deal {index}.format_version must be integer 2"
        )
    if type(deal["schema_version"]) is not int or deal["schema_version"] != 1:
        raise FormalEvalDataError(
            f"standard deal {index}.schema_version must be integer 1"
        )
    for name in ("ruleset_id", "ruleset_version", "ruleset_hash", "first_bidder"):
        if type(deal[name]) is not str:
            raise FormalEvalDataError(
                f"standard deal {index}.{name} must be a JSON string"
            )
    _require_int_list(deal["deck"], label=f"deal {index}.deck")
    bidding_order = deal["bidding_order"]
    if type(bidding_order) is not list or any(
        type(seat) is not str for seat in bidding_order
    ):
        raise FormalEvalDataError(
            f"standard deal {index}.bidding_order must be an array of strings"
        )
    if deal["bidding_script"] is not None:
        raise FormalEvalDataError(
            f"standard deal {index}.bidding_script must be null"
        )
    try:
        _validate_standard_record(deal, index, expected_ruleset)
    except (TypeError, ValueError) as exc:
        raise FormalEvalDataError(str(exc)) from exc


def _validate_payload(
    payload: Any, *, expected_mode: str, expected_ruleset: RuleSet
) -> list[dict[str, Any]]:
    if expected_mode not in ("cardplay_only", "full_game"):
        raise FormalEvalDataError(
            "expected_mode must be 'cardplay_only' or 'full_game'"
        )
    if not isinstance(expected_ruleset, RuleSet):
        raise TypeError("expected_ruleset must be a RuleSet")
    expected_ruleset_id = (
        "legacy" if expected_mode == "cardplay_only" else "standard"
    )
    if expected_ruleset.ruleset_id != expected_ruleset_id:
        raise FormalEvalDataError(
            f"{expected_mode} requires ruleset_id={expected_ruleset_id!r}"
        )
    if type(payload) is not dict:
        raise FormalEvalDataError("formal eval data must be a JSON object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, label="formal eval data")
    if payload["schema_version"] != FORMAL_EVAL_DATA_SCHEMA_VERSION:
        raise FormalEvalDataError(
            "unsupported formal eval data schema_version; expected "
            f"{FORMAL_EVAL_DATA_SCHEMA_VERSION!r}"
        )
    if type(payload["schema_version"]) is not str:
        raise FormalEvalDataError("formal eval data schema_version must be a string")
    if type(payload["mode"]) is not str or payload["mode"] != expected_mode:
        raise FormalEvalDataError(
            f"formal eval data mode must equal {expected_mode!r}"
        )
    _validate_ruleset_identity(
        payload["ruleset"], expected_ruleset=expected_ruleset
    )
    deals = payload["deals"]
    if type(deals) is not list:
        raise FormalEvalDataError("formal eval data deals must be a JSON array")
    if not deals:
        raise FormalEvalDataError("formal eval data must contain at least one deal")
    for index, deal in enumerate(deals):
        if expected_mode == "cardplay_only":
            _validate_legacy_deal(deal, index)
        else:
            _validate_standard_deal(
                deal, index, expected_ruleset=expected_ruleset
            )
    deal_hashes = [canonical_deal_hash(deal) for deal in deals]
    if len(set(deal_hashes)) != len(deal_hashes):
        raise FormalEvalDataError("formal evaluation deals must be unique")
    return deals


def load_formal_eval_data(
    path: str | Path, *, expected_mode: str, expected_ruleset: RuleSet
) -> list[dict[str, Any]]:
    """Load an exact-schema formal JSON deal set without code execution."""

    source = Path(path)
    if source.suffix != ".json":
        raise FormalEvalDataError(
            "formal evaluation data must use the strict .json format; "
            "pickle files are forbidden"
        )
    raw = source.read_bytes()
    if len(raw) > _MAX_FORMAL_DATA_BYTES:
        raise FormalEvalDataError(
            f"formal eval data exceeds {_MAX_FORMAL_DATA_BYTES} bytes"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormalEvalDataError("formal eval data must be UTF-8 JSON") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except FormalEvalDataError:
        raise
    except json.JSONDecodeError as exc:
        raise FormalEvalDataError(
            f"invalid formal eval data JSON: {exc.msg}"
        ) from exc
    return _validate_payload(
        payload,
        expected_mode=expected_mode,
        expected_ruleset=expected_ruleset,
    )


def write_formal_eval_data(
    path: str | Path,
    *,
    mode: str,
    ruleset: RuleSet,
    deals: Sequence[dict[str, Any]],
) -> Path:
    """Validate and deterministically write a formal JSON deal set."""

    destination = Path(path)
    if destination.suffix != ".json":
        raise FormalEvalDataError("formal evaluation data output must end in .json")
    payload = {
        "schema_version": FORMAL_EVAL_DATA_SCHEMA_VERSION,
        "mode": mode,
        "ruleset": ruleset.identity(),
        "deals": list(deals),
    }
    _validate_payload(payload, expected_mode=mode, expected_ruleset=ruleset)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8", newline="")
    return destination
