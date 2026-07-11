"""Legacy evaluation data adapter (P02 Slice 4).

Provides :func:`load_eval_data` that auto-detects the format of a pickled
evaluation dataset:

- **Legacy format** (no ``format_version`` key): a ``list[dict]`` where each
  dict has keys ``landlord`` (20 cards), ``landlord_up`` (17),
  ``landlord_down`` (17), ``three_landlord_cards`` (3). This is the original
  DouZero eval data format, unchanged.
- **Standard format** (``format_version == 2``): a ``list[dict]`` where each
  dict has keys ``deck`` (54-card order), ``first_bidder`` (neutral seat
  "0"/"1"/"2"), ``bidding_order``, ``ruleset_id``, ``ruleset_version``,
  ``ruleset_hash``, ``schema_version``, and optional ``bidding_script``.

The adapter validates that the data format matches the requested ruleset and
raises a precise error on mismatch. For standard data, the ``ruleset_hash``
is checked against the active RuleSet's hash to reject same-ID-different-
params. **Every record is validated**, not just the first.

.. warning::

    Evaluation data files are loaded with ``pickle.load``. Only load files
    from trusted sources — pickle can execute arbitrary code. The standard
    format is designed to be simple enough for a future JSON migration.
"""

from __future__ import annotations

import pickle
from collections import Counter
from typing import Any


_NEUTRAL_SEATS = ("0", "1", "2")


def _validate_standard_deck(deck: list[int]) -> None:
    """Validate that a deck is a legal 54-card DouDizhu deck."""
    if not isinstance(deck, list):
        raise TypeError(f"deck must be a list, got {type(deck).__name__}")
    if len(deck) != 54:
        raise ValueError(f"Standard deck must have 54 cards, got {len(deck)}")
    counts = Counter(deck)
    for rank in range(3, 15):
        if counts[rank] != 4:
            raise ValueError(
                f"Rank {rank} must appear 4 times in the deck, got {counts[rank]}"
            )
    if counts[17] != 4:
        raise ValueError(f"Rank 17 (2) must appear 4 times, got {counts[17]}")
    if counts[20] != 1:
        raise ValueError(f"Small joker (20) must appear once, got {counts[20]}")
    if counts[30] != 1:
        raise ValueError(f"Big joker (30) must appear once, got {counts[30]}")


def _validate_standard_record(deal: dict[str, Any], idx: int,
                              expected_ruleset) -> None:
    """Validate a single v2 standard record.

    Checks: schema_version, ruleset_id, ruleset_version, ruleset_hash
    (required and matching), deck validity, first_bidder/bidding_order
    (neutral seat permutation).
    """
    # schema_version (required).
    sv = deal.get("schema_version")
    if sv is None:
        raise ValueError(
            f"Standard eval data record {idx} is missing 'schema_version'."
        )
    if sv != 1:
        raise ValueError(
            f"Standard eval data record {idx} has unsupported schema_version "
            f"{sv!r}; expected 1."
        )

    # ruleset_hash (required, not optional).
    actual_hash = deal.get("ruleset_hash")
    if actual_hash is None:
        raise ValueError(
            f"Standard eval data record {idx} is missing 'ruleset_hash'. "
            f"All v2 records must include ruleset_hash."
        )
    if expected_ruleset is not None:
        expected_hash = expected_ruleset.stable_hash()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Standard eval data record {idx} has ruleset_hash "
                f"{actual_hash!r} but the active RuleSet hash is "
                f"{expected_hash!r}. The rule parameters do not match."
            )

    # ruleset_id / ruleset_version (informational but must be present).
    for field in ("ruleset_id", "ruleset_version"):
        if field not in deal:
            raise ValueError(
                f"Standard eval data record {idx} is missing {field!r}."
            )

    # Deck validity.
    if "deck" not in deal:
        raise ValueError(
            f"Standard eval data record {idx} is missing 'deck'."
        )
    _validate_standard_deck(deal["deck"])

    # first_bidder / bidding_order (neutral seat permutation).
    first_bidder = deal.get("first_bidder")
    if first_bidder is not None:
        if not isinstance(first_bidder, str) or first_bidder not in _NEUTRAL_SEATS:
            raise ValueError(
                f"Standard eval data record {idx} has first_bidder "
                f"{first_bidder!r}; expected one of {_NEUTRAL_SEATS}."
            )
    bidding_order = deal.get("bidding_order")
    if bidding_order is not None:
        if sorted(bidding_order) != list(_NEUTRAL_SEATS):
            raise ValueError(
                f"Standard eval data record {idx} has bidding_order "
                f"{bidding_order!r}; expected a permutation of {_NEUTRAL_SEATS}."
            )
        if first_bidder is not None and bidding_order[0] != first_bidder:
            raise ValueError(
                f"Standard eval data record {idx}: first_bidder "
                f"{first_bidder!r} != bidding_order[0] {bidding_order[0]!r}."
            )


def load_eval_data(path: str, ruleset: str = "legacy",
                   expected_ruleset=None) -> list[dict[str, Any]]:
    """Load a pickled evaluation dataset, auto-detecting its format.

    Parameters
    ----------
    path
        Path to the ``.pkl`` file. **Only load trusted files** — pickle can
        execute arbitrary code.
    ruleset
        The expected ruleset: ``"legacy"`` or ``"standard"``. If the data
        format does not match, a :class:`ValueError` is raised.
    expected_ruleset
        Optional active :class:`RuleSet` instance. For standard data, the
        ``ruleset_hash`` of every record is validated against this RuleSet's
        hash.

    Returns
    -------
    list[dict]
        The loaded dataset (format depends on the file).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise TypeError(
            f"Eval data at {path} must be a list of dicts, got {type(data).__name__}"
        )
    if len(data) == 0:
        return data

    # Detect format from the first element.
    first = data[0]
    if not isinstance(first, dict):
        raise TypeError(
            f"Eval data elements must be dicts, got {type(first).__name__}"
        )

    is_standard = "format_version" in first and first.get("format_version") == 2

    if ruleset == "standard" and not is_standard:
        raise ValueError(
            f"Eval data at {path} is in legacy format (no format_version=2) "
            f"but ruleset='standard' was requested. Regenerate with "
            f"`generate_eval_data.py --ruleset standard`."
        )
    if ruleset == "legacy" and is_standard:
        raise ValueError(
            f"Eval data at {path} is in standard format (format_version=2) "
            f"but ruleset='legacy' was requested. Use --ruleset standard or "
            f"regenerate with `generate_eval_data.py --ruleset legacy`."
        )

    # Validate EVERY record (not just the first).
    if is_standard:
        for idx, deal in enumerate(data):
            if not isinstance(deal, dict):
                raise TypeError(
                    f"Standard eval data record {idx} must be a dict, got "
                    f"{type(deal).__name__}."
                )
            # Reject mixed legacy records inside a standard dataset.
            if deal.get("format_version") != 2:
                raise ValueError(
                    f"Eval data record {idx} has a different format than "
                    f"record 0 (expected format_version=2). Mixed datasets "
                    f"are not allowed."
                )
            _validate_standard_record(deal, idx, expected_ruleset)

    return data


def is_standard_format(data: list[dict[str, Any]]) -> bool:
    """Return True if the dataset is in standard (v2) format."""
    if not data or not isinstance(data[0], dict):
        return False
    return data[0].get("format_version") == 2


def is_legacy_format(data: list[dict[str, Any]]) -> bool:
    """Return True if the dataset is in legacy (v1) format."""
    if not data or not isinstance(data[0], dict):
        return False
    return "format_version" not in data[0]


def deal_standard_deck(deck: list[int]) -> dict[str, list[int]]:
    """Slice a 54-card deck into the standard 17+17+17+3 dealing.

    Returns a dict with keys ``landlord``, ``landlord_up``,
    ``landlord_down``, ``three_landlord_cards``, each sorted. The
    ``first_bidder`` seat receives the first 17 cards (not necessarily the
    landlord — the landlord is determined by bidding).
    """
    _validate_standard_deck(deck)
    return {
        'landlord': sorted(deck[:17]),
        'landlord_up': sorted(deck[17:34]),
        'landlord_down': sorted(deck[34:51]),
        'three_landlord_cards': sorted(deck[51:54]),
    }
