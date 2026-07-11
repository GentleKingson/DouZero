"""Legacy evaluation data adapter (P02 Slice 4).

Provides :func:`load_eval_data` that auto-detects the format of a pickled
evaluation dataset:

- **Legacy format** (no ``format_version`` key): a ``list[dict]`` where each
  dict has keys ``landlord`` (20 cards), ``landlord_up`` (17),
  ``landlord_down`` (17), ``three_landlord_cards`` (3). This is the original
  DouZero eval data format, unchanged.
- **Standard format** (``format_version == 2``): a ``list[dict]`` where each
  dict has keys ``deck`` (54-card order), ``first_bidder``, ``bidding_order``,
  ``ruleset_id``, ``ruleset_version``, ``ruleset_hash``, and optional
  ``bidding_script``.

The adapter validates that the data format matches the requested ruleset and
raises a precise error on mismatch. For standard data, the ``ruleset_hash``
is checked against the canonical ``RuleSet.standard()`` hash to reject
same-ID-different-params. Legacy eval data remains readable.

.. warning::

    Evaluation data files are loaded with ``pickle.load``. Only load files
    from trusted sources — pickle can execute arbitrary code. The standard
    format is designed to be simple enough for a future JSON migration.
"""

from __future__ import annotations

import pickle
from collections import Counter
from typing import Any


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


def load_eval_data(path: str, ruleset: str = "legacy") -> list[dict[str, Any]]:
    """Load a pickled evaluation dataset, auto-detecting its format.

    Parameters
    ----------
    path
        Path to the ``.pkl`` file. **Only load trusted files** — pickle can
        execute arbitrary code.
    ruleset
        The expected ruleset: ``"legacy"`` or ``"standard"``. If the data
        format does not match, a :class:`ValueError` is raised. For standard
        data, the ``ruleset_hash`` is validated against the canonical
        ``RuleSet.standard()`` hash.

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

    # Validate standard data: check schema_version, ruleset_hash, and deck.
    if is_standard:
        sv = first.get("schema_version")
        if sv is None:
            raise ValueError(
                f"Standard eval data at {path} is missing 'schema_version'. "
                f"The file may be from an incompatible P02 build."
            )
        if sv != 1:
            raise ValueError(
                f"Standard eval data at {path} has unsupported schema_version "
                f"{sv!r}; expected 1."
            )
        # Validate ruleset_hash against the canonical standard ruleset.
        from douzero.env.rules import RuleSet
        expected_hash = RuleSet.standard().stable_hash()[:16]
        actual_hash = first.get("ruleset_hash")
        if actual_hash is not None and actual_hash != expected_hash:
            raise ValueError(
                f"Standard eval data at {path} has ruleset_hash "
                f"{actual_hash!r} but the canonical RuleSet.standard() hash is "
                f"{expected_hash!r}. The rule parameters do not match — "
                f"regenerate with the current ruleset."
            )
        # Validate every deal's deck.
        for i, deal in enumerate(data):
            _validate_standard_deck(deal["deck"])

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
