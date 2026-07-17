"""Versioned P15 evaluation and P17 empirical-readiness policies."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Iterable


EVALUATION_PROTOCOL = "p15_paired_v1"
PROMOTION_MODE = "cardplay_only"
PROMOTION_ESTIMATOR = "cardplay_win_rate_delta"
OFFICIAL_CONFIDENCE_LEVEL = 0.95
OFFICIAL_STATISTICAL_UNIT = "deal"
OFFICIAL_CI_METHOD = "paired_percentile_bootstrap_v1"
# Preserve the closed P15 v1 promotion contract. P17 raises release-evidence
# requirements through a separate policy below; changing these values under
# the same protocol identifier would invalidate previously eligible reports.
MIN_PROMOTION_BOOTSTRAP_SAMPLES = 1000
MIN_PROMOTION_PAIRED_DEALS = 0

P17_READINESS_PROTOCOL = "p17_empirical_readiness_v1"
P17_MIN_BOOTSTRAP_SAMPLES = 2000
P17_MIN_PAIRED_DEALS = 1000

OFFICIAL_PERMUTATIONS = MappingProxyType({
    "cardplay_only": (
        ("candidate", "baseline", "baseline"),
        ("baseline", "candidate", "candidate"),
    ),
    "full_game": (
        ("candidate", "baseline", "baseline"),
        ("baseline", "candidate", "baseline"),
        ("baseline", "baseline", "candidate"),
    ),
})


def permutation_hash(
    mode: str, permutations: Iterable[Iterable[str]]
) -> str:
    """Hash the ordered mode/permutation identity used by a report."""
    payload = json.dumps(
        {"mode": mode, "permutations": [list(row) for row in permutations]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


OFFICIAL_PERMUTATION_HASHES = MappingProxyType({
    mode: permutation_hash(mode, permutations)
    for mode, permutations in OFFICIAL_PERMUTATIONS.items()
})
