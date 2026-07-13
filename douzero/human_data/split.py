"""Deterministic, leak-free dataset splitting (P08).

Splits canonical records into disjoint train / validation (and optional test)
sets partitioned by ``game_id``. AGENTS.md: "Split data by complete game, and
where appropriate by player and time, to prevent leakage." The split is
therefore **by game** — two decisions from the same game never end up in
different splits, which would leak the deal and inflate BC metrics.

The split is deterministic for a fixed ``(seed, ratios)`` and produces no
``game_id`` overlap (asserted at split time). Stratification by ``winner_team``
is offered so the train/val distributions keep a comparable landlord/farmer
balance (this also guards against survivorship bias: both teams are
represented).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .schema import HumanGameRecord


class SplitError(ValueError):
    """Raised when a split configuration is invalid or leakage is detected."""


@dataclass(frozen=True)
class SplitConfig:
    """Split ratios (must sum to 1.0 within tolerance).

    ``val_ratio`` of 0.0 yields an empty validation set. ``test_ratio`` of 0.0
    yields an empty test set. ``seed`` drives a deterministic shuffle.
    """

    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 0
    stratify_by_team: bool = True

    def __post_init__(self) -> None:
        for name, val in (("val_ratio", self.val_ratio),
                          ("test_ratio", self.test_ratio)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise SplitError(
                    f"{name} must be a float, got {type(val).__name__}"
                )
            if not 0.0 <= val < 1.0:
                raise SplitError(
                    f"{name} must be in [0.0, 1.0), got {val}"
                )
        if self.val_ratio + self.test_ratio >= 1.0:
            raise SplitError(
                f"val_ratio + test_ratio must be < 1.0, got "
                f"{self.val_ratio + self.test_ratio}"
            )


@dataclass(frozen=True)
class Split:
    """A three-way split of records (any subset may be empty)."""

    train: tuple[HumanGameRecord, ...]
    val: tuple[HumanGameRecord, ...]
    test: tuple[HumanGameRecord, ...]

    @property
    def all_game_ids(self) -> tuple[str, ...]:
        return tuple(r.game_id for r in (self.train + self.val + self.test))

    def assert_no_overlap(self) -> None:
        """Assert no ``game_id`` appears in more than one split."""
        train_ids = {r.game_id for r in self.train}
        val_ids = {r.game_id for r in self.val}
        test_ids = {r.game_id for r in self.test}
        if train_ids & val_ids:
            raise SplitError(
                f"train/val game_id overlap: {sorted(train_ids & val_ids)}"
            )
        if train_ids & test_ids:
            raise SplitError(
                f"train/test game_id overlap: {sorted(train_ids & test_ids)}"
            )
        if val_ids & test_ids:
            raise SplitError(
                f"val/test game_id overlap: {sorted(val_ids & test_ids)}"
            )


def _slice_counts(
    n: int, val_ratio: float, test_ratio: float
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) covering exactly ``n`` items."""
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 0:
        n_train = 0
        n_val = min(n_val, n - n_test)
    return n_train, n_val, n_test


def split_records(
    records: Iterable[HumanGameRecord],
    config: SplitConfig | None = None,
) -> Split:
    """Split records by ``game_id`` into train / val / test (no overlap).

    Deterministic for a fixed ``(config.seed, config ratios, input order)``.
    When ``config.stratify_by_team`` is True the split is performed within each
    ``winner_team`` stratum so the landlord/farmer balance is preserved across
    splits (guards against survivorship bias).
    """
    cfg = config or SplitConfig()
    record_list = list(records)

    # Reject duplicate game_ids at the input boundary (a duplicate would make
    # the no-overlap guarantee meaningless).
    ids = [r.game_id for r in record_list]
    if len(set(ids)) != len(ids):
        dupes = sorted({g for g in ids if ids.count(g) > 1})
        raise SplitError(
            f"input contains duplicate game_ids {dupes}; de-duplicate before "
            f"splitting (ingest.dedupe_by_game_id)."
        )

    rng = random.Random(cfg.seed)

    if not cfg.stratify_by_team:
        shuffled = sorted(record_list, key=lambda r: r.game_id)
        rng.shuffle(shuffled)
        n_train, n_val, n_test = _slice_counts(
            len(shuffled), cfg.val_ratio, cfg.test_ratio
        )
        split = Split(
            train=tuple(shuffled[:n_train]),
            val=tuple(shuffled[n_train:n_train + n_val]),
            test=tuple(shuffled[n_train + n_val:n_train + n_val + n_test]),
        )
        split.assert_no_overlap()
        return split

    # Stratified by winner_team: split each stratum independently, then merge.
    strata: dict[str, list[HumanGameRecord]] = {
        "landlord": [],
        "farmer": [],
    }
    for r in record_list:
        team = r.final_result.get("winner_team", "landlord")
        strata.setdefault(team, []).append(r)

    train: list[HumanGameRecord] = []
    val: list[HumanGameRecord] = []
    test: list[HumanGameRecord] = []
    for team in sorted(strata.keys()):
        bucket = sorted(strata[team], key=lambda r: r.game_id)
        rng.shuffle(bucket)
        n_train, n_val, n_test = _slice_counts(
            len(bucket), cfg.val_ratio, cfg.test_ratio
        )
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:n_train + n_val + n_test])

    # Sort each split by game_id for stable output.
    train.sort(key=lambda r: r.game_id)
    val.sort(key=lambda r: r.game_id)
    test.sort(key=lambda r: r.game_id)

    split = Split(train=tuple(train), val=tuple(val), test=tuple(test))
    split.assert_no_overlap()
    return split


def split_stats(split: Split) -> dict[str, dict[str, int]]:
    """Return per-split counts by winner_team (for survivorship-bias audit)."""
    buckets = {"train": split.train, "val": split.val, "test": split.test}
    stats: dict[str, dict[str, int]] = {}
    for name, recs in buckets.items():
        team_counts: dict[str, int] = {}
        for r in recs:
            team = r.final_result.get("winner_team", "unknown")
            team_counts[team] = team_counts.get(team, 0) + 1
        stats[name] = {"total": len(recs), **team_counts}
    return stats
