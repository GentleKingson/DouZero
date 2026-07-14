"""Transposition keys and a bounded per-search cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True, slots=True)
class TranspositionKey:
    """Complete identity of a perfect-information card-play state."""

    hands: tuple[tuple[int, ...], ...]
    acting_role: str
    last_move: tuple[int, ...]
    last_non_pass_role: str | None
    consecutive_passes: int
    bomb_count: int
    rocket_count: int
    bid_value: int
    action_counts: tuple[int, ...]
    ruleset_hash: str


K = TypeVar("K")
V = TypeVar("V")


class TranspositionTable(Generic[K, V]):
    """Simple deterministic cache capped to prevent inference memory growth."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._values: dict[K, V] = {}

    def get(self, key: K) -> V | None:
        return self._values.get(key)

    def put(self, key: K, value: V) -> None:
        if self.max_entries and len(self._values) < self.max_entries:
            self._values[key] = value

    def __len__(self) -> int:
        return len(self._values)
