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


T = TypeVar("T")


class TranspositionTable(Generic[T]):
    """Simple deterministic cache capped to prevent inference memory growth."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._values: dict[TranspositionKey, T] = {}

    def get(self, key: TranspositionKey) -> T | None:
        return self._values.get(key)

    def put(self, key: TranspositionKey, value: T) -> None:
        if self.max_entries and len(self._values) < self.max_entries:
            self._values[key] = value

    def __len__(self) -> int:
        return len(self._values)
