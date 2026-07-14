"""Race-free, versioned policy snapshots for actor processes.

Learners publish into an unused shared-memory slot and atomically flip the
active slot only after the complete state dict has been copied. Actors lease a
slot for a whole episode, so a model currently running inference is never
modified in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class PolicyLease(Generic[T]):
    """A stable actor model and its published version for one episode."""

    slot: int
    version: int
    model: T


class VersionedPolicyPool(Generic[T]):
    """Shared model slots with atomic publication and episode leases."""

    def __init__(self, models: Iterable[T], *, mp_context) -> None:
        self.models = list(models)
        if len(self.models) < 2:
            raise ValueError("VersionedPolicyPool requires at least two slots")
        self._lock = mp_context.Lock()
        self._active_slot = mp_context.Value("i", 0, lock=False)
        self._version = mp_context.Value("q", 0, lock=False)
        self._readers = mp_context.Array("i", len(self.models), lock=False)

    @property
    def version(self) -> int:
        """Return the most recently published policy version."""
        with self._lock:
            return int(self._version.value)

    def acquire(self) -> PolicyLease[T]:
        """Lease the active immutable slot until :meth:`release` is called."""
        with self._lock:
            slot = int(self._active_slot.value)
            self._readers[slot] += 1
            version = int(self._version.value)
        return PolicyLease(slot=slot, version=version, model=self.models[slot])

    def release(self, lease: PolicyLease[T]) -> None:
        """Release a previously acquired episode lease."""
        with self._lock:
            if lease.slot < 0 or lease.slot >= len(self.models):
                raise ValueError(f"invalid policy slot {lease.slot}")
            if self._readers[lease.slot] <= 0:
                raise RuntimeError(f"policy slot {lease.slot} is not leased")
            self._readers[lease.slot] -= 1

    def initialize(self, source_models: dict[str, torch.nn.Module]) -> None:
        """Copy the initial learner state into every slot before actors start."""
        with self._lock, torch.no_grad():
            for target in self.models:
                for position, source in source_models.items():
                    target.get_model(position).load_state_dict(source.state_dict())

    def publish(
        self,
        source_models: dict[str, torch.nn.Module],
        *,
        version: int,
    ) -> bool:
        """Publish a complete snapshot, or return ``False`` if no slot is free."""
        with self._lock:
            current = int(self._version.value)
            if version <= current:
                raise ValueError(
                    f"policy versions must increase: current={current}, new={version}"
                )
            active = int(self._active_slot.value)
            writable = next(
                (slot for slot in range(len(self.models))
                 if slot != active and self._readers[slot] == 0),
                None,
            )
            if writable is None:
                return False
            target = self.models[writable]
            with torch.no_grad():
                for position, source in source_models.items():
                    target.get_model(position).load_state_dict(source.state_dict())
            self._active_slot.value = writable
            self._version.value = version
            return True

    def reader_counts(self) -> tuple[int, ...]:
        """Return per-slot lease counts for diagnostics and tests."""
        with self._lock:
            return tuple(int(v) for v in self._readers)
