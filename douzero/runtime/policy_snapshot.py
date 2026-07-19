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
    owner_id: int
    generation: int


class VersionedPolicyPool(Generic[T]):
    """Shared model slots with atomic publication and episode leases."""

    def __init__(self, models: Iterable[T], *, mp_context,
                 max_owners: int = 1024) -> None:
        self.models = list(models)
        if len(self.models) < 2:
            raise ValueError("VersionedPolicyPool requires at least two slots")
        if max_owners < 1:
            raise ValueError("max_owners must be positive")
        self._lock = mp_context.Lock()
        self._active_slot = mp_context.Value("i", 0, lock=False)
        self._version = mp_context.Value("q", 0, lock=False)
        self._readers = mp_context.Array("i", len(self.models), lock=False)
        self._writers = mp_context.Array("i", len(self.models), lock=False)
        self._owner_slots = mp_context.Array("i", max_owners, lock=False)
        self._owner_generations = mp_context.Array("q", max_owners, lock=False)
        for owner in range(max_owners):
            self._owner_slots[owner] = -1
        self._target_states = []
        for target in self.models:
            role_models = (
                target.get_models()
                if hasattr(target, "get_models")
                else target.models
            )
            self._target_states.append({
                position: target.get_model(position).state_dict(keep_vars=True)
                for position in role_models
            })

    def _copy_state(self, slot: int,
                    source_models: dict[str, torch.nn.Module]) -> None:
        """Copy pre-paired state tensors without rebuilding/loading a state_dict."""
        target_roles = self._target_states[slot]
        with torch.no_grad():
            for position, source in source_models.items():
                source_state = source.state_dict(keep_vars=True)
                target_state = target_roles[position]
                if source_state.keys() != target_state.keys():
                    raise ValueError(f"policy state keys changed for {position}")
                for key, target_tensor in target_state.items():
                    source_tensor = source_state[key]
                    if target_tensor.shape != source_tensor.shape:
                        raise ValueError(
                            f"policy state shape changed for {position}.{key}"
                        )
                    target_tensor.copy_(source_tensor)

    @property
    def version(self) -> int:
        """Return the most recently published policy version."""
        with self._lock:
            return int(self._version.value)

    def acquire(self, *, owner_id: int | None = None) -> PolicyLease[T]:
        """Lease the active immutable slot until :meth:`release` is called."""
        with self._lock:
            if owner_id is None:
                owner_id = next(
                    (owner for owner, slot in enumerate(self._owner_slots)
                     if slot == -1),
                    None,
                )
                if owner_id is None:
                    raise RuntimeError("no policy lease owner slots are available")
            if owner_id < 0 or owner_id >= len(self._owner_slots):
                raise ValueError(f"invalid policy lease owner {owner_id}")
            if self._owner_slots[owner_id] != -1:
                raise RuntimeError(
                    f"policy lease owner {owner_id} already holds a lease"
                )
            slot = int(self._active_slot.value)
            self._readers[slot] += 1
            version = int(self._version.value)
            generation = int(self._owner_generations[owner_id]) + 1
            self._owner_generations[owner_id] = generation
            self._owner_slots[owner_id] = slot
        return PolicyLease(
            slot=slot,
            version=version,
            model=self.models[slot],
            owner_id=owner_id,
            generation=generation,
        )

    def release(self, lease: PolicyLease[T]) -> None:
        """Release a previously acquired episode lease."""
        with self._lock:
            if lease.slot < 0 or lease.slot >= len(self.models):
                raise ValueError(f"invalid policy slot {lease.slot}")
            if lease.owner_id < 0 or lease.owner_id >= len(self._owner_slots):
                raise ValueError(f"invalid policy lease owner {lease.owner_id}")
            owner_slot = int(self._owner_slots[lease.owner_id])
            owner_generation = int(self._owner_generations[lease.owner_id])
            if owner_slot != lease.slot or owner_generation != lease.generation:
                raise RuntimeError(
                    f"policy lease owner {lease.owner_id} does not hold "
                    f"slot {lease.slot} generation {lease.generation}"
                )
            if self._readers[lease.slot] <= 0:
                raise RuntimeError(f"policy slot {lease.slot} reader count is corrupt")
            self._owner_slots[lease.owner_id] = -1
            self._readers[lease.slot] -= 1

    def recover_owner(self, owner_id: int) -> bool:
        """Reclaim a lease held by an actor that the parent has reaped."""
        with self._lock:
            if owner_id < 0 or owner_id >= len(self._owner_slots):
                raise ValueError(f"invalid policy lease owner {owner_id}")
            slot = int(self._owner_slots[owner_id])
            if slot == -1:
                return False
            if self._readers[slot] <= 0:
                raise RuntimeError(f"policy slot {slot} reader count is corrupt")
            self._owner_slots[owner_id] = -1
            self._readers[slot] -= 1
            return True

    def initialize(self, source_models: dict[str, torch.nn.Module]) -> None:
        """Copy the initial learner state into every slot before actors start."""
        with self._lock:
            for slot in range(len(self.models)):
                self._copy_state(slot, source_models)

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
                 if slot != active and self._readers[slot] == 0
                 and self._writers[slot] == 0),
                None,
            )
            if writable is None:
                return False
            self._writers[writable] = 1

        # The expensive device/host copies happen outside the lease lock. The
        # slot is inactive and writer-reserved, so actors cannot observe it.
        try:
            self._copy_state(writable, source_models)
        except BaseException:
            with self._lock:
                self._writers[writable] = 0
            raise

        with self._lock:
            if version <= self._version.value:
                self._writers[writable] = 0
                raise ValueError(
                    f"policy version {version} was superseded during publication"
                )
            self._active_slot.value = writable
            self._version.value = version
            self._writers[writable] = 0
        return True

    def reader_counts(self) -> tuple[int, ...]:
        """Return per-slot lease counts for diagnostics and tests."""
        with self._lock:
            return tuple(int(v) for v in self._readers)
