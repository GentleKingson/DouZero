"""Atomic checkpoint registration and conservative league retention."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .manifest import LeagueManifest, PolicyEntry


@dataclass(frozen=True)
class SnapshotRetention:
    keep_recent: int = 5
    milestone_interval: int = 0
    keep_top_rated: int = 3

    def __post_init__(self) -> None:
        for name in ("keep_recent", "milestone_interval", "keep_top_rated"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")


class SnapshotManager:
    """Write every role checkpoint completely before registering its policy."""

    def __init__(
        self,
        manifest_path: str,
        *,
        retention: SnapshotRetention | None = None,
        interval_steps: int = 0,
    ) -> None:
        if (
            isinstance(interval_steps, bool)
            or not isinstance(interval_steps, int)
            or interval_steps < 0
        ):
            raise ValueError("interval_steps must be a non-negative int")
        self.manifest_path = Path(manifest_path)
        self.retention = retention or SnapshotRetention()
        self.interval_steps = interval_steps

    def should_snapshot(self, current_step: int, last_snapshot_step: int) -> bool:
        """Return whether the configured periodic snapshot boundary was crossed."""

        if current_step < 0 or last_snapshot_step < 0:
            raise ValueError("snapshot steps must be non-negative")
        return (
            self.interval_steps > 0
            and current_step > last_snapshot_step
            and current_step // self.interval_steps
            > last_snapshot_step // self.interval_steps
        )

    def load(self) -> LeagueManifest:
        if not self.manifest_path.exists():
            return LeagueManifest()
        return LeagueManifest.load(self.manifest_path)

    def write_and_register(
        self,
        entry: PolicyEntry,
        writers_by_role: Mapping[str, Callable[[str], None]],
        *,
        make_primary: bool = False,
    ) -> LeagueManifest:
        """Atomically publish checkpoints, then atomically publish the manifest.

        Each writer receives a temporary path in the final checkpoint folder.
        A writer must return only after the complete file is durable enough for
        its own format. Registration happens after every final ``os.replace``.
        """

        if set(writers_by_role) != set(entry.checkpoint_paths_by_role):
            raise ValueError("checkpoint writers must match the entry role paths")
        temp_paths: list[Path] = []
        try:
            for role, final_name in entry.checkpoint_paths_by_role.items():
                final_path = Path(final_name)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
                temp_paths.append(temp_path)
                writers_by_role[role](str(temp_path))
                if not temp_path.is_file() or temp_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"snapshot writer for {role} did not create a complete file"
                    )
                os.replace(temp_path, final_path)
            return self.register_complete(entry, make_primary=make_primary)
        finally:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def register_complete(
        self, entry: PolicyEntry, *, make_primary: bool = False
    ) -> LeagueManifest:
        missing = [
            path for path in entry.checkpoint_paths_by_role.values()
            if not Path(path).is_file() or Path(path).stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                f"refusing to register incomplete policy {entry.policy_id!r}: {missing}"
            )
        manifest = self.load().upsert(entry, make_primary=make_primary)
        manifest.save(self.manifest_path)
        return self.apply_retention(manifest)

    def apply_retention(self, manifest: LeagueManifest | None = None) -> LeagueManifest:
        manifest = manifest or self.load()
        policies = list(manifest.policies)
        protected = {manifest.primary_policy_id}
        protected.update(
            policy.policy_id
            for policy in policies
            if any(tag in policy.tags for tag in ("pinned", "user", "milestone"))
        )
        if self.retention.milestone_interval > 0:
            protected.update(
                policy.policy_id for policy in policies
                if policy.created_step % self.retention.milestone_interval == 0
            )
        recent = sorted(policies, key=lambda p: p.created_step, reverse=True)
        protected.update(p.policy_id for p in recent[: self.retention.keep_recent])
        rated = sorted(policies, key=lambda p: p.rating, reverse=True)
        protected.update(p.policy_id for p in rated[: self.retention.keep_top_rated])
        removable = {
            policy.policy_id for policy in policies
            if policy.policy_id not in protected and "builtin" not in policy.tags
        }
        if not removable:
            return manifest
        # Only delete files owned by removed manifest entries. User/pinned and
        # primary policies are protected above; shared paths are protected too.
        kept_paths = {
            Path(path)
            for policy in policies if policy.policy_id not in removable
            for path in policy.checkpoint_paths_by_role.values()
        }
        for policy in policies:
            if policy.policy_id not in removable:
                continue
            for raw_path in policy.checkpoint_paths_by_role.values():
                path = Path(raw_path)
                if path not in kept_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
        updated = manifest.without(removable)
        updated.save(self.manifest_path)
        return updated
