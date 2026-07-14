"""Atomic managed snapshot bundles and recoverable league retention."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from douzero.observation.seats import ALL_ROLES

from .manifest import LeagueManifest, PendingDelete, PolicyEntry

_MANAGED_TAG = "managed-snapshot"


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
    """Publish immutable three-role bundles below one managed root."""

    def __init__(
        self,
        manifest_path: str,
        *,
        snapshot_root: str,
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
        self.snapshot_root = Path(snapshot_root).absolute()
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

    def checkpoint_paths(self, policy_id: str) -> dict[str, str]:
        """Return the only checkpoint layout this manager will own."""

        # PolicyEntry performs the full identifier validation.
        return {
            role: str(self.snapshot_root / "policies" / policy_id / f"{role}.ckpt")
            for role in ALL_ROLES
        }

    def _load_raw(self) -> LeagueManifest:
        if not self.manifest_path.exists():
            return LeagueManifest()
        manifest = LeagueManifest.load(self.manifest_path)
        self._validate_managed_entries(manifest)
        return manifest

    def load(self) -> LeagueManifest:
        """Load league state and replay any interrupted pending cleanup."""

        return self._cleanup_pending_deletes(self._load_raw())

    def write_and_register(
        self,
        entry: PolicyEntry,
        writers_by_role: Mapping[str, Callable[[str], None]],
        *,
        make_primary: bool = False,
    ) -> LeagueManifest:
        """Stage a full bundle, then atomically publish its directory.

        Policy IDs are immutable: an existing bundle directory is never
        overwritten. A crash before the directory rename leaves only staging
        data; a crash after it can leave a complete unregistered orphan, never
        a partially replaced registered policy.
        """

        expected_paths = self.checkpoint_paths(entry.policy_id)
        if dict(entry.checkpoint_paths_by_role) != expected_paths:
            raise ValueError(
                f"policy {entry.policy_id!r} paths must use the manager layout"
            )
        if set(writers_by_role) != set(ALL_ROLES):
            raise ValueError("checkpoint writers must cover all three roles")

        final_dir = self.snapshot_root / "policies" / entry.policy_id
        self._prepare_managed_directories()
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(
                f"immutable snapshot policy_id {entry.policy_id!r} already exists"
            )

        staging_root = self.snapshot_root / ".staging"
        stage_dir = Path(tempfile.mkdtemp(prefix=f".{entry.policy_id}.", dir=staging_root))
        published = False
        try:
            for role in ALL_ROLES:
                stage_path = stage_dir / f"{role}.ckpt"
                writers_by_role[role](str(stage_path))
                self._validate_regular_file(stage_path, role=role)
                with open(stage_path, "rb") as handle:
                    os.fsync(handle.fileno())
            self._fsync_directory(stage_dir)
            os.replace(stage_dir, final_dir)
            published = True
            self._fsync_directory(final_dir.parent)
            return self.register_complete(entry, make_primary=make_primary)
        finally:
            if not published:
                shutil.rmtree(stage_dir, ignore_errors=True)

    def register_complete(
        self, entry: PolicyEntry, *, make_primary: bool = False
    ) -> LeagueManifest:
        managed = entry
        if _MANAGED_TAG not in entry.tags:
            managed = replace(
                entry,
                tags=tuple(dict.fromkeys(entry.tags + (_MANAGED_TAG,))),
            )
        self._validate_managed_entry(managed, require_files=True)
        manifest = self.load().upsert(managed, make_primary=make_primary)
        manifest.save(self.manifest_path)
        return self.apply_retention(manifest)

    def apply_retention(self, manifest: LeagueManifest | None = None) -> LeagueManifest:
        manifest = manifest or self.load()
        self._validate_managed_entries(manifest)
        manifest = self._cleanup_pending_deletes(manifest)
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
            if policy.policy_id not in protected and _MANAGED_TAG in policy.tags
        }
        if not removable:
            return manifest

        # Publish logical removal first. A crash can leave only tombstoned
        # files, never an active manifest entry pointing at a deleted bundle.
        updated = manifest.mark_pending_delete(removable)
        updated.save(self.manifest_path)
        return self._cleanup_pending_deletes(updated)

    def _cleanup_pending_deletes(self, manifest: LeagueManifest) -> LeagueManifest:
        if not manifest.pending_deletes:
            return manifest
        for item in manifest.pending_deletes:
            for role, raw_path in item.checkpoint_paths_by_role.items():
                self._delete_managed_checkpoint(item, role, Path(raw_path))
            policy_dir = self.snapshot_root / "policies" / item.policy_id
            try:
                policy_dir.rmdir()
            except FileNotFoundError:
                pass
        cleaned = manifest.clear_pending_deletes()
        cleaned.save(self.manifest_path)
        return cleaned

    def _delete_managed_checkpoint(
        self, item: PendingDelete, role: str, path: Path
    ) -> None:
        self._validate_managed_path(item.policy_id, role, path)
        if path.is_symlink():
            raise ValueError(f"refusing to delete symlink checkpoint {path}")
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISREG(mode):
            raise ValueError(f"refusing to delete non-regular checkpoint {path}")
        path.unlink()

    def _validate_managed_entries(self, manifest: LeagueManifest) -> None:
        for policy in manifest.policies:
            if _MANAGED_TAG in policy.tags:
                self._validate_managed_entry(policy, require_files=False)
        for item in manifest.pending_deletes:
            for role, raw_path in item.checkpoint_paths_by_role.items():
                self._validate_managed_path(item.policy_id, role, Path(raw_path))

    def _validate_managed_entry(
        self, entry: PolicyEntry, *, require_files: bool
    ) -> None:
        if set(entry.checkpoint_paths_by_role) != set(ALL_ROLES):
            raise ValueError("managed snapshot must contain all three role checkpoints")
        for role, raw_path in entry.checkpoint_paths_by_role.items():
            path = Path(raw_path)
            self._validate_managed_path(entry.policy_id, role, path)
            if path.is_symlink():
                raise ValueError(f"managed checkpoint cannot be a symlink: {path}")
            if require_files:
                self._validate_regular_file(path, role=role)

    def _validate_managed_path(self, policy_id: str, role: str, path: Path) -> None:
        expected = Path(self.checkpoint_paths(policy_id)[role])
        if path.absolute() != expected:
            raise ValueError(
                f"checkpoint for policy {policy_id!r} role {role!r} is outside "
                "the manager-owned layout"
            )
        root = self.snapshot_root.resolve(strict=False)
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"refusing to manage checkpoint outside snapshot_root: {path}"
            ) from exc
        self._assert_no_symlink_components(path)

    def _prepare_managed_directories(self) -> None:
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self._assert_no_symlink_components(self.snapshot_root)
        for path in (
            self.snapshot_root / "policies",
            self.snapshot_root / ".staging",
        ):
            path.mkdir(exist_ok=True)
            self._assert_no_symlink_components(path)
            if not path.is_dir():
                raise ValueError(f"managed snapshot path is not a directory: {path}")

    def _assert_no_symlink_components(self, path: Path) -> None:
        absolute = path.absolute()
        root = self.snapshot_root.absolute()
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path is outside snapshot_root: {path}") from exc
        if root.is_symlink():
            raise ValueError(f"snapshot_root cannot be a symlink: {root}")
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"managed checkpoint path traverses symlink: {current}")

    @staticmethod
    def _validate_regular_file(path: Path, *, role: str) -> None:
        if path.is_symlink():
            raise RuntimeError(f"snapshot writer for {role} created a symlink")
        try:
            metadata = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"snapshot writer for {role} did not create a complete file"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
            raise RuntimeError(
                f"snapshot writer for {role} did not create a complete file"
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
