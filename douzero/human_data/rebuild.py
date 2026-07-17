"""Deterministic, versioned canonical-dataset rebuild and atomic publication."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .identifiers import is_canonical_game_id
from .schema import (
    _dataset_version_payload_path,
    _encode_dataset_pointer,
    _canonical_dataset_path,
    _legacy_dataset_manifest_path,
    _load_verified_jsonl_snapshot,
    _write_jsonl_unlocked,
    dataset_publication_lock,
    dataset_version_root,
)

_VERSION_NAME_PATTERN = re.compile(r"^v-[0-9a-f]{32}$")


@dataclass(frozen=True)
class RebuildReport:
    input_records: int
    output_records: int
    excluded_records: int
    requested_ids: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_records": self.input_records,
            "output_records": self.output_records,
            "excluded_records": self.excluded_records,
            "requested_ids": self.requested_ids,
        }


class ConcurrentDatasetUpdateError(RuntimeError):
    """Raised when an unlocked destination change is detected pre-commit."""


class RebuildCommitUncertainError(RuntimeError):
    """The pointer switch failed and its commit state cannot be proven."""

    committed = None
    durable = None


class RebuildPostCommitError(RuntimeError):
    """A new pointer is visible, but the caller observed a later failure."""

    committed = True

    def __init__(
        self,
        message: str,
        *,
        durable: bool,
        current: bool | None = True,
    ) -> None:
        super().__init__(message)
        self.durable = durable
        self.current = current


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_private_version_root(path: Path) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(
            "dataset version root must be a regular, non-symlink directory"
        )
    if stat.S_IMODE(mode) & 0o077:
        raise ValueError(
            "dataset version root must not grant group or other permissions"
        )


def _remove_unpublished_version(payload: Path) -> None:
    manifest = _legacy_dataset_manifest_path(payload)
    try:
        payload.parent.chmod(0o700)
    except OSError:
        pass
    try:
        manifest.chmod(0o600)
    except OSError:
        pass
    try:
        payload.chmod(0o600)
    except OSError:
        pass
    manifest.unlink(missing_ok=True)
    payload.unlink(missing_ok=True)
    try:
        payload.parent.rmdir()
    except OSError:
        # A concurrent writer never shares this UUID-named directory. If an
        # unexpected artifact is present, leave it unreferenced rather than
        # deleting content that this transaction did not create.
        pass


def _destination_token(path: Path) -> tuple[int, int, str] | None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("dataset destination must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return info.st_dev, info.st_ino, digest.hexdigest()
    finally:
        os.close(descriptor)


def _destination_points_to(path: Path, pointer_bytes: bytes) -> bool:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError:
        return False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(pointer_bytes):
            return False
        return os.read(descriptor, len(pointer_bytes) + 1) == pointer_bytes
    finally:
        os.close(descriptor)


def _publication_state(
    path: Path,
    pointer_bytes: bytes,
    prior_token: tuple[int, int, str] | None,
) -> str:
    """Return ``new``, ``old``, ``other``, or ``unknown`` after replace."""

    try:
        if _destination_points_to(path, pointer_bytes):
            return "new"
        current_token = _destination_token(path)
    except BaseException:
        return "unknown"
    if current_token is not None and current_token[2] == hashlib.sha256(
        pointer_bytes
    ).hexdigest():
        return "new"
    if current_token == prior_token:
        return "old"
    return "other"


def _version_directory_names(destination: Path) -> frozenset[str]:
    version_root = dataset_version_root(destination)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = os.open(version_root, flags)
    try:
        root_mode = os.fstat(root_fd).st_mode
        if not stat.S_ISDIR(root_mode) or stat.S_IMODE(root_mode) & 0o077:
            raise ValueError("dataset version root is not a private directory")
        versions = frozenset(os.listdir(root_fd))
        if any(not _VERSION_NAME_PATTERN.fullmatch(name) for name in versions):
            raise ValueError("dataset version root contains an unknown entry")
        return versions
    finally:
        os.close(root_fd)


def _retire_version_directories(
    destination: Path,
    *,
    keep_version: str,
    retire_versions: frozenset[str],
) -> None:
    version_root = dataset_version_root(destination)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(version_root, flags)
    try:
        root_mode = os.fstat(root_fd).st_mode
        if not stat.S_ISDIR(root_mode) or stat.S_IMODE(root_mode) & 0o077:
            raise ValueError("dataset version root is not a private directory")
        for version in sorted(retire_versions):
            if version == keep_version:
                continue
            if not _VERSION_NAME_PATTERN.fullmatch(version):
                raise ValueError("dataset version root contains an unknown entry")
            version_fd = os.open(version, flags, dir_fd=root_fd)
            try:
                mode = os.fstat(version_fd).st_mode
                if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) & 0o077:
                    raise ValueError("dataset version directory is not private")
                entries = set(os.listdir(version_fd))
                allowed = {"dataset.jsonl", "dataset.jsonl.manifest.json"}
                if not entries <= allowed:
                    raise ValueError("dataset version contains an unknown artifact")
                os.fchmod(version_fd, 0o700)
                for artifact in sorted(entries):
                    info = os.stat(artifact, dir_fd=version_fd, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("dataset version artifact is not regular")
                    os.unlink(artifact, dir_fd=version_fd)
                os.fsync(version_fd)
            finally:
                os.close(version_fd)
            os.rmdir(version, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _freeze_version(payload: Path) -> None:
    manifest = _legacy_dataset_manifest_path(payload)
    payload.chmod(0o400)
    manifest.chmod(0o400)
    payload.parent.chmod(0o500)
    _fsync_file(payload)
    _fsync_file(manifest)
    _fsync_directory(payload.parent)


def rebuild_without_game_ids(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    excluded_game_ids: Iterable[str],
) -> RebuildReport:
    """Publish a complete versioned dataset while excluding complete games.

    Data and manifest are written and fsynced inside one immutable version
    directory. A single atomic pointer replacement then publishes the pair, so
    readers observe either the prior complete version or the new complete
    version. The report contains counts only; it never returns or logs game
    identifiers or record contents. ``input_path`` bootstraps a missing output;
    once the output exists, its active version is authoritative so later
    deletion requests are cumulative and cannot resurrect older input rows.

    Publication uses an advisory lock shared by all supported writers. The
    output parent must be an owner-controlled trusted directory; adversarial
    same-UID replacement of that directory is outside this API's boundary.
    """

    source = Path(input_path)
    destination = Path(output_path)
    excluded = set(excluded_game_ids)
    if not excluded:
        raise ValueError("at least one excluded game_id is required")
    if any(not is_canonical_game_id(game_id) for game_id in excluded):
        raise ValueError("every excluded game_id must be a canonical opaque ID")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source = _canonical_dataset_path(source)
    destination = _canonical_dataset_path(destination)
    with dataset_publication_lock(
        destination,
        exclusive=True,
        create=True,
        require_posix=True,
    ):
        destination_token = _destination_token(destination)
        if destination_token is None:
            if source.resolve() == destination.resolve():
                raise ValueError(
                    "input must exist when rebuilding a destination in place"
                )
            source_manifest, records = _load_verified_jsonl_snapshot(source)
        else:
            # Once an active destination exists it is the authoritative base.
            # This makes separate deletion requests cumulative and prevents an
            # older input export from resurrecting previously removed games.
            source_manifest, records = _load_verified_jsonl_snapshot(
                destination,
                acquire_lock=False,
            )
        if _destination_token(destination) != destination_token:
            raise ConcurrentDatasetUpdateError(
                "dataset destination changed while its base snapshot was loaded"
            )
        retained = [record for record in records if record.game_id not in excluded]

        version_root = dataset_version_root(destination)
        version_root.mkdir(mode=0o700, exist_ok=True)
        _require_private_version_root(version_root)
        retire_versions = _version_directory_names(destination)
        version = f"v-{uuid.uuid4().hex}"
        payload = _dataset_version_payload_path(destination, version)
        payload.parent.mkdir(mode=0o700)
        pointer_bytes = _encode_dataset_pointer(version)
        pointer_temporary: Path | None = None
        committed = False
        durable = False
        retain_version = False
        replace_exception: BaseException | None = None
        try:
            _write_jsonl_unlocked(
                retained,
                str(payload),
                config_identity={
                    "operation": "rebuild_without_game_ids",
                    "requested_ids": len(excluded),
                    "source_dataset_sha256": source_manifest["dataset_sha256"],
                },
            )
            _freeze_version(payload)
            _fsync_directory(version_root)
            # Persist the version-root entry before publishing a pointer to it.
            _fsync_directory(destination.parent)

            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".pointer.tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                pointer_temporary = Path(handle.name)
                handle.write(pointer_bytes)
                handle.flush()
                os.fsync(handle.fileno())

            if _destination_token(destination) != destination_token:
                raise ConcurrentDatasetUpdateError(
                    "dataset destination changed before pointer publication"
                )
            try:
                os.replace(pointer_temporary, destination)
                committed = True
                pointer_temporary = None
            except BaseException as exc:
                publication_state = _publication_state(
                    destination,
                    pointer_bytes,
                    destination_token,
                )
                if publication_state == "old":
                    raise
                if publication_state == "new":
                    committed = True
                    pointer_temporary = None
                    replace_exception = exc
                else:
                    # Never delete a version that an interrupted rename may
                    # already reference. The caller gets an explicit unknown
                    # state rather than a misleading pre-commit exception.
                    retain_version = True
                    raise RebuildCommitUncertainError(
                        "dataset pointer commit state is uncertain; staged "
                        "version was retained"
                    ) from exc

            try:
                _fsync_directory(destination.parent)
            except BaseException as exc:
                raise RebuildPostCommitError(
                    "new dataset is visible but pointer durability is unconfirmed",
                    durable=False,
                ) from exc
            durable = True

            try:
                publication_state = _publication_state(
                    destination,
                    pointer_bytes,
                    destination_token,
                )
                if publication_state != "new":
                    raise RebuildPostCommitError(
                        "new dataset committed but is no longer the active "
                        "pointer; old versions were retained",
                        durable=True,
                        current=(
                            False if publication_state == "other" else None
                        ),
                    )
                _retire_version_directories(
                    destination,
                    keep_version=version,
                    retire_versions=retire_versions,
                )
                _legacy_dataset_manifest_path(destination).unlink(missing_ok=True)
                _fsync_directory(destination.parent)
            except RebuildPostCommitError:
                raise
            except BaseException as exc:
                raise RebuildPostCommitError(
                    "new dataset is durable but old-version retirement failed",
                    durable=True,
                ) from exc

            if replace_exception is not None:
                raise RebuildPostCommitError(
                    "new dataset committed despite an interrupted pointer switch",
                    durable=True,
                ) from replace_exception
        finally:
            if pointer_temporary is not None:
                try:
                    pointer_temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            if not committed and not retain_version:
                _remove_unpublished_version(payload)
            elif not durable:
                # The old version is retained because a crash may still roll
                # the directory entry back when parent fsync did not complete.
                pass

        return RebuildReport(
            input_records=len(records),
            output_records=len(retained),
            excluded_records=len(records) - len(retained),
            requested_ids=len(excluded),
        )
