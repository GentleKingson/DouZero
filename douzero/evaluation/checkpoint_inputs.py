"""Declared checkpoint identities, verified loading, and safe snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, TypeVar


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_T = TypeVar("_T")


class CheckpointIdentityError(ValueError):
    """Raised when checkpoint bytes do not match their approved identity."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointIdentityError(f"duplicate model-matrix key: {key!r}")
        result[key] = value
    return result


def validate_sha256(value: object, *, label: str) -> str:
    """Return a strict lowercase SHA-256 digest or fail closed."""

    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CheckpointIdentityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    """Hash one regular, non-symlink checkpoint file."""

    checkpoint = os.fspath(path)
    descriptor = -1
    try:
        before_path = os.lstat(checkpoint)
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise CheckpointIdentityError(
                f"checkpoint must be a regular non-symlink file: {checkpoint}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(checkpoint, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise CheckpointIdentityError("checkpoint path changed before hashing")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CheckpointIdentityError("checkpoint changed while being hashed")
        return digest.hexdigest()
    except CheckpointIdentityError:
        raise
    except OSError as exc:
        raise CheckpointIdentityError(
            f"checkpoint could not be hashed: {checkpoint}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_checkpoint_file(
    path: str | os.PathLike[str], expected_sha256: object, *, label: str
) -> str:
    """Verify one checkpoint path against a predeclared file digest."""

    expected = validate_sha256(expected_sha256, label=f"{label} SHA-256")
    actual = checkpoint_sha256(path)
    if actual != expected:
        raise CheckpointIdentityError(
            f"{label} checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_verified_checkpoint(
    path: str | os.PathLike[str],
    expected_sha256: object,
    loader: Callable[[str], _T],
    *,
    label: str,
) -> _T:
    """Verify checkpoint bytes immediately before and after a loader call."""

    checkpoint = os.fspath(path)
    expected = validate_sha256(expected_sha256, label=f"{label} SHA-256")
    verify_checkpoint_file(checkpoint, expected, label=label)
    try:
        loaded = loader(checkpoint)
    finally:
        verify_checkpoint_file(checkpoint, expected, label=label)
    return loaded


def _bundle_nodes(matrix: Mapping[str, Any], *, kind: str) -> list[dict[str, Any]]:
    if kind not in {"evaluator", "p17", "auto"}:
        raise CheckpointIdentityError("matrix kind must be evaluator, p17, or auto")
    actual_kind = kind
    if kind == "auto":
        if "bundles" in matrix:
            actual_kind = "evaluator"
        elif "models" in matrix and "schema_version" in matrix:
            actual_kind = "p17"
        else:
            raise CheckpointIdentityError("could not determine model-matrix kind")

    nodes: list[dict[str, Any]] = []
    if actual_kind == "evaluator":
        if set(matrix) != {"bundles", "ablations"}:
            raise CheckpointIdentityError(
                "evaluator matrix must contain only bundles and ablations"
            )
        bundles = matrix.get("bundles")
        if not isinstance(bundles, Mapping):
            raise CheckpointIdentityError("evaluator matrix bundles must be an object")
        for name, raw in bundles.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise CheckpointIdentityError(
                    "evaluator bundle entries must be objects"
                )
            nodes.append(raw)
        return nodes

    models = matrix.get("models")
    if not isinstance(models, Mapping):
        raise CheckpointIdentityError("P17 matrix models must be an object")
    for model_rows in models.values():
        if not isinstance(model_rows, Mapping):
            raise CheckpointIdentityError("P17 model protocol rows must be objects")
        for row in model_rows.values():
            if not isinstance(row, Mapping):
                raise CheckpointIdentityError("P17 model rows must be objects")
            if row.get("status") != "available":
                continue
            bundle = row.get("bundle")
            if not isinstance(bundle, dict):
                raise CheckpointIdentityError(
                    "available P17 model rows require a bundle object"
                )
            nodes.append(bundle)
    return nodes


def require_explicit_matrix_checkpoint_digests(
    matrix: Mapping[str, Any], *, kind: str = "auto"
) -> None:
    """Require every actual matrix checkpoint to carry an approved digest."""

    from .scenario import bundle_from_dict

    for index, raw_bundle in enumerate(_bundle_nodes(matrix, kind=kind)):
        name = raw_bundle.get("name", f"matrix-bundle-{index}")
        try:
            bundle_from_dict(
                {**raw_bundle, "name": name},
                require_checkpoint_digests=True,
            )
        except (TypeError, ValueError, CheckpointIdentityError) as exc:
            raise CheckpointIdentityError(
                f"bundle {name!r} lacks complete predeclared checkpoint "
                f"SHA-256 identities: {exc}"
            ) from exc


def _snapshot_file(
    source: str,
    expected_sha256: str,
    destination_directory: Path,
    cache: dict[tuple[str, str], str],
) -> str:
    cache_key = (source, expected_sha256)
    if cache_key in cache:
        return cache[cache_key]
    expected = validate_sha256(expected_sha256, label=f"checkpoint {source!r}")
    destination = destination_directory / f"{expected}.checkpoint"
    if destination.exists():
        verify_checkpoint_file(destination, expected, label="checkpoint snapshot")
        resolved = str(destination.resolve())
        cache[cache_key] = resolved
        return resolved

    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    source_fd = -1
    destination_fd = -1
    snapshot_complete = False
    try:
        source_fd = os.open(source, read_flags)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise CheckpointIdentityError("checkpoint snapshot source is not regular")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        digest = hashlib.sha256()
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise CheckpointIdentityError(
                "checkpoint changed while its protected snapshot was created"
            )
        if digest.hexdigest() != expected:
            raise CheckpointIdentityError(
                f"checkpoint snapshot SHA-256 mismatch for {source!r}"
            )
        snapshot_complete = True
    except CheckpointIdentityError:
        raise
    except OSError as exc:
        raise CheckpointIdentityError(
            f"checkpoint snapshot failed for {source!r}"
        ) from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if not snapshot_complete:
            destination.unlink(missing_ok=True)
    try:
        os.chmod(destination, 0o400)
        verify_checkpoint_file(destination, expected, label="checkpoint snapshot")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    resolved = str(destination.resolve())
    cache[cache_key] = resolved
    return resolved


def snapshot_model_matrix(
    matrix: Mapping[str, Any],
    checkpoint_directory: str | os.PathLike[str],
    *,
    kind: str = "auto",
) -> dict[str, Any]:
    """Snapshot all declared checkpoints and return a path-rewritten matrix."""

    from .scenario import bundle_from_dict

    if not isinstance(matrix, Mapping):
        raise CheckpointIdentityError("model matrix must be an object")
    rewritten = copy.deepcopy(dict(matrix))
    nodes = _bundle_nodes(rewritten, kind=kind)
    destination = Path(checkpoint_directory)
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise CheckpointIdentityError(
            "checkpoint snapshot directory must be new and private"
        ) from exc
    cache: dict[tuple[str, str], str] = {}
    for index, raw_bundle in enumerate(nodes):
        name = raw_bundle.get("name", f"matrix-bundle-{index}")
        bundle = bundle_from_dict(
            {**raw_bundle, "name": name},
            require_checkpoint_digests=True,
        )
        rewritten_roles = dict(raw_bundle.get("checkpoints", {}))
        for role, source in bundle.checkpoints.items():
            rewritten_roles[role] = _snapshot_file(
                source,
                bundle.checkpoint_sha256[role],
                destination,
                cache,
            )
        raw_bundle["checkpoints"] = rewritten_roles
        if bundle.belief_checkpoint:
            raw_bundle["belief_checkpoint"] = _snapshot_file(
                bundle.belief_checkpoint,
                bundle.belief_checkpoint_sha256,
                destination,
                cache,
            )
        if bundle.bidding_checkpoint:
            raw_bundle["bidding_checkpoint"] = _snapshot_file(
                bundle.bidding_checkpoint,
                bundle.bidding_checkpoint_sha256,
                destination,
                cache,
            )
    return rewritten


def snapshot_model_matrix_file(
    matrix_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    checkpoint_directory: str | os.PathLike[str],
    *,
    kind: str = "auto",
) -> Path:
    """Read, snapshot, and atomically write a structured model matrix."""

    source = Path(matrix_path)
    output = Path(output_path)
    if output.exists():
        raise CheckpointIdentityError("snapshot model-matrix output already exists")
    try:
        matrix = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CheckpointIdentityError(
                    f"non-finite model-matrix JSON number: {value}"
                )
            ),
        )
    except CheckpointIdentityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIdentityError(
            "model matrix is not readable strict JSON"
        ) from exc
    if not isinstance(matrix, Mapping):
        raise CheckpointIdentityError("model matrix must contain an object")
    rewritten = snapshot_model_matrix(matrix, checkpoint_directory, kind=kind)
    try:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(rewritten, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        temporary.replace(output)
    except (OSError, TypeError, ValueError) as exc:
        raise CheckpointIdentityError(
            "snapshot model matrix could not be written"
        ) from exc
    return output.resolve()


__all__ = [
    "CheckpointIdentityError",
    "checkpoint_sha256",
    "load_verified_checkpoint",
    "require_explicit_matrix_checkpoint_digests",
    "snapshot_model_matrix",
    "snapshot_model_matrix_file",
    "validate_sha256",
    "verify_checkpoint_file",
]
