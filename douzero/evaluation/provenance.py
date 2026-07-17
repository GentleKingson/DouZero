"""Fail-closed source and result provenance for formal evaluations.

This module deliberately does not use :func:`douzero._version.git_sha`.  That
helper permits an environment override for source-less, descriptive builds;
formal evaluation instead requires a real, clean Git checkout and a detached
GitHub artifact attestation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


CANONICAL_JSON_VERSION = "douzero-canonical-json-v1"
RESULT_INTEGRITY_VERSION = "douzero-evaluation-result-integrity-v1"
TRACKED_TREE_DIGEST_VERSION = "douzero-tracked-tree-sha256-v1"
RESULT_PAYLOAD_FIELDS = (
    "protocol",
    "ablation",
    "scenario",
    "metrics",
    "games",
    "runtime_identity",
)
RESULT_INTEGRITY_FIELD = "result_integrity"

_HEX_GIT_OID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_VERIFIED_RESULT_SEAL = object()


class ProvenanceError(ValueError):
    """Raised when formal provenance cannot be established exactly."""


@dataclass(frozen=True)
class GitCheckoutIdentity:
    """Identity of the actual checkout used by an evaluator."""

    head_sha: str
    head_tree_oid: str
    tracked_tree_sha256: str
    source_ref: str | None
    tracked_file_count: int
    clean: bool = True

    def to_runtime_fields(self) -> dict[str, Any]:
        """Return stable fields suitable for inclusion in runtime_identity."""

        return {
            "source_git_sha": self.head_sha,
            "source_git_tree_oid": self.head_tree_oid,
            "source_tracked_tree_sha256": self.tracked_tree_sha256,
            "source_git_ref": self.source_ref,
            "source_worktree_clean": self.clean,
            "source_tracked_file_count": self.tracked_file_count,
        }


@dataclass(frozen=True)
class AttestationPolicy:
    """Exact GitHub Actions identity expected to attest a result artifact."""

    repository: str
    signer_workflow: str
    signer_digest: str
    source_digest: str
    source_ref: str
    artifact_sha256: str
    trusted_root_path: str | os.PathLike[str] | None = None
    trusted_root_sha256: str | None = None

    def __post_init__(self) -> None:
        if not _REPOSITORY.fullmatch(self.repository):
            raise ProvenanceError("repository must be an exact owner/name pair")
        if (
            not self.signer_workflow
            or any(char.isspace() for char in self.signer_workflow)
            or ".github/workflows/" not in self.signer_workflow
        ):
            raise ProvenanceError("signer_workflow must name an exact workflow path")
        if not _HEX_GIT_OID.fullmatch(self.signer_digest):
            raise ProvenanceError("signer_digest must be a full lowercase Git OID")
        if not _HEX_GIT_OID.fullmatch(self.source_digest):
            raise ProvenanceError("source_digest must be a full lowercase Git OID")
        if not self.source_ref.startswith("refs/") or any(
            char.isspace() for char in self.source_ref
        ):
            raise ProvenanceError("source_ref must be an exact fully qualified Git ref")
        if not _HEX_SHA256.fullmatch(self.artifact_sha256):
            raise ProvenanceError("artifact_sha256 must be a lowercase SHA-256 digest")
        if (self.trusted_root_path is None) != (self.trusted_root_sha256 is None):
            raise ProvenanceError(
                "trusted_root_path and trusted_root_sha256 must be supplied together"
            )
        if self.trusted_root_path is not None:
            try:
                trusted_root = os.fspath(self.trusted_root_path)
            except TypeError as exc:
                raise ProvenanceError("trusted_root_path must be path-like") from exc
            if not trusted_root or not _HEX_SHA256.fullmatch(
                self.trusted_root_sha256 or ""
            ):
                raise ProvenanceError(
                    "trusted root requires a path and lowercase SHA-256 digest"
                )


@dataclass(frozen=True)
class AttestedEvaluationInput:
    """Paths and immutable policy that a collator must verify itself."""

    result_path: str | os.PathLike[str]
    bundle_path: str | os.PathLike[str]
    policy: AttestationPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.policy, AttestationPolicy):
            raise ProvenanceError("attested input requires an AttestationPolicy")
        try:
            result_path = os.fspath(self.result_path)
            bundle_path = os.fspath(self.bundle_path)
        except TypeError as exc:
            raise ProvenanceError("attested input paths must be path-like") from exc
        if not result_path or not bundle_path:
            raise ProvenanceError("attested input paths must be non-empty")


@dataclass(frozen=True)
class VerifiedEvaluationResult:
    """A result whose bytes, integrity envelope, source, and signer all verify."""

    result: dict[str, Any]
    artifact_sha256: str
    result_digest: str
    source_git_sha: str
    repository: str
    source_ref: str
    signer_workflow: str
    signer_digest: str
    workflow_run_url: str
    runner_environment: str
    attestation_verifications: tuple[dict[str, Any], ...]
    _verification_seal: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._verification_seal is not _VERIFIED_RESULT_SEAL:
            raise ProvenanceError(
                "VerifiedEvaluationResult must come from attestation verification"
            )


def _git_environment() -> dict[str, str]:
    # GIT_DIR, GIT_INDEX_FILE, and the config-injection variables can redirect
    # otherwise innocent-looking commands.  Formal identity always starts from
    # the supplied checkout path, not caller-controlled Git environment state.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["LC_ALL"] = "C"
    return environment


def _run_git(
    cwd: Path,
    *arguments: str,
    allow_failure: bool = False,
) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(cwd), *arguments],
            capture_output=True,
            timeout=15,
            check=False,
            env=_git_environment(),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError(
            "formal evaluation requires an available Git checkout"
        ) from exc
    if completed.returncode != 0:
        if allow_failure:
            return None
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(
            f"Git checkout inspection failed: {detail or arguments[0]}"
        )
    return completed.stdout


def _decode_oid(value: bytes | None, label: str) -> str:
    decoded = (value or b"").strip().decode("ascii", "strict")
    if not _HEX_GIT_OID.fullmatch(decoded):
        raise ProvenanceError(f"{label} is not a full lowercase Git OID")
    return decoded


def _tracked_entries(index: bytes) -> list[tuple[bytes, bytes, bytes]]:
    entries: list[tuple[bytes, bytes, bytes]] = []
    seen: set[bytes] = set()
    for raw_entry in index.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, path = raw_entry.split(b"\t", 1)
            mode, object_id, stage_number = header.split(b" ", 2)
        except ValueError as exc:
            raise ProvenanceError(
                "Git index contains an unparsable tracked entry"
            ) from exc
        if stage_number != b"0":
            raise ProvenanceError("Git index contains an unresolved merge stage")
        if mode not in {b"100644", b"100755", b"120000"}:
            if mode == b"160000":
                raise ProvenanceError(
                    "formal tracked-tree hashing does not accept Git submodules"
                )
            raise ProvenanceError(f"unsupported tracked file mode {mode!r}")
        if not path or path in seen or path.startswith(b"/") or b"\0" in path:
            raise ProvenanceError("Git index contains an invalid or duplicate path")
        if any(component in {b"", b".", b".."} for component in path.split(b"/")):
            raise ProvenanceError("Git index contains a non-canonical path")
        if not _HEX_GIT_OID.fullmatch(object_id.decode("ascii", "strict")):
            raise ProvenanceError("Git index contains an invalid object ID")
        seen.add(path)
        entries.append((path, mode, object_id))
    entries.sort(key=lambda item: item[0])
    return entries


def _update_tracked_digest(
    digest: Any,
    relative_path: bytes,
    mode: bytes,
    content: bytes,
    *,
    present: bool = True,
) -> None:
    digest.update(len(relative_path).to_bytes(8, "big"))
    digest.update(relative_path)
    digest.update(mode)
    digest.update(b"P" if present else b"M")
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def _working_tree_digest(
    root: Path, index: bytes, *, allow_missing: bool
) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(TRACKED_TREE_DIGEST_VERSION.encode("ascii") + b"\0")
    root_bytes = os.fsencode(root)
    entries = _tracked_entries(index)
    for relative_path, mode, _object_id in entries:
        absolute_path = os.path.join(root_bytes, relative_path)
        try:
            metadata = os.lstat(absolute_path)
            if mode == b"120000":
                if not stat.S_ISLNK(metadata.st_mode):
                    raise ProvenanceError(
                        "tracked symlink has the wrong working-tree type"
                    )
                content = os.readlink(absolute_path)
                if isinstance(content, str):
                    content = os.fsencode(content)
            else:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ProvenanceError(
                        "tracked file has the wrong working-tree type"
                    )
                executable = bool(metadata.st_mode & 0o111)
                if executable != (mode == b"100755"):
                    raise ProvenanceError(
                        "tracked file executable mode does not match Git"
                    )
                with open(absolute_path, "rb") as handle:
                    content = handle.read()
        except FileNotFoundError:
            if not allow_missing:
                raise ProvenanceError("tracked file is missing from the working tree")
            _update_tracked_digest(
                digest, relative_path, mode, b"", present=False
            )
            continue
        except ProvenanceError:
            raise
        except (OSError, ValueError) as exc:
            raise ProvenanceError("tracked file could not be hashed") from exc
        _update_tracked_digest(digest, relative_path, mode, content)
    return digest.hexdigest(), len(entries)


def _head_tree_entries(root: Path) -> list[tuple[bytes, bytes, bytes]]:
    raw_tree = _run_git(root, "ls-tree", "-r", "--full-tree", "-z", "HEAD") or b""
    entries: list[tuple[bytes, bytes, bytes]] = []
    seen: set[bytes] = set()
    for raw_entry in raw_tree.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = header.split(b" ", 2)
        except ValueError as exc:
            raise ProvenanceError("Git HEAD tree contains an unparsable entry") from exc
        if object_type != b"blob" or mode not in {b"100644", b"100755", b"120000"}:
            if object_type == b"commit" or mode == b"160000":
                raise ProvenanceError(
                    "formal tracked-tree hashing does not accept Git submodules"
                )
            raise ProvenanceError("Git HEAD tree contains an unsupported object")
        if path in seen:
            raise ProvenanceError("Git HEAD tree contains a duplicate path")
        if not _HEX_GIT_OID.fullmatch(object_id.decode("ascii", "strict")):
            raise ProvenanceError("Git HEAD tree contains an invalid object ID")
        seen.add(path)
        entries.append((path, mode, object_id))
    entries.sort(key=lambda item: item[0])
    return entries


def _read_git_blobs(root: Path, object_ids: list[bytes]) -> list[bytes]:
    if not object_ids:
        return []
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(root), "cat-file", "--batch"],
            input=b"".join(object_id + b"\n" for object_id in object_ids),
            capture_output=True,
            timeout=30,
            check=False,
            env=_git_environment(),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError("Git HEAD blobs could not be inspected") from exc
    if completed.returncode != 0:
        raise ProvenanceError("Git HEAD blobs could not be inspected")
    output = completed.stdout
    offset = 0
    blobs: list[bytes] = []
    for expected_oid in object_ids:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise ProvenanceError("git cat-file returned a truncated header")
        try:
            actual_oid, object_type, raw_size = output[offset:header_end].split(b" ", 2)
            size = int(raw_size)
        except (ValueError, TypeError) as exc:
            raise ProvenanceError("git cat-file returned a malformed header") from exc
        if actual_oid != expected_oid or object_type != b"blob" or size < 0:
            raise ProvenanceError("git cat-file returned an unexpected object")
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(output) or output[content_end : content_end + 1] != b"\n":
            raise ProvenanceError("git cat-file returned truncated blob content")
        blobs.append(output[content_start:content_end])
        offset = content_end + 1
    if offset != len(output):
        raise ProvenanceError("git cat-file returned unexpected trailing data")
    return blobs


def _committed_tree_digest(
    root: Path,
) -> tuple[str, int, list[tuple[bytes, bytes, bytes]]]:
    entries = _head_tree_entries(root)
    blobs = _read_git_blobs(root, [entry[2] for entry in entries])
    digest = hashlib.sha256()
    digest.update(TRACKED_TREE_DIGEST_VERSION.encode("ascii") + b"\0")
    for (relative_path, mode, _object_id), content in zip(entries, blobs):
        _update_tracked_digest(digest, relative_path, mode, content)
    return digest.hexdigest(), len(entries), entries


def inspect_git_checkout(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    require_clean: bool = True,
) -> GitCheckoutIdentity:
    """Inspect the real checkout without accepting an environment SHA.

    With ``require_clean=False``, dirty state is retained as ``clean=False`` and
    the digest describes the current tracked working-tree bytes.  Formal callers
    use the default and fail closed.  Unmerged entries, unsupported file types,
    source-less installs, and concurrent mutations are always rejected.
    """

    if not isinstance(require_clean, bool):
        raise ProvenanceError("require_clean must be a boolean")

    candidate = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    root_raw = _run_git(candidate, "rev-parse", "--show-toplevel")
    try:
        root = Path(
            (root_raw or b"").decode("utf-8", "surrogateescape").strip()
        ).resolve()
    except (OSError, ValueError) as exc:
        raise ProvenanceError("Git reported an invalid checkout root") from exc
    if not root.is_dir():
        raise ProvenanceError("formal evaluation requires a real Git checkout root")

    def sample() -> tuple[str, str, bytes, bytes]:
        head = _decode_oid(
            _run_git(root, "rev-parse", "--verify", "HEAD^{commit}"), "Git HEAD"
        )
        tree = _decode_oid(
            _run_git(root, "rev-parse", "--verify", "HEAD^{tree}"), "Git HEAD tree"
        )
        status = _run_git(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        ) or b""
        index = _run_git(root, "ls-files", "--stage", "-z") or b""
        return head, tree, status, index

    head, tree, status, index = sample()
    if status and require_clean:
        raise ProvenanceError("formal evaluation requires a clean Git working tree")
    tracked_digest, tracked_count = _working_tree_digest(
        root, index, allow_missing=not require_clean
    )
    committed_digest, committed_count, committed_entries = _committed_tree_digest(root)
    tree_matches_head = (
        tracked_digest == committed_digest
        and tracked_count == committed_count
        and _tracked_entries(index) == committed_entries
    )
    if require_clean and not tree_matches_head:
        raise ProvenanceError("working tree bytes do not match the Git HEAD tree")
    second_head, second_tree, second_status, second_index = sample()
    if (head, tree, status, index) != (
        second_head,
        second_tree,
        second_status,
        second_index,
    ):
        raise ProvenanceError("Git checkout changed while provenance was inspected")
    second_digest, second_count = _working_tree_digest(
        root, second_index, allow_missing=not require_clean
    )
    if (tracked_digest, tracked_count) != (second_digest, second_count):
        raise ProvenanceError("Git checkout changed while provenance was inspected")

    ref_raw = _run_git(root, "symbolic-ref", "-q", "HEAD", allow_failure=True)
    source_ref = None
    if ref_raw is not None:
        source_ref = ref_raw.strip().decode("utf-8", "strict")
        if not source_ref.startswith("refs/") or any(
            char.isspace() for char in source_ref
        ):
            raise ProvenanceError("Git checkout has an invalid symbolic ref")
    return GitCheckoutIdentity(
        head_sha=head,
        head_tree_oid=tree,
        tracked_tree_sha256=tracked_digest,
        source_ref=source_ref,
        tracked_file_count=tracked_count,
        clean=not status and tree_matches_head,
    )


def inspect_formal_git_checkout(
    repo_root: str | os.PathLike[str] | None = None,
) -> GitCheckoutIdentity:
    """Inspect a real Git checkout and reject every dirty state."""

    return inspect_git_checkout(repo_root, require_clean=True)


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProvenanceError(f"non-finite JSON number at {path}")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(f"non-string JSON object key at {path}")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ProvenanceError(f"unsupported canonical JSON value at {path}")


def _result_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ProvenanceError("evaluation result must be an object")
    expected = set(RESULT_PAYLOAD_FIELDS)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProvenanceError(
            f"unsigned result field mismatch; missing={missing}, extra={extra}"
        )
    return {
        field: _normalize_json(result[field], path=f"$.{field}")
        for field in RESULT_PAYLOAD_FIELDS
    }


def canonical_result_json(result: Mapping[str, Any]) -> bytes:
    """Return the versioned canonical bytes covered by the result digest."""

    envelope = {
        "canonicalization": CANONICAL_JSON_VERSION,
        "schema_version": RESULT_INTEGRITY_VERSION,
        "result": _result_payload(result),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_result_digest(result: Mapping[str, Any]) -> str:
    """Hash every and only the six allowed evaluation-result sections."""

    return hashlib.sha256(canonical_result_json(result)).hexdigest()


def attach_result_integrity(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-normalized result with a strict integrity envelope."""

    payload = _result_payload(result)
    result_digest = compute_result_digest(payload)
    payload[RESULT_INTEGRITY_FIELD] = {
        "schema_version": RESULT_INTEGRITY_VERSION,
        "canonicalization": CANONICAL_JSON_VERSION,
        "digest_algorithm": "sha256",
        "result_digest": result_digest,
    }
    return payload


def verify_result_integrity(result: Mapping[str, Any]) -> str:
    """Verify a strict integrity envelope and return its result SHA-256."""

    if not isinstance(result, Mapping):
        raise ProvenanceError("evaluation result must be an object")
    expected_top_level = {*RESULT_PAYLOAD_FIELDS, RESULT_INTEGRITY_FIELD}
    if set(result) != expected_top_level:
        raise ProvenanceError("integrity-protected result has missing or extra fields")
    integrity = result.get(RESULT_INTEGRITY_FIELD)
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "schema_version",
        "canonicalization",
        "digest_algorithm",
        "result_digest",
    }:
        raise ProvenanceError("result integrity envelope is malformed")
    if integrity.get("schema_version") != RESULT_INTEGRITY_VERSION:
        raise ProvenanceError("result integrity schema version mismatch")
    if integrity.get("canonicalization") != CANONICAL_JSON_VERSION:
        raise ProvenanceError("result canonicalization version mismatch")
    if integrity.get("digest_algorithm") != "sha256":
        raise ProvenanceError("result digest algorithm must be sha256")
    claimed = integrity.get("result_digest")
    if not isinstance(claimed, str) or not _HEX_SHA256.fullmatch(claimed):
        raise ProvenanceError("result integrity digest is malformed")
    payload = {field: result[field] for field in RESULT_PAYLOAD_FIELDS}
    actual = compute_result_digest(payload)
    if not hmac.compare_digest(claimed, actual):
        raise ProvenanceError("evaluation result integrity check failed")
    return actual


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_strict_json(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProvenanceError(f"non-finite JSON number in {label}: {value}")
            ),
        )
    except ProvenanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{label} is not strict UTF-8 JSON") from exc


def _attested_subject_matches(entry: Mapping[str, Any], digest: str) -> bool:
    verification = entry.get("verificationResult")
    if not isinstance(verification, Mapping):
        return False
    statement = verification.get("statement")
    if not isinstance(statement, Mapping):
        return False
    subjects = statement.get("subject")
    if not isinstance(subjects, list):
        return False
    for subject in subjects:
        if not isinstance(subject, Mapping):
            continue
        digests = subject.get("digest")
        if isinstance(digests, Mapping) and digests.get("sha256") == digest:
            return True
    return False


def _verified_github_run(
    entry: Mapping[str, Any], policy: AttestationPolicy
) -> tuple[str, str] | None:
    """Return immutable run metadata from the verified certificate."""

    verification = entry.get("verificationResult")
    signature = verification.get("signature") if isinstance(verification, Mapping) else None
    certificate = signature.get("certificate") if isinstance(signature, Mapping) else None
    if not isinstance(certificate, Mapping):
        return None
    run_url = certificate.get("runInvocationURI")
    runner_environment = certificate.get("runnerEnvironment")
    expected_prefix = f"https://github.com/{policy.repository}/actions/runs/"
    if (
        certificate.get("issuer") != "https://token.actions.githubusercontent.com"
        or certificate.get("githubWorkflowRepository") != policy.repository
        or certificate.get("buildSignerDigest") != policy.signer_digest
        or certificate.get("sourceRepositoryDigest") != policy.source_digest
        or certificate.get("sourceRepositoryRef") != policy.source_ref
        or not isinstance(run_url, str)
        or not run_url.startswith(expected_prefix)
        or not re.fullmatch(
            re.escape(expected_prefix) + r"[1-9][0-9]*/attempts/[1-9][0-9]*",
            run_url,
        )
        or runner_environment not in {"github-hosted", "self-hosted"}
    ):
        return None
    return run_url, runner_environment


def verify_github_attested_result(
    result_path: str | os.PathLike[str],
    bundle_path: str | os.PathLike[str],
    policy: AttestationPolicy,
    *,
    gh_executable: str = "gh",
    timeout_seconds: float = 60.0,
) -> VerifiedEvaluationResult:
    """Verify a detached GitHub attestation and the complete result payload.

    The artifact is copied to a private snapshot before invoking ``gh`` so the
    bytes parsed here are exactly the bytes whose subject digest is verified.
    """

    artifact = Path(result_path)
    bundle = Path(bundle_path)
    try:
        artifact_bytes = artifact.read_bytes()
    except OSError as exc:
        raise ProvenanceError("evaluation result artifact cannot be read") from exc
    if not bundle.is_file():
        raise ProvenanceError("detached attestation bundle does not exist")
    trusted_root_bytes = None
    if policy.trusted_root_path is not None:
        trusted_root = Path(policy.trusted_root_path)
        try:
            if trusted_root.is_symlink() or not trusted_root.is_file():
                raise ProvenanceError(
                    "attestation trusted root must be a regular non-symlink file"
                )
            trusted_root_bytes = trusted_root.read_bytes()
        except ProvenanceError:
            raise
        except OSError as exc:
            raise ProvenanceError("attestation trusted root cannot be read") from exc
        actual_root_sha = hashlib.sha256(trusted_root_bytes).hexdigest()
        if not hmac.compare_digest(
            actual_root_sha, policy.trusted_root_sha256 or ""
        ):
            raise ProvenanceError("attestation trusted root SHA-256 mismatch")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if not hmac.compare_digest(artifact_sha256, policy.artifact_sha256):
        raise ProvenanceError("evaluation artifact SHA-256 does not match policy")

    decoded = _load_strict_json(artifact_bytes, label="evaluation result artifact")
    if not isinstance(decoded, Mapping):
        raise ProvenanceError("evaluation result artifact must contain an object")
    result_digest = verify_result_integrity(decoded)
    runtime = decoded.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        raise ProvenanceError("evaluation runtime identity is missing")
    source_git_sha = runtime.get("source_git_sha")
    if source_git_sha != policy.source_digest:
        raise ProvenanceError(
            "runtime source_git_sha does not match the attested source digest"
        )

    if not gh_executable or timeout_seconds <= 0:
        raise ProvenanceError("GitHub attestation verifier configuration is invalid")
    with tempfile.TemporaryDirectory(prefix="douzero-attestation-") as directory:
        snapshot = Path(directory, "evaluation-result.json")
        snapshot.write_bytes(artifact_bytes)
        trusted_root_snapshot = None
        if trusted_root_bytes is not None:
            trusted_root_snapshot = Path(directory, "trusted-root.jsonl")
            trusted_root_snapshot.write_bytes(trusted_root_bytes)
        command = [
            gh_executable,
            "attestation",
            "verify",
            os.fspath(snapshot),
            "--bundle",
            os.fspath(bundle),
        ]
        if trusted_root_snapshot is not None:
            command.extend([
                "--custom-trusted-root",
                os.fspath(trusted_root_snapshot),
            ])
        command.extend([
            "--repo",
            policy.repository,
            "--signer-workflow",
            policy.signer_workflow,
            "--signer-digest",
            policy.signer_digest,
            "--source-digest",
            policy.source_digest,
            "--source-ref",
            policy.source_ref,
            "--format",
            "json",
        ])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            raise ProvenanceError(
                "GitHub attestation verification could not run"
            ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise ProvenanceError(
            f"GitHub attestation verification failed: {detail or 'gh exited nonzero'}"
        )
    verification_output = _load_strict_json(
        completed.stdout.encode("utf-8"), label="gh attestation verification output"
    )
    if not isinstance(verification_output, list) or not verification_output:
        raise ProvenanceError("gh returned no verified attestations")
    verified_entries: list[dict[str, Any]] = []
    verified_runs: list[tuple[str, str]] = []
    for entry in verification_output:
        if not isinstance(entry, Mapping):
            raise ProvenanceError("gh returned a malformed attestation result")
        normalized_entry = _normalize_json(entry)
        verified_run = _verified_github_run(normalized_entry, policy)
        if (
            _attested_subject_matches(normalized_entry, artifact_sha256)
            and verified_run is not None
        ):
            verified_entries.append(normalized_entry)
            verified_runs.append(verified_run)
    if not verified_entries:
        raise ProvenanceError(
            "verified attestation does not bind the exact artifact SHA-256 "
            "and immutable GitHub run provenance"
        )

    normalized_result = _normalize_json(decoded)
    workflow_runs = {run for run, _environment in verified_runs}
    runner_environments = {
        environment for _run, environment in verified_runs
    }
    if len(workflow_runs) != 1 or len(runner_environments) != 1:
        raise ProvenanceError(
            "verified attestations disagree on workflow-run provenance"
        )
    return VerifiedEvaluationResult(
        result=normalized_result,
        artifact_sha256=artifact_sha256,
        result_digest=result_digest,
        source_git_sha=source_git_sha,
        repository=policy.repository,
        source_ref=policy.source_ref,
        signer_workflow=policy.signer_workflow,
        signer_digest=policy.signer_digest,
        workflow_run_url=next(iter(workflow_runs)),
        runner_environment=next(iter(runner_environments)),
        attestation_verifications=tuple(verified_entries),
        _verification_seal=_VERIFIED_RESULT_SEAL,
    )


__all__ = [
    "AttestedEvaluationInput",
    "AttestationPolicy",
    "CANONICAL_JSON_VERSION",
    "GitCheckoutIdentity",
    "ProvenanceError",
    "RESULT_INTEGRITY_FIELD",
    "RESULT_INTEGRITY_VERSION",
    "RESULT_PAYLOAD_FIELDS",
    "TRACKED_TREE_DIGEST_VERSION",
    "VerifiedEvaluationResult",
    "attach_result_integrity",
    "canonical_result_json",
    "compute_result_digest",
    "inspect_formal_git_checkout",
    "inspect_git_checkout",
    "verify_github_attested_result",
    "verify_result_integrity",
]
