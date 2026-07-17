"""Validate stable PR scope claims and emit commit-bound CI evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "p17-pr-evidence-v1"
_OBJECT_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?:[0-9a-f]{24})?(?![0-9a-f])")
_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_STALE_CLAIMS = {
    "mutable final-head claim": re.compile(
        r"\b(?:exact[- ]+)?(?:current|final)(?:[- ]+pr)?[- ]+(?:head|sha)\b",
        re.IGNORECASE,
    ),
    "mutable test-count claim": re.compile(
        r"\b\d{1,3}(?:,\d{3})*\s+tests?\s+passed\b", re.IGNORECASE
    ),
    "mutable Docker-head claim": re.compile(
        r"\bcurrent[- ]head\s+docker\b", re.IGNORECASE
    ),
    "mutable final-matrix claim": re.compile(
        r"\bfinal[- ]head\s+github\s+matrix\b", re.IGNORECASE
    ),
    "unbound independent-audit claim": re.compile(
        r"\bindependent\b.{0,80}\baudit(?:s)?\b.{0,80}\bno\s+(?:remaining\s+)?p0",
        re.IGNORECASE | re.DOTALL,
    ),
}


class EvidenceContractError(ValueError):
    """The pull request makes a mutable or unsupported readiness claim."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceContractError(f"duplicate event JSON key: {key}")
        result[key] = value
    return result


def _load_event(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceContractError("GitHub event is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise EvidenceContractError("GitHub event must be a JSON object")
    return payload


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceContractError(f"{name} must be an object")
    return value


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EvidenceContractError(f"{name} must be a full Git object ID")
    return value


def _normalized_markdown(value: str) -> str:
    return re.sub(r"[*_`]", "", value)


def _validate_scope(title: str, body: str) -> None:
    normalized_title = _normalized_markdown(title)
    normalized_body = _normalized_markdown(body)
    if re.search(r"\b(?:release[- ]readiness\s+)?closure\b", normalized_title, re.I):
        raise EvidenceContractError("PR title must not claim release-readiness closure")
    if re.search(r"\b(?:release|production)[- ]ready\b", normalized_title, re.I):
        raise EvidenceContractError("PR title must not claim release readiness")
    if re.search(r"\b(?:infrastructure|scaffolding)\b", normalized_title, re.I) is None:
        raise EvidenceContractError("PR title must identify infrastructure or scaffolding scope")

    if _OBJECT_ID.search(normalized_body):
        raise EvidenceContractError(
            "PR body must not duplicate commit or artifact object IDs; use workflow evidence"
        )
    for description, pattern in _STALE_CLAIMS.items():
        if pattern.search(normalized_body):
            raise EvidenceContractError(f"PR body contains {description}")
    if re.search(r"\brelease\s+candidate\s*:\s*none\b", normalized_body, re.I) is None:
        raise EvidenceContractError("PR body must state 'Release candidate: NONE'")
    if re.search(r"\brelease\s+status\s*:\s*not\s+ready\b", normalized_body, re.I) is None:
        raise EvidenceContractError("PR body must state 'Release status: NOT READY'")


def build_evidence(
    event: dict[str, Any],
    *,
    expected_repository: str,
    expected_head_sha: str,
    merge_sha: str,
    workflow_sha: str,
    run_id: str,
    run_attempt: int,
) -> dict[str, Any]:
    repository = _require_object(event.get("repository"), "repository")
    actual_repository = repository.get("full_name")
    if actual_repository != expected_repository:
        raise EvidenceContractError(
            f"repository mismatch: {actual_repository!r} != {expected_repository!r}"
        )

    pull_request = _require_object(event.get("pull_request"), "pull_request")
    head = _require_object(pull_request.get("head"), "pull_request.head")
    head_sha = _require_sha(head.get("sha"), "pull_request.head.sha")
    expected_head_sha = _require_sha(expected_head_sha, "expected head SHA")
    if head_sha != expected_head_sha:
        raise EvidenceContractError(
            f"head SHA mismatch: event has {head_sha}, checkout expects {expected_head_sha}"
        )

    title = pull_request.get("title")
    body = pull_request.get("body")
    number = pull_request.get("number")
    if not isinstance(title, str) or not title.strip():
        raise EvidenceContractError("pull request title must be non-empty text")
    if body is None:
        body = ""
    if not isinstance(body, str):
        raise EvidenceContractError("pull request body must be text")
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise EvidenceContractError("pull request number must be a positive integer")
    _validate_scope(title, body)

    if not isinstance(run_id, str) or not run_id.isdecimal() or int(run_id) < 1:
        raise EvidenceContractError("workflow run ID must be a positive integer")
    if not isinstance(run_attempt, int) or isinstance(run_attempt, bool) or run_attempt < 1:
        raise EvidenceContractError("workflow run attempt must be a positive integer")

    return {
        "schema_version": SCHEMA_VERSION,
        "repository": expected_repository,
        "pull_request": number,
        "head_sha": head_sha,
        "merge_sha": _require_sha(merge_sha, "merge SHA"),
        "workflow_sha": _require_sha(workflow_sha, "workflow SHA"),
        "workflow_run_id": int(run_id),
        "workflow_run_attempt": run_attempt,
        "title_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "declared_scope": "readiness_infrastructure",
        "release_candidate": "NONE",
        "release_status": "NOT_READY",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--merge-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = build_evidence(
        _load_event(args.event),
        expected_repository=args.repository,
        expected_head_sha=args.head_sha,
        merge_sha=args.merge_sha,
        workflow_sha=args.workflow_sha,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
