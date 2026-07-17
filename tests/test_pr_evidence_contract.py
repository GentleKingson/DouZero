"""PR scope and commit-bound evidence contract tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from tools.validate_pr_evidence import EvidenceContractError, build_evidence, main


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-evidence.yml"
HEAD = "1" * 40
MERGE = "2" * 40
WORKFLOW_SHA = "3" * 40


def _event(*, title: str | None = None, body: str | None = None) -> dict:
    return {
        "repository": {"full_name": "GentleKingson/DouZero"},
        "pull_request": {
            "number": 20,
            "title": title or "P17: full-game training and readiness infrastructure",
            "body": body
            or (
                "This Draft provides fail-closed infrastructure, not model-release "
                "evidence.\n\nRelease candidate: NONE\n\n"
                "Release status: NOT READY\n"
            ),
            "head": {"sha": HEAD},
        },
    }


def _build(event: dict | None = None) -> dict:
    return build_evidence(
        event or _event(),
        expected_repository="GentleKingson/DouZero",
        expected_head_sha=HEAD,
        merge_sha=MERGE,
        workflow_sha=WORKFLOW_SHA,
        run_id="1234",
        run_attempt=2,
    )


def test_stable_infrastructure_scope_emits_bound_evidence() -> None:
    evidence = _build()
    assert evidence == {
        "schema_version": "p17-pr-evidence-v1",
        "repository": "GentleKingson/DouZero",
        "pull_request": 20,
        "head_sha": HEAD,
        "merge_sha": MERGE,
        "workflow_sha": WORKFLOW_SHA,
        "workflow_run_id": 1234,
        "workflow_run_attempt": 2,
        "title_sha256": evidence["title_sha256"],
        "body_sha256": evidence["body_sha256"],
        "declared_scope": "readiness_infrastructure",
        "release_candidate": "NONE",
        "release_status": "NOT_READY",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["title_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["body_sha256"])


@pytest.mark.parametrize(
    "title",
    [
        "P17: release-readiness closure",
        "P17: release ready",
        "P17: full-game training",
    ],
)
def test_title_must_be_honest_infrastructure_scope(title: str) -> None:
    with pytest.raises(EvidenceContractError):
        _build(_event(title=title))


@pytest.mark.parametrize(
    "claim",
    [
        "final PR head: " + "a" * 40,
        "current-head Docker CPU release gate passed",
        "1,781 tests passed",
        "final-head GitHub matrix passed",
        "two independent local boundary audits found no remaining P0-P2 issue",
        "artifact identity: " + "b" * 64,
    ],
)
def test_body_rejects_mutable_or_unbound_validation_claims(claim: str) -> None:
    body = (
        f"{claim}\n\nRelease candidate: NONE\n\n"
        "Release status: NOT READY\n"
    )
    with pytest.raises(EvidenceContractError):
        _build(_event(body=body))


@pytest.mark.parametrize(
    "body",
    [
        "Release status: NOT READY",
        "Release candidate: NONE",
        "Release candidate: rc-1\nRelease status: READY",
    ],
)
def test_body_requires_explicit_none_not_ready_status(body: str) -> None:
    with pytest.raises(EvidenceContractError):
        _build(_event(body=body))


def test_event_head_must_match_checked_out_head() -> None:
    event = _event()
    event["pull_request"]["head"]["sha"] = "4" * 40
    with pytest.raises(EvidenceContractError, match="head SHA mismatch"):
        _build(event)


def test_cli_writes_canonical_evidence(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    output = tmp_path / "evidence" / "pr.json"
    event_path.write_text(json.dumps(_event()), encoding="utf-8")
    assert main(
        [
            "--event",
            str(event_path),
            "--repository",
            "GentleKingson/DouZero",
            "--head-sha",
            HEAD,
            "--merge-sha",
            MERGE,
            "--workflow-sha",
            WORKFLOW_SHA,
            "--run-id",
            "1234",
            "--run-attempt",
            "1",
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["head_sha"] == HEAD
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_workflow_runs_on_every_scope_or_head_change_with_least_privilege() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request"}
    trigger = workflow["on"]["pull_request"]
    assert set(trigger["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "edited",
        "ready_for_review",
    }
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["bind-evidence"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["EXPECTED_HEAD_SHA"] == (
        "${{ github.event.pull_request.head.sha }}"
    )
    steps = {step["name"]: step for step in job["steps"]}
    checkout = steps["Checkout the exact PR head"]
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    assert "--head-sha \"$EXPECTED_HEAD_SHA\"" in steps[
        "Validate scope and bind evidence"
    ]["run"]
    assert steps["Upload commit-bound evidence"]["with"]["if-no-files-found"] == (
        "error"
    )
    for step in job["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
