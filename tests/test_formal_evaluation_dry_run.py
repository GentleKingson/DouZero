"""Contract tests for the public synthetic formal-evaluation smoke."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.capture_baseline import _ci_identity
from tools.formal_evaluation_dry_run import _build_matrices


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "formal-evaluation-dry-run.yml"


def _workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _job() -> dict:
    return _workflow()["jobs"]["synthetic-dry-run"]


def _steps() -> dict[str, dict]:
    return {step["name"]: step for step in _job()["steps"]}


def _scripts() -> str:
    return "\n".join(step["run"] for step in _job()["steps"] if "run" in step)


def test_dry_run_is_public_synthetic_and_cannot_impersonate_formal_producer() -> None:
    workflow = _workflow()
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}
    assert workflow["on"]["pull_request"] == {
        "branches": ["main"],
        "types": ["opened", "synchronize", "reopened"],
    }
    assert set(workflow["on"]["workflow_dispatch"]["inputs"]) == {
        "expected_source_sha"
    }
    job = _job()
    assert job["runs-on"] == "ubuntu-latest"
    assert "environment" not in job
    assert "head.repo.full_name == github.repository" in job["if"]
    raw = WORKFLOW.read_text(encoding="utf-8")
    assert "formal-evaluator" not in raw
    assert "secrets." not in raw
    assert "vars." not in raw
    assert "private_holdout" not in raw
    assert "formal-evaluation.yml" not in raw
    assert "formal-evaluation-dry-run.yml" in raw
    assert "release_eligible" in raw
    assert '== "false"' in raw
    assert '== "insufficient"' in raw


def test_dry_run_binds_attested_merge_separately_from_pr_head() -> None:
    steps = _steps()
    checkout = steps["Checkout exact synthetic source"]
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    identity = steps["Bind dispatch, workflow, and checkout identity"]["run"]
    assert '"$EXPECTED_SOURCE_SHA" == "$TESTED_MERGE_SHA"' in identity
    assert '"$REQUESTED_SOURCE_SHA" == "$EXPECTED_SOURCE_SHA"' in identity
    assert '"$EXPECTED_SOURCE_SHA" == "$GITHUB_SHA"' in identity
    assert "refs/pull/[0-9]+/merge" in identity
    assert "git rev-parse --verify 'HEAD^{commit}'" in identity
    assert "git status --porcelain=v1 --untracked-files=all" in identity


def test_every_runtime_container_is_offline_read_only_and_capability_free() -> None:
    scripts = _scripts()
    definitions = scripts.count("docker run") + scripts.count("docker create")
    assert definitions >= 7
    for flag in (
        '--entrypoint ""',
        "--network none",
        "--read-only",
        "--security-opt no-new-privileges",
        "--cap-drop ALL",
    ):
        assert scripts.count(flag) == definitions
    assert "--pull never" in scripts
    audit = _steps()["Audit hardened container runtime controls"]["run"]
    assert "docker inspect" in audit
    assert "NetworkMode" in audit
    assert "ReadonlyRootfs" in audit
    assert "CapDrop" in audit
    assert "SecurityOpt" in audit


def test_dry_run_exercises_snapshot_formal_source_replay_and_offline_attestation() -> None:
    scripts = _scripts()
    assert scripts.count("douzero.evaluation.snapshot_cli") == 2
    evaluation = _steps()["Run synthetic formal-source evaluation offline"]["run"]
    assert "python evaluate_paired.py" in evaluation
    assert "--formal-release" in evaluation
    assert '--expected-source-git-sha "$EXPECTED_SOURCE_SHA"' in evaluation
    assert "--bootstrap-samples 2000" in evaluation
    assert "--network none" in evaluation
    collate = _steps()[
        "Replay and collate through the production attestation boundary"
    ]["run"]
    assert "python tools/prepare_p17_evaluation.py" in collate
    assert "--cardplay-attestation" in collate
    assert "--attestation-trusted-root" in collate
    assert "--attestation-trusted-root-sha256" in collate
    assert "--attestation-signer-workflow" in collate
    assert "--attestation-signer-digest" in collate
    assert "--attestation-source-ref" in collate
    assert "result-attestation.bundle" in collate
    assert "allow_unverified_results" not in (
        ROOT / "tools" / "formal_evaluation_dry_run.py"
    ).read_text(encoding="utf-8")
    for name in (
        "Verify synthetic result attestation offline",
        "Verify synthetic P17 manifest attestation offline",
    ):
        script = _steps()[name]["run"]
        assert "gh attestation verify" in script
        assert "--bundle" in script
        assert "--custom-trusted-root /tmp/trusted-root.jsonl" in script
        assert "--network none" in script
    for name in (
        "Snapshot trusted root for synthetic result",
        "Snapshot trusted root for synthetic P17 manifest",
    ):
        script = _steps()[name]["run"]
        assert "gh attestation trusted-root" in script
        assert "sha256sum" in script
        assert "chmod 0444" in script


def test_failure_cannot_reach_upload_and_cleanup_is_fail_closed() -> None:
    steps = _job()["steps"]
    names = [step["name"] for step in steps]
    failure = names.index("Prove failed evaluation cannot authorize upload")
    gate = names.index("Authorize synthetic-only artifact upload")
    upload = names.index("Upload synthetic dry-run evidence")
    assert failure < gate < upload
    upload_step = steps[upload]
    assert upload_step["if"] == (
        "success() && steps.synthetic-gate.outputs.authorized == 'true'"
    )
    assert upload_step["with"]["name"].startswith("synthetic-formal-dry-run-")
    assert "formal-p17-" not in upload_step["with"]["name"]

    create = _steps()["Create isolated synthetic run root"]["run"]
    assert "stale=(\"$RUNNER_TEMP\"/douzero-synthetic-formal-*)" in create
    assert "Refusing to run with a stale synthetic run root" in create
    assert "rm -rf" not in create
    cleanup = _steps()["Remove all synthetic run state"]
    assert cleanup["if"] == "always()"
    assert '[[ ! -e "$run_root" && ! -L "$run_root" ]]' in cleanup["run"]


def test_synthetic_matrix_uses_loadable_backend_names_but_remains_small() -> None:
    paths = {role: f"/tmp/{role}.pt" for role in (
        "landlord", "landlord_up", "landlord_down"
    )}
    digests = {role: str(index) * 64 for index, role in enumerate(paths, start=1)}
    evaluator, p17 = _build_matrices(paths, digests)
    assert set(evaluator["bundles"]) == {"v2_full_stack", "legacy_wp"}
    assert evaluator["ablations"] == {}
    for name in evaluator["bundles"]:
        assert evaluator["bundles"][name]["backend"] == "legacy"
        assert p17["models"][name]["cardplay_only"]["status"] == "available"
        assert p17["models"][name]["full_game"]["status"] == "unavailable"


def test_baseline_ci_identity_keeps_head_and_merge_separate(monkeypatch) -> None:
    monkeypatch.delenv("DOUZERO_CI_HEAD_SHA", raising=False)
    monkeypatch.delenv("DOUZERO_CI_MERGE_SHA", raising=False)
    assert _ci_identity() is None
    monkeypatch.setenv("DOUZERO_CI_HEAD_SHA", "1" * 40)
    monkeypatch.setenv("DOUZERO_CI_MERGE_SHA", "2" * 40)
    identity = _ci_identity()
    assert identity["head_sha"] == "1" * 40
    assert identity["merge_sha"] == "2" * 40
    monkeypatch.setenv("DOUZERO_CI_MERGE_SHA", "short")
    with pytest.raises(ValueError, match="must both be full"):
        _ci_identity()


@pytest.mark.parametrize(
    ("filename", "job_name"),
    (("ci.yml", "test"), ("python-package.yml", "build")),
)
def test_ordinary_ci_binds_pr_head_separately_from_tested_merge(
    filename: str, job_name: str
) -> None:
    path = ROOT / ".github" / "workflows" / filename
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    job = workflow["jobs"][job_name]
    env = job["env"]
    assert env["DOUZERO_GIT_SHA"] == "${{ github.sha }}"
    assert env["DOUZERO_CI_MERGE_SHA"] == "${{ github.sha }}"
    assert env["DOUZERO_CI_HEAD_SHA"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )
    bind = next(
        step for step in job["steps"]
        if step.get("name") == "Bind tested head and merge identities"
    )["run"]
    assert 'git rev-parse HEAD)" == "$DOUZERO_CI_MERGE_SHA"' in bind
    upload = next(
        step for step in job["steps"]
        if step.get("name", "").startswith("Upload commit-bound")
    )
    assert "${{ env.DOUZERO_CI_HEAD_SHA }}" in upload["with"]["name"]
    assert "${{ env.DOUZERO_CI_MERGE_SHA }}" in upload["with"]["name"]
