"""Contract tests for the Standard V2 M1 target-hardware merge gate."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "standard-v2-m1-gpu.yml"


def _workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_m1_gpu_workflow_is_a_safe_pull_request_gate():
    workflow = _workflow()
    assert set(workflow["on"]) == {"pull_request", "workflow_dispatch"}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    condition = workflow["jobs"]["validate"]["if"]
    assert "head.repo.full_name == github.repository" in condition
    checkout = workflow["jobs"]["validate"]["steps"][0]
    assert "github.event.pull_request.head.sha || github.sha" in checkout["with"]["ref"]


def test_m1_gpu_workflow_enforces_immutable_complete_evidence():
    workflow = _workflow()
    scripts = "\n".join(
        step.get("run", "") for step in workflow["jobs"]["validate"]["steps"]
    )
    assert "douzero-test:latest" not in scripts
    assert '"${IMAGE_ID}" python' in scripts
    assert "B=32 {metric} exceeds 1.5 ms" in scripts
    assert "--episodes 16 --optimizer_steps 1 --device cuda" in scripts
    assert "standard-v2-r1-single-gpu.json" in scripts
    assert 'report.get("source_git_sha") != target_sha' in scripts
    assert "recorded Docker image ID changed during the run" in scripts
