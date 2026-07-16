"""Security contract for the protected formal-evaluation producer workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "formal-evaluation.yml"
)


def _workflow() -> dict:
    # BaseLoader keeps the YAML 1.1 spelling `on` as a string rather than bool.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _job() -> dict:
    return _workflow()["jobs"]["evaluate-and-attest"]


def _steps_by_name() -> dict[str, dict]:
    return {step["name"]: step for step in _job()["steps"]}


def _all_run_scripts() -> str:
    return "\n".join(
        step["run"] for step in _job()["steps"] if "run" in step
    )


def _request_parser() -> str:
    script = _steps_by_name()[
        "Validate protected evaluation request in immutable image"
    ]["run"]
    return script.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def _valid_request() -> dict:
    return {
        "schema_version": "formal-evaluation-request-v1",
        "mode": "full_game",
        "dataset_scope": "private_holdout",
        "eval_data_path": "/protected/evaluation/eval-data.json",
        "eval_data_sha256": "1" * 64,
        "deal_set_id": "2" * 64,
        "model_matrix_path": "/protected/evaluation/model-matrix.json",
        "model_matrix_sha256": "3" * 64,
        "p17_matrix_path": "/protected/evaluation/p17-matrix.json",
        "p17_matrix_sha256": "4" * 64,
        "candidate": "candidate-v1",
        "baseline": "baseline-v1",
        "bootstrap_samples": 2000,
    }


def _run_request_parser(tmp_path: Path, raw: bytes) -> subprocess.CompletedProcess:
    request_path = tmp_path / "evaluation-request.json"
    request_path.write_bytes(raw)
    (tmp_path / "output").mkdir()
    env = dict(os.environ)
    env.update(
        {
            "FORMAL_EVALUATION_REQUEST": str(request_path),
            "FORMAL_EVALUATION_REQUEST_SHA256": hashlib.sha256(raw).hexdigest(),
            "FORMAL_RUN_ROOT": str(tmp_path),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", _request_parser()],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_workflow_dispatch_accepts_only_the_protected_source_commit() -> None:
    workflow = _workflow()
    assert set(workflow["on"]) == {"workflow_dispatch"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"source_sha"}
    assert inputs["source_sha"]["required"] == "true"

    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "github.event.inputs" not in raw
    assert set(re.findall(r"inputs\.([A-Za-z0-9_-]+)", raw)) == {
        "source_sha"
    }


def test_all_evaluation_identities_come_from_one_protected_request() -> None:
    env = _job()["env"]
    assert env["FORMAL_EVALUATION_REQUEST_PATH"] == (
        "${{ vars.FORMAL_EVALUATION_REQUEST_PATH }}"
    )
    assert env["FORMAL_EVALUATION_REQUEST_SHA256"] == (
        "${{ vars.FORMAL_EVALUATION_REQUEST_SHA256 }}"
    )
    caller_controlled = {
        "EVALUATION_MODE",
        "DATASET_SCOPE",
        "APPROVED_EVAL_DATA_PATH",
        "APPROVED_EVAL_DATA_SHA256",
        "APPROVED_DEAL_SET_ID",
        "APPROVED_MODEL_MATRIX_PATH",
        "APPROVED_MODEL_MATRIX_SHA256",
        "APPROVED_P17_MATRIX_PATH",
        "APPROVED_P17_MATRIX_SHA256",
        "CANDIDATE_BUNDLE",
        "BASELINE_BUNDLE",
        "BOOTSTRAP_SAMPLES",
    }
    assert caller_controlled.isdisjoint(env)

    request = _steps_by_name()[
        "Validate protected evaluation request in immutable image"
    ]
    assert request["id"] == "request"
    script = request["run"]
    assert "docker run --rm --interactive" in script
    assert "--pull never" in script
    assert '--entrypoint ""' in script
    assert "--network none" in script
    assert '"$FORMAL_EVALUATOR_IMAGE"' in script
    assert 'SCHEMA = "formal-evaluation-request-v1"' in script
    for field in (
        "mode",
        "dataset_scope",
        "eval_data_path",
        "eval_data_sha256",
        "deal_set_id",
        "model_matrix_path",
        "model_matrix_sha256",
        "p17_matrix_path",
        "p17_matrix_sha256",
        "candidate",
        "baseline",
        "bootstrap_samples",
    ):
        assert f'"{field}"' in script
    assert "set(payload) != FIELDS" in script
    assert "duplicate JSON key" in script
    assert "object_pairs_hook=strict_object" in script
    assert "parse_constant=reject_constant" in script
    assert "non-finite JSON number" in script
    assert 'type(value) is not str or "\\n" in value or "\\r" in value' in script
    assert 're.compile(r"[0-9a-f]{64}")' in script
    assert 're.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")' in script
    assert "str(parsed) != value" in script
    assert "type(samples) is not int" in script
    assert "2000 <= samples <= 100_000" in script
    assert 'cat "$env_file" >>"$GITHUB_ENV"' in script
    assert 'cat "$outputs_file" >>"$GITHUB_OUTPUT"' in script


def test_embedded_request_parser_emits_only_validated_values(tmp_path: Path) -> None:
    raw = json.dumps(
        _valid_request(), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    result = _run_request_parser(tmp_path, raw)
    assert result.returncode == 0, result.stderr

    env_lines = (tmp_path / "output" / "validated-request.env").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(env_lines) == 13
    assert "EVALUATION_MODE=full_game" in env_lines
    assert "DATASET_SCOPE=private_holdout" in env_lines
    assert "APPROVED_DEAL_SET_ID=" + "2" * 64 in env_lines
    assert "BOOTSTRAP_SAMPLES=2000" in env_lines
    outputs = (
        tmp_path / "output" / "validated-request.outputs"
    ).read_text(encoding="utf-8")
    assert outputs == "mode=full_game\ndataset_scope=private_holdout\n"


def test_embedded_request_parser_rejects_untrusted_json_forms(
    tmp_path: Path,
) -> None:
    valid = _valid_request()
    canonical = json.dumps(valid, separators=(",", ":"), allow_nan=False)

    extra = dict(valid)
    extra["extra"] = "not-allowed"
    newline = dict(valid)
    newline["candidate"] = "candidate\nDATASET_SCOPE=public"
    noncanonical_path = dict(valid)
    noncanonical_path["eval_data_path"] = "//protected/evaluation/deals.json"
    wrong_type = dict(valid)
    wrong_type["bootstrap_samples"] = True
    too_many_bootstraps = dict(valid)
    too_many_bootstraps["bootstrap_samples"] = 100_001
    invalid_scope = dict(valid)
    invalid_scope["dataset_scope"] = "public\nignored"

    invalid_payloads = [
        json.dumps(extra, separators=(",", ":")).encode("utf-8"),
        canonical.replace(
            '"mode":"full_game"',
            '"mode":"full_game","mode":"cardplay_only"',
            1,
        ).encode("utf-8"),
        canonical.replace(
            '"bootstrap_samples":2000', '"bootstrap_samples":NaN'
        ).encode("utf-8"),
        json.dumps(newline, separators=(",", ":")).encode("utf-8"),
        json.dumps(noncanonical_path, separators=(",", ":")).encode("utf-8"),
        json.dumps(wrong_type, separators=(",", ":")).encode("utf-8"),
        json.dumps(too_many_bootstraps, separators=(",", ":")).encode("utf-8"),
        json.dumps(invalid_scope, separators=(",", ":")).encode("utf-8"),
    ]
    for index, raw in enumerate(invalid_payloads):
        case_root = tmp_path / f"case-{index}"
        case_root.mkdir()
        result = _run_request_parser(case_root, raw)
        assert result.returncode != 0, f"accepted invalid request case {index}"


def test_workflow_uses_protected_self_hosted_environment_and_least_privilege() -> None:
    job = _job()
    assert job["environment"] == "formal-evaluation"
    assert job["runs-on"] == ["self-hosted", "linux", "formal-evaluator"]
    assert _workflow()["permissions"] == {}
    assert job["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
        "artifact-metadata": "write",
    }
    script = _all_run_scripts()
    assert '"$GITHUB_REF" == "refs/heads/main"' in script
    assert '"$GITHUB_REF_PROTECTED" == "true"' in script
    assert '"$GITHUB_REPOSITORY" == "$EXPECTED_REPOSITORY"' in script
    assert '"$GITHUB_WORKFLOW_REF" == "$EXPECTED_WORKFLOW_REF"' in script


def test_checkout_and_runtime_source_are_the_same_approved_commit() -> None:
    steps = _steps_by_name()
    checkout = steps["Checkout the approved source commit"]
    assert checkout["with"]["ref"] == "${{ inputs.source_sha }}"
    assert checkout["with"]["fetch-depth"] == "1"
    assert checkout["with"]["clean"] == "true"
    assert checkout["with"]["persist-credentials"] == "false"

    script = _all_run_scripts()
    assert '"$APPROVED_SOURCE_SHA" == "$GITHUB_SHA"' in script
    assert '"$APPROVED_SOURCE_SHA" == "$GITHUB_WORKFLOW_SHA"' in script
    assert "git rev-parse --verify 'HEAD^{commit}'" in script
    assert "--formal-release" in script
    assert '--expected-source-git-sha "$APPROVED_SOURCE_SHA"' in script
    assert 'DOUZERO_GIT_SHA: ""' in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_evaluator_image_and_dependencies_come_only_from_protected_environment() -> None:
    workflow = _workflow()
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert not any("image" in name or "package" in name for name in inputs)

    env = _job()["env"]
    assert env["FORMAL_EVALUATOR_IMAGE"] == (
        "${{ vars.FORMAL_EVALUATOR_IMAGE }}"
    )
    assert env["APPROVED_PYTHON_PACKAGES_SHA256"] == (
        "${{ vars.FORMAL_PYTHON_PACKAGES_SHA256 }}"
    )
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"] == "${{ github.workspace }}"
    assert "DOUZERO_EVALUATOR_IMAGE_DIGEST" not in env

    steps = _steps_by_name()
    image = steps["Pull and verify protected immutable evaluator image"]["run"]
    assert 'docker pull --quiet "$FORMAL_EVALUATOR_IMAGE"' in image
    assert ".RepoDigests" in image
    assert '"$repo_digest" == "$FORMAL_EVALUATOR_IMAGE"' in image
    assert 'image_digest="${FORMAL_EVALUATOR_IMAGE##*@}"' in image
    assert 'DOUZERO_EVALUATOR_IMAGE_DIGEST=$image_digest' in image

    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "actions/setup-python" not in raw
    assert "pip install" not in raw
    assert "python -m venv" not in raw
    assert "douzero-formal-venv" not in raw

    scripts = _all_run_scripts()
    assert scripts.count("docker run") == 7
    assert scripts.count('--entrypoint ""') == 7
    assert scripts.count("--network none") == 7
    assert "--network bridge" not in scripts


def test_evaluation_runs_offline_in_exact_image_with_read_only_evidence() -> None:
    steps = _steps_by_name()
    evaluation = steps[
        "Run formal paired evaluation in the immutable image"
    ]["run"]
    assert "docker run --rm" in evaluation
    assert "--pull never" in evaluation
    assert '--entrypoint ""' in evaluation
    assert "--network none" in evaluation
    assert '"$FORMAL_EVALUATOR_IMAGE" \\' in evaluation
    assert "python evaluate_paired.py" in evaluation
    assert (
        '--mount "type=bind,src=$GITHUB_WORKSPACE,'
        'dst=$GITHUB_WORKSPACE,readonly"'
    ) in evaluation
    for directory in ("inputs", "evaluator-checkpoints", "p17-checkpoints"):
        assert (
            f'src=$FORMAL_RUN_ROOT/{directory},'
            f'dst=$FORMAL_RUN_ROOT/{directory},readonly'
        ) in evaluation
    assert (
        'src=$FORMAL_RUN_ROOT/output,dst=$FORMAL_RUN_ROOT/output"'
    ) in evaluation
    for identity in (
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_WORKFLOW_SHA",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "RUNNER_ENVIRONMENT",
        "DOUZERO_EVALUATOR_IMAGE_DIGEST",
    ):
        assert f"--env {identity}" in evaluation


def test_container_dependency_manifest_must_match_protected_digest() -> None:
    dependency = _steps_by_name()[
        "Verify the immutable image dependency set"
    ]["run"]
    assert "--network none" in dependency
    assert '"$FORMAL_EVALUATOR_IMAGE"' in dependency
    assert "command -v gh" in dependency
    assert "python -m pip freeze --all" in dependency
    assert "LC_ALL=C sort" in dependency
    assert (
        '[[ "$package_sha" == "$APPROVED_PYTHON_PACKAGES_SHA256" ]]'
        in dependency
    )
    assert "PYTHON_PACKAGES_SHA256=$package_sha" in dependency


def test_inputs_are_hash_checked_snapshots_not_shell_interpolation() -> None:
    script = _all_run_scripts()
    assert 'sha256sum "$FORMAL_EVALUATION_REQUEST_PATH"' in script
    assert 'install -m 0400 "$FORMAL_EVALUATION_REQUEST_PATH"' in script
    assert '"${FORMAL_EVALUATION_REQUEST_SHA256:-}" =~ ^[0-9a-f]{64}$' in script
    assert "evaluation-request.json" in script
    assert 'sha256sum "$APPROVED_EVAL_DATA_PATH"' in script
    assert 'sha256sum "$APPROVED_MODEL_MATRIX_PATH"' in script
    assert 'sha256sum "$APPROVED_P17_MATRIX_PATH"' in script
    assert 'install -m 0400 "$APPROVED_EVAL_DATA_PATH"' in script
    assert 'install -m 0400 "$APPROVED_MODEL_MATRIX_PATH"' in script
    assert 'install -m 0400 "$APPROVED_P17_MATRIX_PATH"' in script
    assert '"$APPROVED_EVAL_DATA_PATH" == /*' in script
    assert '"$APPROVED_MODEL_MATRIX_PATH" == /*' in script
    assert '"$APPROVED_P17_MATRIX_PATH" == /*' in script
    assert '"$APPROVED_DEAL_SET_ID" =~ ^[0-9a-f]{64}$' in script
    assert 'scenario.get("num_deals", 0) < 1000' in script
    assert 'scenario.get("bootstrap_samples", 0) < 2000' in script
    assert 'run_root="$RUNNER_TEMP/douzero-formal-' in script
    assert '[[ ! -e "$run_root" ]]' in script
    assert "FORMAL_EVAL_DATA=$run_root/inputs/eval-data.json" in script
    assert "eval-data.pkl" not in script
    assert "FORMAL_RESULT_JSON=$run_root/output/formal-evaluation-result.json" in script
    for step in _job()["steps"]:
        if "run" in step:
            assert "${{ inputs." not in step["run"]


def test_complete_result_binds_run_hardware_inputs_and_dependencies() -> None:
    script = _all_run_scripts()
    assert "verify_result_integrity(protected)" in script
    assert "attach_result_integrity(payload)" in script
    assert 'runtime.get("execution_environment")' in script
    assert '"container_image_digest"' in script
    assert "DOUZERO_EVALUATOR_IMAGE_DIGEST" in script
    for identity in (
        '"repository"',
        '"workflow_ref"',
        '"workflow_sha"',
        '"run_id"',
        '"run_attempt"',
        '"run_url"',
        '"eval_data_sha256"',
        '"model_matrix_sha256"',
        '"p17_matrix_sha256"',
        '"evaluation_request_sha256"',
        '"python_packages_sha256"',
        '"protected_environment"',
        '"runner_identity"',
    ):
        assert identity in script
    assert "validate_evaluation_runtime_identity(" in script
    assert "require_formal_source=True" in script
    assert 'scenario.get("bootstrap_samples") != expected["bootstrap_samples"]' in script
    assert "pip freeze --all" in script
    assert "PYTHON_PACKAGES_SHA256" in script
    assert '"container_image_reference"' in script
    assert '"container_image_id"' in script

    cleanup = _steps_by_name()["Remove protected input snapshots"]
    assert cleanup["if"] == "always()"
    assert 'rm -rf -- "$run_root"' in cleanup["run"]
    assert "venv" not in cleanup["run"]


def test_exact_result_is_sigstore_attested_verified_and_uploaded() -> None:
    steps = _steps_by_name()
    attest = steps["Attest exact evaluation result"]
    assert attest["uses"].startswith("actions/attest@")
    assert attest["with"]["subject-path"] == "${{ env.FORMAL_RESULT_JSON }}"
    assert set(attest["with"]) == {"subject-path", "show-summary"}
    p17_attest = steps["Attest exact P17 artifact manifest"]
    assert p17_attest["uses"].startswith("actions/attest@")
    assert p17_attest["with"]["subject-path"] == (
        "${{ env.FORMAL_P17_OUTPUT }}/manifest.json"
    )

    verification = steps["Verify detached attestation and immutable subject"]["run"]
    assert 'sha256sum "$FORMAL_RESULT_JSON"' in verification
    assert 'gh attestation verify "$FORMAL_RESULT_JSON"' in verification
    for flag in (
        "--bundle",
        "--repo",
        "--signer-workflow",
        "--signer-digest",
        "--source-digest",
        "--source-ref",
        "--format json",
    ):
        assert flag in verification

    public_upload = steps[
        "Upload public result, detached bundle, and audit material"
    ]
    assert public_upload["if"] == (
        "steps.request.outputs.dataset_scope == 'public'"
    )
    assert "${{ steps.request.outputs.mode }}" in public_upload["with"]["name"]
    assert "${{ env.FORMAL_RESULT_JSON }}" in public_upload["with"]["path"]
    assert "${{ env.FORMAL_P17_OUTPUT }}/" in public_upload["with"]["path"]
    assert "${{ steps.attest.outputs.bundle-path }}" in public_upload["with"]["path"]
    assert public_upload["with"]["if-no-files-found"] == "error"
    assert "${{ steps.p17-attest.outputs.bundle-path }}" in public_upload[
        "with"
    ]["path"]

    private_upload = steps[
        "Upload private attestation audit without game traces"
    ]
    assert private_upload["if"] == (
        "steps.request.outputs.dataset_scope == 'private_holdout'"
    )
    assert "${{ env.FORMAL_RESULT_JSON }}" not in private_upload["with"]["path"]
    assert "${{ env.FORMAL_P17_OUTPUT }}/" in private_upload["with"]["path"]
    assert "${{ steps.attest.outputs.bundle-path }}" in private_upload["with"]["path"]
    assert "${{ steps.p17-attest.outputs.bundle-path }}" in private_upload[
        "with"
    ]["path"]

    collation = steps[
        "Collate replayed P17 artifacts inside protected boundary"
    ]["run"]
    assert "--network none" in collation
    assert '"$FORMAL_EVALUATOR_IMAGE"' in collation
    assert "GH_TOKEN" not in collation
    assert 'install -m 0400 "$ATTESTATION_BUNDLE"' in collation
    assert "tools/prepare_p17_evaluation.py" in collation
    assert "--approved-cardplay-eval-data" in collation
    assert "--approved-full-game-eval-data" in collation
    assert 'set(result) != {' in collation
    assert '"p17-private-result-projection-v1"' in collation
    assert 'set(scenario) != {' in collation
    assert 'set(evidence) != {' in collation
    assert 'evidence["published_game_rows"] != 0' in collation
    p17_verification = steps[
        "Verify P17 artifact manifest attestation"
    ]["run"]
    assert 'gh attestation verify "$manifest"' in p17_verification
    assert "--signer-workflow" in p17_verification


def test_signed_manifest_closure_is_rechecked_immediately_before_upload() -> None:
    steps = _job()["steps"]
    names = [step["name"] for step in steps]
    verify_index = names.index("Verify P17 artifact manifest attestation")
    recheck_index = names.index("Recheck signed P17 manifest closure before upload")
    public_upload_index = names.index(
        "Upload public result, detached bundle, and audit material"
    )
    private_upload_index = names.index(
        "Upload private attestation audit without game traces"
    )
    assert verify_index < recheck_index < public_upload_index
    assert verify_index < recheck_index < private_upload_index

    recheck = steps[recheck_index]["run"]
    assert "--network none" in recheck
    assert '"$FORMAL_EVALUATOR_IMAGE"' in recheck
    assert "EXPECTED_MANIFEST_SHA256" in recheck
    assert "p17-artifact-manifest-v1" in recheck
    assert "actual != EXPECTED_FILES | {\"manifest.json\"}" in recheck
    assert "entry.is_symlink()" in recheck
    assert "not stat.S_ISREG" in recheck
    assert "actual_digest != digest or actual_size != size" in recheck
    for artifact in (
        "ablations.json",
        "calibration.json",
        "cardplay_results.json",
        "full_game_results.json",
        "latency.json",
        "model_matrix.json",
        "report.md",
    ):
        assert f'"{artifact}"' in recheck


def test_every_third_party_action_is_pinned_to_a_full_commit() -> None:
    actions = [step["uses"] for step in _job()["steps"] if "uses" in step]
    assert actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in actions)
