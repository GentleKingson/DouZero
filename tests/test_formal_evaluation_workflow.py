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


def _logical_shell_commands(script: str) -> list[str]:
    """Join backslash continuations without trying to interpret Bash."""

    commands: list[str] = []
    pending: list[str] = []
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        continued = raw_line.rstrip().endswith("\\")
        pending.append(line[:-1].rstrip() if continued else line)
        if not continued:
            commands.append(" ".join(part for part in pending if part))
            pending = []
    if pending:
        commands.append(" ".join(part for part in pending if part))
    return commands


def _docker_array_spans(script: str) -> tuple[set[str], dict[int, int]]:
    """Return Docker command-array names and their raw-line spans."""

    lines = script.splitlines()
    names: set[str] = set()
    spans: dict[int, int] = {}
    index = 0
    while index < len(lines):
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)=\(\s*", lines[index])
        if match is None:
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip() != ")":
            end += 1
        if end == len(lines):
            index += 1
            continue
        block = "\n".join(lines[index : end + 1])
        if re.search(r"(?m)^\s*docker\s+run\b", block):
            names.add(match.group(1))
            spans[index] = end
        index = end + 1
    return names, spans


def _host_visible_commands(script: str) -> list[str]:
    """Exclude commands and inline programs that are unambiguously in Docker."""

    lines = script.splitlines()
    docker_arrays, array_spans = _docker_array_spans(script)
    commands: list[str] = []
    index = 0
    while index < len(lines):
        if index in array_spans:
            index = array_spans[index] + 1
            continue
        raw_parts: list[str] = []
        while index < len(lines):
            raw_line = lines[index]
            index += 1
            line = raw_line.strip()
            if not raw_parts and (not line or line.startswith("#")):
                continue
            continued = raw_line.rstrip().endswith("\\")
            raw_parts.append(line[:-1].rstrip() if continued else line)
            if not continued:
                break
        if not raw_parts:
            continue
        command = " ".join(part for part in raw_parts if part)
        in_container = command.startswith("docker run ") or any(
            command.startswith(f'"${{{name}[@]}}"')
            or command.startswith(f"${{{name}[@]}}")
            for name in docker_arrays
        )
        chained_after_container = bool(
            in_container and re.search(r"\s(?:;|&&|\|\||[|&])\s", command)
        )

        heredoc = re.search(
            r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", command
        )
        if not in_container or chained_after_container:
            commands.append(command)
        if heredoc is not None:
            delimiter = heredoc.group(2)
            while index < len(lines):
                closing = lines[index].strip()
                index += 1
                if closing == delimiter:
                    break
            continue
        if (
            in_container
            and re.search(r"\b(?:ba)?sh\s+-[^\s]*c\s+'", command)
            and command.count("'") % 2 == 1
        ):
            while index < len(lines):
                closing = lines[index].strip()
                index += 1
                if closing == "'":
                    break
    return commands


def _command_substitutions(script: str) -> list[str]:
    """Extract balanced command substitutions, including nested forms."""

    substitutions: list[str] = []
    index = 0
    while index < len(script) - 1:
        if script[index] == "\\":
            index += 2
            continue
        if script[index : index + 2] != "$(":
            index += 1
            continue
        start = index + 2
        cursor = start
        depth = 1
        while cursor < len(script) and depth:
            if script[cursor] == "\\":
                cursor += 2
                continue
            if script[cursor] == "(":
                depth += 1
            elif script[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            substitutions.append(script[start:])
            break
        content = script[start : cursor - 1]
        substitutions.append(content)
        substitutions.extend(_command_substitutions(content))
        index = cursor
    return substitutions


def _mounts(script: str) -> set[str]:
    return set(re.findall(r'--mount\s+"([^"]+)"', script))


def _request_parser() -> str:
    script = _steps_by_name()[
        "Validate protected evaluation request in immutable image"
    ]["run"]
    return script.split("python -I -S - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def _valid_request() -> dict:
    return {
        "schema_version": "formal-evaluation-request-v2",
        "mode": "full_game",
        "dataset_scope": "private_holdout",
        "eval_data_path": "/protected/evaluation/eval-data.json",
        "eval_data_sha256": "1" * 64,
        "deal_set_id": "2" * 64,
        "model_matrix_path": "/protected/evaluation/model-matrix.json",
        "model_matrix_sha256": "3" * 64,
        "model_checkpoint_root": "/protected/evaluation/model-checkpoints",
        "p17_matrix_path": "/protected/evaluation/p17-matrix.json",
        "p17_matrix_sha256": "4" * 64,
        "p17_checkpoint_root": "/protected/evaluation/p17-checkpoints",
        "candidate": "candidate-v1",
        "baseline": "baseline-v1",
        "bootstrap_samples": 2000,
    }


def _run_request_parser(tmp_path: Path, raw: bytes) -> subprocess.CompletedProcess:
    request_path = tmp_path / "evaluation-request.json"
    request_path.write_bytes(raw)
    (tmp_path / "control").mkdir()
    env = dict(os.environ)
    env.update(
        {
            "FORMAL_EVALUATION_REQUEST": str(request_path),
            "FORMAL_EVALUATION_REQUEST_SHA256": hashlib.sha256(raw).hexdigest(),
            "FORMAL_CONTROL_ROOT": str(tmp_path / "control"),
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
    assert "FORMAL_EVALUATION_REQUEST_PATH" not in env
    assert "FORMAL_EVALUATION_REQUEST_SHA256" not in env
    snapshot = _steps_by_name()["Snapshot protected evaluation request"]
    assert snapshot["env"]["FORMAL_EVALUATION_REQUEST_PATH"] == (
        "${{ vars.FORMAL_EVALUATION_REQUEST_PATH }}"
    )
    assert snapshot["env"]["FORMAL_EVALUATION_REQUEST_SHA256"] == (
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
        "APPROVED_MODEL_CHECKPOINT_ROOT",
        "APPROVED_P17_MATRIX_PATH",
        "APPROVED_P17_MATRIX_SHA256",
        "APPROVED_P17_CHECKPOINT_ROOT",
        "CANDIDATE_BUNDLE",
        "BASELINE_BUNDLE",
        "BOOTSTRAP_SAMPLES",
    }
    assert caller_controlled.isdisjoint(env)

    request = _steps_by_name()[
        "Validate protected evaluation request in immutable image"
    ]
    assert request["id"] == "request"
    assert request["env"] == {
        "FORMAL_EVALUATION_REQUEST": "${{ steps.request_snapshot.outputs.path }}",
        "FORMAL_EVALUATION_REQUEST_SHA256": (
            "${{ steps.request_snapshot.outputs.sha256 }}"
        ),
    }
    script = request["run"]
    assert "docker run --rm --interactive" in script
    assert "--pull never" in script
    assert '--entrypoint ""' in script
    assert "--network none" in script
    assert '"$FORMAL_EVALUATOR_IMAGE"' in script
    assert 'SCHEMA = "formal-evaluation-request-v2"' in script
    for field in (
        "mode",
        "dataset_scope",
        "eval_data_path",
        "eval_data_sha256",
        "deal_set_id",
        "model_matrix_path",
        "model_matrix_sha256",
        "model_checkpoint_root",
        "p17_matrix_path",
        "p17_matrix_sha256",
        "p17_checkpoint_root",
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
    assert 'cat "$env_file" >>"$GITHUB_ENV"' not in script
    assert 'cat "$outputs_file" >>"$GITHUB_OUTPUT"' in script
    assert _mounts(script) == {
        "type=bind,src=$FORMAL_EVALUATION_REQUEST,"
        "dst=$FORMAL_EVALUATION_REQUEST,readonly",
        "type=bind,src=$FORMAL_CONTROL_ROOT,dst=$FORMAL_CONTROL_ROOT",
    }
    input_snapshot = _steps_by_name()[
        "Validate and snapshot approved evaluation inputs"
    ]["run"]
    assert 'source "$validated_env"' in input_snapshot
    assert "rm -f -- \\" in input_snapshot
    for removed in (
        '"$validated_env"',
        '"$validated_outputs"',
        '"$run_root/inputs/evaluation-request.json"',
        '"$run_root/inputs/model-matrix.approved.json"',
        '"$run_root/inputs/p17-matrix.approved.json"',
    ):
        assert removed in input_snapshot
    for raw_path in (
        "APPROVED_EVAL_DATA_PATH",
        "APPROVED_MODEL_MATRIX_PATH",
        "APPROVED_MODEL_CHECKPOINT_ROOT",
        "APPROVED_P17_MATRIX_PATH",
        "APPROVED_P17_CHECKPOINT_ROOT",
    ):
        assert f'echo "{raw_path}=' not in input_snapshot


def test_embedded_request_parser_emits_only_validated_values(tmp_path: Path) -> None:
    raw = json.dumps(
        _valid_request(), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    result = _run_request_parser(tmp_path, raw)
    assert result.returncode == 0, result.stderr

    env_lines = (tmp_path / "control" / "validated-request.env").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(env_lines) == 15
    assert "EVALUATION_MODE=full_game" in env_lines
    assert "DATASET_SCOPE=private_holdout" in env_lines
    assert "APPROVED_DEAL_SET_ID=" + "2" * 64 in env_lines
    assert (
        "APPROVED_MODEL_CHECKPOINT_ROOT=/protected/evaluation/model-checkpoints"
        in env_lines
    )
    assert (
        "APPROVED_P17_CHECKPOINT_ROOT=/protected/evaluation/p17-checkpoints"
        in env_lines
    )
    assert "BOOTSTRAP_SAMPLES=2000" in env_lines
    outputs = (
        tmp_path / "control" / "validated-request.outputs"
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
    noncanonical_root = dict(valid)
    noncanonical_root["model_checkpoint_root"] = (
        "/protected/evaluation/../model-checkpoints"
    )
    relative_root = dict(valid)
    relative_root["p17_checkpoint_root"] = "protected/p17-checkpoints"
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
        json.dumps(noncanonical_root, separators=(",", ":")).encode("utf-8"),
        json.dumps(relative_root, separators=(",", ":")).encode("utf-8"),
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
    assert "PYTHONPATH" not in env
    assert "DOUZERO_EVALUATOR_IMAGE_DIGEST" not in env

    steps = _steps_by_name()
    for name in (
        "Run formal paired evaluation in the immutable image",
        "Bind run, input, dependency, and hardware identity into result",
        "Collate replayed P17 artifacts inside protected boundary",
    ):
        assert '--env PYTHONPATH="$GITHUB_WORKSPACE"' in steps[name]["run"]
        assert "--env DOUZERO_GIT_SHA=" in steps[name]["run"]

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
    docker_run_definitions = scripts.count("docker run")
    assert docker_run_definitions == 8
    assert scripts.count('--entrypoint ""') == docker_run_definitions
    assert scripts.count("--network none") == docker_run_definitions
    assert "--network bridge" not in scripts
    assert "--volume" not in scripts
    short_volume = r"(?:-v|\"-v\"|'-v')"
    assert re.search(rf"(?m)^\s*{short_volume}(?:\s|=)", scripts) is None
    assert re.search(
        rf"docker\s+run[^\n]*\s{short_volume}(?:\s|=)", scripts
    ) is None
    mount_tokens = re.findall(r"(?<!\S)--mount(?:\s|=)", scripts)
    assert len(mount_tokens) == scripts.count('--mount "')


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
    assert 'src=$FORMAL_EVAL_DATA,dst=$FORMAL_EVAL_DATA,readonly' in evaluation
    assert (
        "src=$FORMAL_RUN_ROOT/evaluator-snapshot,"
        "dst=$FORMAL_RUN_ROOT/evaluator-snapshot,readonly" in evaluation
    )
    assert "$FORMAL_RUN_ROOT/p17-snapshot" not in evaluation
    assert (
        'src=$FORMAL_RUN_ROOT/result,dst=$FORMAL_RUN_ROOT/result"'
    ) in evaluation
    assert _mounts(evaluation) == {
        "type=bind,src=$GITHUB_WORKSPACE,dst=$GITHUB_WORKSPACE,readonly",
        "type=bind,src=$FORMAL_EVAL_DATA,dst=$FORMAL_EVAL_DATA,readonly",
        "type=bind,src=$FORMAL_RUN_ROOT/evaluator-snapshot,"
        "dst=$FORMAL_RUN_ROOT/evaluator-snapshot,readonly",
        "type=bind,src=$FORMAL_RUN_ROOT/result,dst=$FORMAL_RUN_ROOT/result",
    }
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


def test_checkout_code_cannot_write_trusted_audit_or_other_stage_outputs() -> None:
    steps = _steps_by_name()
    evaluation = steps[
        "Run formal paired evaluation in the immutable image"
    ]["run"]
    binder = steps[
        "Bind run, input, dependency, and hardware identity into result"
    ]["run"]
    collation = steps[
        "Collate replayed P17 artifacts inside protected boundary"
    ]["run"]

    for script in (evaluation, binder, collation):
        assert "FORMAL_AUDIT_ROOT" not in script
    assert "FORMAL_P17_OUTPUT" not in evaluation
    assert "$FORMAL_RUN_ROOT/p17-snapshot" not in evaluation

    assert "src=$FORMAL_RUN_ROOT/result,dst=$FORMAL_RUN_ROOT/result" in binder
    assert _mounts(binder) == {
        "type=bind,src=$GITHUB_WORKSPACE,dst=$GITHUB_WORKSPACE,readonly",
        "type=bind,src=$FORMAL_RUN_ROOT/result,dst=$FORMAL_RUN_ROOT/result",
    }
    for forbidden in (
        "FORMAL_RUN_ROOT/inputs",
        "evaluator-snapshot",
        "p17-snapshot",
        "FORMAL_P17_OUTPUT",
    ):
        assert forbidden not in binder
    assert '[[ -f "$FORMAL_RESULT_JSON" && ! -L "$FORMAL_RESULT_JSON" ]]' in binder
    assert 'realpath -e -- "$FORMAL_RESULT_JSON"' in binder
    assert 'stat -c \'%F\' "$FORMAL_RESULT_JSON"' in binder

    for readonly_input in (
        "src=$FORMAL_EVAL_DATA,dst=$FORMAL_EVAL_DATA,readonly",
        "src=$FORMAL_RESULT_JSON,dst=$FORMAL_RESULT_JSON,readonly",
        "src=$attestation_snapshot,dst=$attestation_snapshot,readonly",
        "src=$FORMAL_RUN_ROOT/p17-snapshot,"
        "dst=$FORMAL_RUN_ROOT/p17-snapshot,readonly",
    ):
        assert readonly_input in collation
    assert "src=$FORMAL_P17_OUTPUT,dst=$FORMAL_P17_OUTPUT" in collation
    assert "src=$FORMAL_RUN_ROOT/inputs" not in collation
    assert "src=$FORMAL_RUN_ROOT/result,dst=$FORMAL_RUN_ROOT/result" not in collation

    collation_lines = collation.splitlines()
    docker_arrays, spans = _docker_array_spans(collation)
    assert docker_arrays == {"container"}
    container_start = next(
        index
        for index, line in enumerate(collation_lines)
        if re.fullmatch(r"\s*container=\(\s*", line)
    )
    container = "\n".join(
        collation_lines[container_start : spans[container_start] + 1]
    )
    assert _mounts(container) == {
        "type=bind,src=$GITHUB_WORKSPACE,dst=$GITHUB_WORKSPACE,readonly",
        "type=bind,src=$FORMAL_EVAL_DATA,dst=$FORMAL_EVAL_DATA,readonly",
        "type=bind,src=$FORMAL_RESULT_JSON,dst=$FORMAL_RESULT_JSON,readonly",
        "type=bind,src=$attestation_snapshot,dst=$attestation_snapshot,readonly",
        "type=bind,src=$FORMAL_ATTESTATION_TRUSTED_ROOT,"
        "dst=$FORMAL_ATTESTATION_TRUSTED_ROOT,readonly",
        "type=bind,src=$FORMAL_RUN_ROOT/p17-snapshot,"
        "dst=$FORMAL_RUN_ROOT/p17-snapshot,readonly",
        "type=bind,src=$FORMAL_P17_OUTPUT,dst=$FORMAL_P17_OUTPUT",
    }
    private_commands = [
        command
        for command in _logical_shell_commands(collation)
        if 'python -I -S - "$release_result" "$FORMAL_RESULT_JSON"' in command
    ]
    assert len(private_commands) == 1
    assert _mounts(private_commands[0]) == {
        "type=bind,src=$release_result,dst=$release_result,readonly",
        "type=bind,src=$FORMAL_RESULT_JSON,dst=$FORMAL_RESULT_JSON,readonly",
    }

    recheck = steps[
        "Recheck immutable upload subjects and audit material"
    ]["run"]
    assert "require_regular" in recheck
    assert "PYTHON_PACKAGES_SHA256" in recheck
    assert "EXPECTED_RESULT_SHA256" in recheck
    assert "EXPECTED_MANIFEST_SHA256" in recheck


def test_container_dependency_manifest_must_match_protected_digest() -> None:
    dependency = _steps_by_name()[
        "Verify the immutable image dependency set"
    ]["run"]
    assert "--network none" in dependency
    assert '"$FORMAL_EVALUATOR_IMAGE"' in dependency
    assert "command -v gh" in dependency
    assert "python -I -B -m pip freeze --all" in dependency
    assert "LC_ALL=C sort" in dependency
    assert (
        '[[ "$package_sha" == "$APPROVED_PYTHON_PACKAGES_SHA256" ]]'
        in dependency
    )
    assert "PYTHON_PACKAGES_SHA256=$package_sha" in dependency
    names = [step["name"] for step in _job()["steps"]]
    package_index = names.index("Verify the immutable image dependency set")
    request_snapshot_index = names.index("Snapshot protected evaluation request")
    assert package_index == request_snapshot_index + 1
    assert package_index < names.index(
        "Validate protected evaluation request in immutable image"
    )
    assert package_index < names.index(
        "Validate and snapshot approved evaluation inputs"
    )
    assert "--workdir /tmp" in dependency
    assert "FORMAL_AUDIT_ROOT" in dependency
    for forbidden_mount in (
        "GITHUB_WORKSPACE",
        "FORMAL_EVALUATION_REQUEST",
        "FORMAL_RUN_ROOT/inputs",
        "PYTHONPATH",
    ):
        assert forbidden_mount not in dependency
    assert _mounts(dependency) == {
        "type=bind,src=$FORMAL_AUDIT_ROOT,dst=$FORMAL_AUDIT_ROOT"
    }


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
    assert '"$APPROVED_MODEL_CHECKPOINT_ROOT" == /*' in script
    assert '"$APPROVED_P17_CHECKPOINT_ROOT" == /*' in script
    assert (
        '[[ -d "$APPROVED_MODEL_CHECKPOINT_ROOT" '
        '&& ! -L "$APPROVED_MODEL_CHECKPOINT_ROOT" ]]' in script
    )
    assert (
        '[[ -d "$APPROVED_P17_CHECKPOINT_ROOT" '
        '&& ! -L "$APPROVED_P17_CHECKPOINT_ROOT" ]]' in script
    )
    assert 'realpath -e -- "$APPROVED_MODEL_CHECKPOINT_ROOT"' in script
    assert 'realpath -e -- "$APPROVED_P17_CHECKPOINT_ROOT"' in script
    assert '[[ "$checkpoint_root" != "$GITHUB_WORKSPACE" ]]' in script
    assert '[[ "$checkpoint_root" != "$FORMAL_RUN_ROOT" ]]' in script
    assert '"$APPROVED_DEAL_SET_ID" =~ ^[0-9a-f]{64}$' in script
    assert 'scenario.get("num_deals", 0) < 1000' in script
    assert 'scenario.get("bootstrap_samples", 0) < 2000' in script
    assert 'run_root="$RUNNER_TEMP/douzero-formal-' in script
    assert '[[ ! -e "$run_root" ]]' in script
    assert "FORMAL_EVAL_DATA=$run_root/inputs/eval-data.json" in script
    assert "FORMAL_MODEL_MATRIX=$run_root/evaluator-snapshot/model-matrix.json" in script
    assert "FORMAL_P17_MATRIX=$run_root/p17-snapshot/p17-matrix.json" in script
    assert "eval-data.pkl" not in script
    assert "FORMAL_RESULT_JSON=$run_root/result/formal-evaluation-result.json" in script
    assert "FORMAL_P17_OUTPUT=$run_root/p17" in script
    for step in _job()["steps"]:
        if "run" in step:
            assert "${{ inputs." not in step["run"]


def test_both_checkpoint_snapshots_run_in_one_hardened_container_contract() -> None:
    snapshot = _steps_by_name()[
        "Validate and snapshot approved evaluation inputs"
    ]["run"]
    lines = snapshot.splitlines()
    docker_arrays, spans = _docker_array_spans(snapshot)
    assert "snapshot_container" in docker_arrays
    start = next(
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"\s*snapshot_container=\(\s*", line)
    )
    container = "\n".join(lines[start : spans[start] + 1])
    for hardening in (
        "docker run --rm",
        "--pull never",
        '--entrypoint ""',
        "--network none",
        "--read-only",
        '--user "$(id -u):$(id -g)"',
        "--security-opt no-new-privileges",
        "--cap-drop ALL",
    ):
        assert hardening in container
    assert "GITHUB_WORKSPACE" not in container
    assert "PYTHONPATH" not in container
    assert "$run_root/inputs" not in container
    assert "--workdir /tmp" in container
    assert "tools/snapshot_evaluation_checkpoints.py" not in snapshot

    snapshot_commands = [
        command
        for command in _logical_shell_commands(snapshot)
        if "python -I -B -m douzero.evaluation.snapshot_cli" in command
    ]
    assert len(snapshot_commands) == 2
    assert all(
        command.startswith('"${snapshot_container[@]}" ')
        for command in snapshot_commands
    )

    evaluator = next(
        command for command in snapshot_commands if "--kind evaluator" in command
    )
    p17 = next(
        command for command in snapshot_commands if "--kind p17" in command
    )
    for command, root, checkpoint_dir, matrix, output in (
        (
            evaluator,
            "model_checkpoint_root",
            "evaluator-snapshot",
            "model-matrix.approved.json",
            "model-matrix.json",
        ),
        (
            p17,
            "p17_checkpoint_root",
            "p17-snapshot",
            "p17-matrix.approved.json",
            "p17-matrix.json",
        ),
    ):
        assert (
            f'src=${root},dst=${root},readonly"' in command
        )
        assert (
            f"src=$run_root/{checkpoint_dir},"
            f'dst=$run_root/{checkpoint_dir}"' in command
        )
        assert f'--matrix "$run_root/inputs/{matrix}"' in command
        assert (
            f"src=$run_root/inputs/{matrix},"
            f"dst=$run_root/inputs/{matrix},readonly" in command
        )
        assert f'--source-root "${root}"' in command
        assert (
            f'--checkpoint-dir "$run_root/{checkpoint_dir}/checkpoints"'
            in command
        )
        assert f'--output "$run_root/{checkpoint_dir}/{output}"' in command
        assert (
            '"$FORMAL_EVALUATOR_IMAGE" '
            "python -I -B -m douzero.evaluation.snapshot_cli" in command
        )
        assert _mounts(command) == {
            f"type=bind,src=$run_root/inputs/{matrix},"
            f"dst=$run_root/inputs/{matrix},readonly",
            f"type=bind,src=${root},dst=${root},readonly",
            f"type=bind,src=$run_root/{checkpoint_dir},"
            f"dst=$run_root/{checkpoint_dir}",
        }


def test_protected_inputs_never_reach_checkout_code_on_the_host() -> None:
    assert set(_workflow()["jobs"]) == {"evaluate-and-attest"}
    steps = _job()["steps"]
    interpreter = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        r"python(?:3(?:\.[0-9]+)?)?|bash|sh|make|pytest|tox|node|npm|npx|"
        r"yarn|pnpm|uv|perl|ruby"
        r")(?:\s|$)"
    )
    repo_executable = re.compile(
        r"(?:^|&&|\|\||[;|&]|\bthen\b|\bdo\b)\s*"
        r"(?:(?:command|exec|env)\s+)*"
        r"(?:(?:source|\.)\s+)?"
        r"['\"]?(?:\./|\$\{?GITHUB_WORKSPACE\}?/|\$PWD/|\.github/|"
        r"tools/|scripts/|douzero/|evaluate_paired\.py|generate_eval_data\.py)"
    )
    checkout_literal = re.compile(
        r"(?:\./|\$\{?GITHUB_WORKSPACE\}?/|\$PWD/|"
        r"(?<![A-Za-z0-9_/])\.github/|tools/|scripts/|douzero/|"
        r"evaluate_paired\.py|generate_eval_data\.py)"
    )
    assignment = re.compile(
        r"^(?:(?:export|readonly|local)\s+)?[A-Za-z_][A-Za-z0-9_]*="
    )
    forbidden_examples = (
        "python tools/checkout_tool.py",
        "python -m douzero.checkout_module",
        "bash scripts/from_checkout.sh",
        'sh "$GITHUB_WORKSPACE/scripts/from_checkout.sh"',
        "make release",
        "node tools/from_checkout.js",
        "uv run pytest",
        "./tools/checkout_executable",
        '"${GITHUB_WORKSPACE}/tools/checkout_executable"',
        "source scripts/from_checkout.sh",
        "docker run image true | ./tools/checkout_executable",
        "docker run image true & ./tools/checkout_executable",
    )
    assert all(
        _host_visible_commands(command) == [command]
        and (
            interpreter.search(command) is not None
            or repo_executable.search(command) is not None
        )
        for command in forbidden_examples
    )
    for substitution in (
        "./tools/checkout_executable",
        "python tools/checkout_tool.py",
    ):
        assert (
            interpreter.search(substitution) is not None
            or repo_executable.search(substitution) is not None
        )
    nested = (
        'docker run --env X="$(./tools/checkout_executable "'
        '"$(id -u)")" image true'
    )
    assert any(
        repo_executable.search(substitution) is not None
        for substitution in _command_substitutions(nested)
    )
    indirect = 'runner=./tools/checkout_executable\n"$runner"'
    assert any(
        assignment.search(command) is not None
        and checkout_literal.search(command) is not None
        for command in _host_visible_commands(indirect)
    )

    for step in steps:
        if "uses" in step:
            assert not step["uses"].startswith("./"), (
                f"{step['name']} executes a checkout-owned local action"
            )
        if "run" not in step:
            continue
        assert step.get("shell") == "bash", (
            f"{step['name']} uses an unapproved host shell"
        )
        assert (
            "<(" not in step["run"] and ">(" not in step["run"]
        ), f"{step['name']} uses process substitution"
        for substitution in _command_substitutions(step["run"]):
            assert interpreter.search(substitution) is None, (
                f"{step['name']} executes an interpreter in a host command "
                f"substitution: {substitution}"
            )
            assert repo_executable.search(substitution) is None, (
                f"{step['name']} executes checkout content in a host command "
                f"substitution: {substitution}"
            )
            assert checkout_literal.search(substitution) is None, (
                f"{step['name']} references checkout content in a host command "
                f"substitution: {substitution}"
            )
        for substitution in re.findall(r"(?<!\\)`([^`]*)`", step["run"]):
            assert interpreter.search(substitution) is None, (
                f"{step['name']} executes an interpreter in host backticks: "
                f"{substitution}"
            )
            assert repo_executable.search(substitution) is None, (
                f"{step['name']} executes checkout content in host backticks: "
                f"{substitution}"
            )
            assert checkout_literal.search(substitution) is None, (
                f"{step['name']} references checkout content in host backticks: "
                f"{substitution}"
            )
        for command in _host_visible_commands(step["run"]):
            assert interpreter.search(command) is None, (
                f"{step['name']} executes an interpreter on the host: {command}"
            )
            assert repo_executable.search(command) is None, (
                f"{step['name']} executes checkout content on the host: {command}"
            )
            if assignment.search(command) is not None:
                assert checkout_literal.search(command) is None, (
                    f"{step['name']} assigns checkout content on the host: "
                    f"{command}"
                )


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
        "--custom-trusted-root",
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
        "success() && steps.upload-gate.outputs.authorized == 'true' && "
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
    assert "${{ env.FORMAL_AUDIT_ROOT }}/" in public_upload["with"]["path"]
    assert "${{ env.FORMAL_RUN_ROOT }}/output/" not in public_upload["with"]["path"]

    private_upload = steps[
        "Upload private attestation audit without game traces"
    ]
    assert private_upload["if"] == (
        "success() && steps.upload-gate.outputs.authorized == 'true' && "
        "steps.request.outputs.dataset_scope == 'private_holdout'"
    )
    assert "${{ env.FORMAL_RESULT_JSON }}" not in private_upload["with"]["path"]
    assert "${{ env.FORMAL_P17_OUTPUT }}/" in private_upload["with"]["path"]
    assert "${{ steps.attest.outputs.bundle-path }}" in private_upload["with"]["path"]
    assert "${{ steps.p17-attest.outputs.bundle-path }}" in private_upload[
        "with"
    ]["path"]
    assert "${{ env.FORMAL_AUDIT_ROOT }}/" in private_upload["with"]["path"]
    assert "${{ env.FORMAL_RUN_ROOT }}/output/" not in private_upload["with"]["path"]

    collation = steps[
        "Collate replayed P17 artifacts inside protected boundary"
    ]["run"]
    assert "--network none" in collation
    assert '"$FORMAL_EVALUATOR_IMAGE"' in collation
    assert "GH_TOKEN" not in collation
    assert "--attestation-trusted-root" in collation
    assert "--attestation-trusted-root-sha256" in collation
    assert (
        "src=$FORMAL_ATTESTATION_TRUSTED_ROOT,"
        "dst=$FORMAL_ATTESTATION_TRUSTED_ROOT,readonly" in collation
    )
    assert 'install -m 0400 "$ATTESTATION_BUNDLE"' in collation
    assert "tools/prepare_p17_evaluation.py" in collation
    assert "--approved-cardplay-eval-data" in collation
    assert "--approved-full-game-eval-data" in collation
    assert 'set(result) != {' in collation
    assert '"p17-private-result-projection-v1"' in collation
    assert 'set(scenario) != {' in collation
    assert 'set(evidence) != {' in collation
    assert 'evidence["published_game_rows"] != 0' in collation
    assert 'signed = json.loads(Path(sys.argv[2])' in collation
    assert 'scenario[key] != signed_scenario.get(key)' in collation
    assert 'evidence["source_game_rows"] != len(signed_games)' in collation
    p17_verification = steps[
        "Verify P17 artifact manifest attestation"
    ]["run"]
    assert 'gh attestation verify "$manifest"' in p17_verification
    assert (
        '--custom-trusted-root "$FORMAL_ATTESTATION_TRUSTED_ROOT"'
        in p17_verification
    )
    assert "FORMAL_ATTESTATION_TRUSTED_ROOT_SHA256" in p17_verification
    assert "--signer-workflow" in p17_verification

    trusted_root_snapshot = (
        "${{ env.FORMAL_AUDIT_ROOT }}/attestation-trusted-root.jsonl"
    )
    assert trusted_root_snapshot in public_upload["with"]["path"]
    assert trusted_root_snapshot in private_upload["with"]["path"]


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


def test_stale_runner_state_and_incomplete_results_fail_before_upload() -> None:
    steps = _steps_by_name()
    snapshot = steps["Snapshot protected evaluation request"]["run"]
    assert 'stale_roots=("$RUNNER_TEMP"/douzero-formal-*)' in snapshot
    assert "${#stale_roots[@]} != 0" in snapshot
    assert "Refusing protected evaluation with a stale formal run root" in snapshot
    assert "rm -rf" not in snapshot

    recheck = steps["Recheck signed P17 manifest closure before upload"]["run"]
    assert "release_status" in recheck
    assert 'release_status[os.environ["EVALUATION_MODE"]] != "eligible"' in recheck
    assert "signed manifest is not release eligible" in recheck

    names = [step["name"] for step in _job()["steps"]]
    gate_index = names.index("Authorize completed formal artifact upload")
    assert gate_index < names.index(
        "Upload public result, detached bundle, and audit material"
    )
    assert gate_index < names.index(
        "Upload private attestation audit without game traces"
    )
    cleanup = steps["Remove protected input snapshots"]["run"]
    assert '[[ ! -e "$run_root" && ! -L "$run_root" ]]' in cleanup


def test_every_third_party_action_is_pinned_to_a_full_commit() -> None:
    actions = [step["uses"] for step in _job()["steps"] if "uses" in step]
    assert actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in actions)
