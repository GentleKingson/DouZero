"""Formal P17 source, result-integrity, and attestation provenance tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from douzero.evaluation import provenance
from douzero.evaluation.p17 import result_readiness
from douzero.evaluation.provenance import (
    AttestationPolicy,
    ProvenanceError,
    VerifiedEvaluationResult,
    attach_result_integrity,
    compute_result_digest,
    inspect_formal_git_checkout,
    inspect_git_checkout,
    verify_github_attested_result,
    verify_result_integrity,
)


SOURCE_SHA = "1" * 40
SIGNER_SHA = "2" * 40


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "formal-evaluator@example.invalid")
    _git(repository, "config", "user.name", "Formal Evaluator")
    source = repository / "source.py"
    source.write_text("print('approved')\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-q", "-m", "approved source")
    return repository


def _result(source_sha: str = SOURCE_SHA) -> dict:
    return {
        "protocol": "p15_paired_v1",
        "ablation": "base",
        "scenario": {"mode": "full_game", "confidence_level": 0.95},
        "metrics": {"paired_estimate_ci": {"estimate": 0.125}},
        "games": [{"deal_id": "000000-deadbeefcafe", "candidate_win": 1.0}],
        "runtime_identity": {"source_git_sha": source_sha},
    }


def _write_attested_result(tmp_path: Path, source_sha: str = SOURCE_SHA):
    artifact = tmp_path / "result.json"
    artifact.write_text(
        json.dumps(attach_result_integrity(_result(source_sha)), sort_keys=True),
        encoding="utf-8",
    )
    bundle = tmp_path / "attestation.jsonl"
    bundle.write_text("{}\n", encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    policy = AttestationPolicy(
        repository="GentleKingson/DouZero",
        signer_workflow=(
            "GentleKingson/DouZero/.github/workflows/formal-evaluation.yml"
        ),
        signer_digest=SIGNER_SHA,
        source_digest=source_sha,
        source_ref="refs/heads/codex/formal-evaluation",
        artifact_sha256=artifact_sha,
    )
    return artifact, bundle, policy


def _gh_output(artifact_sha: str) -> str:
    return json.dumps(
        [
            {
                "attestation": {
                    "mediaType": "application/vnd.dev.sigstore.bundle+json"
                },
                "verificationResult": {
                    "signature": {
                        "certificate": {
                            "issuer": "https://token.actions.githubusercontent.com",
                            "githubWorkflowRepository": "GentleKingson/DouZero",
                            "buildSignerDigest": SIGNER_SHA,
                            "sourceRepositoryDigest": SOURCE_SHA,
                            "sourceRepositoryRef": (
                                "refs/heads/codex/formal-evaluation"
                            ),
                            "runInvocationURI": (
                                "https://github.com/GentleKingson/DouZero/"
                                "actions/runs/12345/attempts/2"
                            ),
                            "runnerEnvironment": "github-hosted",
                        }
                    },
                    "statement": {
                        "subject": [
                            {
                                "name": "evaluation-result.json",
                                "digest": {"sha256": artifact_sha},
                            }
                        ]
                    }
                },
            }
        ]
    )


def test_formal_checkout_ignores_environment_sha_and_records_real_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setenv("DOUZERO_GIT_SHA", "0" * 40)

    identity = inspect_formal_git_checkout(repository)

    assert identity.head_sha == _git(repository, "rev-parse", "HEAD")
    assert identity.head_sha != "0" * 40
    assert identity.head_tree_oid == _git(repository, "rev-parse", "HEAD^{tree}")
    assert len(identity.tracked_tree_sha256) == 64
    assert identity.tracked_file_count == 1
    assert identity.clean is True
    assert identity.to_runtime_fields()["source_git_sha"] == identity.head_sha


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_formal_checkout_rejects_every_dirty_state(
    tmp_path: Path, dirty_kind: str
) -> None:
    repository = _repository(tmp_path)
    if dirty_kind == "tracked":
        (repository / "source.py").write_text("print('modified')\n", encoding="utf-8")
    elif dirty_kind == "staged":
        (repository / "source.py").write_text("print('staged')\n", encoding="utf-8")
        _git(repository, "add", "source.py")
    else:
        (repository / "untracked.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="clean Git working tree"):
        inspect_formal_git_checkout(repository)


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked", "deleted"])
def test_local_checkout_records_dirty_identity_without_an_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty_kind: str,
) -> None:
    repository = _repository(tmp_path)
    clean = inspect_git_checkout(repository, require_clean=False)
    if dirty_kind == "tracked":
        (repository / "source.py").write_text("print('modified')\n", encoding="utf-8")
    elif dirty_kind == "staged":
        (repository / "source.py").write_text("print('staged')\n", encoding="utf-8")
        _git(repository, "add", "source.py")
    elif dirty_kind == "untracked":
        (repository / "untracked.py").write_text("pass\n", encoding="utf-8")
    else:
        (repository / "source.py").unlink()
    monkeypatch.setenv("DOUZERO_GIT_SHA", "0" * 40)

    identity = inspect_git_checkout(repository, require_clean=False)

    assert identity.head_sha == clean.head_sha
    assert identity.head_sha != "0" * 40
    assert identity.clean is False
    if dirty_kind != "untracked":
        assert identity.tracked_tree_sha256 != clean.tracked_tree_sha256


def test_formal_checkout_detects_assume_unchanged_content_tampering(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _git(repository, "update-index", "--assume-unchanged", "source.py")
    (repository / "source.py").write_text("print('hidden change')\n", encoding="utf-8")
    assert _git(repository, "status", "--porcelain") == ""

    with pytest.raises(ProvenanceError, match="bytes do not match"):
        inspect_formal_git_checkout(repository)


def test_independent_tracked_tree_digest_changes_with_committed_bytes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    before = inspect_formal_git_checkout(repository)
    (repository / "source.py").write_text("print('next')\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-q", "-m", "next source")

    after = inspect_formal_git_checkout(repository)

    assert after.head_sha != before.head_sha
    assert after.head_tree_oid != before.head_tree_oid
    assert after.tracked_tree_sha256 != before.tracked_tree_sha256


def test_result_integrity_covers_all_six_sections() -> None:
    protected = attach_result_integrity(_result())
    digest = verify_result_integrity(protected)

    assert digest == compute_result_digest(_result())
    assert protected["result_integrity"]["result_digest"] == digest
    for field in (
        "protocol",
        "ablation",
        "scenario",
        "metrics",
        "games",
        "runtime_identity",
    ):
        tampered = json.loads(json.dumps(protected))
        if field in {"protocol", "ablation"}:
            tampered[field] += "-tampered"
        elif field == "games":
            tampered[field][0]["candidate_win"] = 0.0
        elif field == "runtime_identity":
            tampered[field]["source_git_sha"] = "f" * 40
        else:
            tampered[field]["tampered"] = True
        with pytest.raises(ProvenanceError, match="integrity check failed"):
            verify_result_integrity(tampered)


def test_result_digest_rejects_unsigned_fields_and_non_finite_numbers() -> None:
    result = _result()
    result["unsigned_summary"] = {"eligible": True}
    with pytest.raises(ProvenanceError, match="unsigned result field mismatch"):
        compute_result_digest(result)

    result = _result()
    result["metrics"]["latency"] = float("nan")
    with pytest.raises(ProvenanceError, match="non-finite JSON number"):
        attach_result_integrity(result)


def test_verified_result_wrapper_cannot_be_caller_constructed() -> None:
    with pytest.raises(ProvenanceError, match="attestation verification"):
        VerifiedEvaluationResult(
            result=attach_result_integrity(_result()),
            artifact_sha256="a" * 64,
            result_digest="b" * 64,
            source_git_sha=SOURCE_SHA,
            repository="GentleKingson/DouZero",
            source_ref="refs/heads/main",
            signer_workflow=(
                "GentleKingson/DouZero/.github/workflows/formal-evaluation.yml"
            ),
            signer_digest=SIGNER_SHA,
            workflow_run_url=(
                "https://github.com/GentleKingson/DouZero/"
                "actions/runs/1/attempts/1"
            ),
            runner_environment="github-hosted",
            attestation_verifications=({},),
        )


def test_public_readiness_api_cannot_accept_a_preverified_wrapper() -> None:
    with pytest.raises(TypeError, match="verified_result"):
        result_readiness(
            attach_result_integrity(_result()),
            mode="full_game",
            verified_result=object(),
        )


def test_detached_github_attestation_verifies_exact_artifact_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, bundle, policy = _write_attested_result(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        snapshot = Path(command[3])
        assert (
            hashlib.sha256(snapshot.read_bytes()).hexdigest()
            == policy.artifact_sha256
        )
        return subprocess.CompletedProcess(
            command, 0, _gh_output(policy.artifact_sha256), ""
        )

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    verified = verify_github_attested_result(
        artifact, bundle, policy, gh_executable="gh-test"
    )

    assert verified.artifact_sha256 == policy.artifact_sha256
    assert verified.result_digest == protected_digest(artifact)
    assert verified.source_git_sha == SOURCE_SHA
    assert verified.signer_workflow == policy.signer_workflow
    assert verified.signer_digest == SIGNER_SHA
    assert verified.workflow_run_url.endswith("/actions/runs/12345/attempts/2")
    assert verified.runner_environment == "github-hosted"
    assert verified.result["games"] == _result()["games"]
    command = captured["command"]
    assert command[:3] == ["gh-test", "attestation", "verify"]
    assert command[4:] == [
        "--bundle",
        str(bundle),
        "--repo",
        policy.repository,
        "--signer-workflow",
        policy.signer_workflow,
        "--signer-digest",
        SIGNER_SHA,
        "--source-digest",
        SOURCE_SHA,
        "--source-ref",
        policy.source_ref,
        "--format",
        "json",
    ]


def test_detached_attestation_snapshots_digest_bound_offline_trusted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, bundle, base_policy = _write_attested_result(tmp_path)
    trusted_root = tmp_path / "trusted-root.jsonl"
    trusted_root.write_text('{"mediaType":"trusted-root"}\n', encoding="utf-8")
    root_digest = hashlib.sha256(trusted_root.read_bytes()).hexdigest()
    policy = AttestationPolicy(
        repository=base_policy.repository,
        signer_workflow=base_policy.signer_workflow,
        signer_digest=base_policy.signer_digest,
        source_digest=base_policy.source_digest,
        source_ref=base_policy.source_ref,
        artifact_sha256=base_policy.artifact_sha256,
        trusted_root_path=trusted_root,
        trusted_root_sha256=root_digest,
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        root_index = command.index("--custom-trusted-root") + 1
        snapshot = Path(command[root_index])
        assert snapshot != trusted_root
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == root_digest
        return subprocess.CompletedProcess(
            command, 0, _gh_output(policy.artifact_sha256), ""
        )

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    verify_github_attested_result(artifact, bundle, policy)
    assert "--custom-trusted-root" in captured["command"]

    trusted_root.write_text("substituted\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="trusted root SHA-256"):
        verify_github_attested_result(artifact, bundle, policy)


def protected_digest(artifact: Path) -> str:
    return verify_result_integrity(json.loads(artifact.read_text(encoding="utf-8")))


def test_attestation_rejects_artifact_sha_before_invoking_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, bundle, policy = _write_attested_result(tmp_path)
    policy = AttestationPolicy(
        repository=policy.repository,
        signer_workflow=policy.signer_workflow,
        signer_digest=policy.signer_digest,
        source_digest=policy.source_digest,
        source_ref=policy.source_ref,
        artifact_sha256="f" * 64,
    )
    invoked = False

    def fake_run(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("gh must not run")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    with pytest.raises(ProvenanceError, match="artifact SHA-256"):
        verify_github_attested_result(artifact, bundle, policy)
    assert invoked is False


def test_attestation_rejects_runtime_source_sha_mismatch(tmp_path: Path) -> None:
    artifact, bundle, policy = _write_attested_result(tmp_path, source_sha="3" * 40)
    mismatched_policy = AttestationPolicy(
        repository=policy.repository,
        signer_workflow=policy.signer_workflow,
        signer_digest=policy.signer_digest,
        source_digest=SOURCE_SHA,
        source_ref=policy.source_ref,
        artifact_sha256=policy.artifact_sha256,
    )

    with pytest.raises(ProvenanceError, match="runtime source_git_sha"):
        verify_github_attested_result(artifact, bundle, mismatched_policy)


def test_attestation_rejects_gh_subject_that_does_not_match_exact_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, bundle, policy = _write_attested_result(tmp_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, _gh_output("0" * 64), "")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    with pytest.raises(ProvenanceError, match="exact artifact SHA-256"):
        verify_github_attested_result(artifact, bundle, policy)


def test_attestation_policy_requires_all_exact_identities() -> None:
    with pytest.raises(ProvenanceError, match="owner/name"):
        AttestationPolicy(
            repository="GentleKingson",
            signer_workflow="GentleKingson/DouZero/.github/workflows/formal.yml",
            signer_digest=SIGNER_SHA,
            source_digest=SOURCE_SHA,
            source_ref="refs/heads/main",
            artifact_sha256="a" * 64,
        )
