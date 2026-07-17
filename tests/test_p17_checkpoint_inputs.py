"""Checkpoint digest binding, verified loading, and snapshot regression tests."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from douzero.env.rules import RuleSet
from douzero.evaluation.agents import BundleFactory
from douzero.evaluation.checkpoint_inputs import (
    CheckpointIdentityError,
    checkpoint_sha256,
    load_verified_checkpoint,
    require_explicit_matrix_checkpoint_digests,
    snapshot_model_matrix_file,
)
from douzero.evaluation.p17 import (
    P17MatrixError,
    empty_matrix,
    normalize_matrix,
    write_p17_artifacts,
)
from douzero.evaluation.provenance import AttestationPolicy, AttestedEvaluationInput
from douzero.evaluation.scenario import BundleSpec, bundle_from_dict
from evaluate_paired import _load_matrix


ROLES = ("landlord", "landlord_up", "landlord_down")


def _checkpoint(tmp_path: Path, name: str = "model.pt", data: bytes = b"approved"):
    path = tmp_path / name
    path.write_bytes(data)
    return path, checkpoint_sha256(path)


def _bundle_payload(path: Path, digest: str) -> dict:
    return {
        "backend": "legacy",
        "checkpoints": {role: str(path) for role in ROLES},
        "checkpoint_sha256": {role: digest for role in ROLES},
    }


def test_local_bundle_auto_digest_is_captured_once_not_reread(tmp_path: Path) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    bundle = BundleSpec(
        name="local",
        backend="legacy",
        checkpoints={role: str(checkpoint) for role in ROLES},
    )
    assert bundle.checkpoint_digests_explicit is False
    assert set(bundle.checkpoint_sha256.values()) == {approved}

    checkpoint.write_bytes(b"replaced-after-construction")

    identities = bundle.to_dict()["checkpoint_identities"]
    assert set(identities["roles"].values()) == {approved}
    with pytest.raises(CheckpointIdentityError, match="SHA-256 mismatch"):
        BundleFactory(RuleSet.legacy())._load_model_agent(bundle, "landlord")


def test_formal_bundle_requires_complete_predeclared_digests(tmp_path: Path) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    paths = {role: str(checkpoint) for role in ROLES}

    with pytest.raises(ValueError, match="explicit predeclared checkpoint"):
        bundle_from_dict(
            {"name": "formal", "backend": "legacy", "checkpoints": paths},
            require_checkpoint_digests=True,
        )
    with pytest.raises(ValueError, match="must cover every role"):
        BundleSpec(
            name="partial",
            backend="legacy",
            checkpoints=paths,
            checkpoint_sha256={"landlord": approved},
        )

    formal = bundle_from_dict(
        {
            "name": "formal",
            "backend": "legacy",
            "checkpoints": paths,
            "checkpoint_sha256": {role: approved for role in ROLES},
        },
        require_checkpoint_digests=True,
    )
    assert formal.checkpoint_digests_explicit is True
    assert formal.to_dict()["checkpoint_identities"]["explicitly_predeclared"] is True


def test_bidding_and_belief_sidecars_require_their_own_digests(
    tmp_path: Path,
) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    paths = {role: str(checkpoint) for role in ROLES}
    with pytest.raises(ValueError, match="must cover every role"):
        BundleSpec(
            name="missing-sidecar-digests",
            backend="v2",
            checkpoints=paths,
            checkpoint_sha256={role: approved for role in ROLES},
            belief_checkpoint=str(checkpoint),
            bidding_policy="learned",
            bidding_checkpoint=str(checkpoint),
        )

    bundle = BundleSpec(
        name="all-sidecars-bound",
        backend="v2",
        checkpoints=paths,
        checkpoint_sha256={role: approved for role in ROLES},
        belief_checkpoint=str(checkpoint),
        belief_checkpoint_sha256=approved,
        bidding_policy="learned",
        bidding_checkpoint=str(checkpoint),
        bidding_checkpoint_sha256=approved,
    )
    assert bundle.checkpoint_digests_explicit is True
    assert bundle.checkpoint_identities()["belief"] == approved
    assert bundle.checkpoint_identities()["bidding"] == approved


def test_verified_loader_rejects_wrong_digest_before_load(tmp_path: Path) -> None:
    checkpoint, _approved = _checkpoint(tmp_path)
    called = False

    def loader(_path: str):
        nonlocal called
        called = True
        return object()

    with pytest.raises(CheckpointIdentityError, match="SHA-256 mismatch"):
        load_verified_checkpoint(
            checkpoint,
            "0" * 64,
            loader,
            label="candidate.landlord",
        )
    assert called is False


def test_verified_loader_detects_checkpoint_changed_during_load(tmp_path: Path) -> None:
    checkpoint, approved = _checkpoint(tmp_path)

    def mutating_loader(path: str):
        Path(path).write_bytes(b"substituted-during-load")
        return object()

    with pytest.raises(CheckpointIdentityError, match="SHA-256 mismatch"):
        load_verified_checkpoint(
            checkpoint,
            approved,
            mutating_loader,
            label="candidate.landlord",
        )


def test_evaluator_matrix_snapshot_rewrites_every_role_to_read_only_bytes(
    tmp_path: Path,
) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    matrix = {
        "bundles": {"candidate": _bundle_payload(checkpoint, approved)},
        "ablations": {},
    }
    source = tmp_path / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")
    output = tmp_path / "private" / "matrix.json"

    snapshot_model_matrix_file(
        source,
        output,
        tmp_path / "private" / "checkpoints",
        kind="evaluator",
    )

    rewritten = json.loads(output.read_text(encoding="utf-8"))
    paths = set(rewritten["bundles"]["candidate"]["checkpoints"].values())
    assert len(paths) == 1
    snapshot = Path(paths.pop())
    assert snapshot != checkpoint
    assert snapshot.read_bytes() == b"approved"
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400
    assert rewritten["bundles"]["candidate"]["checkpoint_sha256"] == {
        role: approved for role in ROLES
    }
    checkpoint.write_bytes(b"later-source-mutation")
    assert checkpoint_sha256(snapshot) == approved
    _load_matrix(str(output), require_checkpoint_digests=True)


def test_snapshot_cli_runs_as_an_isolated_installed_module(tmp_path: Path) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    matrix = {
        "bundles": {"candidate": _bundle_payload(checkpoint, approved)},
        "ablations": {},
    }
    source = tmp_path / "matrix.json"
    output = tmp_path / "snapshot" / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "douzero.evaluation.snapshot_cli",
            "--matrix",
            str(source),
            "--kind",
            "evaluator",
            "--source-root",
            str(tmp_path),
            "--checkpoint-dir",
            str(tmp_path / "snapshot" / "checkpoints"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"snapshot_matrix": str(output.resolve())}
    assert output.is_file()


def test_snapshot_rejects_wrong_declared_digest_without_writing_matrix(
    tmp_path: Path,
) -> None:
    checkpoint, _approved = _checkpoint(tmp_path)
    matrix = {
        "bundles": {"candidate": _bundle_payload(checkpoint, "0" * 64)},
        "ablations": {},
    }
    source = tmp_path / "matrix.json"
    output = tmp_path / "snapshot" / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(CheckpointIdentityError, match="snapshot SHA-256 mismatch"):
        snapshot_model_matrix_file(
            source,
            output,
            tmp_path / "snapshot" / "checkpoints",
            kind="evaluator",
        )
    assert not output.exists()
    assert not list((tmp_path / "snapshot" / "checkpoints").glob("*.checkpoint"))


def test_snapshot_rejects_checkpoint_outside_approved_source_root(
    tmp_path: Path,
) -> None:
    approved_root = tmp_path / "approved-checkpoints"
    approved_root.mkdir()
    checkpoint, approved = _checkpoint(tmp_path, name="outside.pt")
    matrix = {
        "bundles": {"candidate": _bundle_payload(checkpoint, approved)},
        "ablations": {},
    }
    source = tmp_path / "matrix.json"
    output = tmp_path / "snapshot" / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(CheckpointIdentityError, match="outside the approved root"):
        snapshot_model_matrix_file(
            source,
            output,
            tmp_path / "snapshot" / "checkpoints",
            kind="evaluator",
            source_root=approved_root,
        )


def test_snapshot_source_root_rejects_symlinks(tmp_path: Path) -> None:
    approved_root = tmp_path / "approved-checkpoints"
    approved_root.mkdir()
    checkpoint, approved = _checkpoint(approved_root)
    linked_root = tmp_path / "linked-checkpoints"
    linked_root.symlink_to(approved_root, target_is_directory=True)
    matrix = {
        "bundles": {"candidate": _bundle_payload(checkpoint, approved)},
        "ablations": {},
    }
    source = tmp_path / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(CheckpointIdentityError, match="non-symlink directory"):
        snapshot_model_matrix_file(
            source,
            tmp_path / "snapshot" / "matrix.json",
            tmp_path / "snapshot" / "checkpoints",
            kind="evaluator",
            source_root=linked_root,
        )


def test_p17_matrix_snapshot_rewrites_available_bundle_paths(tmp_path: Path) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    matrix = {
        "schema_version": "p17-model-matrix-v1",
        "models": {
            "model": {
                "cardplay_only": {
                    "status": "available",
                    "reason": "",
                    "bundle": _bundle_payload(checkpoint, approved),
                }
            }
        },
        "ablations": {},
    }
    source = tmp_path / "p17.json"
    output = tmp_path / "p17-snapshot" / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")

    snapshot_model_matrix_file(
        source,
        output,
        tmp_path / "p17-snapshot" / "checkpoints",
        kind="p17",
    )

    rewritten = json.loads(output.read_text(encoding="utf-8"))
    bundle = rewritten["models"]["model"]["cardplay_only"]["bundle"]
    assert all(
        Path(path).parent.name == "checkpoints"
        for path in bundle["checkpoints"].values()
    )
    require_explicit_matrix_checkpoint_digests(rewritten, kind="p17")


def test_matrix_explicit_digest_validator_rejects_auto_computed_identity(
    tmp_path: Path,
) -> None:
    checkpoint, _approved = _checkpoint(tmp_path)
    matrix = {
        "bundles": {
            "candidate": {
                "backend": "legacy",
                "checkpoints": {role: str(checkpoint) for role in ROLES},
            }
        },
        "ablations": {},
    }
    with pytest.raises(CheckpointIdentityError, match="predeclared checkpoint"):
        require_explicit_matrix_checkpoint_digests(matrix, kind="evaluator")

    source = tmp_path / "matrix.json"
    source.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit predeclared checkpoint"):
        _load_matrix(str(source), require_checkpoint_digests=True)


def test_formal_p17_writer_cannot_bypass_predeclared_checkpoint_digests(
    tmp_path: Path,
) -> None:
    checkpoint, _approved = _checkpoint(tmp_path)
    matrix = empty_matrix()
    matrix["models"]["legacy_wp"]["cardplay_only"] = {
        "status": "available",
        "reason": "",
        "bundle": {
            "name": "legacy_wp",
            "backend": "legacy",
            "checkpoints": {role: str(checkpoint) for role in ROLES},
        },
    }
    attested = AttestedEvaluationInput(
        result_path=tmp_path / "missing-result.json",
        bundle_path=tmp_path / "missing-attestation.jsonl",
        policy=AttestationPolicy(
            repository="owner/repository",
            signer_workflow="owner/repository/.github/workflows/formal.yml",
            signer_digest="a" * 40,
            source_digest="b" * 40,
            source_ref="refs/heads/main",
            artifact_sha256="c" * 64,
        ),
    )

    with pytest.raises(P17MatrixError, match="predeclared checkpoint"):
        write_p17_artifacts(
            tmp_path / "formal-output",
            matrix=matrix,
            cardplay_result=attested,
        )


def test_p17_checkpoint_validation_detects_mutation_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, approved = _checkpoint(tmp_path)
    matrix = empty_matrix()
    matrix["models"]["legacy_wp"]["cardplay_only"] = {
        "status": "available",
        "reason": "",
        "bundle": {
            "name": "legacy_wp",
            **_bundle_payload(checkpoint, approved),
        },
    }

    def mutate_checkpoint(path: str, _expected_state: dict) -> None:
        Path(path).write_bytes(b"substituted-during-p17-validation")

    monkeypatch.setattr(
        "douzero.checkpoint.load_position_state_dict_strict",
        mutate_checkpoint,
    )
    with pytest.raises(
        P17MatrixError,
        match="checkpoint identity validation failed.*CheckpointIdentityError",
    ):
        normalize_matrix(matrix)


def test_workflow_snapshots_evaluator_and_p17_checkpoint_paths() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "formal-evaluation.yml"
    ).read_text(encoding="utf-8")
    assert workflow.count("python -I -B -m douzero.evaluation.snapshot_cli") == 2
    assert "--kind evaluator" in workflow
    assert "--kind p17" in workflow
    assert '--source-root "$model_checkpoint_root"' in workflow
    assert '--source-root "$p17_checkpoint_root"' in workflow
    assert (
        "FORMAL_MODEL_MATRIX=$run_root/evaluator-snapshot/model-matrix.json"
        in workflow
    )
    assert (
        "FORMAL_P17_MATRIX=$run_root/p17-snapshot/p17-matrix.json" in workflow
    )
    assert "SNAPSHOT_MODEL_MATRIX_SHA256" in workflow
    assert "SNAPSHOT_P17_MATRIX_SHA256" in workflow
