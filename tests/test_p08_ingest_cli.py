"""CLI adapter loading contracts for external human-data ingest."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import ingest_human_games
from douzero.human_data import (
    AttestedAdapterRecord,
    ExternalGameIdentity,
    pseudonymize_external_game_id,
    read_jsonl,
)
from douzero.human_data.ingest import IngestError
from douzero.human_data.synthetic import generate_synthetic_record


NON_CALLABLE_ADAPTER = 42


def _attested_record(raw, pseudonymizer, *, seed):
    identity = pseudonymizer.pseudonymize(raw["id"])
    record = replace(
        generate_synthetic_record(raw["id"], seed=seed),
        game_id=identity.game_id,
    )
    return AttestedAdapterRecord(record=record, identity=identity)


def function_adapter(raw, *, pseudonymizer):
    return _attested_record(raw, pseudonymizer, seed=1)


class ZeroArgumentAdapter:
    def __call__(self, raw, *, pseudonymizer):
        return _attested_record(raw, pseudonymizer, seed=2)


class ConfiguredAdapter:
    def __init__(self, required):
        self.required = required

    def __call__(self, raw, *, pseudonymizer):
        return _attested_record(raw, pseudonymizer, seed=3)


def wrong_return_adapter(raw, *, pseudonymizer):
    return {"not": "a HumanGameRecord"}


def unsalted_adapter(raw, *, pseudonymizer):
    record = generate_synthetic_record(raw["id"], seed=4)
    identity = ExternalGameIdentity(
        game_id=record.game_id,
        attestation="0" * 64,
    )
    return AttestedAdapterRecord(record=record, identity=identity)


def redaction_probe_adapter(raw, *, pseudonymizer):
    raise ValueError(raw["id"])


def _run_external(tmp_path, adapter_name: str) -> int:
    input_path = tmp_path / f"{adapter_name}.input.jsonl"
    output_path = tmp_path / f"{adapter_name}.output.jsonl"
    input_path.write_text(json.dumps({"id": adapter_name}) + "\n", encoding="utf-8")
    key_path = tmp_path / f"{adapter_name}.key"
    key_path.write_bytes(b"k" * 32)
    return ingest_human_games.main([
        "--input", str(input_path),
        "--adapter", f"{__name__}.{adapter_name}",
        "--hmac-key-file", str(key_path),
        "--output", str(output_path),
    ])


def test_function_adapter_cli(tmp_path):
    assert _run_external(tmp_path, "function_adapter") == 0


def test_zero_argument_class_adapter_cli(tmp_path):
    assert _run_external(tmp_path, "ZeroArgumentAdapter") == 0


def test_configured_class_adapter_rejected():
    with pytest.raises(SystemExit, match="zero-argument constructor"):
        ingest_human_games._load_adapter(f"{__name__}.ConfiguredAdapter")


def test_non_callable_adapter_rejected():
    with pytest.raises(SystemExit, match="not callable"):
        ingest_human_games._load_adapter(f"{__name__}.NON_CALLABLE_ADAPTER")


def test_adapter_wrong_return_type_rejected_by_cli(tmp_path):
    with pytest.raises(IngestError, match="must return AttestedAdapterRecord"):
        _run_external(tmp_path, "wrong_return_adapter")


def test_external_cli_fails_closed_without_project_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DOUZERO_HUMAN_DATA_HMAC_KEY_FILE", raising=False)
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id":"external-1"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="external ingest requires"):
        ingest_human_games.main([
            "--input", str(input_path),
            "--adapter", f"{__name__}.function_adapter",
            "--output", str(tmp_path / "output.jsonl"),
        ])
    assert not (tmp_path / "output.jsonl").exists()


def test_unsalted_regex_shaped_adapter_id_is_rejected(tmp_path):
    with pytest.raises(IngestError, match="not bound"):
        _run_external(tmp_path, "unsalted_adapter")


def test_external_cli_valid_keyed_ingest_and_no_key_leakage(tmp_path):
    external_id = "authorized-game-17"
    project_key = b"secret-project-key-material-32bytes!"
    input_path = tmp_path / "authorized.jsonl"
    output_path = tmp_path / "canonical.jsonl"
    key_path = tmp_path / "project.key"
    input_path.write_text(json.dumps({"id": external_id}) + "\n", encoding="utf-8")
    key_path.write_bytes(project_key)

    assert ingest_human_games.main([
        "--input", str(input_path),
        "--adapter", f"{__name__}.function_adapter",
        "--hmac-key-file", str(key_path),
        "--output", str(output_path),
    ]) == 0
    records = list(read_jsonl(str(output_path)))
    assert records[0].game_id == pseudonymize_external_game_id(
        external_id, project_key=project_key
    )
    serialized = output_path.read_text(encoding="utf-8")
    assert external_id not in serialized
    assert project_key.decode("ascii") not in serialized


def test_adapter_errors_redact_raw_ids_keys_and_key_paths(tmp_path):
    sensitive = "S" * 40
    input_path = tmp_path / "sensitive.jsonl"
    output_path = tmp_path / "output.jsonl"
    key_path = tmp_path / "sensitive-project-key"
    input_path.write_text(json.dumps({"id": sensitive}) + "\n", encoding="utf-8")
    key_path.write_bytes(sensitive.encode("ascii"))
    with pytest.raises(IngestError) as captured:
        ingest_human_games.main([
            "--input", str(input_path),
            "--adapter", f"{__name__}.redaction_probe_adapter",
            "--hmac-key-file", str(key_path),
            "--output", str(output_path),
        ])
    message = str(captured.value)
    assert sensitive not in message
    assert str(key_path) not in message


def test_synthetic_cli_remains_keyless(monkeypatch, tmp_path):
    monkeypatch.delenv("DOUZERO_HUMAN_DATA_HMAC_KEY_FILE", raising=False)
    output = tmp_path / "synthetic.jsonl"
    assert ingest_human_games.main([
        "--synthetic", "--num_synthetic", "1", "--output", str(output)
    ]) == 0
    assert len(list(read_jsonl(str(output)))) == 1
