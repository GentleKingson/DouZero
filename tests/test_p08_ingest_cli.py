"""CLI adapter loading contracts for external human-data ingest."""

from __future__ import annotations

import json

import pytest

import ingest_human_games
from douzero.human_data.ingest import IngestError
from douzero.human_data.synthetic import generate_synthetic_record


NON_CALLABLE_ADAPTER = 42


def function_adapter(raw):
    return generate_synthetic_record(raw["id"], seed=1)


class ZeroArgumentAdapter:
    def __call__(self, raw):
        return generate_synthetic_record(raw["id"], seed=2)


class ConfiguredAdapter:
    def __init__(self, required):
        self.required = required

    def __call__(self, raw):
        return generate_synthetic_record(raw["id"], seed=3)


def wrong_return_adapter(raw):
    return {"not": "a HumanGameRecord"}


def _run_external(tmp_path, adapter_name: str) -> int:
    input_path = tmp_path / f"{adapter_name}.input.jsonl"
    output_path = tmp_path / f"{adapter_name}.output.jsonl"
    input_path.write_text(json.dumps({"id": adapter_name}) + "\n", encoding="utf-8")
    return ingest_human_games.main([
        "--input", str(input_path),
        "--adapter", f"{__name__}.{adapter_name}",
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
    with pytest.raises(IngestError, match="must return HumanGameRecord"):
        _run_external(tmp_path, "wrong_return_adapter")
