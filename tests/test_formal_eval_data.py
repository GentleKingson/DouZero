"""Security and schema tests for formal evaluation deal sets."""

from __future__ import annotations

import copy
import json
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_paired
from douzero.env.rules import RuleSet
from douzero.evaluation.formal_eval_data import (
    FORMAL_EVAL_DATA_SCHEMA_VERSION,
    FormalEvalDataError,
    load_formal_eval_data,
    write_formal_eval_data,
)
from douzero.evaluation.p17 import empty_matrix
from evaluate_paired import generate_deals
from tools import prepare_p17_evaluation


ROOT = Path(__file__).resolve().parents[1]


class _MaliciousPickle:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {shlex.quote(str(self.marker))}",)


def _write_malicious_pickle(path: Path, marker: Path) -> None:
    # Protocol 0 is valid UTF-8 too, so this exercises JSON parsing rather than
    # relying only on binary decoding failure.
    path.write_bytes(pickle.dumps(_MaliciousPickle(marker), protocol=0))


def _valid_file(tmp_path: Path, mode: str) -> tuple[Path, RuleSet, list[dict]]:
    ruleset = RuleSet.legacy() if mode == "cardplay_only" else RuleSet.standard()
    deals = list(generate_deals(mode, 2, 1701, ruleset))
    path = tmp_path / f"{mode}.json"
    write_formal_eval_data(path, mode=mode, ruleset=ruleset, deals=deals)
    return path, ruleset, deals


@pytest.mark.parametrize("mode", ["cardplay_only", "full_game"])
def test_formal_json_round_trip_is_deterministic_and_exact(tmp_path, mode):
    path, ruleset, deals = _valid_file(tmp_path, mode)
    second = tmp_path / f"{mode}-second.json"
    write_formal_eval_data(second, mode=mode, ruleset=ruleset, deals=deals)

    assert path.read_bytes() == second.read_bytes()
    assert load_formal_eval_data(
        path, expected_mode=mode, expected_ruleset=ruleset
    ) == deals
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "mode", "ruleset", "deals"}
    assert payload["schema_version"] == FORMAL_EVAL_DATA_SCHEMA_VERSION
    expected_deal_keys = (
        {"landlord", "landlord_up", "landlord_down", "three_landlord_cards"}
        if mode == "cardplay_only"
        else {
            "format_version",
            "schema_version",
            "ruleset_id",
            "ruleset_version",
            "ruleset_hash",
            "deck",
            "first_bidder",
            "bidding_order",
            "bidding_script",
        }
    )
    assert all(set(deal) == expected_deal_keys for deal in payload["deals"])


def test_formal_loader_rejects_duplicate_keys(tmp_path):
    path, ruleset, _deals = _valid_file(tmp_path, "cardplay_only")
    raw = path.read_text(encoding="utf-8").replace(
        '"mode":"cardplay_only"',
        '"mode":"cardplay_only","mode":"cardplay_only"',
        1,
    )
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(FormalEvalDataError, match="duplicate JSON object key"):
        load_formal_eval_data(
            path,
            expected_mode="cardplay_only",
            expected_ruleset=ruleset,
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_formal_loader_rejects_non_finite_json_numbers(tmp_path, constant):
    path, ruleset, _deals = _valid_file(tmp_path, "cardplay_only")
    raw = path.read_text(encoding="utf-8").replace(
        '"schema_version":"douzero-formal-eval-data-v1"',
        f'"poison":{constant},"schema_version":"douzero-formal-eval-data-v1"',
        1,
    )
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(FormalEvalDataError, match="non-finite JSON number"):
        load_formal_eval_data(
            path,
            expected_mode="cardplay_only",
            expected_ruleset=ruleset,
        )


@pytest.mark.parametrize("location", ["top", "deal", "ruleset"])
def test_formal_loader_rejects_extra_keys_at_every_object_level(
    tmp_path, location
):
    path, ruleset, _deals = _valid_file(tmp_path, "cardplay_only")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if location == "top":
        payload["extra"] = None
    elif location == "deal":
        payload["deals"][0]["extra"] = None
    else:
        payload["ruleset"]["extra"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalEvalDataError, match="must contain exactly"):
        load_formal_eval_data(
            path,
            expected_mode="cardplay_only",
            expected_ruleset=ruleset,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="future"), "schema_version"),
        (lambda payload: payload.update(mode="full_game"), "mode"),
        (
            lambda payload: payload["ruleset"].update(ruleset_hash="0" * 64),
            "ruleset identity",
        ),
    ],
)
def test_formal_loader_rejects_schema_mode_and_ruleset_mismatch(
    tmp_path, mutation, message
):
    path, ruleset, _deals = _valid_file(tmp_path, "cardplay_only")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalEvalDataError, match=message):
        load_formal_eval_data(
            path,
            expected_mode="cardplay_only",
            expected_ruleset=ruleset,
        )


@pytest.mark.parametrize("replacement", [3.0, True, "3"])
def test_formal_loader_rejects_noncanonical_card_types(tmp_path, replacement):
    path, ruleset, _deals = _valid_file(tmp_path, "full_game")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["deals"][0]["deck"][0] = replacement
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalEvalDataError, match="JSON integer"):
        load_formal_eval_data(
            path,
            expected_mode="full_game",
            expected_ruleset=ruleset,
        )


def test_formal_loader_rejects_unsorted_legacy_hands(tmp_path):
    path, ruleset, _deals = _valid_file(tmp_path, "cardplay_only")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["deals"][0]["landlord"].reverse()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalEvalDataError, match="canonical sorted order"):
        load_formal_eval_data(
            path,
            expected_mode="cardplay_only",
            expected_ruleset=ruleset,
        )


def test_malicious_pickle_is_never_executed_by_formal_loader(tmp_path):
    marker = tmp_path / "executed"
    disguised = tmp_path / "deals.json"
    _write_malicious_pickle(disguised, marker)

    with pytest.raises(FormalEvalDataError, match="invalid formal eval data JSON"):
        load_formal_eval_data(
            disguised,
            expected_mode="cardplay_only",
            expected_ruleset=RuleSet.legacy(),
        )
    assert not marker.exists()


def test_evaluate_paired_formal_cli_refuses_pickle_without_unpickling(
    tmp_path, monkeypatch
):
    marker = tmp_path / "executed"
    malicious = tmp_path / "deals.pkl"
    _write_malicious_pickle(malicious, marker)
    approved_sha = "a" * 40
    monkeypatch.setattr(
        evaluate_paired,
        "inspect_formal_git_checkout",
        lambda: SimpleNamespace(head_sha=approved_sha),
    )

    with pytest.raises(FormalEvalDataError, match="pickle files are forbidden"):
        evaluate_paired.main([
            "--formal-release",
            "--expected-source-git-sha",
            approved_sha,
            "--model-matrix",
            str(tmp_path / "not-reached.json"),
            "--eval-data",
            str(malicious),
        ])
    assert not marker.exists()


def test_p17_collator_refuses_pickle_without_unpickling(tmp_path):
    marker = tmp_path / "executed"
    malicious = tmp_path / "deals.pkl"
    _write_malicious_pickle(malicious, marker)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(empty_matrix()), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        prepare_p17_evaluation.main([
            "--matrix",
            str(matrix),
            "--approved-cardplay-eval-data",
            str(malicious),
        ])
    assert exc_info.value.code == 2
    assert not marker.exists()


@pytest.mark.parametrize(
    ("ruleset_name", "mode"),
    [("legacy", "cardplay_only"), ("standard", "full_game")],
)
def test_generate_eval_data_cli_writes_loadable_formal_json(
    tmp_path, ruleset_name, mode
):
    output = tmp_path / f"generated-{ruleset_name}"
    completed = subprocess.run(
        [
            sys.executable,
            "generate_eval_data.py",
            "--output",
            str(output),
            "--num_games",
            "2",
            "--ruleset",
            ruleset_name,
            "--output-format",
            "formal-json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    ruleset = RuleSet.legacy() if ruleset_name == "legacy" else RuleSet.standard()
    loaded = load_formal_eval_data(
        Path(f"{output}.json"),
        expected_mode=mode,
        expected_ruleset=ruleset,
    )
    assert len(loaded) == 2


def test_writer_rejects_duplicate_deals(tmp_path):
    ruleset = RuleSet.legacy()
    deal = generate_deals("cardplay_only", 1, 42, ruleset)[0]
    with pytest.raises(FormalEvalDataError, match="must be unique"):
        write_formal_eval_data(
            tmp_path / "duplicate.json",
            mode="cardplay_only",
            ruleset=ruleset,
            deals=[copy.deepcopy(deal), copy.deepcopy(deal)],
        )
