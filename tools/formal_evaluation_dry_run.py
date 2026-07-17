#!/usr/bin/env python3
"""Prepare and collate a small, explicitly non-release formal-evaluation smoke.

The protected producer cannot be exercised without its private environment and
self-hosted runner.  This tool builds public synthetic inputs that drive the
same checkpoint snapshot, formal evaluator, deterministic replay, and P17
artifact code paths on a GitHub-hosted runner.  Its output is always marked as
synthetic and release-ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping


SYNTHETIC_DRY_RUN_SCHEMA_VERSION = "p17-synthetic-formal-dry-run-v1"
_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}\Z")
_ROLES = ("landlord", "landlord_up", "landlord_down")
_CANDIDATE = "v2_full_stack"
_BASELINE = "legacy_wp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bundle(checkpoints: Mapping[str, str], digests: Mapping[str, str]) -> dict:
    return {
        "backend": "legacy",
        "bidding_policy": "rule",
        "checkpoints": dict(checkpoints),
        "checkpoint_sha256": dict(digests),
    }


def _generate_cardplay_deals(count: int, *, seed: int) -> tuple[dict, ...]:
    rng = random.Random(seed)
    deals = []
    for _ in range(count):
        deck = list(range(3, 15)) * 4 + [17] * 4 + [20, 30]
        rng.shuffle(deck)
        deals.append({
            "landlord": sorted(deck[:20]),
            "landlord_up": sorted(deck[20:37]),
            "landlord_down": sorted(deck[37:54]),
            "three_landlord_cards": sorted(deck[17:20]),
        })
    return tuple(deals)


def _build_matrices(
    checkpoints: Mapping[str, str], digests: Mapping[str, str]
) -> tuple[dict, dict]:
    from douzero.evaluation.p17 import empty_matrix

    candidate = _bundle(checkpoints, digests)
    baseline = _bundle(checkpoints, digests)
    evaluator = {
        "bundles": {
            _CANDIDATE: candidate,
            _BASELINE: baseline,
        },
        "ablations": {},
    }
    p17 = empty_matrix("Unavailable in the public synthetic dry run")
    for name, bundle in ((_CANDIDATE, candidate), (_BASELINE, baseline)):
        p17["models"][name]["cardplay_only"] = {
            "status": "available",
            "reason": "",
            "bundle": bundle,
        }
    return evaluator, p17


def prepare(output: Path, *, source_sha: str, num_deals: int) -> dict[str, Any]:
    if _SHA.fullmatch(source_sha) is None:
        raise ValueError("--source-sha must be a full lowercase Git object ID")
    if not 1 <= num_deals <= 8:
        raise ValueError("--num-deals must be between 1 and 8")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("dry-run output directory must be new or empty")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)

    import torch

    from douzero.dmc.models import model_dict
    from douzero.env.rules import RuleSet
    from douzero.evaluation.formal_eval_data import write_formal_eval_data
    from douzero.evaluation.p17 import normalize_matrix
    from douzero.evaluation.protocol import OFFICIAL_PERMUTATION_HASHES
    from douzero.evaluation.scenario import (
        canonical_deal_hash,
        canonical_deal_set_id,
    )

    checkpoint_root = output / "checkpoint-source"
    checkpoint_root.mkdir(mode=0o700)
    checkpoints: dict[str, str] = {}
    digests: dict[str, str] = {}
    for index, role in enumerate(_ROLES):
        torch.manual_seed(1700 + index)
        path = checkpoint_root / f"{role}.pt"
        torch.save(model_dict[role]().state_dict(), path)
        path.chmod(0o400)
        checkpoints[role] = str(path)
        digests[role] = _sha256(path)

    evaluator, p17 = _build_matrices(checkpoints, digests)
    evaluator_path = output / "model-matrix.approved.json"
    p17_path = output / "p17-matrix.approved.json"
    _write_json(evaluator_path, evaluator)
    _write_json(p17_path, p17)
    # Eagerly prove the synthetic P17 checkpoints are structurally loadable.
    normalize_matrix(p17)

    ruleset = RuleSet.legacy()
    deals = _generate_cardplay_deals(num_deals, seed=1701)
    data_path = output / "eval-data.json"
    write_formal_eval_data(
        data_path,
        mode="cardplay_only",
        ruleset=ruleset,
        deals=deals,
    )
    deal_set_id = canonical_deal_set_id(
        "cardplay_only",
        ruleset,
        [canonical_deal_hash(deal) for deal in deals],
        seat_permutation_hash=OFFICIAL_PERMUTATION_HASHES["cardplay_only"],
    )
    identity = {
        "schema_version": SYNTHETIC_DRY_RUN_SCHEMA_VERSION,
        "synthetic": True,
        "release_eligible": False,
        "source_sha": source_sha,
        "mode": "cardplay_only",
        "dataset_scope": "public",
        "num_deals": num_deals,
        "candidate": _CANDIDATE,
        "baseline": _BASELINE,
        "deal_set_id": deal_set_id,
        "eval_data_path": str(data_path),
        "eval_data_sha256": _sha256(data_path),
        "model_matrix_path": str(evaluator_path),
        "model_matrix_sha256": _sha256(evaluator_path),
        "p17_matrix_path": str(p17_path),
        "p17_matrix_sha256": _sha256(p17_path),
        "checkpoint_root": str(checkpoint_root),
    }
    identity_path = output / "inputs.json"
    _write_json(identity_path, identity)
    return {**identity, "inputs_path": str(identity_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--source-sha", required=True)
    prepare_parser.add_argument("--num-deals", type=int, default=2)
    args = parser.parse_args(argv)
    payload = prepare(
        args.output,
        source_sha=args.source_sha,
        num_deals=args.num_deals,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
