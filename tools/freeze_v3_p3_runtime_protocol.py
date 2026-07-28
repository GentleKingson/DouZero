#!/usr/bin/env python3
"""Freeze the P3 matched runtime decision protocol before measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if not sys.path or sys.path[0] != str(_ROOT):
    sys.path.insert(0, str(_ROOT))

from douzero._version import git_sha
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import build_pilot_resolved_config
from douzero.v3_hybrid.runtime import (
    V3_H7_CHECKPOINT_FORMAT,
    V3_H7_REPLAY_PROTOCOL,
    V3_H7_REQUEST_PROTOCOL,
    V3_H7_RUNTIME_VERSION,
)
from douzero.v3_hybrid.runtime_decision import P3RuntimeProtocol

P3_RUNNER_VERSION = "v3-p3-matched-runtime-runner-v4"
_SCALE_FIELDS = (
    "hidden_size",
    "history_encoder",
    "history_layers",
    "history_heads",
    "shared_fusion_layers",
    "landlord_adapter_layers",
    "farmer_adapter_layers",
    "farmer_channel_gate",
    "farmer_channel_gate_reduction",
)


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")).hexdigest()


def _source_tree() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise SystemExit("P3 source tree identity is invalid")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--measurement-seconds", type=float, default=300.0)
    args = parser.parse_args()

    base = load_formal_config(args.base_config)
    full = load_formal_config(args.full_config)
    if base.variant != "v3_admc" or full.variant != "v3_full_hybrid":
        raise SystemExit("P3 requires v3_admc and v3_full_hybrid configs")
    if base.ruleset["id"] != "legacy" or full.ruleset["id"] != "legacy":
        raise SystemExit("P3 first decision protocol is legacy-only")
    if base.runtime.batch_size != full.runtime.batch_size:
        raise SystemExit("P3 matched configs require the same batch size")
    if base.seeds.derivation != full.seeds.derivation:
        raise SystemExit("P3 matched configs require the same deal seed derivation")
    if (
        base.runtime.checkpoint_cadence_updates
        != full.runtime.checkpoint_cadence_updates
    ):
        raise SystemExit("P3 matched configs require the same checkpoint cadence")
    base_model = dict(base.model["config"])
    full_model = dict(full.model["config"])
    base_scale = {name: base_model[name] for name in _SCALE_FIELDS}
    full_scale = {name: full_model[name] for name in _SCALE_FIELDS}
    if base_scale != full_scale:
        raise SystemExit("P3 matched configs require the same backbone scale")

    base_resolved = build_pilot_resolved_config(base)
    full_resolved = build_pilot_resolved_config(full)
    oracle_schedule = full_resolved.learner.base.base.base.schedule
    guided_update = oracle_schedule.warmup_updates
    if (
        guided_update < 1
        or oracle_schedule.at(guided_update).phase != "guided"
    ):
        raise SystemExit("P3 full-hybrid config has no guided phase boundary")
    protocol = P3RuntimeProtocol(
        source_git_sha=git_sha(),
        source_tree=_source_tree(),
        image_digest=args.image_digest,
        base_config_hash=base.identity_dict()["config_sha256"],
        full_config_hash=full.identity_dict()["config_sha256"],
        shared_scale_hash=_hash(base_scale),
        model_identity_hashes={
            "base": base_resolved.model.stable_hash(),
            "full_hybrid": full_resolved.model.stable_hash(),
        },
        trainer_identity_hash=_hash({
            "runner": P3_RUNNER_VERSION,
            "runtime": V3_H7_RUNTIME_VERSION,
            "checkpoint": V3_H7_CHECKPOINT_FORMAT,
            "request": V3_H7_REQUEST_PROTOCOL,
        }),
        replay_protocol_hash=_hash({
            "runtime_replay": V3_H7_REPLAY_PROTOCOL,
            "full_single_replay": (
                "v3-p3-formal-deals-guided-phase-incomplete-pair-skip-v2"
            ),
        }),
        gpu=args.gpu,
        driver=args.driver,
        pytorch=args.pytorch,
        cuda=args.cuda,
        cpu=args.cpu,
        warmup_seconds=args.warmup_seconds,
        measurement_seconds=args.measurement_seconds,
        checkpoint_cadence_updates=base.runtime.checkpoint_cadence_updates,
        batch_size=base.runtime.batch_size,
        seeds=base.seeds.training,
        repetitions=3,
        full_hybrid_min_base_ratio=0.70,
        max_policy_lag=128,
        checkpoint_enabled=True,
        deal_seed_derivation=base.seeds.derivation,
        episodes_per_learner_update=4,
        full_hybrid_phase="guided",
        full_hybrid_phase_update=guided_update,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol.identity(), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(protocol.stable_hash())


if __name__ == "__main__":
    main()
