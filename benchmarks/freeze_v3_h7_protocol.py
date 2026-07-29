"""Freeze the immutable H7 matched-topology benchmark protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from douzero._version import git_sha
from douzero.v3_hybrid.benchmark import (
    V3H7BenchmarkProtocol,
    h7_trainer_identity_hash,
)
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.h7_smoke import build_v3_h7_smoke_config
from douzero.v3_hybrid.pilot import build_pilot_resolved_config
from douzero.v3_hybrid.runtime import (
    V3_H71A_REPLAY_PROTOCOL,
    V3_H71A_REQUEST_PROTOCOL,
    V3_H7_CHECKPOINT_FORMAT,
    V3_H7_REPLAY_PROTOCOL,
    V3_H7_REQUEST_PROTOCOL,
    V3_H7_RUNTIME_VERSION,
)


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formal-config",
        type=Path,
        help="freeze the H7.1a belief protocol for a committed formal config",
    )
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--measurement-seconds", type=float, default=300.0)
    args = parser.parse_args()
    resolved = (
        build_v3_h7_smoke_config()
        if args.formal_config is None
        else build_pilot_resolved_config(load_formal_config(args.formal_config))
    )
    belief_enabled = resolved.learner.features.belief
    request_protocol = (
        V3_H71A_REQUEST_PROTOCOL if belief_enabled else V3_H7_REQUEST_PROTOCOL
    )
    replay_protocol = (
        V3_H71A_REPLAY_PROTOCOL if belief_enabled else V3_H7_REPLAY_PROTOCOL
    )
    protocol = V3H7BenchmarkProtocol(
        source_git_sha=git_sha(),
        image_digest=args.image_digest,
        config_hash=resolved.stable_hash(),
        model_identity_hash=resolved.model.stable_hash(),
        trainer_identity_hash=h7_trainer_identity_hash(
            runtime_version=V3_H7_RUNTIME_VERSION,
            checkpoint_format=V3_H7_CHECKPOINT_FORMAT,
            request_protocol=request_protocol,
            resolved_learner_hash=(
                resolved.learner.stable_hash()
                if args.formal_config is not None
                else None
            ),
        ),
        replay_protocol_hash=_hash({"replay": replay_protocol}),
        gpu=args.gpu,
        driver=args.driver,
        pytorch=args.pytorch,
        cuda=args.cuda,
        cpu=args.cpu,
        warmup_seconds=args.warmup_seconds,
        measurement_seconds=args.measurement_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol.identity(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(protocol.stable_hash())


if __name__ == "__main__":
    main()
