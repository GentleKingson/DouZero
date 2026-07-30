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
from douzero.v3_hybrid.integration_config import load_v3_hybrid_config
from douzero.v3_hybrid.pilot import build_pilot_resolved_config
from douzero.v3_hybrid.runtime import (
    V3_H71A_REPLAY_PROTOCOL,
    V3_H71A_REQUEST_PROTOCOL,
    V3_H71A_SNAPSHOT_SEMANTICS,
    V3_H71B_REPLAY_PROTOCOL,
    V3_H71B_REQUEST_PROTOCOL,
    V3_H71C_REPLAY_PROTOCOL,
    V3_H71C_REQUEST_PROTOCOL,
    V3_H71D_REPLAY_PROTOCOL,
    V3_H71D_REQUEST_PROTOCOL,
    V3_H7_CHECKPOINT_FORMAT,
    V3_H7_REPLAY_PROTOCOL,
    V3_H7_REQUEST_PROTOCOL,
    V3_H7_RUNTIME_VERSION,
    V3H7RuntimeConfig,
    validate_v3_h7_formal_initialization,
    validate_v3_h7_runtime_config,
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
    config = parser.add_mutually_exclusive_group()
    config.add_argument(
        "--formal-config",
        type=Path,
        help="freeze an H7 sidecar protocol for a committed formal config",
    )
    config.add_argument(
        "--config",
        type=Path,
        help="freeze an H7 protocol for a committed resolved H6 config",
    )
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--measurement-seconds", type=float, default=300.0)
    args = parser.parse_args()
    formal = (
        None
        if args.formal_config is None
        else load_formal_config(args.formal_config)
    )
    if formal is not None:
        validate_v3_h7_formal_initialization(formal.initialization.kind)
    resolved = (
        load_v3_hybrid_config(args.config)
        if args.config is not None
        else (
            build_v3_h7_smoke_config()
            if formal is None
            else build_pilot_resolved_config(formal)
        )
    )
    belief_enabled = resolved.learner.features.belief
    oracle_enabled = resolved.learner.features.oracle
    cooperation_enabled = resolved.learner.features.cooperation
    public_aux_enabled = (
        resolved.learner.features.strategy
        or resolved.learner.features.style
    )
    request_protocol = (
        V3_H71A_REQUEST_PROTOCOL
        if belief_enabled
        else (
            V3_H71B_REQUEST_PROTOCOL
            if oracle_enabled
            else (
                V3_H71C_REQUEST_PROTOCOL
                if cooperation_enabled
                else (
                    V3_H71D_REQUEST_PROTOCOL
                    if public_aux_enabled
                    else V3_H7_REQUEST_PROTOCOL
                )
            )
        )
    )
    replay_protocol = (
        V3_H71A_REPLAY_PROTOCOL
        if belief_enabled
        else (
            V3_H71B_REPLAY_PROTOCOL
            if oracle_enabled
            else (
                V3_H71C_REPLAY_PROTOCOL
                if cooperation_enabled
                else (
                    V3_H71D_REPLAY_PROTOCOL
                    if public_aux_enabled
                    else V3_H7_REPLAY_PROTOCOL
                )
            )
        )
    )
    runtime_config = V3H7RuntimeConfig(
        belief_runtime_enabled=belief_enabled,
        oracle_runtime_enabled=oracle_enabled,
        cooperation_runtime_enabled=cooperation_enabled,
        public_aux_runtime_enabled=public_aux_enabled,
        request_protocol=request_protocol,
        replay_protocol=replay_protocol,
        snapshot_semantics=(
            V3_H71A_SNAPSHOT_SEMANTICS
            if belief_enabled
            else V3H7RuntimeConfig.snapshot_semantics
        ),
    )
    validate_v3_h7_runtime_config(resolved, runtime_config)
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
                if args.formal_config is not None or args.config is not None
                else None
            ),
        ),
        replay_protocol_hash=_hash({"replay": replay_protocol}),
        formal_config_hash=(
            None
            if formal is None
            else str(formal.identity_dict()["config_sha256"])
        ),
        oracle_enabled=oracle_enabled,
        cooperation_enabled=cooperation_enabled,
        public_aux_enabled=public_aux_enabled,
        gpu=args.gpu,
        driver=args.driver,
        pytorch=args.pytorch,
        cuda=args.cuda,
        cpu=args.cpu,
        warmup_seconds=args.warmup_seconds,
        measurement_seconds=args.measurement_seconds,
        seeds=(
            (101, 202, 303)
            if formal is None
            else formal.seeds.training
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol.identity(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(protocol.stable_hash())


if __name__ == "__main__":
    main()
