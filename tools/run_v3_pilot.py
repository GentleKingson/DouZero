#!/usr/bin/env python3
"""Run one commit-bound P2 pilot variant with atomic checkpoint evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Script execution puts tools/ first; bind imports to the attested checkout.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if not sys.path or sys.path[0] != str(_REPOSITORY_ROOT):
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import torch

from douzero._version import environment_info, git_sha
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import (
    P2_PILOT_PROTOCOL,
    P2_PILOT_SCHEMA,
    P2_SEED_DERIVATION,
    P2_VARIANTS,
    _sha256,
    collect_real_pilot_episode,
    create_pilot_learner,
    slice_pilot_batch,
    train_pilot_batch,
    write_pilot_summary,
)

_DOCKER_SOCKET = "/var/run/docker.sock"
_HOST_PROC = "/host/proc"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen P2 real-environment V3 pilot variant."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--root-seed", type=int)
    parser.add_argument("--command-record", default="")
    return parser.parse_args()


def _finite_metrics(value):
    if isinstance(value, dict):
        return {str(key): _finite_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_metrics(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError("pilot metrics contain NaN or Inf")
        return value
    return _finite_metrics(vars(value))


def _train_episode_before_deadline(
    learner, pieces, *, started: float, max_seconds: float,
    clock=time.monotonic, train_fn=train_pilot_batch,
):
    """Train a complete episode or fail before starting a late batch piece."""

    last_metrics = None
    for piece in pieces:
        if not _before_deadline(started, max_seconds, clock()):
            raise RuntimeError(
                "pilot wall-clock deadline elapsed during an atomic episode"
            )
        last_metrics = train_fn(learner, piece).as_dict()
        if not _before_deadline(started, max_seconds, clock()):
            raise RuntimeError(
                "pilot wall-clock deadline elapsed during an atomic episode"
            )
    return last_metrics


def _attest_clean_source(source_sha: str) -> str:
    """Require Git metadata and a clean, commit-matching runtime source tree."""

    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git", "status", "--porcelain=v1", "--untracked-files=all", "--", ".",
                ":(exclude)artifacts/**",
                ":(exclude)baselines/put_pretrained_models_here",
                ":(exclude)imgs/douzero_logo.jpg",
            ],
            cwd=root, check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise SystemExit(f"P2 evidence requires readable Git metadata: {exc}") from exc
    if head != source_sha:
        raise SystemExit("P2 Git HEAD does not match the declared source SHA")
    if status:
        raise SystemExit("P2 evidence requires a clean runtime source tree")
    if len(tree) != 40 or any(c not in "0123456789abcdef" for c in tree):
        raise SystemExit("P2 Git tree identity is invalid")
    return tree


def _driver_version() -> str:
    try:
        value = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise SystemExit(f"P2 CUDA evidence requires NVIDIA driver identity: {exc}") from exc
    if len(value) != 1 or not value[0]:
        raise SystemExit("P2 CUDA evidence requires exactly one NVIDIA GPU")
    return value[0]


def _docker_api_get(path: str, socket_path: str):
    request = (
        f"GET {path} HTTP/1.0\r\n"
        "Host: docker\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(10)
        client.connect(socket_path)
        client.sendall(request)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        raise SystemExit(f"P2 evidence requires Docker image attestation: {exc}") from exc
    finally:
        client.close()
    response = b"".join(chunks)
    try:
        header, body = response.split(b"\r\n\r\n", 1)
        status = int(header.split(b"\r\n", 1)[0].split()[1])
        payload = json.loads(body)
    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        raise SystemExit("Docker image attestation returned an invalid response") from exc
    if status != 200:
        raise SystemExit(f"Docker image attestation API returned status {status}")
    return payload


def _pid_namespace(path: str) -> str:
    try:
        namespace = os.readlink(path)
    except OSError as exc:
        raise SystemExit(f"P2 evidence cannot read PID namespace {path}") from exc
    if not namespace.startswith("pid:[") or not namespace.endswith("]"):
        raise SystemExit("P2 evidence found an invalid PID namespace identity")
    return namespace


def _attest_current_container(
    socket_path: str,
    host_proc: str,
) -> tuple[str, str]:
    """Bind evidence to the container matching this PID namespace's init."""

    current_namespace = _pid_namespace("/proc/1/ns/pid")
    containers = _docker_api_get("/containers/json?all=0", socket_path)
    if not isinstance(containers, list):
        raise SystemExit("Docker image attestation returned an invalid container list")
    matches = []
    for row in containers:
        container_id = row.get("Id") if isinstance(row, dict) else None
        if not isinstance(container_id, str) or len(container_id) != 64:
            continue
        detail = _docker_api_get(f"/containers/{container_id}/json", socket_path)
        host_pid = (
            detail.get("State", {}).get("Pid") if isinstance(detail, dict) else None
        )
        if (
            isinstance(host_pid, int)
            and host_pid > 0
            and _pid_namespace(f"{host_proc}/{host_pid}/ns/pid") == current_namespace
        ):
            matches.append((container_id, detail.get("Image")))
    if len(matches) != 1:
        raise SystemExit("Docker image attestation could not uniquely bind this container")
    container_id, image_id = matches[0]
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(c not in "0123456789abcdef" for c in image_id[7:])
    ):
        raise SystemExit("Docker image attestation did not return a valid image ID")
    return container_id, image_id


def _resolve_bounded_limit(override, default, label: str):
    value = default if override is None else override
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise SystemExit(f"{label} must be positive")
    if value > default:
        raise SystemExit(f"{label} exceeds the frozen pilot ceiling")
    return value


def _fits_sample_budget(consumed: int, batch_size: int, maximum: int) -> bool:
    return consumed + batch_size <= maximum


def _episode_fits_budget(learner, pieces, max_samples: int, max_steps: int) -> bool:
    return (
        learner.samples_consumed + sum(len(piece.transitions) for piece in pieces)
        <= max_samples
        and learner.eligible_updates + len(pieces) <= max_steps
    )


def _before_deadline(started: float, max_seconds: float, now: float) -> bool:
    return now - started < max_seconds


_RUN_STATE_SCHEMA = "v3-p2-run-state-v1"


def _write_run_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_checkpoint_and_run_state(
    learner,
    checkpoint: Path,
    state_path: Path,
    *,
    source_sha: str,
    formal_config_sha256: str,
    variant: str,
    root_seed: int,
    episodes_completed: int,
    decisions_completed: int,
) -> str:
    learner.save_checkpoint(checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    _write_run_state(state_path, {
        "schema": _RUN_STATE_SCHEMA,
        "source_git_sha": source_sha,
        "formal_config_sha256": formal_config_sha256,
        "variant": variant,
        "root_seed": root_seed,
        "episodes_completed": episodes_completed,
        "decisions_completed": decisions_completed,
        "checkpoint_sha256": checkpoint_sha256,
    })
    return checkpoint_sha256


def _load_run_state(
    path: Path,
    *,
    checkpoint_sha256: str,
    source_sha: str,
    formal_config_sha256: str,
    variant: str,
    root_seed: int,
) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--resume requires a valid pilot-state.json: {exc}") from exc
    expected = {
        "schema": _RUN_STATE_SCHEMA,
        "source_git_sha": source_sha,
        "formal_config_sha256": formal_config_sha256,
        "variant": variant,
        "root_seed": root_seed,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if not isinstance(payload, dict) or set(payload) != {
        *expected, "episodes_completed", "decisions_completed"
    }:
        raise SystemExit("pilot run-state envelope mismatch")
    for field, value in expected.items():
        if payload[field] != value:
            raise SystemExit(f"pilot run-state {field} mismatch")
    for field in ("episodes_completed", "decisions_completed"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(f"pilot run-state {field} is invalid")
    return payload


def main() -> int:
    args = _parse_args()
    formal = load_formal_config(args.config)
    if formal.variant not in P2_VARIANTS:
        raise SystemExit("P2 pilot requires one of the six frozen V3 variants")
    budget = formal.budgets["pilot"]
    max_seconds = float(_resolve_bounded_limit(
        args.max_seconds, budget.wall_clock_seconds, "max_seconds"
    ))
    max_samples = int(_resolve_bounded_limit(
        args.max_samples, budget.sample_budget, "max_samples"
    ))
    max_steps = int(_resolve_bounded_limit(
        args.max_optimizer_steps, budget.optimizer_step_budget, "max_optimizer_steps"
    ))
    checkpoint_every = int(_resolve_bounded_limit(
        args.checkpoint_every, formal.runtime.checkpoint_cadence_updates,
        "checkpoint_every",
    ))
    source_sha = git_sha()
    if len(source_sha) != 40:
        raise SystemExit("P2 evidence requires a full commit-bound Git SHA")
    source_tree = _attest_clean_source(source_sha)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    state_path = output / "pilot-state.json"
    learner, resolved = create_pilot_learner(formal)
    formal_identity = formal.identity_dict()
    seed = formal.seeds.training[0]
    root_seed = args.root_seed if args.root_seed is not None else seed
    resumed_from_samples = 0
    resumed_from_steps = 0
    resumed_from_episodes = 0
    resumed_from_decisions = 0
    resumed_checkpoint_sha256 = None
    if args.resume:
        if not checkpoint.is_file() or not state_path.is_file():
            raise SystemExit("--resume requires latest.pt and pilot-state.json")
        resumed_checkpoint_sha256 = _sha256(checkpoint)
        run_state = _load_run_state(
            state_path,
            checkpoint_sha256=resumed_checkpoint_sha256,
            source_sha=source_sha,
            formal_config_sha256=formal_identity["config_sha256"],
            variant=formal.variant,
            root_seed=root_seed,
        )
        learner.load_checkpoint(checkpoint)
        resumed_from_samples = learner.samples_consumed
        resumed_from_steps = learner.eligible_updates
        resumed_from_episodes = run_state["episodes_completed"]
        resumed_from_decisions = run_state["decisions_completed"]
    elif checkpoint.exists():
        raise SystemExit("fresh pilot refuses to overwrite an existing checkpoint")

    stopped = False
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stopped, stop_signal
        stopped = True
        stop_signal = signal.Signals(signum).name

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    episodes = 0
    decisions = 0
    skipped_long_episodes = 0
    winners = {"landlord": 0, "farmer": 0}
    last_metrics = None
    started_wall = time.time()
    started = time.monotonic()
    status = "completed"
    failure = None
    try:
        while (
            not stopped
            and time.monotonic() - started < max_seconds
            and learner.samples_consumed < max_samples
            and learner.eligible_updates < max_steps
        ):
            batch = collect_real_pilot_episode(
                learner,
                episode_number=resumed_from_episodes + episodes,
                root_seed=root_seed,
                worker_id=0,
                epsilon=args.epsilon,
            )
            if stopped or not _before_deadline(started, max_seconds, time.monotonic()):
                break
            episodes += 1
            decisions += batch.decisions
            winners[batch.winner_team] += 1
            batch_size = formal.runtime.batch_size
            if batch.trajectories is not None:
                if not _fits_sample_budget(
                    learner.samples_consumed, len(batch.transitions), max_samples
                ):
                    break
                if len(batch.transitions) > batch_size:
                    skipped_long_episodes += 1
                    continue
                pieces = (batch,)
            else:
                pieces = tuple(
                    slice_pilot_batch(batch, index, min(index + batch_size, len(batch.transitions)))
                    for index in range(0, len(batch.transitions), batch_size)
                )
            if not _episode_fits_budget(learner, pieces, max_samples, max_steps):
                break
            steps_before_episode = learner.eligible_updates
            last_metrics = _train_episode_before_deadline(
                learner,
                pieces,
                started=started,
                max_seconds=max_seconds,
            )
            if (
                learner.eligible_updates // checkpoint_every
                > steps_before_episode // checkpoint_every
            ):
                _save_checkpoint_and_run_state(
                    learner, checkpoint, state_path,
                    source_sha=source_sha,
                    formal_config_sha256=formal_identity["config_sha256"],
                    variant=formal.variant,
                    root_seed=root_seed,
                    episodes_completed=resumed_from_episodes + episodes,
                    decisions_completed=resumed_from_decisions + decisions,
                )
        if stopped:
            status = "stopped"
    except Exception as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        checkpoint_sha256 = _save_checkpoint_and_run_state(
            learner, checkpoint, state_path,
            source_sha=source_sha,
            formal_config_sha256=formal_identity["config_sha256"],
            variant=formal.variant,
            root_seed=root_seed,
            episodes_completed=resumed_from_episodes + episodes,
            decisions_completed=resumed_from_decisions + decisions,
        )
        elapsed = time.monotonic() - started
        env = environment_info()
        container_id, image_digest = _attest_current_container(
            _DOCKER_SOCKET, _HOST_PROC
        )
        env.update({
            "hostname": socket.gethostname(),
            "image_digest": image_digest,
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "command_record": args.command_record or None,
            "container_id": container_id,
            "source_tree": source_tree,
            "driver_version": _driver_version(),
        })
        summary = {
            "schema": P2_PILOT_SCHEMA,
            "protocol": P2_PILOT_PROTOCOL,
            "source_git_sha": source_sha,
            "formal_config_sha256": formal_identity["config_sha256"],
            "training_semantics_hash": formal_identity["training_semantics_hash"],
            "variant": formal.variant,
            "ruleset": formal.ruleset["id"],
            "seed": seed,
            "limits": {
                "max_seconds": max_seconds,
                "max_samples": max_samples,
                "max_optimizer_steps": max_steps,
                "checkpoint_every": checkpoint_every,
            },
            "collection": {
                "root_seed": root_seed,
                "worker_id": 0,
                "derivation": P2_SEED_DERIVATION,
                "epsilon": args.epsilon,
            },
            "status": status,
            "started_at": started_wall,
            "finished_at": time.time(),
            "wall_clock_seconds": elapsed,
            "samples": learner.samples_consumed,
            "optimizer_steps": learner.eligible_updates,
            "episodes": episodes,
            "decisions": decisions,
            "metrics": {
                "last_step": _finite_metrics(last_metrics),
                "samples_per_second": (
                    learner.samples_consumed - resumed_from_samples
                ) / max(elapsed, 1e-9),
                "optimizer_steps_per_second": (
                    learner.eligible_updates - resumed_from_steps
                ) / max(elapsed, 1e-9),
                "games_per_second": episodes / max(elapsed, 1e-9),
                "skipped_long_cooperation_episodes": skipped_long_episodes,
                "winner_counts": winners,
                "resolved_config_hash": resolved.stable_hash(),
                "model_hash": learner.model.config.stable_hash(),
                "policy_version": learner.policy_version,
            },
            "resume": {
                "requested": bool(args.resume),
                "from_samples": resumed_from_samples,
                "from_optimizer_steps": resumed_from_steps,
                "from_episodes": resumed_from_episodes,
                "from_decisions": resumed_from_decisions,
                "checkpoint_sha256": resumed_checkpoint_sha256,
                "continued_update": learner.eligible_updates > resumed_from_steps,
                "stop_signal": stop_signal,
            },
            "evaluation": {
                "paired_deals": 0,
                "status": "not_executed_by_training_runner",
                "playing_strength_claim": False,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
                "saved": True,
            },
            "environment": env,
            "release_candidate": "NONE",
            "release_status": "NOT READY",
            "playing_strength": "NOT MEASURED",
            "failure": failure,
        }
        write_pilot_summary(output / "training-summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
