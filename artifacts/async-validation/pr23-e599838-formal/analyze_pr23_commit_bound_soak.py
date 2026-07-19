from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path("/evidence")
EXPECTED_HEAD = "e59983869ad88370af661d5238f6e50d4841b39e"
SINGLE_BASELINE = {
    "decisions_per_second": 618.722132,
    "transitions_per_second": 450.718010,
}
MIN_FINAL_MEMORY_PLATEAU_CYCLES = 50


def parse_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return 0.0
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator


def tail(values: list, fraction: float = 0.5) -> list:
    return values[max(0, len(values) - max(2, math.ceil(len(values) * fraction))):]


def trailing_equal_run(values: list[float]) -> int:
    if not values:
        return 0
    final = values[-1]
    count = 0
    for value in reversed(values):
        if value != final:
            break
        count += 1
    return count


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if not ordered:
            return 0.0
        return ordered[round((len(ordered) - 1) * q)]

    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
        "max": max(ordered),
    }


def memory_bytes(value: str) -> float:
    value = value.strip()
    units = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    for unit in sorted(units, key=len, reverse=True):
        if value.endswith(unit):
            return float(value[: -len(unit)]) * units[unit]
    return float(value)


def tensor_state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_report(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    stats = payload["stats"]
    losses = stats.get("last_loss", {})
    finite_losses = all(math.isfinite(float(value)) for value in losses.values())
    grad_norm = float(stats.get("grad_norm_last_step", float("nan")))
    return {
        "path": path.name,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_sha256": tensor_state_hash(payload["model_state_dict"]),
        "optimizer_steps": int(stats["optimizer_steps"]),
        "episodes_completed": int(stats["episodes_completed"]),
        "transitions_collected": int(stats["transitions_collected"]),
        "policy_step": int(payload["policy_step"]),
        "losses": {name: float(value) for name, value in losses.items()},
        "losses_finite": finite_losses,
        "grad_norm": grad_norm,
        "grad_norm_finite": math.isfinite(grad_norm),
    }


def resource_report(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    samples = []
    for row in rows:
        samples.append({
            "time": parse_time(row["timestamp_utc"]),
            "gpu_memory_bytes": float(row["gpu_memory_mib"]) * 1024**2,
            "container_memory_bytes": memory_bytes(row["container_memory_used"]),
            "container_pids": float(row["container_pids"] or 0),
            "total_processes": float(row["total_processes"] or 0),
            "python_processes": float(row["python_processes"] or 0),
        })
    tail_samples = tail(samples)
    origin = tail_samples[0]["time"]
    xs = [(sample["time"] - origin).total_seconds() for sample in tail_samples]

    def series(name: str) -> list[float]:
        return [sample[name] for sample in samples]

    def tail_slope_per_hour(name: str) -> float:
        return slope(xs, [sample[name] for sample in tail_samples]) * 3600.0

    duration = (samples[-1]["time"] - samples[0]["time"]).total_seconds()
    container_memory = series("container_memory_bytes")
    gpu_memory = series("gpu_memory_bytes")
    container_slope = tail_slope_per_hour("container_memory_bytes")
    gpu_slope = tail_slope_per_hour("gpu_memory_bytes")
    return {
        "samples": len(samples),
        "duration_seconds": duration,
        "gpu_memory_bytes": distribution(series("gpu_memory_bytes")),
        "container_memory_bytes": distribution(container_memory),
        "container_pids": distribution(series("container_pids")),
        "total_processes": distribution(series("total_processes")),
        "python_processes": distribution(series("python_processes")),
        "tail_gpu_memory_slope_bytes_per_hour": gpu_slope,
        "tail_container_memory_slope_bytes_per_hour": container_slope,
        "tail_gpu_memory_slope_fraction_per_hour": (
            gpu_slope / statistics.median(gpu_memory)
        ),
        "tail_container_memory_slope_fraction_per_hour": (
            container_slope / statistics.median(container_memory)
        ),
        "tail_pid_slope_per_hour": tail_slope_per_hour("container_pids"),
    }


def phase_report(records: list[dict]) -> dict:
    tail_records = tail(records)
    xs = [float(index) for index in range(len(tail_records))]
    peak_ram = [float(record["peak_ram_bytes"]) for record in records]
    peak_vram = [float(record["peak_vram_bytes"]) for record in records]
    tail_peak_ram = tail(peak_ram)
    tail_peak_vram = tail(peak_vram)

    def tail_cycle_slope(name: str) -> float:
        return slope(xs, [float(record[name]) for record in tail_records])

    finite = all(
        math.isfinite(float(value))
        for record in records
        for value in record.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return {
        "cycles": len(records),
        "first_cycle": int(records[0]["cycle"]),
        "last_cycle": int(records[-1]["cycle"]),
        "first_policy_step": int(records[0]["policy_step"]),
        "last_policy_step": int(records[-1]["policy_step"]),
        "first_total_episodes": int(records[0]["total_episodes"]),
        "last_total_episodes": int(records[-1]["total_episodes"]),
        "first_total_transitions": int(records[0]["total_transitions"]),
        "last_total_transitions": int(records[-1]["total_transitions"]),
        "decisions_per_second": distribution(
            [float(record["decisions_per_second"]) for record in records]
        ),
        "transitions_per_second": distribution(
            [float(record["transitions_per_second"]) for record in records]
        ),
        "requests_per_microbatch": distribution(
            [float(record["requests_per_microbatch"]) for record in records]
        ),
        "queue_p50_ms": distribution(
            [float(record["inference_queue_p50_ms"]) for record in records]
        ),
        "queue_p95_ms": distribution(
            [float(record["inference_queue_p95_ms"]) for record in records]
        ),
        "peak_ram_bytes": distribution(peak_ram),
        "peak_vram_bytes": distribution(peak_vram),
        "peak_ram_final_plateau_cycles": trailing_equal_run(peak_ram),
        "peak_vram_final_plateau_cycles": trailing_equal_run(peak_vram),
        "tail_peak_ram_range_bytes": max(tail_peak_ram) - min(tail_peak_ram),
        "tail_peak_vram_range_bytes": max(tail_peak_vram) - min(tail_peak_vram),
        "tail_peak_ram_slope_bytes_per_cycle": tail_cycle_slope(
            "peak_ram_bytes"
        ),
        "tail_peak_vram_slope_bytes_per_cycle": tail_cycle_slope(
            "peak_vram_bytes"
        ),
        "boundary_violations": sum(
            int(record["active_slots"] != 0)
            + int(record["in_flight_slots"] != 0)
            + int(record["pending_requests"] != 0)
            for record in records
        ),
        "checkpoint_failures": sum(
            record["checkpoint_status"] == "failed" for record in records
        ),
        "amp_fallbacks": sum(int(record["amp_fallback"]) for record in records),
        "all_numeric_metrics_finite": finite,
    }


def monotonic(records: list[dict], field: str) -> bool:
    values = [int(record[field]) for record in records]
    return all(left <= right for left, right in zip(values, values[1:]))


def analyze_topology(directory: Path) -> dict:
    status = json.loads((directory / "campaign-status.json").read_text())
    metric_path = next(directory.glob("*-cycles-cycles.jsonl"))
    records = [json.loads(line) for line in metric_path.read_text().splitlines() if line]
    phase1_records = [record for record in records if not record["resume_source"]]
    phase2_records = [record for record in records if record["resume_source"]]

    phase1_checkpoint = checkpoint_report(
        directory / "phase1-signal-checkpoint.pt"
    )
    phase2_checkpoint = checkpoint_report(
        directory / "phase2-first-checkpoint.pt"
    )
    phase1_manifest = json.loads(
        (directory / "phase1-latest-manifest.json").read_text()
    )
    phase2_first_manifest = json.loads(
        (directory / "phase2-first-manifest.json").read_text()
    )
    phase2_final_manifest = json.loads(
        (directory / "phase2-final-manifest.json").read_text()
    )

    phase1_duration = (
        parse_time((directory / "phase1-sigterm-requested-at.txt").read_text())
        - parse_time((directory / "phase1-started-at.txt").read_text())
    ).total_seconds()
    phase2_duration = (
        parse_time((directory / "phase2-sigterm-requested-at.txt").read_text())
        - parse_time((directory / "phase2-started-at.txt").read_text())
    ).total_seconds()

    before_inspect = json.loads(
        (directory / "phase1-container-inspect-before.json").read_text()
    )[0]
    mounts = {
        mount["Destination"]: {
            "source": mount["Source"],
            "rw": bool(mount["RW"]),
        }
        for mount in before_inspect["Mounts"]
    }
    source_identity = (directory / "container-source-identity.txt").read_text()
    clean_marker = "status_porcelain_begin\nstatus_porcelain_end\n"

    report = {
        "status": status,
        "container_source_identity_clean": clean_marker in source_identity,
        "container_image_matches": before_inspect["Image"] == status["image_id"],
        "mounts": mounts,
        "phase1_duration_seconds": phase1_duration,
        "phase2_duration_seconds": phase2_duration,
        "phase1_exit_code": int(
            (directory / "phase1-exit-code.txt").read_text().strip()
        ),
        "phase2_exit_code": int(
            (directory / "phase2-exit-code.txt").read_text().strip()
        ),
        "phase1": phase_report(phase1_records),
        "phase2": phase_report(phase2_records),
        "phase1_resources": resource_report(directory / "phase1-resources.csv"),
        "phase2_resources": resource_report(directory / "phase2-resources.csv"),
        "phase1_checkpoint": phase1_checkpoint,
        "phase2_first_checkpoint": phase2_checkpoint,
        "model_changed_after_resume": (
            phase1_checkpoint["model_sha256"]
            != phase2_checkpoint["model_sha256"]
        ),
        "optimizer_advanced_after_resume": (
            phase2_checkpoint["optimizer_steps"]
            > phase1_checkpoint["optimizer_steps"]
        ),
        "phase1_manifest": phase1_manifest,
        "phase2_first_manifest": phase2_first_manifest,
        "phase2_final_manifest": phase2_final_manifest,
        "cycle_monotonic": monotonic(records, "cycle"),
        "episode_monotonic": monotonic(records, "total_episodes"),
        "transition_monotonic": monotonic(records, "total_transitions"),
        "optimizer_step_monotonic": monotonic(records, "total_optimizer_steps"),
        "policy_step_monotonic": monotonic(records, "policy_step"),
        "task_containers_after_empty": not (
            directory / "task-containers-after.txt"
        ).read_text().strip(),
        "git_status_before_empty": not (
            directory / "git-status-before.txt"
        ).read_text().strip(),
        "git_status_after_empty": not (
            directory / "git-status-after.txt"
        ).read_text().strip(),
    }

    checks = {
        "source_head_bound": status["source_head"] == EXPECTED_HEAD,
        "source_not_mounted": status["source_mount"] is False,
        "container_source_clean": report["container_source_identity_clean"],
        "container_image_bound": report["container_image_matches"],
        "git_metadata_read_only": (
            mounts.get("/workspace/DouZero/.git", {}).get("rw") is False
        ),
        "phase1_ran_30_minutes": phase1_duration >= 1800,
        "phase2_ran_30_minutes": phase2_duration >= 1800,
        "phase1_exit_zero": report["phase1_exit_code"] == 0,
        "phase2_exit_zero": report["phase2_exit_code"] == 0,
        "all_boundaries_clean": (
            report["phase1"]["boundary_violations"] == 0
            and report["phase2"]["boundary_violations"] == 0
        ),
        "microbatch_gate": min(
            report["phase1"]["requests_per_microbatch"]["min"],
            report["phase2"]["requests_per_microbatch"]["min"],
        ) >= 4.0,
        "metrics_finite": (
            report["phase1"]["all_numeric_metrics_finite"]
            and report["phase2"]["all_numeric_metrics_finite"]
        ),
        "loss_and_grad_finite": (
            phase1_checkpoint["losses_finite"]
            and phase1_checkpoint["grad_norm_finite"]
            and phase2_checkpoint["losses_finite"]
            and phase2_checkpoint["grad_norm_finite"]
        ),
        "model_changed_after_resume": report["model_changed_after_resume"],
        "optimizer_advanced_after_resume": report[
            "optimizer_advanced_after_resume"
        ],
        "all_counters_monotonic": all(
            report[name]
            for name in (
                "cycle_monotonic",
                "episode_monotonic",
                "transition_monotonic",
                "optimizer_step_monotonic",
                "policy_step_monotonic",
            )
        ),
        "no_checkpoint_failures": (
            report["phase1"]["checkpoint_failures"] == 0
            and report["phase2"]["checkpoint_failures"] == 0
        ),
        "no_amp_fallbacks": (
            report["phase1"]["amp_fallbacks"] == 0
            and report["phase2"]["amp_fallbacks"] == 0
        ),
        "process_memory_bounded_and_plateaued": all(
            phase["peak_ram_final_plateau_cycles"]
            >= MIN_FINAL_MEMORY_PLATEAU_CYCLES
            and phase["peak_vram_final_plateau_cycles"]
            >= MIN_FINAL_MEMORY_PLATEAU_CYCLES
            and phase["tail_peak_ram_range_bytes"]
            <= max(32 * 1024**2, phase["peak_ram_bytes"]["median"] * 0.02)
            and phase["tail_peak_vram_range_bytes"]
            <= max(32 * 1024**2, phase["peak_vram_bytes"]["median"] * 0.03)
            for phase in (report["phase1"], report["phase2"])
        ),
        "resource_memory_growth_bounded": all(
            resources["tail_container_memory_slope_bytes_per_hour"]
            <= max(
                64 * 1024**2,
                resources["container_memory_bytes"]["median"] * 0.02,
            )
            and resources["tail_gpu_memory_slope_bytes_per_hour"]
            <= max(
                32 * 1024**2,
                resources["gpu_memory_bytes"]["median"] * 0.03,
            )
            for resources in (
                report["phase1_resources"],
                report["phase2_resources"],
            )
        ),
        "no_pid_growth": all(
            abs(resources["tail_pid_slope_per_hour"]) < 0.5
            for resources in (
                report["phase1_resources"],
                report["phase2_resources"],
            )
        ),
        "task_containers_removed": report["task_containers_after_empty"],
        "host_checkout_clean_before_after": (
            report["git_status_before_empty"]
            and report["git_status_after_empty"]
        ),
    }
    report["checks"] = checks
    report["all_checks_passed"] = all(checks.values())
    return report


def mib(value: float) -> float:
    return value / 1024**2


def markdown(report: dict) -> str:
    lines = [
        "# PR #23 commit-bound async recovery soak",
        "",
        f"Source head: `{EXPECTED_HEAD}`",
        f"Image: `{report['image_id']}`",
        "",
        "The image contains the committed source; only read-only `.git` metadata and the evidence output directory were mounted.",
        "",
        "| Topology | Phase | Cycles | Median decisions/s | Median transitions/s | Min requests/microbatch | Queue p50 median ms | Queue p95 median ms | Boundary violations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topology, topology_report in report["topologies"].items():
        for phase in ("phase1", "phase2"):
            data = topology_report[phase]
            lines.append(
                f"| {topology} | {phase} | {data['cycles']} | "
                f"{data['decisions_per_second']['median']:.2f} | "
                f"{data['transitions_per_second']['median']:.2f} | "
                f"{data['requests_per_microbatch']['min']:.2f} | "
                f"{data['queue_p50_ms']['median']:.2f} | "
                f"{data['queue_p95_ms']['median']:.2f} | "
                f"{data['boundary_violations']} |"
            )
    lines.extend([
        "",
        "| Topology | Phase | Peak RAM plateau cycles | Peak VRAM plateau cycles | Container RAM tail slope MiB/hour | GPU memory tail slope MiB/hour | PID tail slope/hour |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for topology, topology_report in report["topologies"].items():
        for phase in ("phase1", "phase2"):
            data = topology_report[phase]
            resources = topology_report[f"{phase}_resources"]
            lines.append(
                f"| {topology} | {phase} | "
                f"{data['peak_ram_final_plateau_cycles']} | "
                f"{data['peak_vram_final_plateau_cycles']} | "
                f"{mib(resources['tail_container_memory_slope_bytes_per_hour']):.3f} | "
                f"{mib(resources['tail_gpu_memory_slope_bytes_per_hour']):.3f} | "
                f"{resources['tail_pid_slope_per_hour']:.3f} |"
            )
    lines.extend(["", "## Checkpoints", ""])
    for topology, topology_report in report["topologies"].items():
        before = topology_report["phase1_checkpoint"]
        after = topology_report["phase2_first_checkpoint"]
        lines.append(
            f"- `{topology}`: optimizer step {before['optimizer_steps']} -> "
            f"{after['optimizer_steps']}; model hash changed: "
            f"{topology_report['model_changed_after_resume']}; finite loss/grad: "
            f"{before['losses_finite'] and before['grad_norm_finite'] and after['losses_finite'] and after['grad_norm_finite']}."
        )
        lines.append(
            f"  Signal checkpoint SHA-256: `{before['file_sha256']}`"
        )
        lines.append(
            f"  First resumed checkpoint SHA-256: `{after['file_sha256']}`"
        )
    lines.extend(["", "## Gates", ""])
    for topology, topology_report in report["topologies"].items():
        lines.append(
            f"- `{topology}`: "
            + ("PASS" if topology_report["all_checks_passed"] else "FAIL")
        )
    lines.extend([
        "",
        f"Overall commit-bound soak: **{'PASS' if report['all_checks_passed'] else 'FAIL'}**",
        "",
        "Independent approving review remains a separate merge gate.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    topologies = {
        "async4x4": analyze_topology(ROOT / "async4x4"),
        "async8x4": analyze_topology(ROOT / "async8x4"),
    }
    image_ids = {
        value["status"]["image_id"] for value in topologies.values()
    }
    report = {
        "schema_version": "pr23-commit-bound-soak-analysis-v1",
        "source_head": EXPECTED_HEAD,
        "single_process_directional_baseline": SINGLE_BASELINE,
        "memory_gate_thresholds": {
            "minimum_final_peak_plateau_cycles": MIN_FINAL_MEMORY_PLATEAU_CYCLES,
            "maximum_back_half_peak_ram_range": "max(32 MiB, 2% of median)",
            "maximum_back_half_peak_vram_range": "max(32 MiB, 3% of median)",
            "maximum_container_ram_tail_slope_per_hour": "max(64 MiB, 2% of median)",
            "maximum_gpu_memory_tail_slope_per_hour": "max(32 MiB, 3% of median)",
            "maximum_absolute_pid_tail_slope_per_hour": 0.5,
        },
        "image_id": next(iter(image_ids)) if len(image_ids) == 1 else "mismatch",
        "topologies": topologies,
        "all_checks_passed": (
            len(image_ids) == 1
            and all(value["all_checks_passed"] for value in topologies.values())
        ),
    }
    (ROOT / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "all_checks_passed": report["all_checks_passed"],
        "image_id": report["image_id"],
        "topologies": {
            name: value["all_checks_passed"]
            for name, value in topologies.items()
        },
    }, indent=2, sort_keys=True))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
