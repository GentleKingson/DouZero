#!/usr/bin/env python3
"""Write a machine-readable, privacy-minimized GPU validation environment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


def _run(command: Sequence[str], *, timeout: int = 15) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, type(exc).__name__
    text = (completed.stdout or completed.stderr).strip()
    return completed.returncode, text


def _nvidia_environment() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "driver_version": None, "gpus": []}
    returncode, output = _run([
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if returncode != 0:
        return {
            "available": False,
            "driver_version": None,
            "gpus": [],
            # Never persist raw stderr: driver tools may include hostnames,
            # device identifiers, process details, or filesystem paths.
            "probe_error_class": (
                output if returncode is None else "NonZeroExit"
            ),
        }
    gpus = []
    driver = None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, memory_mib, row_driver = parts
        driver = driver or row_driver
        try:
            memory_value: int | None = int(memory_mib)
        except ValueError:
            memory_value = None
        gpus.append({
            "index": int(index) if index.isdigit() else index,
            "name": name,
            "memory_total_mib": memory_value,
        })
    return {"available": bool(gpus), "driver_version": driver, "gpus": gpus}


def _docker_environment() -> dict[str, Any]:
    executable = shutil.which("docker")
    if not executable:
        return {"cli_available": False, "daemon_available": False, "runtimes": []}
    returncode, output = _run([
        executable,
        "info",
        "--format",
        "{{json .Runtimes}}",
    ])
    runtimes: list[str] = []
    if returncode == 0:
        try:
            payload = json.loads(output)
            if isinstance(payload, dict):
                runtimes = sorted(str(name) for name in payload)
        except json.JSONDecodeError:
            pass
    return {
        "cli_available": True,
        "daemon_available": returncode == 0,
        "runtimes": runtimes,
        "nvidia_runtime_available": "nvidia" in runtimes,
    }


def probe_environment() -> dict[str, Any]:
    import torch

    nvidia = _nvidia_environment()
    cuda_available = bool(torch.cuda.is_available())
    nccl_available = bool(
        torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
    )
    nccl_version: int | list[int] | None = None
    if cuda_available and hasattr(torch.cuda, "nccl") and nccl_available:
        raw_nccl_version = torch.cuda.nccl.version()
        nccl_version = (
            list(raw_nccl_version)
            if isinstance(raw_nccl_version, (tuple, list))
            else int(raw_nccl_version)
        )
    torch_devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                bf16_supported = bool(
                    getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                )
            torch_devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_mib": properties.total_memory // (1024 * 1024),
                "capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": bf16_supported,
            })
    return {
        "schema_version": "p17-gpu-environment-v1",
        "status": "available" if cuda_available else "blocked_no_cuda_device",
        # Intentionally excludes hostname, username, environment variables,
        # serial numbers, UUIDs, process lists, and filesystem paths.
        "privacy": "sanitized_no_host_or_device_identifiers",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "nccl_available": nccl_available,
            "nccl_version": nccl_version,
            "devices": torch_devices,
        },
        "nvidia_smi": nvidia,
        "docker": _docker_environment(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = probe_environment()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
