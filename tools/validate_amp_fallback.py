"""Exercise the guarded AMP non-finite retry on the requested device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from douzero.runtime.amp import SafeMixedPrecision


def validate_amp_fallback(*, device: str, dtype: str) -> dict[str, object]:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    parameter = torch.nn.Parameter(torch.tensor([0.5], device=resolved))
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    precision = SafeMixedPrecision(
        resolved,
        enabled=True,
        dtype=dtype,
        fallback_on_nonfinite=True,
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        loss = (parameter - 1.0).square().sum()
        if closure_calls == 1:
            # A controlled validation fault: no production-data input is used.
            return loss * torch.tensor(float("nan"), device=resolved)
        return loss

    result = precision.step(
        closure,
        optimizer,
        [parameter],
        max_grad_norm=1.0,
    )
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    parameter_finite = bool(torch.isfinite(parameter.detach()).all().item())
    parameter_changed = not torch.equal(before, parameter.detach())
    if not (
        closure_calls == 2
        and result.fell_back
        and not result.amp_used
        and precision.fallback_count == 1
        and parameter_finite
        and parameter_changed
    ):
        raise RuntimeError("guarded AMP fallback did not complete as expected")

    return {
        "schema_version": "p17-gpu-run-v1",
        "status": "passed",
        "validation_type": "guarded_amp_nonfinite_fallback",
        "device_type": resolved.type,
        "dtype": dtype,
        "injected_event": "nonfinite_loss",
        "closure_calls": closure_calls,
        "fallback_count": precision.fallback_count,
        "fallback_exercised": True,
        "optimizer_step_completed": True,
        "parameter_finite": parameter_finite,
        "parameter_update_observed": parameter_changed,
        "privacy": "sanitized_no_host_or_device_identifiers",
    }


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="float16"
    )
    args = parser.parse_args()
    _write_atomic(
        args.output,
        validate_amp_fallback(device=args.device, dtype=args.dtype),
    )


if __name__ == "__main__":
    main()
