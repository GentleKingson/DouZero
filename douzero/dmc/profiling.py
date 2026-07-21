"""Low-overhead named ranges for opt-in Legacy A1 profiling."""

from __future__ import annotations

from contextlib import contextmanager

from torch.autograd.profiler import record_function


@contextmanager
def legacy_profile_range(enabled: bool, name: str):
    """Emit a stable PyTorch profiler/NVTX range only when requested."""
    if not enabled:
        yield
        return
    with record_function(f"douzero.legacy.{name}"):
        yield
