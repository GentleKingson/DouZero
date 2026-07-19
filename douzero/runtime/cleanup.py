"""Ordered, non-short-circuiting runtime cleanup."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def run_cleanup_steps(
    steps: Iterable[Callable[[], object]],
    *,
    preserve_active_exception: bool,
) -> None:
    """Run every cleanup step and apply a deterministic error priority.

    The first cleanup failure is raised only when no exception was already
    unwinding into the cleanup block. Later steps always run, so one failed
    resource release cannot strand the resources owned by another step.
    """
    first_error: BaseException | None = None
    for step in steps:
        try:
            step()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not preserve_active_exception:
        raise first_error
