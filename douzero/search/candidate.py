"""Stable top-k filtering from the base Model V2 output."""

from __future__ import annotations

from dataclasses import dataclass

from douzero.models_v2.output import ModelOutput


@dataclass(frozen=True, slots=True)
class Candidate:
    """A legal action together with its base-model estimates."""

    index: int
    action: tuple[int, ...]
    base_win_probability: float
    base_expected_score: float


def select_top_k(
    legal_actions: tuple[tuple[int, ...], ...],
    output: ModelOutput,
    top_k: int,
    mode: str = "win_then_score",
) -> tuple[Candidate, ...]:
    """Return valid actions in the configured base-model objective order."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if mode not in ("win", "score", "win_then_score"):
        raise ValueError("mode must be win, score, or win_then_score")
    if len(legal_actions) > output.num_actions:
        raise ValueError("ModelOutput has fewer rows than legal actions")
    p_win = output.p_win.detach().cpu().squeeze(-1).tolist()
    score = output.score_mean.detach().cpu().squeeze(-1).tolist()
    mask = output.action_mask.detach().cpu().tolist()
    candidates = [
        Candidate(i, tuple(action), float(p_win[i]), float(score[i]))
        for i, action in enumerate(legal_actions)
        if mask[i]
    ]
    if mode == "score":
        candidates.sort(
            key=lambda item: (-item.base_expected_score, item.index)
        )
    elif mode == "win":
        candidates.sort(
            key=lambda item: (-item.base_win_probability, item.index)
        )
    else:
        candidates.sort(
            key=lambda item: (
                -item.base_win_probability,
                -item.base_expected_score,
                item.index,
            )
        )
    return tuple(candidates[:top_k])
