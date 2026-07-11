"""Privileged observation for training-only use (P03).

AGENTS.md imperfect-information boundary:

    "Privileged training-only data may include: exact hidden hands,
    ``all_handcards``, target hidden-card labels, perfect-information teacher
    inputs, future trajectory labels and terminal outcomes."

    "Represent public and privileged data with separate, explicitly named
    types or dictionaries."

    "Production ``act()`` and exported models must accept public data only."

This module — named ``privileged`` — and its type :class:`PrivilegedObservation`
are the ONLY place true hidden hands live in the V2 observation layer. The
public encoder (``encode_v2.get_obs_v2``) never constructs or returns a
:class:`PrivilegedObservation`; only the training data pipeline does.

The deployment type guard (``tests/test_observation_v2.py``) asserts that
``DeepAgentV2`` cannot accept a ``PrivilegedObservation`` by type, and that the
public encoder's return type is ``ObservationV2`` (public only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The literal string "privileged" stamped into every privileged container so a
#: downstream consumer can reject it at a type boundary without introspection.
PRIVILEGED_KIND: str = "privileged"


@dataclass(frozen=True)
class PrivilegedObservation:
    """True hidden hands + training labels — NEVER passed to a deployment model.

    Attributes
    ----------
    all_handcards
        Mapping ``{role: tuple[int, ...]}`` of every role's true hand. This is
        the perfect-information allocation; it MUST NOT appear in any public
        observation or model input.
    hidden_hand_labels
        Optional training-only labels for the belief model (e.g. the per-rank
        count allocation for a chosen opponent). ``None`` when not applicable.
    terminal_target_win
        Optional terminal win label (from the acting team's perspective) used
        for Monte-Carlo value training. ``None`` mid-episode.
    terminal_target_score
        Optional terminal signed-score label (from the acting team's
        perspective). ``None`` mid-episode.
    kind
        Always :data:`PRIVILEGED_KIND`. Lets a guard reject this object without
        inspecting its contents.
    """

    all_handcards: dict[str, tuple[int, ...]]
    acting_role: str
    hidden_hand_labels: dict[str, Any] | None = None
    terminal_target_win: int | None = None
    terminal_target_score: float | None = None
    kind: str = field(default=PRIVILEGED_KIND, init=False)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        if self.kind != PRIVILEGED_KIND:  # defensive; default already set
            object.__setattr__(self, "kind", PRIVILEGED_KIND)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict. Carries ``kind="privileged"`` for guards."""
        return {
            "kind": self.kind,
            "acting_role": self.acting_role,
            "all_handcards": {k: list(v) for k, v in self.all_handcards.items()},
            "hidden_hand_labels": self.hidden_hand_labels,
            "terminal_target_win": self.terminal_target_win,
            "terminal_target_score": self.terminal_target_score,
        }


def is_privileged(obj: Any) -> bool:
    """Return True if ``obj`` is a privileged container (by type or kind)."""
    if isinstance(obj, PrivilegedObservation):
        return True
    if isinstance(obj, dict) and obj.get("kind") == PRIVILEGED_KIND:
        return True
    return getattr(obj, "kind", None) == PRIVILEGED_KIND
