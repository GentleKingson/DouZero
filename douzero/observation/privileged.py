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

The imperfect-information boundary for the V2 path is currently enforced by:

- the public encoder recomputing the unseen pool from public info and ignoring
  ``infoset.all_handcards`` (the leakage test replaces ``all_handcards`` with
  an access-throws sentinel and asserts ``get_obs_v2`` still succeeds);
- the public encoder not importing this ``privileged`` module;
- ``PublicObservation`` serialization containing no hidden-hand field.

A canonical type guard (a ``DeepAgentV2`` rejecting ``PrivilegedObservation``
by type) is **not implemented in P03**. It remains a P05/P16 acceptance
requirement and will be added together with ``DeepAgentV2`` itself.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

#: The literal string "privileged" stamped into every privileged container so a
#: downstream consumer can reject it at a type boundary without introspection.
PRIVILEGED_KIND: str = "privileged"


@dataclass(frozen=True, slots=True)
class PrivilegedObservation:
    """True hidden hands + training labels — NEVER passed to a deployment model.

    Attributes
    ----------
    all_handcards
        Mapping ``{role: tuple[int, ...]}`` of every role's true hand. This is
        the perfect-information allocation; it MUST NOT appear in any public
        observation or model input. Exposed as a read-only
        :class:`types.MappingProxyType`.
    hidden_hand_labels
        Optional training-only labels for the belief model (e.g. the per-rank
        count allocation for a chosen opponent). ``None`` when not applicable.
        Deep-frozen (recursively) at construction.
    terminal_target_win
        Optional terminal win label (from the acting team's perspective) used
        for Monte-Carlo value training. ``None`` mid-episode.
    terminal_target_score
        Optional terminal signed-score label (from the acting team's
        perspective). ``None`` mid-episode.
    kind
        Always :data:`PRIVILEGED_KIND`. Lets a guard reject this object without
        inspecting its contents.

    Deep immutability (item 5): ``frozen`` + ``slots``; caller-supplied
    ``all_handcards`` is deep-copied and exposed as a read-only mapping;
    ``hidden_hand_labels`` is deep-copied and frozen so nested list/dict/array
    values cannot be mutated either. The container shares no ndarray with any
    public container.
    """

    all_handcards: Mapping[str, tuple[int, ...]]
    acting_role: str
    hidden_hand_labels: Mapping[str, Any] | None = None
    terminal_target_win: int | None = None
    terminal_target_score: float | None = None
    kind: str = field(default=PRIVILEGED_KIND, init=False)

    def __post_init__(self) -> None:
        # Deep-copy caller-supplied mutable inputs, then expose them read-only.
        # Tuples are immutable; we copy the outer dict, re-wrap each hand as a
        # sorted tuple of ints, and wrap the result in a MappingProxyType so
        # ``all_handcards[role] = ...`` raises TypeError (item 5).
        copied_hands = {
            str(role): tuple(sorted(int(c) for c in hand))
            for role, hand in self.all_handcards.items()
        }
        object.__setattr__(self, "all_handcards", MappingProxyType(copied_hands))
        if self.hidden_hand_labels is not None:
            # Deep-copy then freeze nested dicts (one level) so label values
            # cannot be mutated in place. ``copy.deepcopy`` handles nested
            # list/dict; we additionally wrap the top level read-only.
            frozen = copy.deepcopy(dict(self.hidden_hand_labels))
            object.__setattr__(self, "hidden_hand_labels", MappingProxyType(frozen))
        if self.kind != PRIVILEGED_KIND:  # defensive; default already set
            object.__setattr__(self, "kind", PRIVILEGED_KIND)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict. Carries ``kind="privileged"`` for guards.

        Returns plain (mutable) copies for serialization; the container itself
        remains read-only.
        """
        return {
            "kind": self.kind,
            "acting_role": self.acting_role,
            "all_handcards": {k: list(v) for k, v in self.all_handcards.items()},
            "hidden_hand_labels": (
                None if self.hidden_hand_labels is None
                else {k: _to_plain(v) for k, v in self.hidden_hand_labels.items()}
            ),
            "terminal_target_win": self.terminal_target_win,
            "terminal_target_score": self.terminal_target_score,
        }


def _to_plain(value: Any) -> Any:
    """Recursively convert a nested dict/list structure to plain Python."""
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def is_privileged(obj: Any) -> bool:
    """Return True if ``obj`` is a privileged container (by type or kind)."""
    if isinstance(obj, PrivilegedObservation):
        return True
    if isinstance(obj, dict) and obj.get("kind") == PRIVILEGED_KIND:
        return True
    return getattr(obj, "kind", None) == PRIVILEGED_KIND
