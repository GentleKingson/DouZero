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


def deep_freeze(value: Any) -> Any:
    """Recursively produce an immutable deep copy of ``value``.

    - mappings -> :class:`types.MappingProxyType` over a frozen dict
    - list/tuple -> tuple of frozen items (list becomes tuple, so append/del
      fail)
    - set/frozenset -> frozenset of frozen items
    - numpy ndarray -> read-only copy (``write=False``)
    - everything else -> a deep copy (ints/strs/floats/None are returned as-is
      by ``copy.deepcopy``)

    This freezes nesting at every level, so
    ``priv.hidden_hand_labels["counts"].append(9)`` raises ``AttributeError``
    (a tuple has no ``append``) and
    ``priv.hidden_hand_labels["meta"]["k"] = 0`` raises ``TypeError``
    (a MappingProxyType is read-only).
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            result = value.copy()
            result.setflags(write=False)
            return result
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    return copy.deepcopy(value)


def _to_plain(value: Any) -> Any:
    """Recursively convert a frozen structure back to plain (mutable) Python.

    The inverse of :func:`deep_freeze` for serialization: MappingProxyType ->
    dict, tuple -> list, frozenset -> list, ndarray -> list. The container
    itself stays immutable; only the serialization copy is mutable.
    """
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, frozenset):
        return [_to_plain(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    return value


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
        Recursively deep-frozen at construction (nested list/dict/set/ndarray
        are all immutable).
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
    ``all_handcards`` is deep-frozen and exposed read-only;
    ``hidden_hand_labels`` is recursively deep-frozen via :func:`deep_freeze`,
    so nested list/dict/set/array values cannot be mutated in place either.
    The container shares no ndarray with any public container.
    """

    all_handcards: Mapping[str, tuple[int, ...]]
    acting_role: str
    hidden_hand_labels: Mapping[str, Any] | None = None
    terminal_target_win: int | None = None
    terminal_target_score: float | None = None
    kind: str = field(default=PRIVILEGED_KIND, init=False)

    def __post_init__(self) -> None:
        # Deep-freeze caller-supplied mutable inputs at every nesting level
        # (review round 4, blocker 1). MappingProxyType only freezes the top
        # mapping; nested list/dict/set/ndarray stayed mutable until we
        # recursed.
        frozen_hands = deep_freeze(dict(self.all_handcards))
        object.__setattr__(self, "all_handcards", frozen_hands)
        if self.hidden_hand_labels is not None:
            object.__setattr__(
                self, "hidden_hand_labels", deep_freeze(dict(self.hidden_hand_labels))
            )
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
            "all_handcards": {k: _to_plain(v) for k, v in self.all_handcards.items()},
            "hidden_hand_labels": (
                None if self.hidden_hand_labels is None
                else {k: _to_plain(v) for k, v in self.hidden_hand_labels.items()}
            ),
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
