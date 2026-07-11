"""Canonical relative-seat mapping for the observation V2 schema (P03).

AGENTS.md / P03 spec: "Establish canonical relative-seat mapping, so the model
can uniformly express: SELF, NEXT, PREVIOUS, LANDLORD, TEAMMATE, OPPONENT."

The legacy game uses absolute role labels — ``landlord``, ``landlord_down``
(acts immediately after the landlord), and ``landlord_up`` (acts immediately
before the landlord). The clockwise turn order is::

    landlord -> landlord_down -> landlord_up -> landlord

so from any acting role's perspective:

- **NEXT** is the seat that acts immediately after it.
- **PREVIOUS** is the seat that acts immediately before it.

For a farmer, the teammate is the *other* farmer; for the landlord there is no
teammate. The landlord is always an opponent to both farmers, and each farmer
is an opponent to the landlord.

This module derives every relationship from the single canonical turn order in
``douzero.env.rules.PLAYER_POSITIONS`` so there is no duplicated seat logic.
"""

from __future__ import annotations

from douzero.env.rules import PLAYER_POSITIONS

#: The acting player's own seat.
SEAT_SELF = "self"

#: The seat that acts immediately after the acting player (clockwise).
SEAT_NEXT = "next"

#: The seat that acts immediately before the acting player (counter-clockwise).
SEAT_PREVIOUS = "previous"

#: The landlord seat (role label).
SEAT_LANDLORD = "landlord"

#: A teammate seat (the other farmer, only defined for farmer actors).
SEAT_TEAMMATE = "teammate"

#: An opponent seat.
SEAT_OPPONENT = "opponent"

#: The canonical relative-seat labels a model may express.
RELATIVE_SEATS: tuple[str, ...] = (
    SEAT_SELF,
    SEAT_NEXT,
    SEAT_PREVIOUS,
    SEAT_LANDLORD,
    SEAT_TEAMMATE,
    SEAT_OPPONENT,
)

#: The landlord role.
LANDLORD_ROLE: str = "landlord"

#: The two farmer roles.
FARMER_ROLES: tuple[str, ...] = ("landlord_up", "landlord_down")

#: All three role labels.
ALL_ROLES: tuple[str, ...] = PLAYER_POSITIONS  # ("landlord", "landlord_down", "landlord_up")

# Canonical clockwise turn order (matches GameEnv.get_acting_player_position).
# landlord -> landlord_down -> landlord_up -> landlord.
_TURN_ORDER: tuple[str, ...] = (LANDLORD_ROLE, "landlord_down", "landlord_up")


def _index(role: str) -> int:
    return _TURN_ORDER.index(role)


def next_seat(role: str) -> str:
    """Return the seat that acts immediately after ``role`` (clockwise)."""
    return _TURN_ORDER[(_index(role) + 1) % len(_TURN_ORDER)]


def previous_seat(role: str) -> str:
    """Return the seat that acts immediately before ``role``."""
    return _TURN_ORDER[(_index(role) - 1) % len(_TURN_ORDER)]


def teammate(role: str) -> str | None:
    """Return the other farmer for a farmer ``role``, else ``None``.

    The landlord has no teammate. For ``landlord_up`` the teammate is
    ``landlord_down`` and vice versa.
    """
    if role == LANDLORD_ROLE:
        return None
    if role not in FARMER_ROLES:
        raise ValueError(f"Unknown role {role!r}")
    return "landlord_down" if role == "landlord_up" else "landlord_up"


def is_landlord(role: str) -> bool:
    """Return True if ``role`` is the landlord."""
    return role == LANDLORD_ROLE


def is_farmer(role: str) -> bool:
    """Return True if ``role`` is a farmer."""
    return role in FARMER_ROLES


def relative_seat(acting_role: str, target_role: str) -> str:
    """Classify ``target_role`` relative to ``acting_role``.

    Returns one of the canonical relative-seat labels. ``target_role == acting_role``
    yields :data:`SEAT_SELF`. The landlord is always :data:`SEAT_LANDLORD` from
    a farmer's view (it is also an opponent, but the more specific landlord
    label wins). For the landlord, each farmer is :data:`SEAT_NEXT` or
    :data:`SEAT_PREVIOUS`. For a farmer, the other farmer is
    :data:`SEAT_TEAMMATE`.
    """
    if acting_role not in ALL_ROLES:
        raise ValueError(f"Unknown acting role {acting_role!r}")
    if target_role not in ALL_ROLES:
        raise ValueError(f"Unknown target role {target_role!r}")
    if target_role == acting_role:
        return SEAT_SELF
    # Farmer looking at the landlord.
    if target_role == LANDLORD_ROLE:
        return SEAT_LANDLORD
    # Landlord looking at a farmer: classify by turn adjacency.
    if acting_role == LANDLORD_ROLE:
        if target_role == next_seat(acting_role):
            return SEAT_NEXT
        if target_role == previous_seat(acting_role):
            return SEAT_PREVIOUS
        return SEAT_OPPONENT
    # Farmer looking at the other farmer.
    if target_role == teammate(acting_role):
        return SEAT_TEAMMATE
    # Defensive: the only remaining target is a non-teammate farmer, which
    # cannot occur in three-player DouDizhu. Treat as opponent.
    return SEAT_OPPONENT


def seats_from(acting_role: str) -> dict[str, str]:
    """Return a mapping ``{role_label: relative_seat}`` for ``acting_role``.

    Covers all three absolute roles with their canonical relative-seat label.
    This is the stable per-decision seat context a model consumes.
    """
    return {role: relative_seat(acting_role, role) for role in ALL_ROLES}
