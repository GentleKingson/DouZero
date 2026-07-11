"""Versioned feature schema manifest for observation V2 (P03).

AGENTS.md: "Version every nontrivial observation schema. Derive dimensions from
a schema or named constants. Do not scatter magic widths such as role-specific
flattened feature sizes across files."

The legacy encoder hard-codes per-role widths — landlord ``x_no_action=319``,
``x_batch=373``; farmers ``x_no_action=430``, ``x_batch=484`` — which the
architecture doc (``docs/architecture/current.md``) calls out as something P03
must derive instead. This module expresses every V2 field width as a named
combination of :mod:`douzero.observation.cards` and :mod:`.seats` constants, and
records the full manifest (version + field specs) for checkpoint stamping.

A :class:`FeatureSchemaManifest` is:

- deterministic (a pure function of the configuration knobs), so two encoders
  built with the same ``feature_version`` / ``max_history_len`` produce the same
  manifest;
- JSON-serialisable, so it can be stamped into a checkpoint and compared on
  load to reject an incompatible observation schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cards import CARD_VECTOR_DIM, NUM_JOKER_SLOTS
from .seats import RELATIVE_SEATS

#: Current observation V2 schema version. Bumped on any breaking change to the
#: field set or field widths. ``"v2-1"`` is the first V2 schema (P03).
FEATURE_VERSION_V2: str = "v2"

#: Schema sub-version, bumped when the field set/widths change within v2.
SCHEMA_VERSION_V2: str = "v2-1"

#: Legacy feature-version identifier (for manifest comparison).
FEATURE_VERSION_LEGACY: str = "legacy"

#: One-hot width for a player's remaining-card count. The landlord holds up to
#: 20 cards (17 + 3 bottom); farmers up to 17. Use the larger bound so one
#: constant covers all roles.
MAX_CARDS_LEFT: int = 20

#: One-hot width for the bomb counter. The legacy encoder reserves 15 slots
#: (``_get_one_hot_bomb``). We keep the same width for legacy-adapter parity.
BOMB_ONEHOT_WIDTH: int = 15

#: Width of the relative-seat one-hot (SELF/NEXT/PREVIOUS/LANDLORD/TEAMMATE/
#: OPPONENT). Defined here rather than read from ``seats.RELATIVE_SEATS`` at
#: build time so the manifest records the exact width it was authored against.
SEAT_ONEHOT_WIDTH: int = len(RELATIVE_SEATS)  # 6

#: Width of the move-type one-hot. The legacy detector defines TYPE_0..TYPE_15
#: (16 labels, including TYPE_15_WRONG). Pass is TYPE_0.
MOVE_TYPE_ONEHOT_WIDTH: int = 16


@dataclass(frozen=True)
class FieldSpec:
    """Description of one tensor field in the observation schema.

    ``shape`` excludes the leading legal-action batch dimension (``N``), which
    is variable and recorded separately. ``dtype`` is the numpy dtype string.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "description": self.description,
        }


@dataclass(frozen=True)
class FeatureSchemaManifest:
    """Complete, versioned description of the observation V2 schema.

    Every field width is derived from named constants in
    :mod:`douzero.observation.cards` / :mod:`.seats` / this module, never a
    bare integer literal. Two manifests compare equal iff every field matches,
    so a checkpoint can reject an incompatible schema precisely.
    """

    feature_version: str
    schema_version: str
    max_history_len: int
    card_vector_dim: int
    seat_onehot_width: int
    move_type_onehot_width: int
    bomb_onehot_width: int
    max_cards_left: int
    state_fields: tuple[FieldSpec, ...]
    action_fields: tuple[FieldSpec, ...]
    history_token_fields: tuple[FieldSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "max_history_len": self.max_history_len,
            "card_vector_dim": self.card_vector_dim,
            "seat_onehot_width": self.seat_onehot_width,
            "move_type_onehot_width": self.move_type_onehot_width,
            "bomb_onehot_width": self.bomb_onehot_width,
            "max_cards_left": self.max_cards_left,
            "state_fields": [f.to_dict() for f in self.state_fields],
            "action_fields": [f.to_dict() for f in self.action_fields],
            "history_token_fields": [f.to_dict() for f in self.history_token_fields],
        }

    def field_by_name(self, name: str, group: str = "state") -> FieldSpec:
        """Look up a field spec by name within a group (state/action/history)."""
        groups = {
            "state": self.state_fields,
            "action": self.action_fields,
            "history": self.history_token_fields,
        }
        if group not in groups:
            raise ValueError(f"Unknown field group {group!r}")
        for spec in groups[group]:
            if spec.name == name:
                return spec
        raise KeyError(f"No {group!r} field named {name!r}")


def build_v2_schema(max_history_len: int = 100) -> FeatureSchemaManifest:
    """Construct the canonical V2 feature schema.

    ``max_history_len`` bounds the history-token sequence length. The default
    (100) comfortably covers a full DouDizhu game (a game rarely exceeds ~60
    actions). A padding mask marks unused slots.

    The state block is encoded **once per decision** (not per legal action).
    It contains, per acting role:

    - ``my_handcards`` (54): the acting player's hand.
    - ``other_handcards`` (54): the union of unseen cards (the legacy
      swap-invariant opponent pool). This is public because it is fully
      determined by the acting hand + played cards + public bottom cards.
    - ``landlord_played`` / ``landlord_down_played`` / ``landlord_up_played``
      (54 each): cumulative cards played by each absolute role.
    - ``last_move`` (54): the most recent valid action.
    - ``num_cards_left`` per seat (MAX_CARDS_LEFT each): one-hot remaining
      counts for the three roles.
    - ``bomb_num`` (BOMB_ONEHOT_WIDTH): one-hot bomb counter.
    - ``acting_role`` (SEAT_ONEHOT_WIDTH): one-hot acting role.

    The action block is encoded once per legal action:

    - ``action_cards`` (54): the candidate action's card vector.
    - ``move_type`` (MOVE_TYPE_ONEHOT_WIDTH): one-hot move type.
    - ``main_rank`` (1): the action's main rank.
    - ``length`` (1): serial length / card count.
    - ``is_pass`` (1): pass flag.
    - ``is_bomb`` (1): bomb/rocket flag.

    Each history token carries:

    - ``actor_role`` (SEAT_ONEHOT_WIDTH): one-hot actor role.
    - ``is_pass`` (1), ``move_type`` (MOVE_TYPE_ONEHOT_WIDTH),
      ``main_rank`` (1), ``length`` (1), ``card_count`` (1),
      ``cards_encoding`` (CARD_VECTOR_DIM), ``cards_left_after`` (1),
      ``bomb_flag`` (1), ``phase`` (1), ``valid`` (1, the padding mask).
    """
    state_fields: tuple[FieldSpec, ...] = (
        FieldSpec("my_handcards", (CARD_VECTOR_DIM,), "int8", "acting player's hand"),
        FieldSpec("other_handcards", (CARD_VECTOR_DIM,), "int8",
                  "union of unseen cards (public, swap-invariant)"),
        FieldSpec("landlord_played", (CARD_VECTOR_DIM,), "int8",
                  "cumulative cards played by the landlord"),
        FieldSpec("landlord_down_played", (CARD_VECTOR_DIM,), "int8",
                  "cumulative cards played by landlord_down"),
        FieldSpec("landlord_up_played", (CARD_VECTOR_DIM,), "int8",
                  "cumulative cards played by landlord_up"),
        FieldSpec("last_move", (CARD_VECTOR_DIM,), "int8",
                  "most recent valid (non-pass) action"),
        FieldSpec("num_cards_left_landlord", (MAX_CARDS_LEFT,), "int8",
                  "one-hot landlord remaining-card count"),
        FieldSpec("num_cards_left_landlord_down", (MAX_CARDS_LEFT,), "int8",
                  "one-hot landlord_down remaining-card count"),
        FieldSpec("num_cards_left_landlord_up", (MAX_CARDS_LEFT,), "int8",
                  "one-hot landlord_up remaining-card count"),
        FieldSpec("bomb_num", (BOMB_ONEHOT_WIDTH,), "int8", "one-hot bomb counter"),
        FieldSpec("acting_role", (SEAT_ONEHOT_WIDTH,), "int8",
                  "one-hot acting role (landlord/landlord_down/landlord_up)"),
    )

    action_fields: tuple[FieldSpec, ...] = (
        FieldSpec("action_cards", (CARD_VECTOR_DIM,), "int8", "candidate action cards"),
        FieldSpec("move_type", (MOVE_TYPE_ONEHOT_WIDTH,), "int8",
                  "one-hot move type (TYPE_0..TYPE_15)"),
        FieldSpec("main_rank", (1,), "int8", "action main rank"),
        FieldSpec("length", (1,), "int8", "serial length / card count"),
        FieldSpec("is_pass", (1,), "int8", "1 if the action is a pass"),
        FieldSpec("is_bomb", (1,), "int8", "1 if the action is a bomb or rocket"),
    )

    history_token_fields: tuple[FieldSpec, ...] = (
        FieldSpec("actor_role", (SEAT_ONEHOT_WIDTH,), "int8",
                  "one-hot role of the acting player for this token"),
        FieldSpec("is_pass", (1,), "int8", "1 if the token is a pass"),
        FieldSpec("move_type", (MOVE_TYPE_ONEHOT_WIDTH,), "int8", "one-hot move type"),
        FieldSpec("main_rank", (1,), "int8", "main rank of the move"),
        FieldSpec("length", (1,), "int8", "serial length / card count"),
        FieldSpec("card_count", (1,), "int8", "number of cards in the move"),
        FieldSpec("cards_encoding", (CARD_VECTOR_DIM,), "int8", "card vector of the move"),
        FieldSpec("cards_left_after", (1,), "int8",
                  "actor's remaining-card count after the move"),
        FieldSpec("bomb_flag", (1,), "int8",
                  "1 if the move was a bomb or rocket"),
        FieldSpec("phase", (1,), "int8", "game phase when the move was made"),
        FieldSpec("valid", (1,), "int8",
                  "padding mask: 1 for a real token, 0 for padding"),
    )

    return FeatureSchemaManifest(
        feature_version=FEATURE_VERSION_V2,
        schema_version=SCHEMA_VERSION_V2,
        max_history_len=max_history_len,
        card_vector_dim=CARD_VECTOR_DIM,
        seat_onehot_width=SEAT_ONEHOT_WIDTH,
        move_type_onehot_width=MOVE_TYPE_ONEHOT_WIDTH,
        bomb_onehot_width=BOMB_ONEHOT_WIDTH,
        max_cards_left=MAX_CARDS_LEFT,
        state_fields=state_fields,
        action_fields=action_fields,
        history_token_fields=history_token_fields,
    )


def state_width(schema: FeatureSchemaManifest) -> int:
    """Total flat width of the per-decision state block (no batch dim)."""
    return sum(int(np_prod(spec.shape)) for spec in schema.state_fields)


def action_width(schema: FeatureSchemaManifest) -> int:
    """Total flat width of one legal-action feature row (no batch dim)."""
    return sum(int(np_prod(spec.shape)) for spec in schema.action_fields)


def history_token_width(schema: FeatureSchemaManifest) -> int:
    """Total flat width of one history token (no sequence dim)."""
    return sum(int(np_prod(spec.shape)) for spec in schema.history_token_fields)


def np_prod(shape: tuple[int, ...]) -> int:
    """Product of a shape tuple, with the empty product = 1."""
    p = 1
    for d in shape:
        p *= d
    return p


# Eagerly compute the legacy widths from the same constants, so the legacy
# adapter and the architecture doc can cite them without re-deriving. These are
# NOT used by the V2 encoder; they document that the V2 schema reproduces the
# legacy information content from named constants rather than magic numbers.
def legacy_landlord_state_width() -> int:
    """Legacy landlord ``x_no_action`` width, derived from constants (319).

    54*5 (my/other/last/up-played/down-played) + 17 (up cards-left) +
    17 (down cards-left) + 15 (bomb one-hot).
    """
    return (
        CARD_VECTOR_DIM * 5
        + 17  # landlord_up cards-left one-hot (legacy uses 17, not MAX_CARDS_LEFT)
        + 17  # landlord_down cards-left one-hot
        + BOMB_ONEHOT_WIDTH
    )


def legacy_farmer_state_width() -> int:
    """Legacy farmer ``x_no_action`` width, derived from constants (430).

    54*7 (my/other/landlord-played/teammate-played/last/last-landlord/
    last-teammate) + 20 (landlord cards-left) + 17 (teammate cards-left) +
    15 (bomb one-hot).
    """
    return (
        CARD_VECTOR_DIM * 7
        + MAX_CARDS_LEFT  # landlord cards-left one-hot (20)
        + 17  # teammate cards-left one-hot
        + BOMB_ONEHOT_WIDTH
    )
