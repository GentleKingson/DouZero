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

import hashlib
import json
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

#: One-hot width for a bid value. Standard bidding uses 0/1/2/3 (4 slots).
BID_VALUE_ONEHOT_WIDTH: int = 4

#: One-hot width for the game phase (bidding/reveal_bottom/playing + reserved).
PHASE_ONEHOT_WIDTH: int = 4

#: Maximum number of bidding tokens (3 bidders per round × bounded redeals;
#: the encoded tensor is variable-length but the schema records the cap used
#: for any fixed-width projection).
MAX_BIDDING_TOKENS: int = 16

#: Width of one bidding token (seat one-hot + bid-value one-hot + is_pass).
#: Kept in sync with ``public.BIDDING_TOKEN_WIDTH``.
BIDDING_TOKEN_WIDTH: int = 3 + BID_VALUE_ONEHOT_WIDTH + 1

#: Length of the rule-identity fingerprint embedded in the context block. It
#: is NOT the full ruleset_hash; it is a fixed-width compact id (the ruleset_id
#: one-hot over legacy/standard + a numeric multiplier scalar).
RULESET_ID_ONEHOT_WIDTH: int = 2

# --------------------------------------------------------------------------- #
# Semantic version stamps for the schema's invariants (item 2).
#
# Each stamp identifies a *contract*, not a width. Two schemas with the same
# stamps produce observations that a model can consume interchangeably. Bump a
# stamp when the corresponding contract changes in a way that breaks a model:
# e.g. the card encoding layout changes, the seat one-hot order changes, the
# padding-mask polarity flips, or the history truncation side changes.
#
# A pure description-text edit does NOT bump any stamp, so stable_hash() is
# stable under documentation churn (item 2: "description wording changes must
# not change the compatibility hash").
# --------------------------------------------------------------------------- #
#: Version of the 54-dim card encoding contract (rank-major multiplicities +
#: trailing jokers). Matches the legacy ``_cards2array`` layout.
CARD_ENCODING_VERSION: str = "rank-multiplicity-54-v1"

#: Version of the move-type enumeration contract (TYPE_0..TYPE_15).
MOVE_TYPE_ENCODING_VERSION: str = "detector-type0-15-v1"

#: Version of the relative-seat enumeration order used in one-hot fields.
SEAT_MAPPING_VERSION: str = "self-next-previous-landlord-teammate-opponent-v1"

#: Version of the history-token field set/order contract.
HISTORY_ENCODING_VERSION: str = "token-v1"

#: Version of the mask semantics: ``valid_mask`` uses 1=valid / 0=padding and
#: ``key_padding_mask`` uses True=padding (PyTorch convention).
MASK_SEMANTICS_VERSION: str = "valid1-pad0-v1"

#: Version of the history truncation contract: bounded to ``max_history_len``,
#: left-truncated (oldest moves dropped), real tokens left-aligned.
TRUNCATION_SEMANTICS_VERSION: str = "left-truncate-v1"

#: Version of the public-context block contract (item 3): the schema-described
#: tensor block carrying bottom-card identity, bid, phase, rocket count, total
#: multiplier, and rule-identity fingerprint.
CONTEXT_ENCODING_VERSION: str = "context-v1"

#: Version of the bidding-token field set/order contract (item 3).
BIDDING_ENCODING_VERSION: str = "bidding-token-v1"


@dataclass(frozen=True)
class FieldSpec:
    """Description of one tensor field in the observation schema.

    ``shape`` excludes the leading legal-action batch dimension (``N``), which
    is variable and recorded separately. ``dtype`` is the numpy dtype string.

    ``description`` is documentation only; it is excluded from the schema
    compatibility hash (:func:`FeatureSchemaManifest.stable_hash`) so wording
    edits do not change a model's identity contract (item 2).
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

    def identity_dict(self) -> dict[str, Any]:
        """The identity-relevant subset (excludes the description text)."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class FeatureSchemaManifest:
    """Complete, versioned description of the observation V2 schema.

    Every field width is derived from named constants in
    :mod:`douzero.observation.cards` / :mod:`.seats` / this module, never a
    bare integer literal. Two manifests compare equal iff every field matches,
    so a checkpoint can reject an incompatible schema precisely.

    The schema identity (used for checkpoint compatibility and ObservationV2
    stamping) is captured by :meth:`compatibility_dict` / :meth:`stable_hash`.
    Identity includes the semantic version stamps (card encoding, seat mapping,
    mask semantics, truncation semantics), the field name/shape/dtype for every
    group, and ``max_history_len``. It deliberately EXCLUDES ``description``
    text, so documentation churn does not change the hash (item 2).
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
    # Item 3: schema-described public-context + bidding-token groups.
    context_fields: tuple[FieldSpec, ...] = ()
    bidding_token_fields: tuple[FieldSpec, ...] = ()
    bid_value_onehot_width: int = BID_VALUE_ONEHOT_WIDTH
    phase_onehot_width: int = PHASE_ONEHOT_WIDTH
    ruleset_id_onehot_width: int = RULESET_ID_ONEHOT_WIDTH
    bidding_token_width: int = BIDDING_TOKEN_WIDTH
    max_bidding_tokens: int = MAX_BIDDING_TOKENS
    # Semantic version stamps (item 2). Defaults bind to the current contracts.
    card_encoding_version: str = CARD_ENCODING_VERSION
    move_type_encoding_version: str = MOVE_TYPE_ENCODING_VERSION
    seat_mapping_version: str = SEAT_MAPPING_VERSION
    history_encoding_version: str = HISTORY_ENCODING_VERSION
    mask_semantics_version: str = MASK_SEMANTICS_VERSION
    truncation_semantics_version: str = TRUNCATION_SEMANTICS_VERSION
    context_encoding_version: str = CONTEXT_ENCODING_VERSION
    bidding_encoding_version: str = BIDDING_ENCODING_VERSION

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
            "bid_value_onehot_width": self.bid_value_onehot_width,
            "phase_onehot_width": self.phase_onehot_width,
            "ruleset_id_onehot_width": self.ruleset_id_onehot_width,
            "bidding_token_width": self.bidding_token_width,
            "max_bidding_tokens": self.max_bidding_tokens,
            "state_fields": [f.to_dict() for f in self.state_fields],
            "action_fields": [f.to_dict() for f in self.action_fields],
            "history_token_fields": [f.to_dict() for f in self.history_token_fields],
            "context_fields": [f.to_dict() for f in self.context_fields],
            "bidding_token_fields": [f.to_dict() for f in self.bidding_token_fields],
            "card_encoding_version": self.card_encoding_version,
            "move_type_encoding_version": self.move_type_encoding_version,
            "seat_mapping_version": self.seat_mapping_version,
            "history_encoding_version": self.history_encoding_version,
            "mask_semantics_version": self.mask_semantics_version,
            "truncation_semantics_version": self.truncation_semantics_version,
            "context_encoding_version": self.context_encoding_version,
            "bidding_encoding_version": self.bidding_encoding_version,
        }

    def field_by_name(self, name: str, group: str = "state") -> FieldSpec:
        """Look up a field spec by name within a group."""
        groups = {
            "state": self.state_fields,
            "action": self.action_fields,
            "history": self.history_token_fields,
            "context": self.context_fields,
            "bidding": self.bidding_token_fields,
        }
        if group not in groups:
            raise ValueError(f"Unknown field group {group!r}")
        for spec in groups[group]:
            if spec.name == name:
                return spec
        raise KeyError(f"No {group!r} field named {name!r}")

    def compatibility_dict(self) -> dict[str, Any]:
        """The identity-relevant subset of the schema (description-stable).

        Two schemas with equal ``compatibility_dict()`` produce observations a
        model can consume interchangeably. Excludes ``description`` text and
        the documentation-only ``feature_version`` label (the ``schema_version``
        and semantic stamps carry the real identity).

        Item 3: the context and bidding-token groups are covered, so a model
        cannot silently gain/lose a public input.
        """
        return {
            "schema_version": self.schema_version,
            "max_history_len": self.max_history_len,
            "card_vector_dim": self.card_vector_dim,
            "seat_onehot_width": self.seat_onehot_width,
            "move_type_onehot_width": self.move_type_onehot_width,
            "bomb_onehot_width": self.bomb_onehot_width,
            "max_cards_left": self.max_cards_left,
            "bid_value_onehot_width": self.bid_value_onehot_width,
            "phase_onehot_width": self.phase_onehot_width,
            "ruleset_id_onehot_width": self.ruleset_id_onehot_width,
            "bidding_token_width": self.bidding_token_width,
            "max_bidding_tokens": self.max_bidding_tokens,
            "card_encoding_version": self.card_encoding_version,
            "move_type_encoding_version": self.move_type_encoding_version,
            "seat_mapping_version": self.seat_mapping_version,
            "history_encoding_version": self.history_encoding_version,
            "mask_semantics_version": self.mask_semantics_version,
            "truncation_semantics_version": self.truncation_semantics_version,
            "context_encoding_version": self.context_encoding_version,
            "bidding_encoding_version": self.bidding_encoding_version,
            "state_fields": [f.identity_dict() for f in self.state_fields],
            "action_fields": [f.identity_dict() for f in self.action_fields],
            "history_token_fields": [f.identity_dict() for f in self.history_token_fields],
            "context_fields": [f.identity_dict() for f in self.context_fields],
            "bidding_token_fields": [f.identity_dict() for f in self.bidding_token_fields],
        }

    def stable_hash(self) -> str:
        """Deterministic SHA-256 of :meth:`compatibility_dict`.

        Stable under ``description`` text edits (item 2). Changes when a field's
        name/shape/dtype changes, when a field is added/removed/reordered, when
        ``max_history_len`` changes, or when any semantic version stamp changes.
        Covers the context and bidding-token groups (item 3).
        """
        payload = json.dumps(self.compatibility_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    # Item 3: the public-context block. A V2 model consumes every field here
    # from the schema-described PublicContextBlock tensor, so P05 never needs
    # to reach into obs.public ad hoc (which would create an unversioned input).
    context_fields: tuple[FieldSpec, ...] = (
        FieldSpec("bottom_cards_revealed", (CARD_VECTOR_DIM,), "int8",
                  "public bottom cards (original identity), landlord-owned"),
        FieldSpec("bottom_cards_unplayed", (CARD_VECTOR_DIM,), "int8",
                  "public bottom cards not yet played by the landlord"),
        FieldSpec("bid_value_onehot", (BID_VALUE_ONEHOT_WIDTH,), "int8",
                  "one-hot final bid value (0 in legacy)"),
        FieldSpec("phase_onehot", (PHASE_ONEHOT_WIDTH,), "int8",
                  "one-hot game phase"),
        FieldSpec("rocket_count", (1,), "int8",
                  "number of rockets played (separate from bomb_count)"),
        FieldSpec("total_multiplier", (1,), "int32",
                  "public total score multiplier (int32: unbounded in standard ruleset)"),
        FieldSpec("ruleset_id_onehot", (RULESET_ID_ONEHOT_WIDTH,), "int8",
                  "one-hot ruleset id (legacy/standard)"),
    )

    # Item 3: the bidding-token group (variable-length, schema-described).
    bidding_token_fields: tuple[FieldSpec, ...] = (
        FieldSpec("bid_seat", (3,), "int8",
                  "one-hot bidder seat index (0/1/2)"),
        FieldSpec("bid_value", (BID_VALUE_ONEHOT_WIDTH,), "int8",
                  "one-hot bid value (0=pass/1/2/3)"),
        FieldSpec("is_pass", (1,), "int8", "1 if the bid was a pass"),
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
        context_fields=context_fields,
        bidding_token_fields=bidding_token_fields,
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


def context_width(schema: FeatureSchemaManifest) -> int:
    """Total flat width of the public-context block (item 3)."""
    return sum(int(np_prod(spec.shape)) for spec in schema.context_fields)


def bidding_token_width(schema: FeatureSchemaManifest) -> int:
    """Total flat width of one bidding token (item 3)."""
    return sum(int(np_prod(spec.shape)) for spec in schema.bidding_token_fields)


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
