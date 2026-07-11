"""Observation V2: versioned public and privileged observations (P03).

This package implements the P03 observation schema:

- :mod:`douzero.observation.cards` — versioned 54-dim card encoding (the single
  source of truth for the legacy layout).
- :mod:`douzero.observation.seats` — canonical relative-seat mapping
  (SELF/NEXT/PREVIOUS/LANDLORD/TEAMMATE/OPPONENT).
- :mod:`douzero.observation.schema` — :class:`FeatureSchemaManifest` deriving
  every field width from named constants (no magic 319/373/430/484).
- :mod:`douzero.observation.history` — :class:`HistoryTokenBatch` with a
  configurable ``max_history_len`` and an explicit padding mask.
- :mod:`douzero.observation.public` — :class:`PublicObservation` (the ONLY
  thing a deployment model may see) + the public unseen-pool helper.
- :mod:`douzero.observation.privileged` — :class:`PrivilegedObservation`
  (training-only; true hidden hands live here and nowhere else in this layer).
- :mod:`douzero.observation.encode_v2` — :func:`get_obs_v2` encoding the state
  once per decision plus a :class:`LegalActionBatch`.
- :mod:`douzero.observation.legacy_adapter` — reconstruct the legacy
  ``x_batch``/``z_batch`` tensors from a V2 observation (transition bridge).

The legacy encoder (``douzero.env.env.get_obs``) is unchanged and remains the
default; V2 is opt-in via ``feature_version="v2"``.
"""

from douzero.observation.cards import (
    BIG_JOKER,
    BIG_JOKER_OFFSET,
    CARD_VECTOR_DIM,
    DECK,
    NUMERIC_RANKS,
    SMALL_JOKER,
    SMALL_JOKER_OFFSET,
    cards_to_vector,
)
from douzero.observation.encode_v2 import (
    LegalActionBatch,
    ObservationV2,
    StateBlock,
    get_obs_v2,
)
from douzero.observation.history import (
    HistoryMove,
    HistoryTokenBatch,
    encode_history,
    encode_history_token,
)
from douzero.observation.legacy_adapter import legacy_observation_from_v2
from douzero.observation.privileged import (
    PRIVILEGED_KIND,
    PrivilegedObservation,
    is_privileged,
)
from douzero.observation.public import (
    PUBLIC_KIND,
    PublicBottomCards,
    PublicObservation,
    build_public_observation,
    compute_belief_unknown_pool,
    compute_unseen_pool,
)
from douzero.observation.schema import (
    FEATURE_VERSION_LEGACY,
    FEATURE_VERSION_V2,
    FeatureSchemaManifest,
    FieldSpec,
    build_v2_schema,
)
from douzero.observation.seats import (
    ALL_ROLES,
    FARMER_ROLES,
    LANDLORD_ROLE,
    RELATIVE_SEATS,
    SEAT_LANDLORD,
    SEAT_NEXT,
    SEAT_OPPONENT,
    SEAT_PREVIOUS,
    SEAT_SELF,
    SEAT_TEAMMATE,
    is_farmer,
    is_landlord,
    next_seat,
    previous_seat,
    relative_seat,
    seats_from,
    teammate,
)

__all__ = [
    # cards
    "BIG_JOKER", "BIG_JOKER_OFFSET", "CARD_VECTOR_DIM", "DECK",
    "NUMERIC_RANKS", "SMALL_JOKER", "SMALL_JOKER_OFFSET", "cards_to_vector",
    # seats
    "ALL_ROLES", "FARMER_ROLES", "LANDLORD_ROLE", "RELATIVE_SEATS",
    "SEAT_LANDLORD", "SEAT_NEXT", "SEAT_OPPONENT", "SEAT_PREVIOUS",
    "SEAT_SELF", "SEAT_TEAMMATE", "is_farmer", "is_landlord", "next_seat",
    "previous_seat", "relative_seat", "seats_from", "teammate",
    # schema
    "FEATURE_VERSION_LEGACY", "FEATURE_VERSION_V2", "FeatureSchemaManifest",
    "FieldSpec", "build_v2_schema",
    # history
    "HistoryMove", "HistoryTokenBatch", "encode_history",
    "encode_history_token",
    # public
    "PUBLIC_KIND", "PublicBottomCards", "PublicObservation",
    "build_public_observation", "compute_belief_unknown_pool",
    "compute_unseen_pool",
    # privileged
    "PRIVILEGED_KIND", "PrivilegedObservation", "is_privileged",
    # encoder
    "LegalActionBatch", "ObservationV2", "StateBlock", "get_obs_v2",
    # legacy adapter
    "legacy_observation_from_v2",
]
