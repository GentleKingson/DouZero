"""Conversion from :class:`ObservationV2` to Model V2 tensor inputs (P05).

The model's :meth:`~douzero.models_v2.model.ModelV2.forward` takes the state and
context blocks split into their card-vector and flat-field portions. This
module performs that split deterministically from an
:class:`~douzero.observation.encode_v2.ObservationV2`, so a caller (DeepAgentV2,
a test, or a future training loop) does not have to know the schema's field
layout.

Field-type identification (bug #2 fix)
--------------------------------------
A field is a "card-vector field" if and only if BOTH hold:

1. its trailing shape equals ``card_vector_dim`` (54), AND
2. its name is in the canonical card-field name set defined by the schema
   (``CARD_VECTOR_FIELD_NAMES`` below).

Relying on width alone is forbidden: two unrelated one-hot/count fields could
someday share a width of 54 by coincidence, and silently routing one of them
through the card projection would corrupt the state trunk. The name check makes
the card-field set an explicit, schema-documented contract; the width check is
a redundant guard that catches a schema typo.

The card-field name set is fixed by the V2 schema (see
``douzero/observation/schema.py`` ``build_v2_schema``) and participates in the
schema's ``stable_hash``, so a model is bound to the exact card-field layout it
was constructed against.

The split is order-preserving within each group (card fields keep their schema
order; flat fields keep their schema order), so swapping two card-field VALUES
at the call site changes which value lands in which concatenation slot in the
state encoder — field identity is preserved end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from douzero.observation.encode_v2 import ObservationV2

#: The canonical set of card-vector field names, as documented by the V2 schema
#: (``build_v2_schema`` in ``douzero/observation/schema.py``). A field is routed
#: through the card projection ONLY if its name is in this set AND its trailing
#: shape equals ``card_vector_dim``. This is an explicit contract, not a
#: width-based guess. The set is frozen so it cannot drift at runtime.
_CARD_VECTOR_FIELD_NAMES: frozenset[str] = frozenset({
    # state block card fields
    "my_handcards",
    "other_handcards",
    "landlord_played",
    "landlord_down_played",
    "landlord_up_played",
    "last_move",
    # public-context block card fields
    "bottom_cards_revealed",
    "bottom_cards_unplayed",
})

#: The number of card-vector fields the model's state encoder expects. Must
#: match the value used to construct the StateEncoder (model.py). Asserted at
#: split time so a schema that adds/removes a card field fails loudly here
#: rather than producing a width mismatch deep in the forward pass.
EXPECTED_NUM_CARD_FIELDS = len(_CARD_VECTOR_FIELD_NAMES)


def _is_card_vector_field(name: str, shape: tuple[int, ...], card_vector_dim: int) -> bool:
    """Return True iff ``name`` is a canonical card-vector field of the right width.

    Both conditions (name in the canonical set AND trailing shape ==
    ``card_vector_dim``) must hold. This refuses to guess from width alone.
    """
    if name not in _CARD_VECTOR_FIELD_NAMES:
        return False
    return len(shape) >= 1 and shape[-1] == card_vector_dim


def _split_card_and_flat(
    field_specs,
    field_getter,
    card_vector_dim: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Split a schema field group into (card vectors, flat non-card fields).

    ``field_specs`` is the ordered tuple of :class:`FieldSpec`. Card-vector
    fields are routed to the card projection; everything else is flattened and
    concatenated in order. Both groups preserve schema order.
    """
    card_vecs: list[torch.Tensor] = []
    flat_parts: list[np.ndarray] = []
    for spec in field_specs:
        arr = np.asarray(field_getter(spec.name)).astype(np.float32)
        if _is_card_vector_field(spec.name, tuple(spec.shape), card_vector_dim):
            card_vecs.append(torch.from_numpy(arr.reshape(-1)))
        else:
            flat_parts.append(arr.reshape(-1))
    flat = (
        torch.from_numpy(np.concatenate(flat_parts).astype(np.float32))
        if flat_parts
        else torch.zeros(0, dtype=torch.float32)
    )
    return tuple(card_vecs), flat


@dataclass
class ModelInputBundle:
    """Converted tensor inputs for one decision, ready for ``ModelV2.forward``."""

    state_card_vectors: tuple[torch.Tensor, ...]
    state_context_flat: torch.Tensor
    context_card_vectors: tuple[torch.Tensor, ...]
    context_flat: torch.Tensor
    history_tokens: torch.Tensor
    history_key_padding_mask: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    acting_role: str
    feature_schema_hash: str
    strategy_features: torch.Tensor | None = None

    def to(self, device) -> "ModelInputBundle":
        """Move every tensor in the bundle to ``device``. Returns self."""
        self.state_card_vectors = tuple(t.to(device) for t in self.state_card_vectors)
        self.state_context_flat = self.state_context_flat.to(device)
        self.context_card_vectors = tuple(t.to(device) for t in self.context_card_vectors)
        self.context_flat = self.context_flat.to(device)
        self.history_tokens = self.history_tokens.to(device)
        self.history_key_padding_mask = self.history_key_padding_mask.to(device)
        self.action_features = self.action_features.to(device)
        self.action_mask = self.action_mask.to(device)
        if self.strategy_features is not None:
            self.strategy_features = self.strategy_features.to(device)
        return self


def observation_to_model_inputs(
    obs: ObservationV2,
    strategy_config=None,
) -> ModelInputBundle:
    """Convert an :class:`ObservationV2` into a :class:`ModelInputBundle`.

    This is the canonical bridge from the V2 observation container to the
    model's tensor contract. It performs NO privileged-field access: it reads
    only the public tensor blocks (``state``, ``context``, ``history``,
    ``actions``) and the public acting role. It also carries the observation's
    ``feature_schema_hash`` so :class:`DeepAgentV2` can reject a model/obs
    schema mismatch before forwarding.

    Raises ``ValueError`` if the observation has zero legal actions (the model
    cannot select from an empty action set) or if the schema's card-field
    layout does not match the model's expectation.
    """
    schema = obs.schema
    cvd = schema.card_vector_dim

    # State block: read each field array off the StateBlock by name, in the
    # schema's fixed field order.
    state_card_vecs, state_flat = _split_card_and_flat(
        schema.state_fields,
        lambda name: getattr(obs.state, name),
        cvd,
    )

    # Public-context block.
    context_card_vecs, context_flat = _split_card_and_flat(
        schema.context_fields,
        lambda name: getattr(obs.context, name),
        cvd,
    )

    total_card = len(state_card_vecs) + len(context_card_vecs)
    if total_card != EXPECTED_NUM_CARD_FIELDS:
        raise ValueError(
            f"Observation schema yields {total_card} card-vector fields "
            f"({len(state_card_vecs)} state + {len(context_card_vecs)} context), "
            f"but the model expects {EXPECTED_NUM_CARD_FIELDS}. The schema's "
            f"card-field layout has drifted from the model's contract."
        )

    # Zero legal actions is a caller error (a decision with no legal actions is
    # undefined). Reject here so the model never receives an empty action batch.
    n_actions = obs.actions.features.shape[0]
    if n_actions == 0:
        raise ValueError(
            "ObservationV2 has zero legal actions; the model cannot select "
            "from an empty action set."
        )

    # History tokens + padding mask.
    history_tokens = torch.from_numpy(
        np.asarray(obs.history.tokens).astype(np.float32)
    )
    history_kpm = torch.from_numpy(
        np.asarray(obs.history.key_padding_mask).astype(bool)
    )

    # Legal actions.
    action_features = torch.from_numpy(
        np.asarray(obs.actions.features).astype(np.float32)
    )
    if action_features.ndim != 2:
        raise ValueError(
            f"obs.actions.features must be 2-D (N, action_width), got "
            f"{tuple(action_features.shape)}"
        )
    action_mask = torch.from_numpy(
        np.asarray(obs.actions.action_mask).astype(bool)
    )

    strategy_features = None
    if strategy_config is not None:
        from douzero.strategy.features import build_strategy_feature_matrix

        strategy_features = torch.from_numpy(
            np.asarray(
                build_strategy_feature_matrix(obs.public, strategy_config)
            ).astype(np.float32)
        )

    return ModelInputBundle(
        state_card_vectors=state_card_vecs,
        state_context_flat=state_flat,
        context_card_vectors=context_card_vecs,
        context_flat=context_flat,
        history_tokens=history_tokens,
        history_key_padding_mask=history_kpm,
        action_features=action_features,
        action_mask=action_mask,
        acting_role=obs.public.acting_role,
        feature_schema_hash=obs.feature_schema_hash,
        strategy_features=strategy_features,
    )
