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
from douzero.models_v2.config import SUPPORTED_ROLES

SUPPORTED_ROLE_TO_INDEX = {role: index for index, role in enumerate(SUPPORTED_ROLES)}

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
    style_features: torch.Tensor | None = None

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
        if self.style_features is not None:
            self.style_features = self.style_features.to(device)
        return self


@dataclass
class BatchedModelInputBundle:
    """Padded tensor inputs for multiple independent V2 decisions.

    The scalar :class:`ModelInputBundle` contract remains unchanged.  This
    companion type adds only a leading decision dimension and pads legal
    actions to ``Amax``; ``action_mask`` is the sole authority for which rows
    are real.  ``chosen_action_index`` is optional for inference and required
    by the learner's gathered loss path.
    """

    state_card_vectors: tuple[torch.Tensor, ...]
    state_context_flat: torch.Tensor
    context_card_vectors: tuple[torch.Tensor, ...]
    context_flat: torch.Tensor
    history_tokens: torch.Tensor
    history_key_padding_mask: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    acting_role: torch.Tensor
    chosen_action_index: torch.Tensor | None
    feature_schema_hashes: tuple[str, ...]
    strategy_features: torch.Tensor | None = None
    style_features: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.action_features.shape[0])

    @property
    def max_actions(self) -> int:
        return int(self.action_features.shape[1])

    def to(
        self, device, *, non_blocking: bool = False
    ) -> "BatchedModelInputBundle":
        """Move tensor fields to ``device`` and return this bundle."""
        move = lambda tensor: tensor.to(device, non_blocking=non_blocking)
        self.state_card_vectors = tuple(move(t) for t in self.state_card_vectors)
        self.state_context_flat = move(self.state_context_flat)
        self.context_card_vectors = tuple(move(t) for t in self.context_card_vectors)
        self.context_flat = move(self.context_flat)
        self.history_tokens = move(self.history_tokens)
        self.history_key_padding_mask = move(self.history_key_padding_mask)
        self.action_features = move(self.action_features)
        self.action_mask = move(self.action_mask)
        self.acting_role = move(self.acting_role)
        if self.chosen_action_index is not None:
            self.chosen_action_index = move(self.chosen_action_index)
        if self.strategy_features is not None:
            self.strategy_features = move(self.strategy_features)
        if self.style_features is not None:
            self.style_features = move(self.style_features)
        return self


def observation_batch_to_model_inputs(
    observations: list[ObservationV2] | tuple[ObservationV2, ...],
    chosen_action_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    *,
    strategy_config=None,
    style_enabled: bool = False,
    pad_to_actions: int | None = None,
) -> BatchedModelInputBundle:
    """Tensorize and pad a decision batch without changing scalar semantics.

    ``pad_to_actions`` is used by action-count buckets.  It may exceed the
    largest decision but may never truncate it.  A batch containing an empty
    legal-action set fails closed before any model work is attempted.
    """
    if not observations:
        raise ValueError("observation batch must contain at least one decision")
    bundles = [
        observation_to_model_inputs(
            obs, strategy_config=strategy_config, style_enabled=style_enabled
        )
        for obs in observations
    ]
    return model_input_bundles_to_batch(
        bundles, chosen_action_indices, pad_to_actions=pad_to_actions
    )


def model_input_bundles_to_batch(
    bundles: list[ModelInputBundle] | tuple[ModelInputBundle, ...],
    chosen_action_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    *,
    pad_to_actions: int | None = None,
) -> BatchedModelInputBundle:
    """Stack already tensorized scalar bundles into one padded batch."""
    if not bundles:
        raise ValueError("model input bundle batch must not be empty")
    hashes = tuple(bundle.feature_schema_hash for bundle in bundles)
    if len(set(hashes)) != 1:
        raise ValueError("all observations in a model batch must share one schema")
    counts = [int(bundle.action_features.shape[0]) for bundle in bundles]
    if any(count <= 0 for count in counts):
        raise ValueError("every batched decision must have at least one legal action")
    for index, (bundle, count) in enumerate(zip(bundles, counts)):
        if bundle.action_mask.shape != (count,) or bundle.action_mask.dtype != torch.bool:
            raise ValueError(
                f"bundle {index} action_mask must be bool with shape ({count},)"
            )
        if bundle.action_mask.device.type == "cuda":
            torch._assert_async(
                bundle.action_mask.any(),
                f"bundle {index} must contain a legal action",
            )
        elif not bool(bundle.action_mask.any()):
            raise ValueError(f"bundle {index} must contain a legal action")
    max_actions = max(counts)
    if pad_to_actions is not None:
        if pad_to_actions < max_actions:
            raise ValueError(
                f"pad_to_actions={pad_to_actions} would truncate {max_actions} actions"
            )
        max_actions = int(pad_to_actions)

    action_width = int(bundles[0].action_features.shape[1])
    actions = bundles[0].action_features.new_zeros(
        (len(bundles), max_actions, action_width)
    )
    action_mask = bundles[0].action_mask.new_zeros((len(bundles), max_actions))
    strategy = None
    if bundles[0].strategy_features is not None:
        strategy_width = int(bundles[0].strategy_features.shape[1])
        strategy = bundles[0].strategy_features.new_zeros(
            (len(bundles), max_actions, strategy_width)
        )
    for index, bundle in enumerate(bundles):
        count = counts[index]
        actions[index, :count] = bundle.action_features
        action_mask[index, :count] = bundle.action_mask
        if strategy is not None:
            if bundle.strategy_features is None:
                raise ValueError("strategy features are partially populated")
            strategy[index, :count] = bundle.strategy_features

    chosen = None
    if chosen_action_indices is not None:
        chosen = torch.as_tensor(
            chosen_action_indices,
            dtype=torch.long,
            device=bundles[0].action_features.device,
        )
        if chosen.shape != (len(bundles),):
            raise ValueError(
                f"chosen_action_indices must have shape ({len(bundles)},), "
                f"got {tuple(chosen.shape)}"
            )
        if chosen.device.type == "cuda":
            action_counts = chosen.new_tensor(counts)
            torch._assert_async(
                ((chosen >= 0) & (chosen < action_counts)).all(),
                "chosen action is outside its row's legal range",
            )
        else:
            for row, count in enumerate(counts):
                value = int(chosen[row].item())
                if value < 0 or value >= count:
                    raise ValueError(
                        f"chosen action {value} is outside row {row}'s legal range "
                        f"[0, {count})"
                    )

    try:
        role_indices = bundles[0].action_features.new_tensor(
            [SUPPORTED_ROLE_TO_INDEX[bundle.acting_role] for bundle in bundles],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"unsupported acting role: {exc.args[0]!r}") from exc
    styles = None
    if bundles[0].style_features is not None:
        if any(bundle.style_features is None for bundle in bundles):
            raise ValueError("style features are partially populated")
        styles = torch.stack([bundle.style_features for bundle in bundles])
    return BatchedModelInputBundle(
        state_card_vectors=tuple(
            torch.stack([bundle.state_card_vectors[i] for bundle in bundles])
            for i in range(len(bundles[0].state_card_vectors))
        ),
        state_context_flat=torch.stack([b.state_context_flat for b in bundles]),
        context_card_vectors=tuple(
            torch.stack([bundle.context_card_vectors[i] for bundle in bundles])
            for i in range(len(bundles[0].context_card_vectors))
        ),
        context_flat=torch.stack([b.context_flat for b in bundles]),
        history_tokens=torch.stack([b.history_tokens for b in bundles]),
        history_key_padding_mask=torch.stack(
            [b.history_key_padding_mask for b in bundles]
        ),
        action_features=actions,
        action_mask=action_mask,
        acting_role=role_indices,
        chosen_action_index=chosen,
        feature_schema_hashes=hashes,
        strategy_features=strategy,
        style_features=styles,
    )


def observation_to_model_inputs(
    obs: ObservationV2,
    strategy_config=None,
    style_enabled: bool = False,
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

    style_features = None
    if style_enabled:
        from douzero.style.features import build_style_features

        style_features = torch.from_numpy(build_style_features(obs.public))

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
        style_features=style_features,
    )
