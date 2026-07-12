"""Conversion from :class:`ObservationV2` to Model V2 tensor inputs (P05).

The model's :meth:`~douzero.models_v2.model.ModelV2.forward` takes the state and
context blocks split into their card-vector and flat-field portions. This
module performs that split deterministically from an
:class:`~douzero.observation.encode_v2.ObservationV2`, so a caller (DeepAgentV2,
a test, or a future training loop) does not have to know the schema's field
layout.

The split is schema-driven: the card-vector fields are those whose trailing
shape equals ``card_vector_dim``; everything else is flattened into the
context path. This means a future public field is routed correctly without a
code change here, as long as the schema declares its shape.

A :class:`ModelInputBundle` holds the converted tensors for one decision. It is
a plain dataclass (not frozen) because the tensors are torch tensors that the
caller may move across devices; freezing would not prevent in-place ops on the
tensors themselves and would only complicate device moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from douzero.observation.encode_v2 import ObservationV2


def _split_card_and_flat(
    field_names: tuple[str, ...],
    field_getter,
    card_vector_dim: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Split a schema field group into (card vectors, flat non-card fields).

    ``field_getter(name)`` returns the numpy array for that field. Fields whose
    trailing shape equals ``card_vector_dim`` are treated as card vectors; the
    rest are flattened and concatenated in order.
    """
    card_vecs: list[torch.Tensor] = []
    flat_parts: list[np.ndarray] = []
    for name in field_names:
        arr = np.asarray(field_getter(name)).astype(np.float32)
        if arr.ndim >= 1 and arr.shape[-1] == card_vector_dim and arr.size == card_vector_dim:
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
        return self


def observation_to_model_inputs(obs: ObservationV2) -> ModelInputBundle:
    """Convert an :class:`ObservationV2` into a :class:`ModelInputBundle`.

    This is the canonical bridge from the V2 observation container to the
    model's tensor contract. It performs NO privileged-field access: it reads
    only the public tensor blocks (``state``, ``context``, ``history``,
    ``actions``) and the public acting role.
    """
    schema = obs.schema
    cvd = schema.card_vector_dim

    # State block: read each field array off the StateBlock by name.
    state_field_names = tuple(f.name for f in schema.state_fields)

    def state_getter(name: str) -> np.ndarray:
        return getattr(obs.state, name)

    state_card_vecs, state_flat = _split_card_and_flat(
        state_field_names, state_getter, cvd
    )

    # Public-context block.
    context_field_names = tuple(f.name for f in schema.context_fields)

    def context_getter(name: str) -> np.ndarray:
        return getattr(obs.context, name)

    context_card_vecs, context_flat = _split_card_and_flat(
        context_field_names, context_getter, cvd
    )

    # History tokens + padding mask. The HistoryTokenBatch stores tokens with a
    # trailing valid flag as the LAST token field; we rely on the batch's own
    # key_padding_mask rather than re-deriving it, so a future schema change to
    # the token layout cannot desync the mask.
    history_tokens = torch.from_numpy(
        np.asarray(obs.history.tokens).astype(np.float32)
    )
    # key_padding_mask is True for padding (PyTorch convention).
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

    acting_role = obs.public.acting_role

    return ModelInputBundle(
        state_card_vectors=state_card_vecs,
        state_context_flat=state_flat,
        context_card_vectors=context_card_vecs,
        context_flat=context_flat,
        history_tokens=history_tokens,
        history_key_padding_mask=history_kpm,
        action_features=action_features,
        action_mask=action_mask,
        acting_role=acting_role,
    )
