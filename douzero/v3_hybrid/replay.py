"""Public-only replay records with immutable-snapshot Q provenance for H2."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import ClassVar, Iterable, Mapping

import torch

from douzero.models_v2.batch import (
    ModelInputBundle,
    _is_card_vector_field,
    observation_to_model_inputs,
)
from douzero.observation.encode_v2 import ObservationV2
from douzero.observation.schema import (
    action_width,
    build_v2_schema,
    context_width,
    history_token_width,
    state_width,
)
from douzero.runtime.policy_snapshot import PolicyLease

from .belief_policy import V3BeliefPolicy
from .config import DMC_TARGET_RAW, DMC_TARGET_SIGNED_LOG
from .model import V3_HYBRID_ROLES, V3HybridModel

V3_H2_REPLAY_SCHEMA_VERSION = 3
V3_H2_REPLAY_SEMANTICS = "selected_action_actor_snapshot_q_old_ruleset_v2"

_TRANSFORMS = frozenset({DMC_TARGET_RAW, DMC_TARGET_SIGNED_LOG})
_RULESET_IDENTITY_KEYS = frozenset({"ruleset_id", "ruleset_version", "ruleset_hash"})
_FEATURE_SCHEMA = build_v2_schema()
_FEATURE_SCHEMA_HASH = _FEATURE_SCHEMA.stable_hash()


def _normalize_ruleset_identity(identity: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(identity, Mapping) or set(identity) != _RULESET_IDENTITY_KEYS:
        raise ValueError("V3 replay ruleset identity fields mismatch")
    normalized: dict[str, str] = {}
    for name in ("ruleset_id", "ruleset_version", "ruleset_hash"):
        value = identity[name]
        if not isinstance(value, str) or not value:
            raise TypeError(f"V3 replay {name} must be a non-empty string")
        normalized[name] = value
    ruleset_hash = normalized["ruleset_hash"]
    if len(ruleset_hash) != 64 or any(
        character not in "0123456789abcdef" for character in ruleset_hash
    ):
        raise ValueError("V3 replay ruleset_hash must be a full SHA-256")
    return normalized


def _clone_public_bundle(observation: ObservationV2) -> ModelInputBundle:
    if not isinstance(observation, ObservationV2):
        raise TypeError("V3 replay capture requires a public ObservationV2")
    source = observation_to_model_inputs(observation)
    if source.strategy_features is not None or source.style_features is not None:
        raise ValueError("H2 replay does not support strategy or style features")
    return ModelInputBundle(
        state_card_vectors=tuple(value.detach().cpu().clone() for value in source.state_card_vectors),
        state_context_flat=source.state_context_flat.detach().cpu().clone(),
        context_card_vectors=tuple(value.detach().cpu().clone() for value in source.context_card_vectors),
        context_flat=source.context_flat.detach().cpu().clone(),
        history_tokens=source.history_tokens.detach().cpu().clone(),
        history_key_padding_mask=source.history_key_padding_mask.detach().cpu().clone(),
        action_features=source.action_features.detach().cpu().clone(),
        action_mask=source.action_mask.detach().cpu().clone(),
        acting_role=source.acting_role,
        feature_schema_hash=source.feature_schema_hash,
    )


def _validate_bundle(bundle: ModelInputBundle, expected_schema_hash: str) -> None:
    if not isinstance(bundle, ModelInputBundle):
        raise TypeError("V3 replay model_inputs must be a ModelInputBundle")
    if bundle.feature_schema_hash != expected_schema_hash:
        raise ValueError("V3 replay feature schema hash mismatch")
    if expected_schema_hash != _FEATURE_SCHEMA_HASH:
        raise ValueError("V3 replay requires the frozen Observation V2 schema")
    if bundle.acting_role not in V3_HYBRID_ROLES:
        raise ValueError("V3 replay contains an unsupported acting role")
    if bundle.strategy_features is not None or bundle.style_features is not None:
        raise ValueError("H2 replay cannot contain later-stage feature tensors")
    if not isinstance(bundle.state_card_vectors, tuple) or not isinstance(
        bundle.context_card_vectors, tuple
    ):
        raise TypeError("V3 replay card-vector groups must be tuples")
    schema = _FEATURE_SCHEMA

    def is_card(spec) -> bool:
        return _is_card_vector_field(
            spec.name, tuple(spec.shape), schema.card_vector_dim
        )

    expected_state_cards = sum(is_card(spec) for spec in schema.state_fields)
    expected_context_cards = sum(is_card(spec) for spec in schema.context_fields)
    if len(bundle.state_card_vectors) != expected_state_cards:
        raise ValueError("V3 replay state card-vector count mismatch")
    if len(bundle.context_card_vectors) != expected_context_cards:
        raise ValueError("V3 replay context card-vector count mismatch")

    card_vectors = (*bundle.state_card_vectors, *bundle.context_card_vectors)
    if any(
        not isinstance(value, torch.Tensor)
        or value.shape != (schema.card_vector_dim,)
        or value.dtype != torch.float32
        for value in card_vectors
    ):
        raise ValueError("V3 replay card-vector shape or dtype mismatch")
    expected_shapes = {
        "state_context_flat": (
            state_width(schema) - expected_state_cards * schema.card_vector_dim,
        ),
        "context_flat": (
            context_width(schema) - expected_context_cards * schema.card_vector_dim,
        ),
        "history_tokens": (
            schema.max_history_len,
            history_token_width(schema),
        ),
    }
    for name, shape in expected_shapes.items():
        value = getattr(bundle, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != torch.float32
        ):
            raise ValueError(f"V3 replay {name} shape or dtype mismatch")
    if (
        not isinstance(bundle.history_key_padding_mask, torch.Tensor)
        or bundle.history_key_padding_mask.shape != (schema.max_history_len,)
        or bundle.history_key_padding_mask.dtype != torch.bool
    ):
        raise ValueError("V3 replay history padding mask shape or dtype mismatch")
    if (
        not isinstance(bundle.action_features, torch.Tensor)
        or bundle.action_features.ndim != 2
        or bundle.action_features.shape[0] < 1
        or bundle.action_features.shape[1] != action_width(schema)
        or bundle.action_features.dtype != torch.float32
    ):
        raise ValueError("V3 replay action feature shape or dtype mismatch")
    count = int(bundle.action_features.shape[0])
    if (
        not isinstance(bundle.action_mask, torch.Tensor)
        or bundle.action_mask.shape != (count,)
        or bundle.action_mask.dtype != torch.bool
        or not bool(bundle.action_mask.all())
    ):
        raise ValueError("V3 replay action mask shape, dtype, or validity mismatch")

    tensors = (
        *card_vectors,
        bundle.state_context_flat,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features,
        bundle.action_mask,
    )
    for value in tensors:
        if not isinstance(value, torch.Tensor):
            raise TypeError("V3 replay public inputs must be tensors")
        if value.device.type != "cpu":
            raise ValueError("V3 replay public inputs must remain on CPU")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("V3 replay public inputs must be finite")


@dataclass(frozen=True)
class AdaptiveSnapshotProvenance:
    """Exact immutable policy lease and Q value used at actor decision time."""

    q_old: float
    policy_version: int
    snapshot_slot: int
    owner_id: int
    generation: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.q_old, bool)
            or not isinstance(self.q_old, (int, float))
            or not math.isfinite(self.q_old)
        ):
            raise ValueError("q_old must be finite")
        for name in ("policy_version", "snapshot_slot", "owner_id", "generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")

    def state_dict(self) -> dict[str, float | int]:
        return {
            "q_old": float(self.q_old),
            "policy_version": self.policy_version,
            "snapshot_slot": self.snapshot_slot,
            "owner_id": self.owner_id,
            "generation": self.generation,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "AdaptiveSnapshotProvenance":
        expected = {
            "q_old", "policy_version", "snapshot_slot", "owner_id", "generation"
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("adaptive snapshot provenance fields mismatch")
        return cls(
            q_old=payload["q_old"],
            policy_version=payload["policy_version"],
            snapshot_slot=payload["snapshot_slot"],
            owner_id=payload["owner_id"],
            generation=payload["generation"],
        )


@dataclass(frozen=True)
class PendingV3Transition:
    """A decision-time public record awaiting the terminal MC return."""

    model_inputs: ModelInputBundle
    selected_action_index: int
    role: str
    episode_id: str
    deal_id: str
    target_transform: str
    ruleset_id: str
    ruleset_version: str
    ruleset_hash: str
    adaptive_provenance: AdaptiveSnapshotProvenance | None = None

    @property
    def ruleset_identity(self) -> dict[str, str]:
        return {
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_hash": self.ruleset_hash,
        }

    def finalize(self, mc_return: float) -> "V3ReplayTransition":
        transition = V3ReplayTransition(
            model_inputs=self.model_inputs,
            selected_action_index=self.selected_action_index,
            role=self.role,
            episode_id=self.episode_id,
            deal_id=self.deal_id,
            target_transform=self.target_transform,
            ruleset_id=self.ruleset_id,
            ruleset_version=self.ruleset_version,
            ruleset_hash=self.ruleset_hash,
            mc_return=float(mc_return),
            adaptive_provenance=self.adaptive_provenance,
        )
        transition.validate(
            expected_schema_hash=self.model_inputs.feature_schema_hash,
            expected_target_transform=self.target_transform,
            expected_ruleset_identity=self.ruleset_identity,
            adaptive_required=self.adaptive_provenance is not None,
        )
        return transition


def _pending_from_observation(
    observation: ObservationV2,
    *,
    selected_action_index: int,
    episode_id: str,
    deal_id: str,
    target_transform: str,
    provenance: AdaptiveSnapshotProvenance | None,
) -> PendingV3Transition:
    bundle = _clone_public_bundle(observation)
    ruleset_identity = _normalize_ruleset_identity({
        "ruleset_id": observation.public.ruleset_id,
        "ruleset_version": observation.public.ruleset_version,
        "ruleset_hash": observation.public.ruleset_hash,
    })
    if isinstance(selected_action_index, bool) or not isinstance(selected_action_index, int):
        raise TypeError("selected_action_index must be an int")
    count = int(bundle.action_features.shape[0])
    if not 0 <= selected_action_index < count:
        raise ValueError("selected_action_index is outside the legal-action list")
    if not bool(bundle.action_mask[selected_action_index]):
        raise ValueError("selected_action_index references a masked action")
    if target_transform not in _TRANSFORMS:
        raise ValueError("unsupported replay target transform")
    for name, value in (("episode_id", episode_id), ("deal_id", deal_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    return PendingV3Transition(
        model_inputs=bundle,
        selected_action_index=selected_action_index,
        role=bundle.acting_role,
        episode_id=episode_id,
        deal_id=deal_id,
        target_transform=target_transform,
        **ruleset_identity,
        adaptive_provenance=provenance,
    )


def capture_plain_transition(
    observation: ObservationV2,
    *,
    selected_action_index: int,
    episode_id: str,
    deal_id: str,
    target_transform: str,
) -> PendingV3Transition:
    """Capture ordinary DMC replay without any Adaptive-DMC dependency."""

    return _pending_from_observation(
        observation,
        selected_action_index=selected_action_index,
        episode_id=episode_id,
        deal_id=deal_id,
        target_transform=target_transform,
        provenance=None,
    )


def capture_adaptive_transition(
    lease: PolicyLease[V3HybridModel | V3BeliefPolicy],
    observation: ObservationV2,
    *,
    selected_action_index: int,
    episode_id: str,
    deal_id: str,
    target_transform: str,
) -> PendingV3Transition:
    """Compute ``q_old`` from the exact immutable lease used by the actor."""

    if not isinstance(lease, PolicyLease):
        raise TypeError("Adaptive DMC capture requires a PolicyLease")
    if isinstance(lease.model, V3HybridModel):
        student = lease.model
    elif isinstance(lease.model, V3BeliefPolicy):
        student = lease.model.model
    else:
        raise TypeError(
            "Adaptive DMC lease must contain a V3HybridModel or V3BeliefPolicy"
        )
    if lease.model.training:
        raise ValueError("actor policy snapshot must be in eval mode")
    if target_transform != student.config.dmc_target_transform:
        raise ValueError("replay target transform does not match the snapshot model")
    with torch.inference_mode():
        output = lease.model.forward_observation(observation)
        if not 0 <= selected_action_index < output.num_actions:
            raise ValueError("selected_action_index is outside the legal-action list")
        if not bool(output.action_mask[selected_action_index]):
            raise ValueError("selected_action_index references a masked action")
        q_old = float(output.dmc_q[selected_action_index, 0].float().item())
    provenance = AdaptiveSnapshotProvenance(
        q_old=q_old,
        policy_version=lease.version,
        snapshot_slot=lease.slot,
        owner_id=lease.owner_id,
        generation=lease.generation,
    )
    return _pending_from_observation(
        observation,
        selected_action_index=selected_action_index,
        episode_id=episode_id,
        deal_id=deal_id,
        target_transform=target_transform,
        provenance=provenance,
    )


@dataclass(frozen=True)
class V3ReplayTransition:
    """Final H2 replay row containing one real selected legal action."""

    SCHEMA_VERSION: ClassVar[int] = V3_H2_REPLAY_SCHEMA_VERSION

    model_inputs: ModelInputBundle
    selected_action_index: int
    role: str
    episode_id: str
    deal_id: str
    target_transform: str
    ruleset_id: str
    ruleset_version: str
    ruleset_hash: str
    mc_return: float
    adaptive_provenance: AdaptiveSnapshotProvenance | None

    @property
    def ruleset_identity(self) -> dict[str, str]:
        return {
            "ruleset_id": self.ruleset_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_hash": self.ruleset_hash,
        }

    def validate(
        self,
        *,
        expected_schema_hash: str,
        expected_target_transform: str,
        expected_ruleset_identity: Mapping[str, object],
        adaptive_required: bool,
    ) -> None:
        _validate_bundle(self.model_inputs, expected_schema_hash)
        actual_ruleset = _normalize_ruleset_identity(self.ruleset_identity)
        expected_ruleset = _normalize_ruleset_identity(expected_ruleset_identity)
        if actual_ruleset != expected_ruleset:
            raise ValueError("V3 replay ruleset identity mismatch")
        if self.role != self.model_inputs.acting_role or self.role not in V3_HYBRID_ROLES:
            raise ValueError("V3 replay role does not match its public inputs")
        count = int(self.model_inputs.action_features.shape[0])
        if (
            isinstance(self.selected_action_index, bool)
            or not isinstance(self.selected_action_index, int)
            or not 0 <= self.selected_action_index < count
        ):
            raise ValueError("V3 replay selected action is outside the legal range")
        if not bool(self.model_inputs.action_mask[self.selected_action_index]):
            raise ValueError("V3 replay selected action is padded or masked")
        if self.target_transform != expected_target_transform:
            raise ValueError("V3 replay target transform mismatch")
        if not math.isfinite(self.mc_return):
            raise ValueError("V3 replay MC return must be finite")
        for name in ("episode_id", "deal_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"V3 replay {name} must be non-empty")
        if adaptive_required and self.adaptive_provenance is None:
            raise ValueError("Adaptive DMC replay is missing actor-snapshot q_old")
        if not adaptive_required and self.adaptive_provenance is not None:
            raise ValueError("ordinary DMC replay must not depend on q_old")

    def state_dict(self) -> dict[str, object]:
        bundle = self.model_inputs
        return {
            "schema_version": self.SCHEMA_VERSION,
            "semantics": V3_H2_REPLAY_SEMANTICS,
            "model_inputs": {
                "state_card_vectors": bundle.state_card_vectors,
                "state_context_flat": bundle.state_context_flat,
                "context_card_vectors": bundle.context_card_vectors,
                "context_flat": bundle.context_flat,
                "history_tokens": bundle.history_tokens,
                "history_key_padding_mask": bundle.history_key_padding_mask,
                "action_features": bundle.action_features,
                "action_mask": bundle.action_mask,
            },
            "acting_role": bundle.acting_role,
            "feature_schema_hash": bundle.feature_schema_hash,
            "selected_action_index": self.selected_action_index,
            "role": self.role,
            "episode_id": self.episode_id,
            "deal_id": self.deal_id,
            "target_transform": self.target_transform,
            "ruleset_identity": self.ruleset_identity,
            "mc_return": float(self.mc_return),
            "adaptive_provenance": (
                None
                if self.adaptive_provenance is None
                else self.adaptive_provenance.state_dict()
            ),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "V3ReplayTransition":
        expected = {
            "schema_version", "semantics", "model_inputs", "acting_role",
            "feature_schema_hash", "selected_action_index", "role", "episode_id",
            "deal_id", "target_transform", "ruleset_identity", "mc_return",
            "adaptive_provenance",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("V3 replay envelope fields mismatch")
        if payload["schema_version"] != V3_H2_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported V3 replay schema version")
        if payload["semantics"] != V3_H2_REPLAY_SEMANTICS:
            raise ValueError("V3 replay semantics mismatch")
        ruleset_identity = _normalize_ruleset_identity(payload["ruleset_identity"])
        tensors = payload["model_inputs"]
        tensor_keys = {
            "state_card_vectors", "state_context_flat", "context_card_vectors",
            "context_flat", "history_tokens", "history_key_padding_mask",
            "action_features", "action_mask",
        }
        if not isinstance(tensors, Mapping) or set(tensors) != tensor_keys:
            raise ValueError("V3 replay public tensor fields mismatch")
        for name in (
            "acting_role", "feature_schema_hash", "role", "episode_id",
            "deal_id", "target_transform",
        ):
            if not isinstance(payload[name], str):
                raise TypeError(f"V3 replay {name} must be a string")
        if (
            isinstance(payload["selected_action_index"], bool)
            or not isinstance(payload["selected_action_index"], int)
        ):
            raise TypeError("V3 replay selected_action_index must be an int")
        if (
            isinstance(payload["mc_return"], bool)
            or not isinstance(payload["mc_return"], (int, float))
        ):
            raise TypeError("V3 replay mc_return must be numeric")
        bundle = ModelInputBundle(
            state_card_vectors=tuple(tensors["state_card_vectors"]),
            state_context_flat=tensors["state_context_flat"],
            context_card_vectors=tuple(tensors["context_card_vectors"]),
            context_flat=tensors["context_flat"],
            history_tokens=tensors["history_tokens"],
            history_key_padding_mask=tensors["history_key_padding_mask"],
            action_features=tensors["action_features"],
            action_mask=tensors["action_mask"],
            acting_role=payload["acting_role"],
            feature_schema_hash=payload["feature_schema_hash"],
        )
        provenance_payload = payload["adaptive_provenance"]
        return cls(
            model_inputs=bundle,
            selected_action_index=payload["selected_action_index"],
            role=payload["role"],
            episode_id=payload["episode_id"],
            deal_id=payload["deal_id"],
            target_transform=payload["target_transform"],
            **ruleset_identity,
            mc_return=payload["mc_return"],
            adaptive_provenance=(
                None
                if provenance_payload is None
                else AdaptiveSnapshotProvenance.from_state_dict(provenance_payload)
            ),
        )


class V3ReplayBuffer:
    """Bounded H2 replay that refuses the wrong q_old schema mode."""

    def __init__(
        self,
        capacity: int,
        *,
        feature_schema_hash: str,
        target_transform: str,
        ruleset_identity: Mapping[str, object],
        adaptive_required: bool,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("V3 replay capacity must be a positive int")
        if not feature_schema_hash:
            raise ValueError("V3 replay requires a feature schema hash")
        if target_transform not in _TRANSFORMS:
            raise ValueError("V3 replay target transform is unsupported")
        if not isinstance(adaptive_required, bool):
            raise TypeError("adaptive_required must be bool")
        self.capacity = capacity
        self.feature_schema_hash = feature_schema_hash
        self.target_transform = target_transform
        self.ruleset_identity = _normalize_ruleset_identity(ruleset_identity)
        self.adaptive_required = adaptive_required
        self._records: deque[V3ReplayTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, transition: V3ReplayTransition) -> None:
        if not isinstance(transition, V3ReplayTransition):
            raise TypeError("V3 replay accepts only V3ReplayTransition")
        transition.validate(
            expected_schema_hash=self.feature_schema_hash,
            expected_target_transform=self.target_transform,
            expected_ruleset_identity=self.ruleset_identity,
            adaptive_required=self.adaptive_required,
        )
        self._records.append(transition)

    def extend(self, transitions: Iterable[V3ReplayTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, count: int, *, rng: random.Random) -> list[V3ReplayTransition]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("V3 replay sample count must be positive")
        if count > len(self._records):
            raise ValueError("V3 replay does not contain enough samples")
        if not isinstance(rng, random.Random):
            raise TypeError("V3 replay sampling requires an explicit Random")
        return rng.sample(list(self._records), count)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": V3_H2_REPLAY_SCHEMA_VERSION,
            "semantics": V3_H2_REPLAY_SEMANTICS,
            "capacity": self.capacity,
            "feature_schema_hash": self.feature_schema_hash,
            "target_transform": self.target_transform,
            "ruleset_identity": dict(self.ruleset_identity),
            "adaptive_required": self.adaptive_required,
            "records": [record.state_dict() for record in self._records],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "V3ReplayBuffer":
        expected = {
            "schema_version", "semantics", "capacity", "feature_schema_hash",
            "target_transform", "ruleset_identity", "adaptive_required", "records",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("V3 replay buffer fields mismatch")
        if payload["schema_version"] != V3_H2_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported V3 replay buffer schema version")
        if payload["semantics"] != V3_H2_REPLAY_SEMANTICS:
            raise ValueError("V3 replay buffer semantics mismatch")
        if (
            isinstance(payload["capacity"], bool)
            or not isinstance(payload["capacity"], int)
        ):
            raise TypeError("V3 replay buffer capacity must be an int")
        if not isinstance(payload["feature_schema_hash"], str):
            raise TypeError("V3 replay buffer schema hash must be a string")
        if not isinstance(payload["target_transform"], str):
            raise TypeError("V3 replay buffer target transform must be a string")
        if not isinstance(payload["adaptive_required"], bool):
            raise TypeError("V3 replay buffer adaptive_required must be bool")
        buffer = cls(
            payload["capacity"],
            feature_schema_hash=payload["feature_schema_hash"],
            target_transform=payload["target_transform"],
            ruleset_identity=payload["ruleset_identity"],
            adaptive_required=payload["adaptive_required"],
        )
        records = payload["records"]
        if not isinstance(records, list):
            raise ValueError("V3 replay records must be a list")
        if len(records) > buffer.capacity:
            raise ValueError("V3 replay records exceed the declared capacity")
        for raw in records:
            record = V3ReplayTransition.from_state_dict(raw)
            buffer.add(record)
        return buffer
