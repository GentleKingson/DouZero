"""H6 public replay schema with optional public strategy/style tensors."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping

import torch

from douzero.models_v2.batch import ModelInputBundle

from .config import (
    DMC_TARGET_RAW,
    DMC_TARGET_SIGNED_LOG,
    V3HybridModelConfig,
)
from .replay import (
    AdaptiveSnapshotProvenance,
    V3ReplayTransition,
    _normalize_ruleset_identity,
)

V3_H6_REPLAY_SCHEMA_VERSION = 1
V3_H6_REPLAY_SEMANTICS = (
    "selected_public_action_h2_provenance_plus_public_h6_features_v1"
)


def _transition_state(transition: V3ReplayTransition) -> dict[str, object]:
    bundle = transition.model_inputs
    return {
        "schema_version": V3_H6_REPLAY_SCHEMA_VERSION,
        "semantics": V3_H6_REPLAY_SEMANTICS,
        "model_inputs": {
            "state_card_vectors": bundle.state_card_vectors,
            "state_context_flat": bundle.state_context_flat,
            "context_card_vectors": bundle.context_card_vectors,
            "context_flat": bundle.context_flat,
            "history_tokens": bundle.history_tokens,
            "history_key_padding_mask": bundle.history_key_padding_mask,
            "action_features": bundle.action_features,
            "action_mask": bundle.action_mask,
            "strategy_features": bundle.strategy_features,
            "style_features": bundle.style_features,
        },
        "acting_role": bundle.acting_role,
        "feature_schema_hash": bundle.feature_schema_hash,
        "selected_action_index": transition.selected_action_index,
        "role": transition.role,
        "episode_id": transition.episode_id,
        "deal_id": transition.deal_id,
        "target_transform": transition.target_transform,
        "ruleset_identity": transition.ruleset_identity,
        "mc_return": float(transition.mc_return),
        "adaptive_provenance": (
            None
            if transition.adaptive_provenance is None
            else transition.adaptive_provenance.state_dict()
        ),
    }


def _transition_from_state(payload: Mapping[str, object]) -> V3ReplayTransition:
    expected = {
        "schema_version", "semantics", "model_inputs", "acting_role",
        "feature_schema_hash", "selected_action_index", "role", "episode_id",
        "deal_id", "target_transform", "ruleset_identity", "mc_return",
        "adaptive_provenance",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("H6 replay transition fields mismatch")
    if payload["schema_version"] != V3_H6_REPLAY_SCHEMA_VERSION:
        raise ValueError("H6 replay schema version mismatch")
    if payload["semantics"] != V3_H6_REPLAY_SEMANTICS:
        raise ValueError("H6 replay semantics mismatch")
    tensors = payload["model_inputs"]
    tensor_fields = {
        "state_card_vectors", "state_context_flat", "context_card_vectors",
        "context_flat", "history_tokens", "history_key_padding_mask",
        "action_features", "action_mask", "strategy_features", "style_features",
    }
    if not isinstance(tensors, Mapping) or set(tensors) != tensor_fields:
        raise ValueError("H6 replay public tensor fields mismatch")
    ruleset = payload["ruleset_identity"]
    if not isinstance(ruleset, Mapping) or set(ruleset) != {
        "ruleset_id", "ruleset_version", "ruleset_hash"
    }:
        raise ValueError("H6 replay ruleset identity fields mismatch")
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
        strategy_features=tensors["strategy_features"],
        style_features=tensors["style_features"],
    )
    provenance = payload["adaptive_provenance"]
    return V3ReplayTransition(
        model_inputs=bundle,
        selected_action_index=payload["selected_action_index"],
        role=payload["role"],
        episode_id=payload["episode_id"],
        deal_id=payload["deal_id"],
        target_transform=payload["target_transform"],
        ruleset_id=ruleset["ruleset_id"],
        ruleset_version=ruleset["ruleset_version"],
        ruleset_hash=ruleset["ruleset_hash"],
        mc_return=payload["mc_return"],
        adaptive_provenance=(
            None
            if provenance is None
            else AdaptiveSnapshotProvenance.from_state_dict(provenance)
        ),
    )


class V3H6ReplayBuffer:
    """Bounded public replay; privileged labels stay in separate sidecars."""

    STATE_FORMAT = "v3-hybrid-h6-public-replay-buffer-v1"

    def __init__(
        self,
        capacity: int,
        *,
        model_config: V3HybridModelConfig,
        feature_schema_hash: str,
        target_transform: str,
        ruleset_identity: Mapping[str, object],
        adaptive_required: bool,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("H6 replay capacity must be a positive int")
        if not isinstance(model_config, V3HybridModelConfig):
            raise TypeError("H6 replay requires V3HybridModelConfig")
        if not isinstance(adaptive_required, bool):
            raise TypeError("H6 replay adaptive_required must be bool")
        if target_transform not in {DMC_TARGET_RAW, DMC_TARGET_SIGNED_LOG}:
            raise ValueError("H6 replay target transform is unsupported")
        normalized_ruleset = _normalize_ruleset_identity(ruleset_identity)
        self.capacity = capacity
        self.model_config = model_config
        self.feature_schema_hash = feature_schema_hash
        self.target_transform = target_transform
        self.ruleset_identity = normalized_ruleset
        self.adaptive_required = adaptive_required
        self._records: deque[V3ReplayTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def _validate(self, transition: V3ReplayTransition) -> None:
        transition.validate(
            expected_schema_hash=self.feature_schema_hash,
            expected_target_transform=self.target_transform,
            expected_ruleset_identity=self.ruleset_identity,
            adaptive_required=self.adaptive_required,
            strategy_features_allowed=self.model_config.strategy_features_enabled,
            style_features_allowed=self.model_config.style_enabled,
        )

    def add(self, transition: V3ReplayTransition) -> None:
        if not isinstance(transition, V3ReplayTransition):
            raise TypeError("H6 replay accepts only V3ReplayTransition")
        self._validate(transition)
        self._records.append(transition)

    def extend(self, transitions: Iterable[V3ReplayTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self.STATE_FORMAT,
            "capacity": self.capacity,
            "model_config_hash": self.model_config.stable_hash(),
            "feature_schema_hash": self.feature_schema_hash,
            "target_transform": self.target_transform,
            "ruleset_identity": dict(self.ruleset_identity),
            "adaptive_required": self.adaptive_required,
            "records": [_transition_state(row) for row in self._records],
            "privileged_sidecars": "excluded",
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        expected = {
            "format", "capacity", "model_config_hash", "feature_schema_hash",
            "target_transform", "ruleset_identity", "adaptive_required",
            "records", "privileged_sidecars",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("H6 replay buffer fields mismatch")
        identity = {
            "format": self.STATE_FORMAT,
            "capacity": self.capacity,
            "model_config_hash": self.model_config.stable_hash(),
            "feature_schema_hash": self.feature_schema_hash,
            "target_transform": self.target_transform,
            "ruleset_identity": self.ruleset_identity,
            "adaptive_required": self.adaptive_required,
            "privileged_sidecars": "excluded",
        }
        for name, value in identity.items():
            if payload[name] != value:
                raise ValueError(f"H6 replay buffer {name} mismatch")
        records = payload["records"]
        if not isinstance(records, list) or len(records) > self.capacity:
            raise ValueError("H6 replay record count is invalid")
        parsed = [_transition_from_state(item) for item in records]
        for transition in parsed:
            self._validate(transition)
        self._records = deque(parsed, maxlen=self.capacity)


def assert_public_replay_payload(payload: object) -> None:
    """Reject privileged names recursively before publishing replay evidence."""

    forbidden = {
        "all_handcards", "hidden_hand", "privileged_observation",
        "belief_label", "belief_labels", "belief_samples",
        "oracle_sample", "oracle_samples", "bc_samples", "strategy_targets",
        "bidding_batch", "trajectories", "cooperation_trajectories",
        "privileged_mixer_state", "mixer_state", "cooperation_state_dict",
        "optimizer", "optimizer_state", "optimizer_state_dict",
        "teacher_state_dict",
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            violations = forbidden & {str(key).lower() for key in value}
            if violations:
                raise ValueError(
                    f"public H6 replay contains privileged fields: {sorted(violations)}"
                )
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, torch.Tensor):
            return

    visit(payload)


__all__ = [
    "V3_H6_REPLAY_SCHEMA_VERSION",
    "V3_H6_REPLAY_SEMANTICS",
    "V3H6ReplayBuffer",
    "assert_public_replay_payload",
]
