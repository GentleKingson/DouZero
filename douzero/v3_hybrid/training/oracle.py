"""Training-only perfect-information Oracle for V3 Hybrid H3.

Importing this module intentionally imports the privileged observation type.
Public model, exporter, loader, agent, and search modules never import it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn

from douzero.distillation.cache import TeacherCacheIdentity
from douzero.distillation.teacher_model import (
    ActionKey,
    TeacherOutput,
    canonical_action_keys,
    state_dict_sha256,
)
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import ModelInputBundle, observation_to_model_inputs
from douzero.observation.cards import CARD_VECTOR_DIM, cards_to_vector
from douzero.observation.encode_v2 import ObservationV2
from douzero.observation.privileged import PrivilegedObservation
from douzero.observation.seats import ALL_ROLES

from ..config import V3HybridModelConfig
from ..model import V3HybridModel


@dataclass(frozen=True)
class V3OracleConfig:
    """Architecture identity for the separate privileged residual branch."""

    hidden_size: int = 128
    value_delta_clamp: float = 32.0

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.hidden_size, bool)
            or not isinstance(self.hidden_size, int)
            or self.hidden_size < 1
        ):
            raise ValueError("Oracle hidden_size must be a positive int")
        if (
            isinstance(self.value_delta_clamp, bool)
            or not isinstance(self.value_delta_clamp, (int, float))
            or not math.isfinite(self.value_delta_clamp)
            or self.value_delta_clamp <= 0.0
        ):
            raise ValueError("Oracle value_delta_clamp must be positive and finite")

    def compatibility_dict(self, public_model_config_hash: str) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            "oracle": asdict(self),
            "public_model_config_hash": public_model_config_hash,
            "input": "public_model_inputs_plus_privileged_all_handcards_v1",
            "legal_action_alignment": "p10_canonical_action_keys_v1",
            "access_class": "privileged_training_only",
        }

    def stable_hash(self, public_model_config_hash: str) -> str:
        encoded = json.dumps(
            self.compatibility_dict(public_model_config_hash),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3OracleConfig":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("V3 Oracle config fields mismatch")
        return cls(**dict(payload))


def _copy_bundle_to_model(bundle: ModelInputBundle, model: V3HybridModel) -> ModelInputBundle:
    parameter = next(model.parameters())

    def move(value: torch.Tensor) -> torch.Tensor:
        dtype = parameter.dtype if value.is_floating_point() else value.dtype
        return value.to(device=parameter.device, dtype=dtype)

    return ModelInputBundle(
        state_card_vectors=tuple(move(value) for value in bundle.state_card_vectors),
        state_context_flat=move(bundle.state_context_flat),
        context_card_vectors=tuple(move(value) for value in bundle.context_card_vectors),
        context_flat=move(bundle.context_flat),
        history_tokens=move(bundle.history_tokens),
        history_key_padding_mask=move(bundle.history_key_padding_mask),
        action_features=move(bundle.action_features),
        action_mask=move(bundle.action_mask),
        acting_role=bundle.acting_role,
        feature_schema_hash=bundle.feature_schema_hash,
        strategy_features=(
            None if bundle.strategy_features is None else move(bundle.strategy_features)
        ),
        style_features=(
            None if bundle.style_features is None else move(bundle.style_features)
        ),
    )


class V3PrivilegedOracle(nn.Module):
    """A wholly separate V3 backbone plus gated perfect-information residual."""

    model_access = "privileged"
    model_version = "v3_hybrid_oracle_h3"

    def __init__(
        self,
        public_model: V3HybridModel,
        config: V3OracleConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(public_model, V3HybridModel):
            raise TypeError("V3 Oracle initialization requires a V3HybridModel")
        self.config = config or V3OracleConfig()
        self.public_backbone = V3HybridModel(public_model.schema, public_model.config)
        self.public_backbone.load_state_dict(public_model.state_dict(), strict=True)
        hidden = self.config.hidden_size
        privileged_width = len(ALL_ROLES) * CARD_VECTOR_DIM
        self.privileged_encoder = nn.Sequential(
            nn.Linear(privileged_width, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.privileged_action_head = nn.Sequential(
            nn.Linear(hidden + self.public_backbone._action_width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    @property
    def schema(self):
        return self.public_backbone.schema

    @property
    def public_model_config(self) -> V3HybridModelConfig:
        return self.public_backbone.config

    def config_hash(self) -> str:
        return self.config.stable_hash(self.public_backbone.config.stable_hash())

    def cache_identity(self, ruleset: RuleSet) -> TeacherCacheIdentity:
        """Reuse the existing strict P10 cache identity without a new format."""

        if not isinstance(ruleset, RuleSet):
            raise TypeError("Oracle cache identity requires a RuleSet")
        return TeacherCacheIdentity(
            feature_schema_hash=self.schema.stable_hash(),
            ruleset_hash=ruleset.stable_hash(),
            teacher_model_sha=state_dict_sha256(self),
            teacher_config_hash=self.config_hash(),
        )

    def _privileged_features(
        self,
        privileged: PrivilegedObservation,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if set(privileged.all_handcards) != set(ALL_ROLES):
            raise ValueError("Oracle privileged hand roles mismatch")
        vectors = [cards_to_vector(privileged.all_handcards[role]) for role in ALL_ROLES]
        return torch.from_numpy(np.concatenate(vectors)).to(device=device, dtype=dtype)

    def forward(
        self,
        public_input: ObservationV2 | ModelInputBundle,
        privileged_observation: PrivilegedObservation,
        *,
        action_keys: tuple[ActionKey, ...] | None = None,
        privileged_gate: float = 1.0,
        belief_features: torch.Tensor | None = None,
    ) -> TeacherOutput:
        if not isinstance(privileged_observation, PrivilegedObservation):
            raise TypeError("V3 Oracle requires a PrivilegedObservation")
        if (
            isinstance(privileged_gate, bool)
            or not isinstance(privileged_gate, (int, float))
            or not math.isfinite(privileged_gate)
            or not 0.0 <= privileged_gate <= 1.0
        ):
            raise ValueError("privileged_gate must be finite and in [0, 1]")
        if isinstance(public_input, ObservationV2):
            keys = canonical_action_keys(public_input.actions.legal_actions)
            bundle = observation_to_model_inputs(
                public_input,
                self.public_backbone.strategy_feature_config(),
                style_enabled=self.public_backbone.config.style_enabled,
            )
        elif isinstance(public_input, ModelInputBundle):
            if action_keys is None:
                raise ValueError("tensorized Oracle input requires action_keys")
            keys = tuple(tuple(key) for key in action_keys)
            bundle = public_input
        else:
            raise TypeError("Oracle public_input must be ObservationV2 or ModelInputBundle")
        if len(set(keys)) != len(keys) or any(key != tuple(sorted(key)) for key in keys):
            raise ValueError("Oracle action_keys must be unique canonical tuples")
        if len(keys) != bundle.action_features.shape[0]:
            raise ValueError("Oracle action key count does not match public action rows")
        if privileged_observation.acting_role != bundle.acting_role:
            raise ValueError("Oracle public/privileged acting role mismatch")
        if set(privileged_observation.all_handcards) != set(ALL_ROLES):
            raise ValueError("Oracle privileged hand roles mismatch")
        acting_hand = torch.from_numpy(
            cards_to_vector(privileged_observation.all_handcards[bundle.acting_role])
        ).to(torch.float32)
        if not torch.equal(bundle.state_card_vectors[0].detach().cpu(), acting_hand):
            raise ValueError("Oracle privileged acting hand does not match public input")

        moved = _copy_bundle_to_model(bundle, self.public_backbone)
        base = self.public_backbone(
            moved.state_card_vectors,
            moved.state_context_flat,
            moved.context_card_vectors,
            moved.context_flat,
            moved.history_tokens,
            moved.history_key_padding_mask,
            moved.action_features,
            moved.action_mask,
            moved.acting_role,
            belief_features,
            moved.strategy_features,
            moved.style_features,
        )
        parameter = next(self.parameters())
        hidden = self.privileged_encoder(
            self._privileged_features(
                privileged_observation, device=parameter.device, dtype=parameter.dtype
            )
        )
        gated = hidden * float(privileged_gate)
        repeated = gated.unsqueeze(0).expand(moved.action_features.shape[0], -1)
        delta = self.privileged_action_head(
            torch.cat((repeated, moved.action_features), dim=-1)
        )
        clamp = float(self.config.value_delta_clamp)
        delta = torch.clamp(delta, min=-clamp, max=clamp)
        dmc_q = torch.clamp(
            base.dmc_q + delta[:, 0:1],
            min=-self.public_backbone.config.dmc_target_clamp,
            max=self.public_backbone.config.dmc_target_clamp,
        )
        win_logit = base.win_logit + delta[:, 1:2]
        score_if_win = torch.clamp(
            base.score_if_win + delta[:, 2:3],
            min=-self.public_backbone.config.score_clamp,
            max=self.public_backbone.config.score_clamp,
        )
        score_if_loss = torch.clamp(
            base.score_if_loss + delta[:, 3:4],
            min=-self.public_backbone.config.score_clamp,
            max=self.public_backbone.config.score_clamp,
        )
        p_win = torch.sigmoid(win_logit)
        expected_score = p_win * score_if_win + (1.0 - p_win) * score_if_loss
        values = (dmc_q, win_logit, score_if_win, score_if_loss, p_win, expected_score)
        if not all(bool(torch.isfinite(value[moved.action_mask]).all()) for value in values):
            raise FloatingPointError("V3 Oracle produced NaN or Inf")
        return TeacherOutput(
            action_keys=keys,
            win_logit=win_logit,
            p_win=p_win,
            score_if_win=score_if_win,
            score_if_loss=score_if_loss,
            expected_score=expected_score,
            action_logits=dmc_q,
            action_mask=moved.action_mask,
        )
