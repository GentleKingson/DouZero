"""Strictly public H1 role-residual policy for variable legal actions."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from douzero.models_v2.action_encoder import ActionEncoder
from douzero.models_v2.batch import (
    BatchedBiddingInput,
    BatchedModelInputBundle,
    ModelInputBundle,
    observation_batch_to_model_inputs,
    observation_to_model_inputs,
)
from douzero.models_v2.history_encoder import build_history_encoder
from douzero.models_v2.heads import BiddingHeads, PriorHead, StrategyAuxiliaryHeads
from douzero.models_v2.numerical import NumericalError
from douzero.models_v2.output import BatchedBiddingOutput, BiddingModelOutput
from douzero.models_v2.state_encoder import StateEncoder
from douzero.observation.encode_v2 import ObservationV2
from douzero.observation.schema import (
    FeatureSchemaManifest,
    action_width,
    context_width,
    history_token_width,
    state_width,
)

from .config import (
    BELIEF_FEEDBACK_ALL,
    BELIEF_FEEDBACK_FARMERS,
    BELIEF_FEEDBACK_NONE,
    CHANNEL_GATE_SE,
    V3HybridModelConfig,
)
from .contract import V3_HYBRID_OBSERVATION_SCHEMA_HASH
from .layers import RoleAdapter, RoleValueHeads, SharedStateActionFusion
from .output import BatchedV3HybridModelOutput, V3HybridModelOutput

V3_HYBRID_ROLES = ("landlord", "landlord_up", "landlord_down")
V3_HYBRID_ROLE_TO_INDEX = {
    role: index for index, role in enumerate(V3_HYBRID_ROLES)
}

_STATE_CARD_FIELDS = 6
_CONTEXT_CARD_FIELDS = 2


class V3HybridModel(nn.Module):
    """Shared public encoders plus three independent role adapter/head paths."""

    model_access = "public"
    model_version = "v3_hybrid"

    def __init__(
        self,
        schema: FeatureSchemaManifest,
        config: V3HybridModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(schema, FeatureSchemaManifest):
            raise TypeError("V3HybridModel requires a FeatureSchemaManifest")
        if schema.stable_hash() != V3_HYBRID_OBSERVATION_SCHEMA_HASH:
            raise ValueError(
                "V3 H1 requires the frozen Observation V2 schema hash; "
                f"got {schema.stable_hash()}"
            )
        self.schema = schema
        self.config = config or V3HybridModelConfig()
        cfg = self.config

        self._state_width = state_width(schema)
        self._context_width = context_width(schema)
        self._action_width = action_width(schema)
        self._history_width = history_token_width(schema)
        card_width = schema.card_vector_dim
        non_card_state = self._state_width - _STATE_CARD_FIELDS * card_width
        non_card_context = self._context_width - _CONTEXT_CARD_FIELDS * card_width
        if non_card_state < 0 or non_card_context < 0:
            raise ValueError("Observation V2 card-field layout is incompatible with H1")

        self.state_encoder = StateEncoder(
            card_vector_dim=card_width,
            num_card_fields=_STATE_CARD_FIELDS + _CONTEXT_CARD_FIELDS,
            flat_context_width=non_card_state + non_card_context,
            hidden_size=cfg.hidden_size,
        )
        self.history_encoder = build_history_encoder(
            token_width=self._history_width,
            hidden_size=cfg.hidden_size,
            max_history_len=schema.max_history_len,
            backend=cfg.history_encoder,
            num_layers=cfg.history_layers,
            num_heads=cfg.history_heads,
            dropout=cfg.history_dropout,
        )
        strategy_width = 0
        if cfg.strategy_features_enabled:
            from douzero.strategy.features import STRATEGY_FEATURE_WIDTH

            strategy_width = STRATEGY_FEATURE_WIDTH
        self.action_encoder = ActionEncoder(
            action_width=self._action_width,
            hidden_size=cfg.hidden_size,
            strategy_width=strategy_width,
        )
        self.shared_fusion = SharedStateActionFusion(
            cfg.hidden_size,
            cfg.shared_fusion_layers,
            cfg.adapter_dropout,
        )
        farmer_gate = cfg.farmer_channel_gate == CHANNEL_GATE_SE
        self.role_adapters = nn.ModuleDict({
            "landlord": RoleAdapter(
                cfg.hidden_size,
                cfg.landlord_adapter_layers,
                cfg.adapter_dropout,
                channel_gate=False,
                gate_reduction=cfg.farmer_channel_gate_reduction,
            ),
            "landlord_up": RoleAdapter(
                cfg.hidden_size,
                cfg.farmer_adapter_layers,
                cfg.adapter_dropout,
                channel_gate=farmer_gate,
                gate_reduction=cfg.farmer_channel_gate_reduction,
            ),
            "landlord_down": RoleAdapter(
                cfg.hidden_size,
                cfg.farmer_adapter_layers,
                cfg.adapter_dropout,
                channel_gate=farmer_gate,
                gate_reduction=cfg.farmer_channel_gate_reduction,
            ),
        })
        self.role_heads = nn.ModuleDict({
            role: RoleValueHeads(
                cfg.hidden_size,
                score_clamp=cfg.score_clamp,
                dmc_clamp=cfg.dmc_target_clamp,
            )
            for role in V3_HYBRID_ROLES
        })
        self.belief_projection: nn.Module | None = None
        self._belief_feature_dim = 0
        if cfg.belief_feedback != BELIEF_FEEDBACK_NONE:
            from douzero.belief.model import BELIEF_FEATURE_DIM

            self._belief_feature_dim = BELIEF_FEATURE_DIM
            self.belief_projection = nn.Sequential(
                nn.Linear(BELIEF_FEATURE_DIM, cfg.hidden_size, bias=False),
                nn.LayerNorm(cfg.hidden_size),
            )
        self.style_encoder: nn.Module | None = None
        if cfg.style_enabled:
            from douzero.style.encoder import StyleEncoder

            self.style_encoder = StyleEncoder(
                output_dim=cfg.hidden_size,
                hidden_dim=cfg.style_embedding_dim,
            )
        self.prior_head: PriorHead | None = (
            PriorHead(cfg.hidden_size) if cfg.human_prior_enabled else None
        )
        self.strategy_aux_heads: StrategyAuxiliaryHeads | None = (
            StrategyAuxiliaryHeads(cfg.hidden_size)
            if cfg.strategy_aux_enabled
            else None
        )
        self.bidding_schema = None
        self.bidding_heads: BiddingHeads | None = None
        if cfg.bidding_enabled:
            from douzero.observation.bidding import BIDDING_ACTIONS, build_bidding_schema

            self.bidding_schema = build_bidding_schema()
            self.bidding_heads = BiddingHeads(
                self.bidding_schema.input_width,
                cfg.bidding_hidden_size,
                num_bid_actions=len(BIDDING_ACTIONS),
                score_clamp=cfg.score_clamp,
                uncertainty_enabled=cfg.bidding_uncertainty_enabled,
            )

    def role_index(self, role: str) -> int:
        try:
            return V3_HYBRID_ROLE_TO_INDEX[role]
        except KeyError as exc:
            raise ValueError(
                f"unsupported V3 role {role!r}; expected {V3_HYBRID_ROLES}"
            ) from exc

    def strategy_feature_config(self):
        if not self.config.strategy_features_enabled:
            return None
        from douzero.strategy.config import StrategyFeatureConfig

        return StrategyFeatureConfig(
            hand_enabled=self.config.strategy_hand_enabled,
            structure_enabled=self.config.strategy_structure_enabled,
            control_enabled=self.config.strategy_control_enabled,
            cooperation_enabled=self.config.strategy_cooperation_enabled,
            risk_enabled=self.config.strategy_risk_enabled,
            node_budget=self.config.strategy_node_budget,
            time_budget_ms=self.config.strategy_time_budget_ms,
        )

    def _apply_style(
        self, state: torch.Tensor, style_features: torch.Tensor | None
    ) -> torch.Tensor:
        if self.style_encoder is None:
            if style_features is not None:
                raise ValueError("style features were passed to a style-disabled V3 model")
            return state
        from douzero.style.features import STYLE_FEATURE_WIDTH

        expected = (
            (STYLE_FEATURE_WIDTH,)
            if state.ndim == 1
            else (state.shape[0], STYLE_FEATURE_WIDTH)
        )
        if style_features is None or tuple(style_features.shape) != expected:
            raise ValueError(f"style_features must have shape {expected}")
        parameter = next(self.style_encoder.parameters())
        encoded = self.style_encoder(
            style_features.to(device=state.device, dtype=parameter.dtype)
        ).to(state.dtype)
        return state + encoded

    def _shared_scalar(
        self,
        state_card_vectors: tuple[torch.Tensor, ...],
        state_context_flat: torch.Tensor,
        context_card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        strategy_features: torch.Tensor | None,
        style_features: torch.Tensor | None,
    ) -> torch.Tensor:
        cards = state_card_vectors + context_card_vectors
        context = torch.cat((state_context_flat, context_flat), dim=-1)
        state = self.state_encoder(cards, context)
        state = self._apply_style(state, style_features)
        history = self.history_encoder(history_tokens, history_key_padding_mask)
        actions = self.action_encoder(action_features, strategy_features)
        return self.shared_fusion(state, history, actions)

    def _validate_scalar_mask(
        self, action_features: torch.Tensor, action_mask: torch.Tensor
    ) -> None:
        if action_features.ndim != 2 or action_features.shape[1] != self._action_width:
            raise ValueError(
                f"action_features must have shape (A, {self._action_width})"
            )
        if action_features.shape[0] < 1:
            raise ValueError("V3HybridModel requires at least one legal action")
        if action_mask.shape != (action_features.shape[0],):
            raise ValueError("action_mask must have shape (A,)")
        if action_mask.dtype != torch.bool:
            raise ValueError("action_mask must have bool dtype")
        if not bool(action_mask.any()):
            raise ValueError("action_mask must contain a valid action")

    def _check_finite(
        self,
        values: dict[str, torch.Tensor],
        mask: torch.Tensor,
        *,
        prefix: str,
    ) -> None:
        if not self.config.nan_guard:
            return
        for name, value in values.items():
            valid = value[mask]
            if not bool(torch.isfinite(valid).all()):
                raise NumericalError(f"{prefix}{name} contains NaN or Inf")

    def _role_uses_belief(self, role: str) -> bool:
        return self.config.belief_feedback == BELIEF_FEEDBACK_ALL or (
            self.config.belief_feedback == BELIEF_FEEDBACK_FARMERS
            and role != "landlord"
        )

    def _apply_scalar_belief(
        self,
        shared: torch.Tensor,
        acting_role: str,
        belief_features: torch.Tensor | None,
    ) -> torch.Tensor:
        uses_belief = self._role_uses_belief(acting_role)
        if not uses_belief:
            if belief_features is not None and self.config.belief_feedback == BELIEF_FEEDBACK_NONE:
                raise ValueError("belief features were passed to a belief-disabled V3 model")
            return shared
        if belief_features is None:
            raise ValueError("belief features are required by this V3 role")
        if belief_features.shape != (self._belief_feature_dim,):
            raise ValueError(
                f"belief_features must have shape ({self._belief_feature_dim},)"
            )
        parameter = next(self.belief_projection.parameters())
        features = belief_features.detach().to(
            device=shared.device, dtype=parameter.dtype
        )
        return shared + self.belief_projection(features).to(shared.dtype).unsqueeze(0)

    def _apply_batched_belief(
        self,
        shared: torch.Tensor,
        acting_role: torch.Tensor,
        belief_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.belief_feedback == BELIEF_FEEDBACK_NONE:
            if belief_features is not None:
                raise ValueError("belief features were passed to a belief-disabled V3 model")
            return shared
        if belief_features is None:
            raise ValueError("belief features are required by this V3 batch")
        if belief_features.shape != (shared.shape[0], self._belief_feature_dim):
            raise ValueError(
                "belief_features must have shape "
                f"({shared.shape[0]}, {self._belief_feature_dim})"
            )
        parameter = next(self.belief_projection.parameters())
        projected = self.belief_projection(
            belief_features.detach().to(device=shared.device, dtype=parameter.dtype)
        ).to(shared.dtype)
        if self.config.belief_feedback == BELIEF_FEEDBACK_FARMERS:
            enabled = (acting_role != V3_HYBRID_ROLE_TO_INDEX["landlord"]).to(
                shared.dtype
            )
            projected = projected * enabled.unsqueeze(-1)
        return shared + projected.unsqueeze(1)

    def forward(
        self,
        state_card_vectors: tuple[torch.Tensor, ...],
        state_context_flat: torch.Tensor,
        context_card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        acting_role: str,
        belief_features: torch.Tensor | None = None,
        strategy_features: torch.Tensor | None = None,
        style_features: torch.Tensor | None = None,
    ) -> V3HybridModelOutput:
        """Score one public decision without assuming a maximum action count."""

        self._validate_scalar_mask(action_features, action_mask)
        self.role_index(acting_role)
        shared = self._shared_scalar(
            state_card_vectors,
            state_context_flat,
            context_card_vectors,
            context_flat,
            history_tokens,
            history_key_padding_mask,
            action_features,
            strategy_features,
            style_features,
        )
        shared = self._apply_scalar_belief(
            shared, acting_role, belief_features
        )
        adapted = self.role_adapters[acting_role](shared)
        values = self.role_heads[acting_role](adapted)
        self._check_finite(values, action_mask, prefix="")
        optional: dict[str, torch.Tensor | None] = {
            "prior_logit": None,
            "min_turns_after": None,
            "regain_initiative_logit": None,
            "teammate_finish_logit": None,
            "spring_probability_logit": None,
            "structure_cost": None,
        }
        if self.prior_head is not None:
            optional["prior_logit"] = self.prior_head(adapted)
        if self.strategy_aux_heads is not None:
            optional.update(self.strategy_aux_heads(adapted))
        self._check_finite(
            {name: value for name, value in optional.items() if value is not None},
            action_mask,
            prefix="",
        )
        return V3HybridModelOutput(
            **values, action_mask=action_mask, **optional
        )

    def _validate_batched(
        self,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        acting_role: torch.Tensor,
    ) -> tuple[int, int]:
        if action_features.ndim != 3 or action_features.shape[-1] != self._action_width:
            raise ValueError(
                f"batched action_features must have shape (B, A, {self._action_width})"
            )
        batch, actions = action_features.shape[:2]
        if batch < 1 or actions < 1:
            raise ValueError("batched V3 input must not be empty")
        if action_mask.shape != (batch, actions) or action_mask.dtype != torch.bool:
            raise ValueError("batched action_mask must be bool with shape (B, A)")
        if not bool(action_mask.any(dim=1).all()):
            raise ValueError("each decision must contain a valid action")
        if acting_role.shape != (batch,) or acting_role.dtype != torch.long:
            raise ValueError("acting_role must be long with shape (B,)")
        if not bool(((acting_role >= 0) & (acting_role < len(V3_HYBRID_ROLES))).all()):
            raise ValueError("batched acting_role contains an unsupported index")
        return batch, actions

    def forward_batched(
        self,
        state_card_vectors: tuple[torch.Tensor, ...],
        state_context_flat: torch.Tensor,
        context_card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        acting_role: torch.Tensor,
        belief_features: torch.Tensor | None = None,
        strategy_features: torch.Tensor | None = None,
        style_features: torch.Tensor | None = None,
    ) -> BatchedV3HybridModelOutput:
        """Score a padded heterogeneous-role batch with independent adapters."""

        adapted = self._encode_batched_actions(
            state_card_vectors,
            state_context_flat,
            context_card_vectors,
            context_flat,
            history_tokens,
            history_key_padding_mask,
            action_features,
            action_mask,
            acting_role,
            belief_features,
            strategy_features,
            style_features,
        )
        batch, actions = action_mask.shape

        outputs = {
            name: adapted.new_zeros((batch, actions, 1))
            for name in (
                "dmc_q",
                "win_logit",
                "score_if_win",
                "score_if_loss",
                "p_win",
                "score_mean",
            )
        }
        for role_index, role in enumerate(V3_HYBRID_ROLES):
            rows = torch.nonzero(acting_role == role_index, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            role_values = self.role_heads[role](adapted.index_select(0, rows))
            role_mask = action_mask.index_select(0, rows)
            self._check_finite(role_values, role_mask, prefix=f"{role}_")
            for name, value in role_values.items():
                outputs[name] = outputs[name].index_copy(0, rows, value)
        optional: dict[str, torch.Tensor | None] = {
            "prior_logit": None,
            "min_turns_after": None,
            "regain_initiative_logit": None,
            "teammate_finish_logit": None,
            "spring_probability_logit": None,
            "structure_cost": None,
        }
        if self.prior_head is not None:
            optional["prior_logit"] = self.prior_head(adapted)
        if self.strategy_aux_heads is not None:
            optional.update(self.strategy_aux_heads(adapted))
        self._check_finite(
            {name: value for name, value in optional.items() if value is not None},
            action_mask,
            prefix="batched_",
        )
        return BatchedV3HybridModelOutput(
            **outputs, action_mask=action_mask, **optional
        )

    def _encode_batched_actions(
        self,
        state_card_vectors: tuple[torch.Tensor, ...],
        state_context_flat: torch.Tensor,
        context_card_vectors: tuple[torch.Tensor, ...],
        context_flat: torch.Tensor,
        history_tokens: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        acting_role: torch.Tensor,
        belief_features: torch.Tensor | None,
        strategy_features: torch.Tensor | None,
        style_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return role-adapted public action embeddings for training auxiliaries."""

        batch, _actions = self._validate_batched(
            action_features, action_mask, acting_role
        )
        cards = state_card_vectors + context_card_vectors
        context = torch.cat((state_context_flat, context_flat), dim=-1)
        state = self.state_encoder(cards, context)
        state = self._apply_style(state, style_features)
        history = self.history_encoder(history_tokens, history_key_padding_mask)
        action_embeddings = self.action_encoder(action_features, strategy_features)
        shared = self.shared_fusion.forward_batched(state, history, action_embeddings)
        shared = self._apply_batched_belief(
            shared, acting_role, belief_features
        )

        adapted = torch.zeros_like(shared)
        for role_index, role in enumerate(V3_HYBRID_ROLES):
            rows = torch.nonzero(acting_role == role_index, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            role_adapted = self.role_adapters[role](shared.index_select(0, rows))
            adapted = adapted.index_copy(0, rows, role_adapted)
        if self.config.nan_guard and not bool(torch.isfinite(adapted[action_mask]).all()):
            raise NumericalError("role-adapted action embedding contains NaN or Inf")
        return adapted

    def _check_public_observation(self, observation: object) -> ObservationV2:
        if getattr(observation, "kind", None) == "privileged" or (
            isinstance(observation, dict) and observation.get("kind") == "privileged"
        ):
            raise TypeError("V3 Hybrid deployment rejects privileged observations")
        if not isinstance(observation, ObservationV2):
            raise TypeError("V3 Hybrid deployment requires an ObservationV2")
        actual_hash = observation.schema.stable_hash()
        if observation.feature_schema_hash != actual_hash:
            raise ValueError("ObservationV2 carries a false feature schema hash")
        if actual_hash != self.schema.stable_hash():
            raise ValueError("ObservationV2 schema does not match the V3 model")
        expected_ruleset = getattr(self, "expected_ruleset_identity", None)
        if expected_ruleset is not None:
            actual_ruleset = (
                observation.public.ruleset_id,
                observation.public.ruleset_version,
                observation.public.ruleset_hash,
            )
            if actual_ruleset != expected_ruleset:
                raise ValueError("ObservationV2 ruleset does not match the V3 checkpoint")
        return observation

    def _move_bundle(self, bundle: ModelInputBundle) -> ModelInputBundle:
        parameter = next(self.parameters())
        bundle.to(parameter.device)
        for name, value in vars(bundle).items():
            if torch.is_tensor(value) and value.is_floating_point():
                setattr(bundle, name, value.to(dtype=parameter.dtype))
            elif isinstance(value, tuple):
                setattr(
                    bundle,
                    name,
                    tuple(
                        item.to(dtype=parameter.dtype)
                        if torch.is_tensor(item) and item.is_floating_point()
                        else item
                        for item in value
                    ),
                )
        return bundle

    def forward_observation(
        self,
        observation: object,
        *,
        belief_features: torch.Tensor | None = None,
    ) -> V3HybridModelOutput:
        obs = self._check_public_observation(observation)
        bundle = self._move_bundle(observation_to_model_inputs(
            obs,
            self.strategy_feature_config(),
            style_enabled=self.config.style_enabled,
        ))
        return self(
            bundle.state_card_vectors,
            bundle.state_context_flat,
            bundle.context_card_vectors,
            bundle.context_flat,
            bundle.history_tokens,
            bundle.history_key_padding_mask,
            bundle.action_features,
            bundle.action_mask,
            bundle.acting_role,
            belief_features,
            bundle.strategy_features,
            bundle.style_features,
        )

    def forward_observation_batch(
        self,
        observations: Iterable[object],
        *,
        pad_to_actions: int | None = None,
        belief_features: torch.Tensor | None = None,
    ) -> BatchedV3HybridModelOutput:
        public = [self._check_public_observation(obs) for obs in observations]
        if not public:
            raise ValueError("observation batch must not be empty")
        inputs = observation_batch_to_model_inputs(
            public,
            strategy_config=self.strategy_feature_config(),
            style_enabled=self.config.style_enabled,
            pad_to_actions=pad_to_actions,
        )
        return self.forward_input_batch(inputs, belief_features=belief_features)

    def forward_input_batch(
        self,
        inputs: BatchedModelInputBundle,
        *,
        belief_features: torch.Tensor | None = None,
    ) -> BatchedV3HybridModelOutput:
        """Forward an already tensorized public replay or inference batch."""

        if not isinstance(inputs, BatchedModelInputBundle):
            raise TypeError("V3 input batch must be a BatchedModelInputBundle")
        expected_hash = self.schema.stable_hash()
        if not inputs.feature_schema_hashes or any(
            value != expected_hash for value in inputs.feature_schema_hashes
        ):
            raise ValueError("V3 input batch feature schema mismatch")
        parameter = next(self.parameters())
        inputs.to(parameter.device)
        return self.forward_batched(
            tuple(value.to(dtype=parameter.dtype) for value in inputs.state_card_vectors),
            inputs.state_context_flat.to(dtype=parameter.dtype),
            tuple(value.to(dtype=parameter.dtype) for value in inputs.context_card_vectors),
            inputs.context_flat.to(dtype=parameter.dtype),
            inputs.history_tokens.to(dtype=parameter.dtype),
            inputs.history_key_padding_mask,
            inputs.action_features.to(dtype=parameter.dtype),
            inputs.action_mask,
            inputs.acting_role,
            belief_features=belief_features,
            strategy_features=(
                None
                if inputs.strategy_features is None
                else inputs.strategy_features.to(dtype=parameter.dtype)
            ),
            style_features=(
                None
                if inputs.style_features is None
                else inputs.style_features.to(dtype=parameter.dtype)
            ),
        )

    def encode_input_batch_context(
        self, inputs: BatchedModelInputBundle
    ) -> torch.Tensor:
        """Encode one public context vector per decision for belief supervision."""

        if not isinstance(inputs, BatchedModelInputBundle):
            raise TypeError("V3 context encoding requires BatchedModelInputBundle")
        expected_hash = self.schema.stable_hash()
        if not inputs.feature_schema_hashes or any(
            value != expected_hash for value in inputs.feature_schema_hashes
        ):
            raise ValueError("V3 context input feature schema mismatch")
        parameter = next(self.parameters())
        inputs.to(parameter.device)
        cards = tuple(
            value.to(dtype=parameter.dtype) for value in inputs.state_card_vectors
        ) + tuple(
            value.to(dtype=parameter.dtype) for value in inputs.context_card_vectors
        )
        context = torch.cat(
            (
                inputs.state_context_flat.to(dtype=parameter.dtype),
                inputs.context_flat.to(dtype=parameter.dtype),
            ),
            dim=-1,
        )
        state = self.state_encoder(cards, context)
        history = self.history_encoder(
            inputs.history_tokens.to(dtype=parameter.dtype),
            inputs.history_key_padding_mask,
        )
        return (state + history) * 0.5

    def encode_input_batch_actions(
        self,
        inputs: BatchedModelInputBundle,
        *,
        belief_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode padded public actions through their role-specific adapters.

        This parameter-free API is consumed only by training auxiliaries. It
        preserves the public model graph and the authoritative action mask.
        """

        if not isinstance(inputs, BatchedModelInputBundle):
            raise TypeError("V3 action encoding requires BatchedModelInputBundle")
        expected_hash = self.schema.stable_hash()
        if not inputs.feature_schema_hashes or any(
            value != expected_hash for value in inputs.feature_schema_hashes
        ):
            raise ValueError("V3 action encoding feature schema mismatch")
        parameter = next(self.parameters())
        inputs.to(parameter.device)
        return self._encode_batched_actions(
            tuple(value.to(dtype=parameter.dtype) for value in inputs.state_card_vectors),
            inputs.state_context_flat.to(dtype=parameter.dtype),
            tuple(value.to(dtype=parameter.dtype) for value in inputs.context_card_vectors),
            inputs.context_flat.to(dtype=parameter.dtype),
            inputs.history_tokens.to(dtype=parameter.dtype),
            inputs.history_key_padding_mask,
            inputs.action_features.to(dtype=parameter.dtype),
            inputs.action_mask,
            inputs.acting_role,
            belief_features,
            (
                None
                if inputs.strategy_features is None
                else inputs.strategy_features.to(dtype=parameter.dtype)
            ),
            (
                None
                if inputs.style_features is None
                else inputs.style_features.to(dtype=parameter.dtype)
            ),
        )

    def act(self, observation: object, *, output: str = "dmc_q") -> tuple[int, ...]:
        obs = self._check_public_observation(observation)
        legal_actions = obs.actions.legal_actions
        if not legal_actions:
            raise ValueError("cannot act without a legal action")
        if len(legal_actions) == 1:
            return legal_actions[0]
        with torch.inference_mode():
            selected = self.forward_observation(obs).argmax(output)
        return legal_actions[selected]

    def _forward_bidding_head(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        if self.bidding_heads is None:
            raise RuntimeError("V3 bidding heads are disabled")
        values = self.bidding_heads(features)
        if self.config.nan_guard:
            tensors = [value for value in values.values() if value is not None]
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise NumericalError("V3 bidding output contains NaN or Inf")
        return values

    def forward_bidding(self, observation: object) -> BiddingModelOutput:
        from douzero.observation.bidding import BiddingObservationV2

        if self.bidding_heads is None or self.bidding_schema is None:
            raise RuntimeError(
                "forward_bidding requires V3HybridModelConfig(bidding_enabled=True)"
            )
        if not isinstance(observation, BiddingObservationV2):
            raise TypeError("forward_bidding requires a public BiddingObservationV2")
        if observation.feature_schema_hash != self.bidding_schema.stable_hash():
            raise ValueError("V3 bidding feature schema mismatch")
        parameter = next(self.bidding_heads.parameters())
        features = observation.to_tensor(parameter.device).to(parameter.dtype)
        values = self._forward_bidding_head(features)
        mask = torch.from_numpy(observation.bid_action_mask.copy()).to(
            device=parameter.device, dtype=torch.bool
        )
        return BiddingModelOutput(
            bid_logits=values["bid_logits"],
            bid_action_mask=mask,
            landlord_win_logit=values["landlord_win_logit"],
            expected_landlord_score=values["expected_landlord_score"],
            uncertainty=values["uncertainty"],
        )

    def forward_bidding_batched(
        self, inputs: BatchedBiddingInput
    ) -> BatchedBiddingOutput:
        if self.bidding_heads is None or self.bidding_schema is None:
            raise RuntimeError(
                "forward_bidding_batched requires "
                "V3HybridModelConfig(bidding_enabled=True)"
            )
        if not isinstance(inputs, BatchedBiddingInput):
            raise TypeError("forward_bidding_batched requires BatchedBiddingInput")
        if inputs.feature_schema_hash != self.bidding_schema.stable_hash():
            raise ValueError("V3 batched bidding feature schema mismatch")
        parameter = next(self.bidding_heads.parameters())
        features = inputs.features.to(device=parameter.device, dtype=parameter.dtype)
        values = self._forward_bidding_head(features)
        return BatchedBiddingOutput(
            bid_logits=values["bid_logits"],
            bid_action_mask=inputs.legal_mask.to(parameter.device),
            landlord_win_logit=values["landlord_win_logit"],
            expected_landlord_score=values["expected_landlord_score"],
            uncertainty=values["uncertainty"],
        )

    def parameter_count(self) -> dict[str, int]:
        modules: list[tuple[str, nn.Module]] = [
            ("state_encoder", self.state_encoder),
            ("history_encoder", self.history_encoder),
            ("action_encoder", self.action_encoder),
            ("shared_fusion", self.shared_fusion),
        ]
        modules.extend(
            (f"adapter.{role}", self.role_adapters[role])
            for role in V3_HYBRID_ROLES
        )
        modules.extend(
            (f"heads.{role}", self.role_heads[role])
            for role in V3_HYBRID_ROLES
        )
        if self.belief_projection is not None:
            modules.append(("belief_projection", self.belief_projection))
        if self.style_encoder is not None:
            modules.append(("style_encoder", self.style_encoder))
        if self.prior_head is not None:
            modules.append(("prior_head", self.prior_head))
        if self.strategy_aux_heads is not None:
            modules.append(("strategy_aux_heads", self.strategy_aux_heads))
        if self.bidding_heads is not None:
            modules.append(("bidding_heads", self.bidding_heads))
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in modules
        }
        counts["total"] = sum(counts.values())
        return counts
