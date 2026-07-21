"""Public-only V3 policy composed with the existing conservative belief model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from douzero.belief.features import build_belief_input
from douzero.belief.model import BeliefModel, belief_features_from_probs
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import (
    model_input_bundles_to_batch,
    observation_to_model_inputs,
)

from .config import BELIEF_FEEDBACK_NONE
from .model import V3HybridModel
from .output import V3HybridModelOutput


class V3BeliefPolicy(nn.Module):
    """Deployment policy whose two components both consume public data only."""

    model_access = "public"
    model_version = "v3_hybrid"

    def __init__(
        self,
        model: V3HybridModel,
        belief_model: BeliefModel,
        *,
        ruleset: RuleSet,
    ) -> None:
        super().__init__()
        if not isinstance(model, V3HybridModel):
            raise TypeError("V3BeliefPolicy requires a V3HybridModel")
        if not isinstance(belief_model, BeliefModel):
            raise TypeError("V3BeliefPolicy requires the existing BeliefModel")
        if not isinstance(ruleset, RuleSet):
            raise TypeError("V3BeliefPolicy requires a RuleSet")
        if model.config.belief_feedback == BELIEF_FEEDBACK_NONE:
            raise ValueError("V3BeliefPolicy requires enabled model belief feedback")
        if belief_model.config.shared_context_dim not in (0, model.config.hidden_size):
            raise ValueError(
                "belief shared_context_dim must be zero or the V3 hidden size"
            )
        model_device = next(model.parameters()).device
        belief_device = next(belief_model.parameters()).device
        if model_device != belief_device:
            raise ValueError("public V3 and belief models must share one device")
        identity = (ruleset.ruleset_id, ruleset.ruleset_version, ruleset.stable_hash())
        existing = getattr(model, "expected_ruleset_identity", None)
        if existing is not None and existing != identity:
            raise ValueError("V3 model and belief policy rulesets differ")
        model.expected_ruleset_identity = identity
        self.model = model
        self.belief_model = belief_model
        self.ruleset = ruleset

    def _belief_features(self, observation: object) -> torch.Tensor:
        obs = self.model._check_public_observation(observation)
        binput = build_belief_input(obs.public)
        shared_context = None
        if self.belief_model.config.shared_context_dim:
            bundle = observation_to_model_inputs(obs)
            batch = model_input_bundles_to_batch([bundle], [0])
            shared_context = self.model.encode_input_batch_context(batch)
        output = self.belief_model(
            [binput], shared_context=shared_context
        )
        features = belief_features_from_probs(
            output.constrained_probs,
            output.opponent_a_total,
            np.stack([binput.unseen_counts]),
        )
        parameter = next(self.model.parameters())
        return torch.from_numpy(features[0]).to(
            device=parameter.device, dtype=parameter.dtype
        )

    def forward_observation(self, observation: object) -> V3HybridModelOutput:
        features = self._belief_features(observation)
        return self.model.forward_observation(
            observation, belief_features=features
        )

    def act(self, observation: object, *, output: str = "dmc_q") -> tuple[int, ...]:
        obs = self.model._check_public_observation(observation)
        legal_actions = obs.actions.legal_actions
        if not legal_actions:
            raise ValueError("cannot act without a legal action")
        if len(legal_actions) == 1:
            return legal_actions[0]
        with torch.inference_mode():
            selected = self.forward_observation(obs).argmax(output)
        return legal_actions[selected]


__all__ = ["V3BeliefPolicy"]
