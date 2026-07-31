"""Strict public-only V3 deployment adapter for evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from douzero.belief.model import BeliefConfig
from douzero.env.rules import RuleSet
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.schema import FeatureSchemaManifest
from douzero.v3_hybrid.belief_checkpoint import load_v3_h4_public_checkpoint
from douzero.v3_hybrid.belief_policy import V3BeliefPolicy
from douzero.v3_hybrid.checkpoint import load_v3_hybrid_public_checkpoint
from douzero.v3_hybrid.config import (
    BELIEF_FEEDBACK_NONE,
    V3HybridModelConfig,
)
from douzero.v3_hybrid.model import V3HybridModel
from douzero.search.budget import SearchConfig


def parse_v3_evaluation_config(
    value: Mapping[str, Any],
) -> tuple[V3HybridModelConfig, BeliefConfig | None]:
    """Parse the explicit public graph contract used by evaluator bundles."""

    if not isinstance(value, Mapping):
        raise TypeError("V3 evaluation model_config must be a mapping")
    unknown = set(value) - {"policy", "belief"}
    if unknown:
        raise ValueError(f"unknown V3 evaluation config keys: {sorted(unknown)}")
    if "policy" not in value or not isinstance(value["policy"], Mapping):
        raise ValueError("V3 evaluation model_config requires a policy mapping")
    policy = V3HybridModelConfig.from_dict(dict(value["policy"]))
    raw_belief = value.get("belief")
    belief = None if raw_belief is None else BeliefConfig(**dict(raw_belief))
    if policy.belief_feedback == BELIEF_FEEDBACK_NONE and belief is not None:
        raise ValueError("belief config supplied for a belief-disabled V3 policy")
    if policy.belief_feedback != BELIEF_FEEDBACK_NONE and belief is None:
        raise ValueError("belief-enabled V3 policy requires a belief config")
    return policy, belief


def load_v3_evaluation_policy(
    path: str,
    *,
    schema: FeatureSchemaManifest,
    ruleset: RuleSet,
    model_config: V3HybridModelConfig,
    belief_config: BeliefConfig | None,
    device: str | torch.device = "cpu",
) -> V3HybridModel | V3BeliefPolicy:
    """Strictly load exactly one supported public V3 checkpoint kind."""

    if belief_config is None:
        return load_v3_hybrid_public_checkpoint(
            path,
            schema=schema,
            ruleset=ruleset,
            config=model_config,
            device=device,
        )
    return load_v3_h4_public_checkpoint(
        path,
        schema=schema,
        ruleset=ruleset,
        model_config=model_config,
        belief_config=belief_config,
        device=device,
    )


class DeepAgentV3:
    """Bridge a verified public V3 policy to the legacy infoset harness."""

    def __init__(
        self,
        position: str,
        policy: V3HybridModel | V3BeliefPolicy,
        *,
        ruleset: RuleSet,
        search_config: SearchConfig | None = None,
    ) -> None:
        if position not in ("landlord", "landlord_up", "landlord_down"):
            raise ValueError(f"unsupported V3 position {position!r}")
        if not isinstance(policy, (V3HybridModel, V3BeliefPolicy)):
            raise TypeError("DeepAgentV3 requires a public V3 policy")
        self.position = position
        self.policy = policy
        self.model = policy.model if isinstance(policy, V3BeliefPolicy) else policy
        self.ruleset = ruleset
        self.search_config = search_config or SearchConfig()
        if not isinstance(self.search_config, SearchConfig):
            raise TypeError("DeepAgentV3 search_config must be a SearchConfig")
        if self.search_config.enabled and not isinstance(policy, V3BeliefPolicy):
            raise ValueError("V3 search requires a coupled public belief policy")
        self.last_p_win: float | None = None
        self.last_search_log = None

    def act(self, infoset):
        if infoset.player_position != self.position:
            raise ValueError("V3 agent position does not match the acting infoset")
        observation = get_obs_v2(
            infoset,
            schema=self.model.schema,
            ruleset=self.ruleset,
        )
        with torch.inference_mode():
            output = self.policy.forward_observation(observation)
        selected = output.argmax("dmc_q")
        if self.search_config.enabled:
            from douzero.search.belief_rollout import BeliefSearch

            decision = BeliefSearch(self.search_config, self.ruleset).select(
                observation=observation,
                model_output=output,
                base_action_index=selected,
                belief_model=self.policy.belief_model,
            )
            selected = decision.action_index
            self.last_search_log = decision.log
        else:
            self.last_search_log = None
        if not bool(output.action_mask[selected]):
            raise RuntimeError("V3 policy selected a padded action")
        self.last_p_win = float(output.p_win[selected, 0].detach().cpu().item())
        if selected >= len(infoset.legal_actions):
            raise RuntimeError("V3 action alignment is outside the legal action list")
        return infoset.legal_actions[selected]


__all__ = [
    "DeepAgentV3",
    "load_v3_evaluation_policy",
    "parse_v3_evaluation_config",
]
