"""Public-policy export guard for P10 student checkpoints."""

from __future__ import annotations

from typing import Any

from douzero.checkpoint.v2 import save_v2_position_weights
from douzero.env.rules import RuleSet
from douzero.models_v2.batch import ModelInputBundle, observation_to_model_inputs
from douzero.models_v2.model import ModelV2
from douzero.observation.encode_v2 import ObservationV2


def build_public_example_input(
    public_input: ObservationV2 | ModelInputBundle,
) -> dict[str, Any]:
    """Return an export example containing only public tensors and role data."""

    bundle = (
        observation_to_model_inputs(public_input)
        if isinstance(public_input, ObservationV2)
        else public_input
    )
    if not isinstance(bundle, ModelInputBundle):
        raise TypeError(
            "public_input must be ObservationV2 or ModelInputBundle, got "
            f"{type(public_input).__name__}"
        )
    return {
        "state_card_vectors": tuple(t.detach().cpu() for t in bundle.state_card_vectors),
        "state_context_flat": bundle.state_context_flat.detach().cpu(),
        "context_card_vectors": tuple(t.detach().cpu() for t in bundle.context_card_vectors),
        "context_flat": bundle.context_flat.detach().cpu(),
        "history_tokens": bundle.history_tokens.detach().cpu(),
        "history_key_padding_mask": bundle.history_key_padding_mask.detach().cpu(),
        "action_features": bundle.action_features.detach().cpu(),
        "action_mask": bundle.action_mask.detach().cpu(),
        "acting_role": bundle.acting_role,
        "feature_schema_hash": bundle.feature_schema_hash,
        "strategy_features": (
            None
            if bundle.strategy_features is None
            else bundle.strategy_features.detach().cpu()
        ),
        "style_features": (
            None
            if bundle.style_features is None
            else bundle.style_features.detach().cpu()
        ),
    }


def export_public_student(
    path: str,
    model: ModelV2,
    *,
    ruleset: RuleSet,
    flags: dict | None = None,
):
    """Export only a public Model V2 as a production policy sidecar.

    A ``TeacherModel`` fails the strict ``ModelV2`` type check and a future
    model that advertises ``model_access='privileged'`` is also rejected.
    """

    if not isinstance(model, ModelV2):
        raise TypeError(
            f"production export requires a public ModelV2, got {type(model).__name__}"
        )
    if getattr(model, "model_access", "public") != "public":
        raise ValueError("privileged models cannot be exported as production agents")
    return save_v2_position_weights(path, model, ruleset=ruleset, flags=flags)
