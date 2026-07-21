"""H0 compatibility boundary for DouZero V3 Hybrid.

This module freezes names and identity axes only. It deliberately does not
register a runnable model or widen the legacy/V2 configuration allowlists.
Later V3 phases must build their checkpoints from
``V3HybridCompatibilityIdentity`` and must not omit an identity section.
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

V3_HYBRID_MODEL_VERSION = "v3_hybrid"
V3_HYBRID_FEATURE_VERSION = "v2"
V3_HYBRID_CHECKPOINT_KIND = "public_policy"
V3_HYBRID_CONTRACT_VERSION = "v3-hybrid-h0-contract-v1"
V3_HYBRID_OBSERVATION_SCHEMA_VERSION = "v2-1"
V3_HYBRID_OBSERVATION_SCHEMA_HASH = (
    "aac17ecbce0795048d44b296500a5f390e03ebbd07661f27354ce70d9c85d148"
)

V3_HYBRID_PHASES = (
    "h0_contract",
    "h1_role_model",
    "h2_adaptive_dmc",
    "h3_oracle_guiding",
    "h4_joint_belief",
    "h5_farmer_cooperation",
    "h6_hybrid_integration",
    "h7_runtime_search",
    "h8_formal_evaluation",
)

V3_HYBRID_LOSS_TERMS = MappingProxyType({
    "dmc": "lambda_dmc",
    "win": "lambda_win",
    "score": "lambda_score",
    "oracle": "lambda_oracle",
    "belief": "lambda_belief",
    "cooperation": "lambda_coop",
    "bc": "lambda_bc",
    "strategy": "lambda_strategy",
    "bidding": "lambda_bidding",
})

_IDENTITY_SECTIONS = (
    "feature_flags",
    "model_graph",
    "output_semantics",
    "optimizer_config",
    "loss_config",
    "loss_schedules",
    "belief_layout",
    "cooperation_mixer",
    "trainer_config",
    "training_topology",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("V3 Hybrid identity must be finite canonical JSON") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _copy_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = json.loads(_canonical_json(value))
    if not copied:
        raise ValueError(f"{name} must not be empty")
    return _freeze_json(copied)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def v3_hybrid_semantic_contract() -> dict[str, Any]:
    """Return the frozen cross-phase semantics included in every V3 identity."""

    return {
        "contract_version": V3_HYBRID_CONTRACT_VERSION,
        "model_version": V3_HYBRID_MODEL_VERSION,
        "checkpoint_kind": V3_HYBRID_CHECKPOINT_KIND,
        "observation": {
            "feature_version": V3_HYBRID_FEATURE_VERSION,
            "schema_version": V3_HYBRID_OBSERVATION_SCHEMA_VERSION,
            "schema_hash": V3_HYBRID_OBSERVATION_SCHEMA_HASH,
            "deployment_input": (
                "douzero.observation.encode_v2.ObservationV2"
            ),
            "public_container": (
                "douzero.observation.public.PublicObservation"
            ),
            "legal_action_authority": "environment_rules_engine_only_v1",
            "candidate_consumers": "rank_existing_legal_actions_only_v1",
        },
        "model": {
            "shared_encoders": (
                "cards_state_history_legal_action_shared_v1"
            ),
            "role_adapters": "landlord_farmer_residual_adapters_v1",
            "dmc_head": "independent_action_conditioned_scalar_q_v1",
            "multi_objective_heads": (
                "v2_win_conditional_score_preserved_v1"
            ),
        },
        "adaptive_dmc": {
            "algorithm": "per_role_popart_huber_monte_carlo_v1",
            "statistics": "checkpointed_float64_running_moments_v1",
            "normalization": "valid_selected_samples_only_v1",
            "role_weighting": "single_application_observable_v1",
        },
        "privileged_training": {
            "oracle_input": (
                "douzero.observation.privileged.PrivilegedObservation"
            ),
            "oracle_scope": "training_only_no_deployment_import_v1",
            "teacher_action_scope": "rank_existing_legal_actions_only_v1",
            "hidden_labels_scope": "loss_builder_only_v1",
        },
        "belief": {
            "layout": "public_joint_rank_count_conservative_dp_v1",
            "policy_feedback": "optional_public_posterior_features_only_v1",
            "true_hand_scope": "supervised_label_builder_only_v1",
        },
        "cooperation": {
            "scope": "farmer_team_only_v1",
            "mixer": "public_state_monotonic_farmer_mixer_v1",
            "credit": "terminal_team_value_counterfactual_aux_v1",
        },
        "loss": {
            "formula_order": list(V3_HYBRID_LOSS_TERMS),
            "weight_fields": dict(V3_HYBRID_LOSS_TERMS),
            "normalization": "sum_over_real_valid_samples_v1",
            "disabled_term": "no_parameters_no_data_dependency_or_exact_noop_v1",
        },
        "resume": {
            "mode": "strict_fail_closed_v1",
            "required_state": [
                "model",
                "optimizer",
                "loss_schedules",
                "adaptive_dmc_statistics",
                "policy_version",
                "python_numpy_torch_rng",
                "trainer_counters",
            ],
            "partial_load": "forbidden",
        },
        "deployment": {
            "forbidden_namespaces": [
                "douzero.observation.privileged",
                "douzero.distillation.teacher_model",
            ],
            "forbidden_payloads": [
                "all_handcards",
                "hidden_hand_labels",
                "oracle_state_dict",
                "teacher_state_dict",
                "training_labels",
            ],
            "strength_without_formal_evaluation": "playing strength not measured",
        },
        "phases": list(V3_HYBRID_PHASES),
    }


@dataclass(frozen=True)
class V3HybridCompatibilityIdentity:
    """Complete, fail-closed identity payload for V3 checkpoints and exports."""

    ruleset: Mapping[str, Any]
    feature_flags: Mapping[str, Any]
    model_graph: Mapping[str, Any]
    output_semantics: Mapping[str, Any]
    optimizer_config: Mapping[str, Any]
    loss_config: Mapping[str, Any]
    loss_schedules: Mapping[str, Any]
    belief_layout: Mapping[str, Any]
    cooperation_mixer: Mapping[str, Any]
    trainer_config: Mapping[str, Any]
    training_topology: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("ruleset",) + _IDENTITY_SECTIONS:
            object.__setattr__(self, name, _copy_mapping(name, getattr(self, name)))
        ruleset_keys = {"ruleset_id", "ruleset_version", "ruleset_hash"}
        if set(self.ruleset) != ruleset_keys:
            raise ValueError(
                "ruleset identity fields must be exactly "
                "ruleset_id, ruleset_version, and ruleset_hash"
            )
        if not all(
            isinstance(self.ruleset[key], str) and self.ruleset[key]
            for key in ruleset_keys
        ):
            raise ValueError("ruleset identity values must be non-empty strings")
        if not _is_sha256(self.ruleset["ruleset_hash"]):
            raise ValueError("ruleset_hash must be a full SHA-256")

    def compatibility_dict(self) -> dict[str, Any]:
        return {
            "semantic_contract": v3_hybrid_semantic_contract(),
            "ruleset": _plain_json(self.ruleset),
            **{
                name: _plain_json(getattr(self, name))
                for name in _IDENTITY_SECTIONS
            },
        }

    def stable_hash(self) -> str:
        payload = _canonical_json(self.compatibility_dict())
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V3HybridCompatibilityIdentity":
        if not isinstance(payload, Mapping):
            raise TypeError("V3 Hybrid identity payload must be a mapping")
        expected = {"semantic_contract", "ruleset", *_IDENTITY_SECTIONS}
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                "V3 Hybrid identity fields mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        if payload["semantic_contract"] != v3_hybrid_semantic_contract():
            raise ValueError("V3 Hybrid semantic contract mismatch")
        return cls(
            ruleset=payload["ruleset"],
            **{name: payload[name] for name in _IDENTITY_SECTIONS},
        )


def assert_v3_hybrid_compatible(
    expected: V3HybridCompatibilityIdentity,
    actual_payload: Mapping[str, Any],
    *,
    actual_hash: str,
) -> V3HybridCompatibilityIdentity:
    """Validate exact V3 identity and reject missing, partial, or stale loads."""

    if not _is_sha256(actual_hash):
        raise ValueError("V3 Hybrid compatibility hash must be a full SHA-256")
    actual = V3HybridCompatibilityIdentity.from_dict(actual_payload)
    computed = actual.stable_hash()
    if computed != actual_hash:
        raise ValueError("V3 Hybrid compatibility payload/hash mismatch")
    if computed != expected.stable_hash():
        raise ValueError("V3 Hybrid checkpoint identity mismatch")
    return actual
