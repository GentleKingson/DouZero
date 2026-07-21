"""Identity-bound architecture configuration for the H1 public policy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

HISTORY_ENCODER_LSTM = "lstm"
HISTORY_ENCODER_TRANSFORMER = "transformer"
CHANNEL_GATE_NONE = "none"
CHANNEL_GATE_SE = "se"
BELIEF_FEEDBACK_NONE = "none"
BELIEF_FEEDBACK_FARMERS = "farmers"
BELIEF_FEEDBACK_ALL = "all_roles"
DMC_TARGET_RAW = "raw"
DMC_TARGET_SIGNED_LOG = "signed_log"

_HISTORY_ENCODERS = frozenset({HISTORY_ENCODER_LSTM, HISTORY_ENCODER_TRANSFORMER})
_CHANNEL_GATES = frozenset({CHANNEL_GATE_NONE, CHANNEL_GATE_SE})
_BELIEF_FEEDBACK = frozenset({
    BELIEF_FEEDBACK_NONE,
    BELIEF_FEEDBACK_FARMERS,
    BELIEF_FEEDBACK_ALL,
})
_DMC_TARGETS = frozenset({DMC_TARGET_RAW, DMC_TARGET_SIGNED_LOG})


@dataclass(frozen=True)
class V3HybridModelConfig:
    """Card-play-only H1 architecture.

    Every field affects either the parameter graph or forward semantics and is
    therefore included in :meth:`stable_hash`. Later-stage auxiliary features
    are intentionally absent rather than represented by dormant parameters.
    """

    hidden_size: int = 256
    history_encoder: str = HISTORY_ENCODER_LSTM
    history_layers: int = 2
    history_heads: int = 8
    history_dropout: float = 0.0
    shared_fusion_layers: int = 2
    landlord_adapter_layers: int = 2
    farmer_adapter_layers: int = 4
    farmer_channel_gate: str = CHANNEL_GATE_NONE
    farmer_channel_gate_reduction: int = 4
    adapter_dropout: float = 0.0
    attention_type: str = "none"
    score_clamp: float = 32.0
    dmc_target_transform: str = DMC_TARGET_RAW
    dmc_target_clamp: float = 32.0
    nan_guard: bool = True
    belief_feedback: str = BELIEF_FEEDBACK_NONE

    IDENTITY_VERSION = 2

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "history_layers",
            "history_heads",
            "farmer_channel_gate_reduction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        for name in (
            "shared_fusion_layers",
            "landlord_adapter_layers",
            "farmer_adapter_layers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.history_encoder not in _HISTORY_ENCODERS:
            raise ValueError(
                f"history_encoder must be one of {sorted(_HISTORY_ENCODERS)}, "
                f"got {self.history_encoder!r}"
            )
        if (
            self.history_encoder == HISTORY_ENCODER_TRANSFORMER
            and self.hidden_size % self.history_heads != 0
        ):
            raise ValueError(
                "hidden_size must be divisible by history_heads for transformer"
            )
        if self.farmer_channel_gate not in _CHANNEL_GATES:
            raise ValueError(
                f"farmer_channel_gate must be one of {sorted(_CHANNEL_GATES)}, "
                f"got {self.farmer_channel_gate!r}"
            )
        if (
            self.farmer_channel_gate == CHANNEL_GATE_SE
            and self.hidden_size // self.farmer_channel_gate_reduction < 1
        ):
            raise ValueError(
                "farmer_channel_gate_reduction must not reduce hidden_size below 1"
            )
        if self.belief_feedback not in _BELIEF_FEEDBACK:
            raise ValueError(
                f"belief_feedback must be one of {sorted(_BELIEF_FEEDBACK)}, "
                f"got {self.belief_feedback!r}"
            )
        for name in ("history_dropout", "adapter_dropout"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        if self.attention_type != "none":
            raise ValueError("H1 supports attention_type='none' only")
        if self.dmc_target_transform not in _DMC_TARGETS:
            raise ValueError(
                f"dmc_target_transform must be one of {sorted(_DMC_TARGETS)}"
            )
        for name in ("score_clamp", "dmc_target_clamp"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive, got {value!r}")
        if not isinstance(self.nan_guard, bool):
            raise TypeError("nan_guard must be bool")

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            **asdict(self),
            "role_layout": {
                "landlord": "independent_adapter_and_heads",
                "landlord_up": "independent_adapter_and_heads",
                "landlord_down": "independent_adapter_and_heads",
            },
            "output_semantics": {
                "dmc_q": "acting_team_monte_carlo_return",
                "win": "acting_team_probability",
                "score": "acting_team_conditional_signed_score",
            },
            "belief_feedback": {
                "mode": self.belief_feedback,
                "layout": (
                    "disabled_no_parameters"
                    if self.belief_feedback == BELIEF_FEEDBACK_NONE
                    else "detached_exact_constrained_posterior_features_v1"
                ),
            },
        }

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.compatibility_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "V3HybridModelConfig":
        if not isinstance(payload, dict):
            raise TypeError("V3 Hybrid model config must be a dict")
        expected = set(cls.__dataclass_fields__)
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"unknown V3 Hybrid model config keys: {sorted(unknown)}")
        missing = expected - set(payload)
        if missing:
            raise ValueError(f"missing V3 Hybrid model config keys: {sorted(missing)}")
        return cls(**payload)
