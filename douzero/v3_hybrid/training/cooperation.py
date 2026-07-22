"""Training-only sequential farmer credit assignment for V3 H5."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from douzero.belief.model import BELIEF_FEATURE_DIM
from douzero.observation.encode_v2 import ObservationV2
from douzero.strategy.features import (
    STRATEGY_FEATURE_LAYOUT_HASH,
    build_strategy_feature_matrix,
)
from douzero.v3_hybrid.replay import V3ReplayTransition

FARMER_ROLES = ("landlord_up", "landlord_down")
FARMER_ROLE_TO_INDEX = {role: index for index, role in enumerate(FARMER_ROLES)}

H5_ALIGNMENT_VERSION = "sequential_trace_index_pair_v1"
H5_MIXER_SEMANTICS_VERSION = "masked_trajectory_mean_monotonic_qmix_v1"
H5_REWARD_PERSPECTIVE = "farmer_team_raw_terminal_return_v1"
H5_PADDING_SEMANTICS = "false_rows_are_zero_and_excluded_v1"
H5_PUBLIC_FEATURE_NAMES = (
    "takes_initiative",
    "control_strength",
    "teammate_cards_left",
    "landlord_cards_left",
    "feeds_teammate",
    "bomb_opportunity_cost",
    "teammate_expected_cards",
    "teammate_high_control",
    "belief_confidence",
    "belief_entropy",
)
H5_PUBLIC_FEATURE_DIM = len(H5_PUBLIC_FEATURE_NAMES)

MIXER_DISABLED = "disabled"
MIXER_PUBLIC = "public"
MIXER_PRIVILEGED = "privileged_training_only"
_MIXER_MODES = frozenset({MIXER_DISABLED, MIXER_PUBLIC, MIXER_PRIVILEGED})


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class V3H5CooperationConfig:
    """Identity-bound, opt-in H5 sidecar and loss schedule."""

    enabled: bool = False
    hidden_size: int = 64
    trajectory_layers: int = 1
    dropout: float = 0.0
    lambda_coop: float = 0.0
    lambda_team_value: float = 1.0
    lambda_trajectory_consistency: float = 0.25
    lambda_mixer: float = 0.0
    warmup_updates: int = 0
    ramp_updates: int = 0
    mixer_mode: str = MIXER_DISABLED
    privileged_state_dim: int = 0
    learning_rate: float = 1e-4
    max_grad_norm: float = 10.0
    update_public_model: bool = True

    IDENTITY_VERSION = 1

    def __post_init__(self) -> None:
        for name in ("enabled", "update_public_model"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in ("hidden_size", "trajectory_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        for name in ("warmup_updates", "ramp_updates", "privileged_state_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        for name in (
            "dropout", "lambda_coop", "lambda_team_value",
            "lambda_trajectory_consistency", "lambda_mixer",
            "learning_rate", "max_grad_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mixer_mode not in _MIXER_MODES:
            raise ValueError(f"mixer_mode must be one of {sorted(_MIXER_MODES)}")
        if not self.enabled:
            dormant = (
                self.lambda_coop,
                self.lambda_mixer,
                self.privileged_state_dim,
            )
            if any(value != 0 for value in dormant) or self.mixer_mode != MIXER_DISABLED:
                raise ValueError("disabled H5 must not configure losses or mixer state")
        else:
            if self.lambda_coop <= 0.0:
                raise ValueError("enabled H5 requires lambda_coop > 0")
            if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
                raise ValueError("enabled H5 requires positive optimizer settings")
            if self.mixer_mode == MIXER_DISABLED:
                if self.lambda_mixer != 0.0 or self.privileged_state_dim != 0:
                    raise ValueError("disabled mixer must have zero loss and state width")
            elif self.lambda_mixer <= 0.0:
                raise ValueError("enabled mixer requires lambda_mixer > 0")
            if self.mixer_mode == MIXER_PUBLIC and self.privileged_state_dim != 0:
                raise ValueError("public mixer cannot configure privileged state")
            if self.mixer_mode == MIXER_PRIVILEGED and self.privileged_state_dim <= 0:
                raise ValueError("privileged mixer requires privileged_state_dim > 0")
            active_loss = self.lambda_team_value + self.lambda_trajectory_consistency
            if self.mixer_enabled:
                active_loss += self.lambda_mixer
            if active_loss <= 0.0:
                raise ValueError("enabled H5 requires at least one active sidecar loss")

    @property
    def mixer_enabled(self) -> bool:
        return self.mixer_mode != MIXER_DISABLED

    def schedule_weight(self, eligible_update: int) -> float:
        if isinstance(eligible_update, bool) or not isinstance(eligible_update, int):
            raise TypeError("eligible_update must be an int")
        if eligible_update < 0:
            raise ValueError("eligible_update must be non-negative")
        if not self.enabled or eligible_update < self.warmup_updates:
            return 0.0
        if self.ramp_updates == 0:
            return float(self.lambda_coop)
        progress = min(
            eligible_update - self.warmup_updates + 1, self.ramp_updates
        )
        return float(self.lambda_coop) * progress / self.ramp_updates

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "identity_version": self.IDENTITY_VERSION,
            **asdict(self),
            "alignment_version": H5_ALIGNMENT_VERSION,
            "mixer_semantics_version": H5_MIXER_SEMANTICS_VERSION,
            "reward_perspective": H5_REWARD_PERSPECTIVE,
            "padding_semantics": H5_PADDING_SEMANTICS,
            "public_feature_names": list(H5_PUBLIC_FEATURE_NAMES),
            "strategy_layout_hash": STRATEGY_FEATURE_LAYOUT_HASH,
            "belief_summary": "detached_public_conservative_posterior_v1",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.compatibility_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V3H5CooperationConfig":
        if not isinstance(payload, Mapping) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("H5 cooperation config fields mismatch")
        return cls(**dict(payload))


def teammate_belief_summary(
    belief_features: np.ndarray,
    unseen_counts: np.ndarray,
    *,
    opponent_a_is_teammate: bool,
) -> np.ndarray:
    """Reduce the detached conservative posterior to four public scalars."""

    values = np.asarray(belief_features, dtype=np.float64)
    unseen = np.asarray(unseen_counts, dtype=np.float64)
    if values.shape != (BELIEF_FEATURE_DIM,) or unseen.shape != (15,):
        raise ValueError("belief summary input layout mismatch")
    if not isinstance(opponent_a_is_teammate, bool):
        raise TypeError("opponent_a_is_teammate must be bool")
    if not np.isfinite(values).all() or not np.isfinite(unseen).all():
        raise ValueError("belief summary inputs must be finite")
    expected_a = values[:15]
    expected = expected_a if opponent_a_is_teammate else unseen - expected_a
    if np.any(expected < -1e-4) or np.any(expected - unseen > 1e-4):
        raise ValueError("teammate belief violates the public unknown pool")
    confidence = values[30:45]
    result = np.asarray(
        [
            expected.sum() / 20.0,
            expected[-5:].sum() / 8.0,
            confidence.mean(),
            values[-1] / 15.0,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ValueError("teammate belief summary must be finite")
    return result


def build_h5_public_features(
    observation: ObservationV2,
    selected_action_index: int,
    *,
    belief_features: np.ndarray | None = None,
    unseen_counts: np.ndarray | None = None,
    opponent_a_role: str | None = None,
) -> torch.Tensor:
    """Build selected-action H5 inputs exclusively from public quantities."""

    if not isinstance(observation, ObservationV2):
        raise TypeError("H5 public features require ObservationV2")
    if observation.public.acting_role not in FARMER_ROLES:
        raise ValueError("H5 public features are farmer-only")
    if (
        isinstance(selected_action_index, bool)
        or not isinstance(selected_action_index, int)
        or not 0 <= selected_action_index < len(observation.actions.legal_actions)
    ):
        raise ValueError("selected action is outside the environment legal list")
    row = build_strategy_feature_matrix(observation.public)[selected_action_index]
    strategy = row[[15, 16, 20, 21, 23, 27]].astype(np.float32)
    if belief_features is None:
        if unseen_counts is not None or opponent_a_role is not None:
            raise ValueError("partial belief summary inputs are forbidden")
        belief = np.zeros(4, dtype=np.float32)
    else:
        if unseen_counts is None or opponent_a_role is None:
            raise ValueError("belief summary requires pool and opponent-A role")
        teammate = (
            "landlord_down"
            if observation.public.acting_role == "landlord_up"
            else "landlord_up"
        )
        if opponent_a_role not in {"landlord", teammate}:
            raise ValueError(
                "opponent_a_role must identify the acting farmer's teammate or landlord"
            )
        belief = teammate_belief_summary(
            belief_features,
            unseen_counts,
            opponent_a_is_teammate=opponent_a_role == teammate,
        )
    result = torch.from_numpy(np.concatenate((strategy, belief))).float()
    if result.shape != (H5_PUBLIC_FEATURE_DIM,) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("H5 public feature layout drift")
    return result


@dataclass(frozen=True)
class V3H5FarmerTrajectory:
    """One ordered farmer trajectory and public-only auxiliary side channel."""

    episode_id: str
    deal_id: str
    role: str
    policy_id: str
    teammate_policy_id: str
    decision_indices: tuple[int, ...]
    transitions: tuple[V3ReplayTransition, ...]
    public_features: torch.Tensor
    selected_action_is_pass: torch.Tensor
    team_return: float

    def __post_init__(self) -> None:
        if self.role not in FARMER_ROLES:
            raise ValueError("H5 trajectories are farmer-only")
        for name in ("episode_id", "deal_id", "policy_id", "teammate_policy_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        count = len(self.transitions)
        if count < 1 or len(self.decision_indices) != count:
            raise ValueError("H5 trajectory lengths must match and be non-empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.decision_indices
        ) or len(set(self.decision_indices)) != count:
            raise ValueError("H5 decision indices must be unique and non-negative")
        if self.public_features.shape != (count, H5_PUBLIC_FEATURE_DIM):
            raise ValueError("H5 public feature shape mismatch")
        if self.public_features.device.type != "cpu" or not bool(
            torch.isfinite(self.public_features).all()
        ):
            raise ValueError("H5 public features must be finite CPU tensors")
        if (
            self.selected_action_is_pass.shape != (count,)
            or self.selected_action_is_pass.dtype != torch.bool
            or self.selected_action_is_pass.device.type != "cpu"
        ):
            raise ValueError("H5 pass flags must be a bool CPU vector")
        order = tuple(
            sorted(range(count), key=lambda position: self.decision_indices[position])
        )
        if order != tuple(range(count)):
            tensor_order = torch.tensor(order, dtype=torch.long)
            object.__setattr__(
                self,
                "decision_indices",
                tuple(self.decision_indices[position] for position in order),
            )
            object.__setattr__(
                self,
                "transitions",
                tuple(self.transitions[position] for position in order),
            )
            object.__setattr__(
                self,
                "public_features",
                self.public_features.index_select(0, tensor_order),
            )
            object.__setattr__(
                self,
                "selected_action_is_pass",
                self.selected_action_is_pass.index_select(0, tensor_order),
            )
        if (
            isinstance(self.team_return, bool)
            or not isinstance(self.team_return, (int, float))
            or not math.isfinite(self.team_return)
        ):
            raise ValueError("H5 farmer team return must be finite")
        for transition in self.transitions:
            if not isinstance(transition, V3ReplayTransition):
                raise TypeError("H5 trajectory contains an invalid replay row")
            if (
                transition.episode_id != self.episode_id
                or transition.deal_id != self.deal_id
                or transition.role != self.role
            ):
                raise ValueError("H5 trajectory/replay identity mismatch")
            if not math.isclose(
                transition.mc_return, self.team_return, rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError("H5 farmer decisions do not share one team return")


def validate_farmer_pairs(
    trajectories: Sequence[V3H5FarmerTrajectory],
) -> tuple[tuple[V3H5FarmerTrajectory, V3H5FarmerTrajectory], ...]:
    """Group exact up/down pairs without assuming aligned decision times."""

    grouped: dict[tuple[str, str], dict[str, V3H5FarmerTrajectory]] = {}
    for trajectory in trajectories:
        if not isinstance(trajectory, V3H5FarmerTrajectory):
            raise TypeError("H5 trajectory batch contains an invalid item")
        roles = grouped.setdefault((trajectory.episode_id, trajectory.deal_id), {})
        if trajectory.role in roles:
            raise ValueError("H5 episode contains a duplicate farmer trajectory")
        roles[trajectory.role] = trajectory
    pairs = []
    for identity in sorted(grouped):
        roles = grouped[identity]
        if set(roles) != set(FARMER_ROLES):
            raise ValueError("H5 episode requires both farmer trajectories")
        up, down = (roles[role] for role in FARMER_ROLES)
        if not math.isclose(up.team_return, down.team_return, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("the two farmers have inconsistent terminal returns")
        if up.teammate_policy_id != down.policy_id or down.teammate_policy_id != up.policy_id:
            raise ValueError("H5 teammate policy provenance is inconsistent")
        pairs.append((up, down))
    if not pairs:
        raise ValueError("H5 requires at least one complete farmer episode")
    return tuple(pairs)


@dataclass(frozen=True)
class FarmerCooperationOutput:
    team_value: torch.Tensor
    trajectory_value: torch.Tensor
    trajectory_embedding: torch.Tensor
    local_q_summary: torch.Tensor
    mixed_team_value: torch.Tensor | None
    mixer_weights: torch.Tensor | None


class MonotonicSequentialMixer(nn.Module):
    """Mix two unequal sequential local-Q summaries with non-negative weights."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2)
        )
        self.bias_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(
        self, local_q_summary: torch.Tensor, training_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_q_summary.ndim != 2 or local_q_summary.shape[1] != 2:
            raise ValueError("mixer local Q must have shape (E, 2)")
        if training_state.ndim != 2 or training_state.shape[0] != local_q_summary.shape[0]:
            raise ValueError("mixer state batch shape mismatch")
        if not bool(torch.isfinite(local_q_summary).all()) or not bool(
            torch.isfinite(training_state).all()
        ):
            raise FloatingPointError("mixer inputs contain NaN or Inf")
        weights = torch.nn.functional.softplus(self.weight_net(training_state))
        mixed = (weights * local_q_summary).sum(dim=-1) + self.bias_net(
            training_state
        ).squeeze(-1)
        if not bool(torch.isfinite(mixed).all()):
            raise FloatingPointError("mixer output contains NaN or Inf")
        return mixed, weights


class FarmerCooperationModule(nn.Module):
    """Farmer-only team heads, sequential embeddings, and optional mixer."""

    def __init__(self, model_hidden_size: int, config: V3H5CooperationConfig) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("disabled H5 must not construct cooperation parameters")
        self.config = config
        step_width = model_hidden_size + H5_PUBLIC_FEATURE_DIM
        self.trajectory_encoder = nn.GRU(
            step_width,
            config.hidden_size,
            num_layers=config.trajectory_layers,
            batch_first=True,
            dropout=(config.dropout if config.trajectory_layers > 1 else 0.0),
        )
        self.team_value_heads = nn.ModuleDict({
            role: nn.Sequential(
                nn.LayerNorm(step_width),
                nn.Linear(step_width, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1),
            )
            for role in FARMER_ROLES
        })
        self.trajectory_value_heads = nn.ModuleDict({
            role: nn.Linear(config.hidden_size, 1) for role in FARMER_ROLES
        })
        self.mixer = None
        if config.mixer_enabled:
            state_dim = 2 * (config.hidden_size + H5_PUBLIC_FEATURE_DIM)
            if config.mixer_mode == MIXER_PRIVILEGED:
                state_dim += config.privileged_state_dim
            self.mixer = MonotonicSequentialMixer(state_dim, config.hidden_size)

    def forward(
        self,
        chosen_embeddings: torch.Tensor,
        public_features: torch.Tensor,
        local_q: torch.Tensor,
        sequence_mask: torch.Tensor,
        role_index: torch.Tensor,
        *,
        privileged_mixer_state: torch.Tensor | None = None,
    ) -> FarmerCooperationOutput:
        if chosen_embeddings.ndim != 3:
            raise ValueError("chosen embeddings must have shape (P, T, H)")
        pairs, steps = chosen_embeddings.shape[:2]
        expected = (pairs, steps)
        if pairs < 2 or pairs % 2 != 0:
            raise ValueError("H5 sidecar requires ordered farmer pairs")
        if public_features.shape != (*expected, H5_PUBLIC_FEATURE_DIM):
            raise ValueError("H5 padded public feature shape mismatch")
        if local_q.shape != expected or sequence_mask.shape != expected:
            raise ValueError("H5 local Q or sequence mask shape mismatch")
        if sequence_mask.dtype != torch.bool:
            raise ValueError("H5 sequence mask must be bool")
        if role_index.shape != (pairs,) or role_index.dtype != torch.long:
            raise ValueError("H5 role index must be long with shape (P,)")
        if not bool((sequence_mask.sum(dim=1) > 0).all()):
            raise ValueError("each farmer trajectory requires a real decision")
        expected_roles = torch.tensor(
            [0, 1] * (pairs // 2), device=role_index.device, dtype=torch.long
        )
        if not torch.equal(role_index, expected_roles):
            raise ValueError("H5 farmer pairs must be ordered up then down")
        for tensor in (chosen_embeddings, public_features, local_q):
            if not bool(torch.isfinite(tensor[sequence_mask]).all()):
                raise FloatingPointError("H5 real trajectory values contain NaN or Inf")
        step_input = torch.cat((chosen_embeddings, public_features), dim=-1)
        step_input = step_input.masked_fill(~sequence_mask.unsqueeze(-1), 0.0)
        lengths = sequence_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            step_input, lengths, batch_first=True, enforce_sorted=False
        )
        _encoded, hidden = self.trajectory_encoder(packed)
        trajectory_embedding = hidden[-1]
        team_value = step_input.new_zeros((*expected,))
        trajectory_value = step_input.new_zeros((pairs,))
        for index, role in enumerate(FARMER_ROLES):
            rows = torch.nonzero(role_index == index, as_tuple=False).squeeze(-1)
            values = self.team_value_heads[role](step_input.index_select(0, rows)).squeeze(-1)
            team_value = team_value.index_copy(0, rows, values)
            trajectory_values = self.trajectory_value_heads[role](
                trajectory_embedding.index_select(0, rows)
            ).squeeze(-1)
            trajectory_value = trajectory_value.index_copy(0, rows, trajectory_values)
        team_value = team_value.masked_fill(~sequence_mask, 0.0)
        denominator = sequence_mask.sum(dim=1).clamp_min(1).to(local_q.dtype)
        local_q_summary = (
            local_q.masked_fill(~sequence_mask, 0.0).sum(dim=1) / denominator
        ).reshape(-1, 2)
        mixed = None
        weights = None
        if self.mixer is not None:
            public_summary = (
                public_features.masked_fill(~sequence_mask.unsqueeze(-1), 0.0).sum(dim=1)
                / denominator.unsqueeze(-1)
            )
            state = torch.cat(
                (trajectory_embedding, public_summary), dim=-1
            ).reshape(pairs // 2, -1)
            if self.config.mixer_mode == MIXER_PRIVILEGED:
                expected_shape = (pairs // 2, self.config.privileged_state_dim)
                if privileged_mixer_state is None or privileged_mixer_state.shape != expected_shape:
                    raise ValueError("privileged mixer state shape mismatch")
                if not bool(torch.isfinite(privileged_mixer_state).all()):
                    raise FloatingPointError("privileged mixer state contains NaN or Inf")
                state = torch.cat((state, privileged_mixer_state), dim=-1)
            elif privileged_mixer_state is not None:
                raise ValueError("public mixer rejects privileged state")
            mixed, weights = self.mixer(local_q_summary, state)
        elif privileged_mixer_state is not None:
            raise ValueError("disabled mixer rejects training-only state")
        return FarmerCooperationOutput(
            team_value=team_value,
            trajectory_value=trajectory_value,
            trajectory_embedding=trajectory_embedding,
            local_q_summary=local_q_summary,
            mixed_team_value=mixed,
            mixer_weights=weights,
        )


__all__ = [
    "FARMER_ROLES",
    "FARMER_ROLE_TO_INDEX",
    "H5_ALIGNMENT_VERSION",
    "H5_MIXER_SEMANTICS_VERSION",
    "H5_PADDING_SEMANTICS",
    "H5_PUBLIC_FEATURE_DIM",
    "H5_PUBLIC_FEATURE_NAMES",
    "H5_REWARD_PERSPECTIVE",
    "MIXER_DISABLED",
    "MIXER_PRIVILEGED",
    "MIXER_PUBLIC",
    "FarmerCooperationModule",
    "FarmerCooperationOutput",
    "MonotonicSequentialMixer",
    "V3H5CooperationConfig",
    "V3H5FarmerTrajectory",
    "build_h5_public_features",
    "teammate_belief_summary",
    "validate_farmer_pairs",
]
