"""Training-only perfect-information teacher for P10.

The teacher composes the public Model V2 backbone with a separate privileged
branch over the exact three remaining hands. Its forward boundary requires a
``PrivilegedObservation``. Nothing in this module is imported by deployment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from douzero.models_v2.batch import ModelInputBundle, observation_to_model_inputs
from douzero.models_v2.model import ModelV2
from douzero.models_v2.output import ModelOutput
from douzero.observation.cards import CARD_VECTOR_DIM, cards_to_vector
from douzero.observation.encode_v2 import ObservationV2
from douzero.observation.privileged import PrivilegedObservation
from douzero.observation.seats import ALL_ROLES

ActionKey = tuple[int, ...]


def canonical_action_key(action: Iterable[int]) -> ActionKey:
    """Return the stable, order-independent key for one legal action."""

    return tuple(sorted(int(card) for card in action))


def canonical_action_keys(actions: Iterable[Iterable[int]]) -> tuple[ActionKey, ...]:
    """Canonicalize a legal-action sequence without changing row order."""

    keys = tuple(canonical_action_key(action) for action in actions)
    if len(set(keys)) != len(keys):
        raise ValueError("legal actions contain duplicate canonical action keys")
    return keys


@dataclass(frozen=True)
class TeacherModelConfig:
    """Architecture settings unique to the privileged teacher branch."""

    hidden_size: int = 128
    score_delta_clamp: float = 32.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.hidden_size, bool)
            or not isinstance(self.hidden_size, int)
            or self.hidden_size < 1
        ):
            raise ValueError(f"hidden_size must be a positive int, got {self.hidden_size!r}")
        if self.score_delta_clamp <= 0.0:
            raise ValueError(
                f"score_delta_clamp must be positive, got {self.score_delta_clamp}"
            )

    def stable_hash(self, public_model_config_hash: str) -> str:
        """Hash teacher-only settings together with the public backbone identity."""

        payload = {
            "teacher": asdict(self),
            "public_model_config_hash": public_model_config_hash,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TeacherOutput:
    """Teacher predictions aligned to explicit canonical legal-action keys."""

    action_keys: tuple[ActionKey, ...]
    win_logit: torch.Tensor
    p_win: torch.Tensor
    expected_score: torch.Tensor
    action_logits: torch.Tensor
    action_mask: torch.Tensor

    def __post_init__(self) -> None:
        n = len(self.action_keys)
        if n < 1:
            raise ValueError("TeacherOutput requires at least one legal action")
        if len(set(self.action_keys)) != n:
            raise ValueError("TeacherOutput action_keys must be unique")
        if any(key != tuple(sorted(key)) for key in self.action_keys):
            raise ValueError("TeacherOutput action_keys must be canonical sorted tuples")
        for name in ("win_logit", "p_win", "expected_score", "action_logits"):
            tensor = getattr(self, name)
            if tensor.shape != (n, 1):
                raise ValueError(f"{name} must have shape ({n}, 1), got {tuple(tensor.shape)}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if self.action_mask.shape != (n,) or self.action_mask.dtype != torch.bool:
            raise ValueError(
                f"action_mask must be bool with shape ({n},), got "
                f"{tuple(self.action_mask.shape)} {self.action_mask.dtype}"
            )

    def detached_cpu(self) -> "TeacherOutput":
        """Return a graph-free CPU copy suitable for a teacher cache."""

        return TeacherOutput(
            action_keys=self.action_keys,
            win_logit=self.win_logit.detach().cpu(),
            p_win=self.p_win.detach().cpu(),
            expected_score=self.expected_score.detach().cpu(),
            action_logits=self.action_logits.detach().cpu(),
            action_mask=self.action_mask.detach().cpu(),
        )


def forward_public_model(model: ModelV2, bundle: ModelInputBundle) -> ModelOutput:
    """Forward Model V2 from its canonical public-only tensor bundle."""

    expected_hash = model.schema.stable_hash()
    if bundle.feature_schema_hash != expected_hash:
        raise ValueError(
            f"public input feature_schema_hash mismatch: bundle has "
            f"{bundle.feature_schema_hash!r}, model expects {expected_hash!r}"
        )
    return model(
        bundle.state_card_vectors,
        bundle.state_context_flat,
        bundle.context_card_vectors,
        bundle.context_flat,
        bundle.history_tokens,
        bundle.history_key_padding_mask,
        bundle.action_features,
        bundle.action_mask,
        bundle.acting_role,
        strategy_features=bundle.strategy_features,
    )


class TeacherModel(nn.Module):
    """Perfect-information teacher; training-only and never deployable."""

    model_access = "privileged"
    model_version = "teacher-v1"

    def __init__(
        self,
        public_model: ModelV2,
        config: TeacherModelConfig | None = None,
    ) -> None:
        super().__init__()
        if public_model.config.belief_enabled:
            raise ValueError(
                "TeacherModel does not accept a belief-enabled public backbone. "
                "The teacher already receives exact hidden hands; use a public "
                "backbone with belief_enabled=False."
            )
        self.public_model = public_model
        self.config = config or TeacherModelConfig()
        privileged_width = len(ALL_ROLES) * CARD_VECTOR_DIM
        action_width = public_model._action_width
        hidden = self.config.hidden_size
        self.privileged_encoder = nn.Sequential(
            nn.Linear(privileged_width, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.privileged_action_head = nn.Sequential(
            nn.Linear(hidden + action_width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    @property
    def schema(self):
        """Feature schema of the public backbone."""

        return self.public_model.schema

    def config_hash(self) -> str:
        """Stable architecture identity for strict teacher checkpoints."""

        return self.config.stable_hash(self.public_model.config.stable_hash())

    def _encode_privileged(
        self,
        privileged: PrivilegedObservation,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        missing = set(ALL_ROLES) - set(privileged.all_handcards)
        extra = set(privileged.all_handcards) - set(ALL_ROLES)
        if missing or extra:
            raise ValueError(
                f"PrivilegedObservation all_handcards role mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        vectors = [cards_to_vector(privileged.all_handcards[role]) for role in ALL_ROLES]
        flat = torch.from_numpy(np.concatenate(vectors)).to(
            device=device, dtype=dtype
        )
        return self.privileged_encoder(flat)

    def forward(
        self,
        public_input: ObservationV2 | ModelInputBundle,
        privileged_observation: PrivilegedObservation,
        *,
        action_keys: tuple[ActionKey, ...] | None = None,
    ) -> TeacherOutput:
        """Score legal actions using public state plus exact training-only hands.

        ``action_keys`` is required for a tensorized offline sample. For a live
        ``ObservationV2`` it is derived from the observation's legal-action row
        order. The explicit key list is returned with the predictions and must
        be used to align teacher and student rows.
        """

        if not isinstance(privileged_observation, PrivilegedObservation):
            raise TypeError(
                "TeacherModel.forward requires a PrivilegedObservation as its "
                f"second argument, got {type(privileged_observation).__name__}"
            )
        if isinstance(public_input, ObservationV2):
            keys = canonical_action_keys(public_input.actions.legal_actions)
            bundle = observation_to_model_inputs(
                public_input, self.public_model.strategy_feature_config()
            )
        elif isinstance(public_input, ModelInputBundle):
            if action_keys is None:
                raise ValueError(
                    "action_keys are required when TeacherModel receives a "
                    "tensorized ModelInputBundle"
                )
            keys = tuple(tuple(key) for key in action_keys)
            bundle = public_input
        else:
            raise TypeError(
                "public_input must be ObservationV2 or ModelInputBundle, got "
                f"{type(public_input).__name__}"
            )
        if privileged_observation.acting_role != bundle.acting_role:
            raise ValueError(
                f"acting_role mismatch: public={bundle.acting_role!r}, "
                f"privileged={privileged_observation.acting_role!r}"
            )
        if privileged_observation.acting_role not in privileged_observation.all_handcards:
            raise ValueError(
                f"PrivilegedObservation has no hand for acting role "
                f"{privileged_observation.acting_role!r}"
            )
        public_hand = bundle.state_card_vectors[0].detach().cpu().to(torch.int8)
        privileged_hand = torch.from_numpy(
            cards_to_vector(
                privileged_observation.all_handcards[privileged_observation.acting_role]
            )
        )
        if not torch.equal(public_hand, privileged_hand):
            raise ValueError(
                "PrivilegedObservation acting hand does not match the public "
                "my_handcards tensor"
            )
        if len(keys) != bundle.action_features.shape[0]:
            raise ValueError(
                f"action key count {len(keys)} does not match action rows "
                f"{bundle.action_features.shape[0]}"
            )

        device = next(self.parameters()).device
        bundle.to(device)
        public_output = forward_public_model(self.public_model, bundle)
        privileged_hidden = self._encode_privileged(
            privileged_observation,
            device=bundle.action_features.device,
            dtype=bundle.action_features.dtype,
        )
        repeated = privileged_hidden.unsqueeze(0).expand(bundle.action_features.shape[0], -1)
        deltas = self.privileged_action_head(
            torch.cat([repeated, bundle.action_features], dim=-1)
        )
        win_logit = public_output.win_logit + deltas[:, 0:1]
        p_win = torch.sigmoid(win_logit)
        expected_score = public_output.score_mean + torch.clamp(
            deltas[:, 1:2],
            -self.config.score_delta_clamp,
            self.config.score_delta_clamp,
        )
        public_logits = (
            public_output.prior_logit
            if public_output.prior_logit is not None
            else public_output.win_logit
        )
        return TeacherOutput(
            action_keys=keys,
            win_logit=win_logit,
            p_win=p_win,
            expected_score=expected_score,
            action_logits=public_logits + deltas[:, 2:3],
            action_mask=public_output.action_mask,
        )


def state_dict_sha256(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and bytes for cache identity."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()
