"""Small, independently-versioned coach model and checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .labels import CoachLabel
from .records import CARD_RANKS, OpeningRecord


COACH_MODEL_VERSION = "opening_coach_v1"
COACH_FEATURE_VERSION = "opening_features_v1"
_POLICY_HASH_WIDTH = 8
_RANK_INDEX = {rank: index for index, rank in enumerate(CARD_RANKS)}


def _rank_counts(cards: Iterable[int]) -> list[float]:
    counts = [0.0] * len(CARD_RANKS)
    for card in cards:
        counts[_RANK_INDEX[card]] += 0.25 if card not in (20, 30) else 1.0
    return counts


def encode_opening(record: OpeningRecord, policy_version: str) -> torch.Tensor:
    """Encode a full training-only opening and policy identity to a vector."""

    if not policy_version:
        raise ValueError("policy_version must be non-empty")
    data = record.to_card_play_data()
    features: list[float] = []
    for key in ("landlord", "landlord_up", "landlord_down", "three_landlord_cards"):
        features.extend(_rank_counts(data[key]))
    ruleset = record.ruleset_obj
    features.extend(
        [
            float(ruleset.ruleset_id == "standard"),
            float(record.landlord_candidate in ("landlord", "0")),
            float(record.landlord_candidate == "1"),
            float(record.landlord_candidate == "2"),
            float(max(record.bids, default=0)) / 3.0,
            float(len(record.bids)) / 3.0,
        ]
    )
    digest = hashlib.sha256(policy_version.encode("utf-8")).digest()
    features.extend(byte / 255.0 for byte in digest[:_POLICY_HASH_WIDTH])
    return torch.tensor(features, dtype=torch.float32)


COACH_INPUT_SIZE = len(CARD_RANKS) * 4 + 6 + _POLICY_HASH_WIDTH


@dataclass(frozen=True)
class CoachModelConfig:
    """Architecture identity for :class:`CoachModel`."""

    input_size: int = COACH_INPUT_SIZE
    hidden_size: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_size != COACH_INPUT_SIZE:
            raise ValueError(
                f"input_size must match {COACH_FEATURE_VERSION} ({COACH_INPUT_SIZE})"
            )
        if (
            isinstance(self.hidden_size, bool)
            or not isinstance(self.hidden_size, int)
            or self.hidden_size < 1
        ):
            raise ValueError("hidden_size must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


class CoachModel(nn.Module):
    """Predict landlord win probability for an opening and policy version."""

    def __init__(self, config: CoachModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or CoachModelConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return landlord win logits for ``(..., COACH_INPUT_SIZE)`` input."""

        if features.shape[-1] != self.config.input_size:
            raise ValueError(
                f"coach features width must be {self.config.input_size}, "
                f"got {features.shape[-1]}"
            )
        return self.network(features).squeeze(-1)

    def predict(self, opening: OpeningRecord, policy_version: str) -> float:
        """Return a calibrated-probability candidate in ``[0, 1]``."""

        was_training = self.training
        self.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(self(encode_opening(opening, policy_version)))
        self.train(was_training)
        return float(probability.item())


def train_coach(
    model: CoachModel,
    labels: Iterable[CoachLabel],
    *,
    epochs: int = 1,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Fit the coach on fresh self-play labels and return epoch losses."""

    samples = list(labels)
    if not samples:
        raise ValueError("train_coach requires at least one label")
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive")
    features = torch.stack(
        [encode_opening(item.opening, item.policy_version) for item in samples]
    )
    targets = torch.tensor([item.landlord_win for item in samples], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(features), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("coach loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def save_coach_checkpoint(
    path: str,
    model: CoachModel,
    *,
    policy_version: str,
    policy_step: int,
    ruleset_hash: str,
    calibration: dict[str, float] | None = None,
) -> None:
    """Atomically save the coach separately from all policy checkpoints."""

    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("policy_version must be non-empty")
    if isinstance(policy_step, bool) or not isinstance(policy_step, int) or policy_step < 0:
        raise ValueError("policy_step must be a non-negative int")
    if not isinstance(ruleset_hash, str) or not ruleset_hash:
        raise ValueError("coach checkpoint provenance is incomplete")
    manifest = {
        "schema_version": 1,
        "model_version": COACH_MODEL_VERSION,
        "feature_version": COACH_FEATURE_VERSION,
        "model_config_hash": model.config.stable_hash(),
        "policy_version": policy_version,
        "policy_step": policy_step,
        "ruleset_hash": ruleset_hash,
        "calibration": dict(calibration or {}),
    }
    bundle = {
        "coach_state_dict": model.state_dict(),
        "coach_config": asdict(model.config),
        "manifest": manifest,
    }
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute) or ".", exist_ok=True)
    temporary = absolute + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, absolute)


def load_coach_checkpoint(
    path: str,
    *,
    expected_ruleset_hash: str,
    map_location: Any = "cpu",
) -> tuple[CoachModel, dict[str, Any]]:
    """Safely load a coach and validate all independent identity fields."""

    bundle = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(bundle, dict) or set(bundle) != {
        "coach_state_dict", "coach_config", "manifest"
    }:
        raise ValueError("invalid coach checkpoint bundle")
    manifest = bundle["manifest"]
    if not isinstance(manifest, dict):
        raise TypeError("coach checkpoint manifest must be a dict")
    expected_manifest = {
        "schema_version", "model_version", "feature_version", "model_config_hash",
        "policy_version", "policy_step", "ruleset_hash", "calibration",
    }
    if set(manifest) != expected_manifest:
        raise ValueError("coach checkpoint manifest fields are invalid")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported coach checkpoint schema")
    if manifest["model_version"] != COACH_MODEL_VERSION:
        raise ValueError("coach model_version mismatch")
    if manifest["feature_version"] != COACH_FEATURE_VERSION:
        raise ValueError("coach feature_version mismatch")
    if manifest["ruleset_hash"] != expected_ruleset_hash:
        raise ValueError("coach ruleset_hash mismatch")
    if not isinstance(manifest["policy_version"], str) or not manifest["policy_version"]:
        raise ValueError("coach policy_version is invalid")
    if (
        isinstance(manifest["policy_step"], bool)
        or not isinstance(manifest["policy_step"], int)
        or manifest["policy_step"] < 0
    ):
        raise ValueError("coach policy_step is invalid")
    if not isinstance(manifest["calibration"], dict):
        raise TypeError("coach calibration metadata must be a dict")
    if not isinstance(bundle["coach_config"], dict):
        raise TypeError("coach_config must be a dict")
    if not isinstance(bundle["coach_state_dict"], dict):
        raise TypeError("coach_state_dict must be a dict")
    config = CoachModelConfig(**bundle["coach_config"])
    if config.stable_hash() != manifest["model_config_hash"]:
        raise ValueError("coach model_config_hash mismatch")
    model = CoachModel(config)
    model.load_state_dict(bundle["coach_state_dict"], strict=True)
    model.eval()
    return model, dict(manifest)
