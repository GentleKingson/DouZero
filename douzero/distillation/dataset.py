"""Offline, safely serialized samples for privileged-teacher distillation."""

from __future__ import annotations

import hashlib
import math
import numbers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch

from douzero.models_v2.batch import ModelInputBundle, observation_to_model_inputs
from douzero.observation.encode_v2 import ObservationV2
from douzero.observation.privileged import PrivilegedObservation

from .teacher_model import ActionKey, canonical_action_keys

DATASET_VERSION = 1


@dataclass(frozen=True)
class DistillationSample:
    """One live public decision paired with training-only hidden hands/labels."""

    public_observation: ObservationV2
    privileged_observation: PrivilegedObservation
    action_index: int
    target_win: float
    target_score: float
    sample_id: str = ""

    def tensorize(
        self,
        strategy_config=None,
        *,
        style_enabled: bool = False,
    ) -> "OfflineDistillationSample":
        """Convert the public observation to the offline tensor format."""

        keys = canonical_action_keys(self.public_observation.actions.legal_actions)
        return OfflineDistillationSample(
            public_inputs=observation_to_model_inputs(
                self.public_observation,
                strategy_config,
                style_enabled=style_enabled,
            ),
            privileged_observation=self.privileged_observation,
            action_keys=keys,
            action_index=self.action_index,
            target_win=self.target_win,
            target_score=self.target_score,
            sample_id=self.sample_id,
        )


@dataclass(frozen=True)
class OfflineDistillationSample:
    """Tensorized public input plus a separately named privileged container."""

    public_inputs: ModelInputBundle
    privileged_observation: PrivilegedObservation
    action_keys: tuple[ActionKey, ...]
    action_index: int
    target_win: float
    target_score: float
    sample_id: str = ""

    def __post_init__(self) -> None:
        n = self.public_inputs.action_features.shape[0]
        if len(self.action_keys) != n:
            raise ValueError(
                f"action_keys has {len(self.action_keys)} rows but public action "
                f"features have {n}"
            )
        if len(set(self.action_keys)) != len(self.action_keys):
            raise ValueError("action_keys must be unique")
        if any(key != tuple(sorted(key)) for key in self.action_keys):
            raise ValueError("action_keys must be canonical sorted tuples")
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise TypeError(
                f"action_index must be an int, got {type(self.action_index).__name__}"
            )
        if not 0 <= self.action_index < n:
            raise ValueError(
                f"action_index {self.action_index} outside legal-action range [0, {n})"
            )
        if not bool(self.public_inputs.action_mask[self.action_index]):
            raise ValueError("action_index points to a masked action row")
        if self.privileged_observation.acting_role != self.public_inputs.acting_role:
            raise ValueError(
                f"acting_role mismatch: public={self.public_inputs.acting_role!r}, "
                f"privileged={self.privileged_observation.acting_role!r}"
            )
        if (
            isinstance(self.target_win, bool)
            or not isinstance(self.target_win, numbers.Real)
            or not math.isfinite(float(self.target_win))
            or float(self.target_win) not in (0.0, 1.0)
        ):
            raise ValueError(
                f"target_win must be finite and exactly 0.0 or 1.0, got "
                f"{self.target_win!r}"
            )
        if (
            isinstance(self.target_score, bool)
            or not isinstance(self.target_score, numbers.Real)
            or not math.isfinite(float(self.target_score))
        ):
            raise ValueError(
                f"target_score must be a finite number, got {self.target_score!r}"
            )
        if not isinstance(self.sample_id, str):
            raise TypeError(
                f"sample_id must be str, got {type(self.sample_id).__name__}"
            )
        object.__setattr__(self, "target_win", float(self.target_win))
        object.__setattr__(self, "target_score", float(self.target_score))


class DistillationDataset(Sequence[OfflineDistillationSample]):
    """Small offline dataset with deterministic iteration order."""

    def __init__(self, samples: Iterable[OfflineDistillationSample]) -> None:
        self._samples = tuple(samples)
        if not self._samples:
            raise ValueError("DistillationDataset requires at least one sample")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> OfflineDistillationSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[OfflineDistillationSample]:
        return iter(self._samples)


def _bundle_to_dict(bundle: ModelInputBundle) -> dict:
    return {
        "state_card_vectors": [tensor.detach().cpu() for tensor in bundle.state_card_vectors],
        "state_context_flat": bundle.state_context_flat.detach().cpu(),
        "context_card_vectors": [tensor.detach().cpu() for tensor in bundle.context_card_vectors],
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


def _bundle_from_dict(raw: dict) -> ModelInputBundle:
    return ModelInputBundle(
        state_card_vectors=tuple(raw["state_card_vectors"]),
        state_context_flat=raw["state_context_flat"],
        context_card_vectors=tuple(raw["context_card_vectors"]),
        context_flat=raw["context_flat"],
        history_tokens=raw["history_tokens"],
        history_key_padding_mask=raw["history_key_padding_mask"],
        action_features=raw["action_features"],
        action_mask=raw["action_mask"],
        acting_role=str(raw["acting_role"]),
        feature_schema_hash=str(raw["feature_schema_hash"]),
        strategy_features=raw.get("strategy_features"),
        style_features=raw.get("style_features"),
    )


def _sample_to_dict(sample: OfflineDistillationSample) -> dict:
    return {
        "public_inputs": _bundle_to_dict(sample.public_inputs),
        "privileged": {
            "acting_role": sample.privileged_observation.acting_role,
            "all_handcards": {
                role: list(cards)
                for role, cards in sample.privileged_observation.all_handcards.items()
            },
        },
        "action_keys": [list(key) for key in sample.action_keys],
        "action_index": sample.action_index,
        "target_win": float(sample.target_win),
        "target_score": float(sample.target_score),
        "sample_id": sample.sample_id,
    }


def _sample_from_dict(raw: dict) -> OfflineDistillationSample:
    privileged_raw = raw["privileged"]
    privileged = PrivilegedObservation(
        all_handcards={
            str(role): tuple(int(card) for card in cards)
            for role, cards in privileged_raw["all_handcards"].items()
        },
        acting_role=str(privileged_raw["acting_role"]),
    )
    return OfflineDistillationSample(
        public_inputs=_bundle_from_dict(raw["public_inputs"]),
        privileged_observation=privileged,
        action_keys=tuple(tuple(int(card) for card in key) for key in raw["action_keys"]),
        action_index=raw["action_index"],
        target_win=raw["target_win"],
        target_score=raw["target_score"],
        sample_id=raw.get("sample_id", ""),
    )


def save_offline_dataset(
    path: str | os.PathLike[str],
    dataset: DistillationDataset,
    *,
    feature_schema_hash: str,
    ruleset_hash: str,
    producer_model_sha: str,
) -> None:
    """Atomically save a weights-only-loadable offline self-play dataset."""

    manifest = {
        "dataset_version": DATASET_VERSION,
        "feature_schema_hash": feature_schema_hash,
        "ruleset_hash": ruleset_hash,
        "producer_model_sha": producer_model_sha,
        "num_samples": len(dataset),
        "access": "privileged_training_only",
    }
    bundle = {
        "manifest": manifest,
        "samples": [_sample_to_dict(sample) for sample in dataset],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, destination)


def load_offline_dataset(
    path: str | os.PathLike[str],
    *,
    expected_feature_schema_hash: str,
    expected_ruleset_hash: str,
    allow_unsafe_pickle: bool = False,
) -> DistillationDataset:
    """Load an offline dataset and reject schema/ruleset/version drift."""

    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=not allow_unsafe_pickle,
    )
    if not isinstance(bundle, dict) or set(bundle) != {"manifest", "samples"}:
        raise ValueError("distillation dataset is not a recognized bundle")
    manifest = bundle["manifest"]
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"dataset_version mismatch: {manifest.get('dataset_version')!r} "
            f"!= {DATASET_VERSION}"
        )
    if manifest.get("feature_schema_hash") != expected_feature_schema_hash:
        raise ValueError("distillation dataset feature_schema_hash mismatch")
    if manifest.get("ruleset_hash") != expected_ruleset_hash:
        raise ValueError("distillation dataset ruleset_hash mismatch")
    samples = [_sample_from_dict(raw) for raw in bundle["samples"]]
    if manifest.get("num_samples") != len(samples):
        raise ValueError("distillation dataset num_samples does not match payload")
    return DistillationDataset(samples)


def teacher_observation_hash(sample: OfflineDistillationSample) -> str:
    """Hash public tensors plus exact hands for a collision-safe teacher key.

    A teacher prediction depends on the hidden allocation. Hashing only the
    public observation would incorrectly alias different privileged states
    that are deliberately identical to the student.
    """

    digest = hashlib.sha256()
    bundle = _bundle_to_dict(sample.public_inputs)
    for name in sorted(bundle):
        value = bundle[name]
        digest.update(name.encode("utf-8"))
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(item).encode("utf-8"))
    for role, cards in sorted(sample.privileged_observation.all_handcards.items()):
        digest.update(role.encode("utf-8"))
        digest.update(repr(tuple(cards)).encode("ascii"))
    digest.update(repr(sample.action_keys).encode("ascii"))
    return digest.hexdigest()
