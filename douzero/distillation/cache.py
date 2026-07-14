"""Versioned JSON teacher-output cache with strict identity checks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .dataset import OfflineDistillationSample, teacher_observation_hash
from .teacher_model import TeacherOutput

CACHE_VERSION = 2


@dataclass(frozen=True)
class TeacherCacheIdentity:
    """All identities that can change the meaning of cached teacher rows."""

    feature_schema_hash: str
    ruleset_hash: str
    teacher_model_sha: str
    teacher_config_hash: str
    cache_version: int = CACHE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "feature_schema_hash", "ruleset_hash", "teacher_model_sha",
            "teacher_config_hash",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.cache_version != CACHE_VERSION:
            raise ValueError(
                f"cache_version {self.cache_version} is unsupported; expected {CACHE_VERSION}"
            )


class TeacherCache:
    """Persistent teacher predictions keyed by public+privileged observation hash."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        identity: TeacherCacheIdentity,
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or set(payload) != {"identity", "entries"}:
            raise ValueError("teacher cache is not a recognized versioned bundle")
        actual = payload["identity"]
        expected = asdict(self.identity)
        if actual != expected:
            mismatches = {
                key: (actual.get(key), value)
                for key, value in expected.items()
                if actual.get(key) != value
            }
            raise ValueError(f"teacher cache identity mismatch: {mismatches}")
        if not isinstance(payload["entries"], dict):
            raise ValueError("teacher cache entries must be a mapping")
        self._entries = dict(payload["entries"])

    def get(self, sample: OfflineDistillationSample) -> TeacherOutput | None:
        """Return a detached teacher output, or ``None`` on a cache miss."""

        raw = self._entries.get(teacher_observation_hash(sample))
        if raw is None:
            return None
        keys = tuple(tuple(int(card) for card in key) for key in raw["action_keys"])
        mask = torch.tensor(raw["action_mask"], dtype=torch.bool)
        return TeacherOutput(
            action_keys=keys,
            win_logit=torch.tensor(raw["win_logit"], dtype=torch.float32).reshape(-1, 1),
            p_win=torch.tensor(raw["p_win"], dtype=torch.float32).reshape(-1, 1),
            score_if_win=torch.tensor(
                raw["score_if_win"], dtype=torch.float32
            ).reshape(-1, 1),
            score_if_loss=torch.tensor(
                raw["score_if_loss"], dtype=torch.float32
            ).reshape(-1, 1),
            expected_score=torch.tensor(
                raw["expected_score"], dtype=torch.float32
            ).reshape(-1, 1),
            action_logits=torch.tensor(
                raw["action_logits"], dtype=torch.float32
            ).reshape(-1, 1),
            action_mask=mask,
        )

    def put(
        self,
        sample: OfflineDistillationSample,
        output: TeacherOutput,
    ) -> None:
        """Insert one output; the output keys must match the sample set."""

        if set(output.action_keys) != set(sample.action_keys):
            raise ValueError("cannot cache teacher output for a different legal-action set")
        detached = output.detached_cpu()
        self._entries[teacher_observation_hash(sample)] = {
            "action_keys": [list(key) for key in detached.action_keys],
            "win_logit": detached.win_logit.squeeze(-1).tolist(),
            "p_win": detached.p_win.squeeze(-1).tolist(),
            "score_if_win": detached.score_if_win.squeeze(-1).tolist(),
            "score_if_loss": detached.score_if_loss.squeeze(-1).tolist(),
            "expected_score": detached.expected_score.squeeze(-1).tolist(),
            "action_logits": detached.action_logits.squeeze(-1).tolist(),
            "action_mask": detached.action_mask.tolist(),
        }

    def save(self) -> None:
        """Atomically write cache metadata and entries."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"identity": asdict(self.identity), "entries": self._entries}
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, self.path)

    def __len__(self) -> int:
        return len(self._entries)
