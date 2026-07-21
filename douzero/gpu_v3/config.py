from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


INDEPENDENT_ROLE_DUAL_TOWER = "independent_role_dual_tower"
SHARED_TRUNK_ROLE_HEADS = "shared_trunk_role_heads"
GPU_V3_ARCHITECTURES = frozenset({
    INDEPENDENT_ROLE_DUAL_TOWER,
    SHARED_TRUNK_ROLE_HEADS,
})


@dataclass(frozen=True)
class GPUV3Config:
    architecture: str = INDEPENDENT_ROLE_DUAL_TOWER
    hidden_size: int = 512
    action_hidden_size: int = 256
    trunk_layers: int = 4
    role_head_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        if self.architecture not in GPU_V3_ARCHITECTURES:
            raise ValueError(f"unsupported gpu_v3 architecture {self.architecture!r}")
        if min(
            self.hidden_size,
            self.action_hidden_size,
            self.trunk_layers,
            self.role_head_layers,
        ) < 1:
            raise ValueError("gpu_v3 widths and layer counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("gpu_v3 dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)

    def stable_hash(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
