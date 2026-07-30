"""Machine-readable H6/H7 capability and topology support contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

V3_H6_SUPPORT_MATRIX_VERSION = "v3-hybrid-h7-1e-support-matrix-v8"

TOPOLOGY_SINGLE_PROCESS = "single_process"
TOPOLOGY_ASYNC_SINGLE_GPU = "async_single_gpu"
TOPOLOGY_DDP = "ddp"

RULESET_LEGACY = "legacy"
RULESET_STANDARD = "standard"


@dataclass(frozen=True)
class CapabilitySupport:
    """One stable row in the V3 support matrix."""

    single_process: bool
    async_single_gpu: bool
    ddp: bool
    legacy_rules: bool
    standard_rules: bool
    checkpoint_resume: bool
    export: bool
    deployment: bool
    search: bool
    note: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "note":
                if not isinstance(value, str) or not value:
                    raise ValueError("support-matrix note must be a non-empty string")
            elif not isinstance(value, bool):
                raise TypeError(f"support-matrix field {name} must be bool")


_ROWS = {
    "role_model": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7 reuses the bounded V2 async protocol for public V3 card play",
    ),
    "adaptive_dmc": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7 async replay binds q_old to the immutable served snapshot",
    ),
    "oracle": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7.1b runs Oracle only in the learner; actors and export use the student",
    ),
    "belief": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7.1a serves coupled public belief snapshots; labels stay sidecar-only",
    ),
    "cooperation": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7.1c aggregates farmer trajectories learner-side; mixer stays training-only",
    ),
    "human_bc": CapabilitySupport(
        True, False, False, True, False, True, True, True, False,
        "validated human-data replay is currently bound to legacy rules",
    ),
    "strategy": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7.1d transports public strategy features and learner trajectory labels",
    ),
    "style": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "H7.1d binds public-history style features to the served policy snapshot",
    ),
    "league": CapabilitySupport(
        False, False, False, True, True, False, False, False, False,
        "V3 policy provenance/runtime integration is deferred to H7",
    ),
    "curriculum": CapabilitySupport(
        False, False, False, True, True, False, False, False, False,
        "V3 actor/coach runtime integration is deferred to H7",
    ),
    "bidding": CapabilitySupport(
        True, True, False, False, True, True, True, True, False,
        "H7.1e transports separate neutral-seat bid decisions and replay",
    ),
    "selective_search": CapabilitySupport(
        True, False, False, True, True, True, True, True, True,
        "H7 public-only composite gate wraps existing budgeted belief search",
    ),
    "public_export": CapabilitySupport(
        True, True, False, True, True, True, True, True, False,
        "strict public-only model sidecar; formal release package is H8 scope",
    ),
}

V3_H6_SUPPORT_MATRIX: Mapping[str, CapabilitySupport] = MappingProxyType(_ROWS)
V3_H6_UNSUPPORTED_COMBINATIONS = (
    ("belief", TOPOLOGY_ASYNC_SINGLE_GPU, RULESET_STANDARD),
    ("oracle", TOPOLOGY_ASYNC_SINGLE_GPU, RULESET_STANDARD),
    ("cooperation", TOPOLOGY_ASYNC_SINGLE_GPU, RULESET_STANDARD),
    ("strategy", TOPOLOGY_ASYNC_SINGLE_GPU, RULESET_STANDARD),
    ("style", TOPOLOGY_ASYNC_SINGLE_GPU, RULESET_STANDARD),
)


def v3_h6_support_matrix_dict() -> dict[str, object]:
    return {
        "version": V3_H6_SUPPORT_MATRIX_VERSION,
        "capabilities": {
            name: asdict(row) for name, row in sorted(V3_H6_SUPPORT_MATRIX.items())
        },
        "unsupported_combinations": [
            {
                "capability": capability,
                "topology": topology,
                "ruleset": ruleset,
            }
            for capability, topology, ruleset in V3_H6_UNSUPPORTED_COMBINATIONS
        ],
    }


def v3_h6_support_matrix_hash() -> str:
    payload = json.dumps(
        v3_h6_support_matrix_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def validate_capability_support(
    capability: str,
    *,
    topology: str,
    ruleset: str,
    checkpoint_resume: bool,
    export: bool,
    deployment: bool,
    search: bool,
) -> None:
    """Fail before runtime side effects when a requested cell is unsupported."""

    try:
        row = V3_H6_SUPPORT_MATRIX[capability]
    except KeyError as exc:
        raise ValueError(f"unknown V3 H6 capability {capability!r}") from exc
    topology_field = {
        TOPOLOGY_SINGLE_PROCESS: "single_process",
        TOPOLOGY_ASYNC_SINGLE_GPU: "async_single_gpu",
        TOPOLOGY_DDP: "ddp",
    }.get(topology)
    if topology_field is None:
        raise ValueError(f"unknown V3 H6 topology {topology!r}")
    if (capability, topology, ruleset) in V3_H6_UNSUPPORTED_COMBINATIONS:
        raise ValueError(
            f"V3 H6 capability {capability!r} does not support the "
            f"{topology!r} + {ruleset!r} combination; {row.note}"
        )
    checks = [
        (topology_field, True),
        ("legacy_rules" if ruleset == RULESET_LEGACY else "standard_rules", True),
        ("checkpoint_resume", checkpoint_resume),
        ("export", export),
        ("deployment", deployment),
        ("search", search),
    ]
    if ruleset not in {RULESET_LEGACY, RULESET_STANDARD}:
        raise ValueError(f"unknown V3 H6 ruleset {ruleset!r}")
    for field, requested in checks:
        if requested and not getattr(row, field):
            raise ValueError(
                f"V3 H6 capability {capability!r} does not support {field}; "
                f"{row.note}"
            )


__all__ = [
    "CapabilitySupport",
    "RULESET_LEGACY",
    "RULESET_STANDARD",
    "TOPOLOGY_ASYNC_SINGLE_GPU",
    "TOPOLOGY_DDP",
    "TOPOLOGY_SINGLE_PROCESS",
    "V3_H6_SUPPORT_MATRIX",
    "V3_H6_SUPPORT_MATRIX_VERSION",
    "v3_h6_support_matrix_dict",
    "v3_h6_support_matrix_hash",
    "validate_capability_support",
]
