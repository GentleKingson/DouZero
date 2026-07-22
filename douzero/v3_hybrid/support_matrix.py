"""Machine-readable H6/H7 capability and topology support contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

V3_H6_SUPPORT_MATRIX_VERSION = "v3-hybrid-h7-support-matrix-v1"

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
        True, False, False, True, True, True, True, True, False,
        "H3 training-only Oracle; export/deployment use the public student only",
    ),
    "belief": CapabilitySupport(
        True, False, False, True, True, True, True, True, False,
        "H4 conservative public belief with privileged labels kept training-only",
    ),
    "cooperation": CapabilitySupport(
        True, False, False, True, True, True, True, True, False,
        "H5 sidecar and mixer are training-only and excluded from public export",
    ),
    "human_bc": CapabilitySupport(
        True, False, False, True, False, True, True, True, False,
        "validated human-data replay is currently bound to legacy rules",
    ),
    "strategy": CapabilitySupport(
        True, False, False, True, True, True, True, True, False,
        "public strategy features and auxiliary heads are supported in H6",
    ),
    "style": CapabilitySupport(
        True, False, False, True, True, True, True, True, False,
        "style encoding consumes public action history only",
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
        True, False, False, False, True, True, True, True, False,
        "learned bidding is a separate standard-rules decision head",
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


def v3_h6_support_matrix_dict() -> dict[str, object]:
    return {
        "version": V3_H6_SUPPORT_MATRIX_VERSION,
        "capabilities": {
            name: asdict(row) for name, row in sorted(V3_H6_SUPPORT_MATRIX.items())
        },
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
