"""Frozen configuration for the Model V2 architecture (P05).

This is the single source of truth for the V2 model's hyperparameters. It is
deliberately kept separate from :class:`douzero.config.schemas.ModelConfig`
(which selects the *model family*) so the architecture knobs live next to the
modules that consume them. :class:`ModelV2Config.from_model_config` bridges the
two so a YAML config under ``model:`` can drive construction without duplicating
the field set.

Design notes
------------
- Widths are NOT configured here. Every input width is derived from the
  :class:`~douzero.observation.schema.FeatureSchemaManifest` (the V2 schema is
  the single source of truth — see ``douzero/observation/schema.py``). The
  model queries the schema it was constructed with, so a schema change is
  caught as a shape mismatch rather than a silent misconfiguration.
- No BatchNorm. Actor batches at inference are size-1, and BatchNorm running
  statistics under such batches are unstable. LayerNorm is used instead
  (AGENTS.md "Model rules": avoid BatchNorm unless thoroughly tested).
- ``hidden_size`` is the unified trunk width. Encoders project their raw
  per-token/per-field widths into ``hidden_size``; the fusion and heads operate
  in this space.
- Auxiliary knobs (``belief_enabled``, ``human_prior_enabled``) are carried so
  P07/P09 can attach heads behind flags; P05 keeps the structural skeleton and
  they default to off, leaving the base value/belief-free behaviour unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import ClassVar

HISTORY_ENCODER_TRANSFORMER = "transformer"
HISTORY_ENCODER_LSTM = "lstm"
_VALID_HISTORY_ENCODERS = frozenset({HISTORY_ENCODER_TRANSFORMER, HISTORY_ENCODER_LSTM})

#: Roles the V2 model supports (mirrors ``douzero.observation.seats.ALL_ROLES``).
#: Kept here to avoid importing the observation package at config-construction
#: time (config is pure plumbing).
SUPPORTED_ROLES: tuple[str, ...] = ("landlord", "landlord_up", "landlord_down")


@dataclass(frozen=True)
class ModelV2Config:
    """Architecture configuration for :class:`~douzero.models_v2.model.ModelV2`.

    Defaults keep the model small enough for a CPU forward/backward smoke test
    while still representing a credible shared backbone. Production tunings
    belong in ``configs/enhanced.yaml``, not in these defaults.
    """

    hidden_size: int = 256
    history_encoder: str = HISTORY_ENCODER_TRANSFORMER
    history_layers: int = 4
    history_heads: int = 8
    history_dropout: float = 0.0
    role_embedding_dim: int = 32
    mlp_layers: int = 2
    mlp_dropout: float = 0.0
    # Auxiliary heads (P05 keeps the skeleton; heads wired in P07/P09).
    belief_enabled: bool = False
    human_prior_enabled: bool = False
    # P09 public tactical features and auxiliary heads. Every option defaults
    # off at the architecture boundary, preserving P08 checkpoints/behaviour.
    strategy_features_enabled: bool = False
    strategy_hand_enabled: bool = True
    strategy_structure_enabled: bool = True
    strategy_control_enabled: bool = True
    strategy_cooperation_enabled: bool = True
    strategy_risk_enabled: bool = True
    strategy_aux_enabled: bool = False
    strategy_node_budget: int = 500
    strategy_time_budget_ms: int = 0
    # Score-head stability: outputs are clamped to a finite range so a wild
    # initialization cannot produce Inf that poisons the multi-objective loss.
    score_clamp: float = 32.0
    # P06 r5: score-output semantics identity. This records which target
    # transform the model's score heads were trained against, so a model
    # trained with ``"raw"`` scores cannot be loaded as if its ``score_mean``
    # were on the ``"signed_log"`` scale (or vice versa). It enters
    # :meth:`compatibility_dict` and therefore the model-config hash the
    # checkpoint loader validates, making cross-semantics loading fail with
    # a precise error. ``"raw"`` is the default (matches the P05 architecture
    # contract).
    score_target_transform: str = "raw"
    # Runtime NaN/Inf guard (bug #5). When True (default), the model forward
    # asserts its fused representation and head outputs are finite and raises
    # NumericalError otherwise. This catches both bad inputs and bad weights
    # (a NaN weight produces a NaN output regardless of the input). A caller
    # that has already validated inputs may disable it for speed.
    nan_guard: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.history_encoder not in _VALID_HISTORY_ENCODERS:
            raise ValueError(
                f"history_encoder must be one of {sorted(_VALID_HISTORY_ENCODERS)}, "
                f"got {self.history_encoder!r}"
            )
        if self.history_layers <= 0:
            raise ValueError(f"history_layers must be positive, got {self.history_layers}")
        if self.history_heads <= 0:
            raise ValueError(f"history_heads must be positive, got {self.history_heads}")
        # Transformer: hidden_size must be divisible by the head count so the
        # projected Q/K/V split is even. This is the only divisibility coupling.
        if self.history_encoder == HISTORY_ENCODER_TRANSFORMER:
            if self.hidden_size % self.history_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"history_heads ({self.history_heads}) for the transformer encoder"
                )
        if self.role_embedding_dim < 0:
            raise ValueError(
                f"role_embedding_dim must be non-negative, got {self.role_embedding_dim}"
            )
        if self.mlp_layers < 1:
            raise ValueError(f"mlp_layers must be >= 1, got {self.mlp_layers}")
        if not (0.0 <= self.history_dropout < 1.0):
            raise ValueError(f"history_dropout must be in [0, 1), got {self.history_dropout}")
        if not (0.0 <= self.mlp_dropout < 1.0):
            raise ValueError(f"mlp_dropout must be in [0, 1), got {self.mlp_dropout}")
        if self.score_clamp <= 0.0:
            raise ValueError(f"score_clamp must be positive, got {self.score_clamp}")
        if self.score_target_transform not in ("raw", "signed_log"):
            raise ValueError(
                f"score_target_transform must be 'raw' or 'signed_log', "
                f"got {self.score_target_transform!r}"
            )
        if self.strategy_aux_enabled and not self.strategy_features_enabled:
            raise ValueError(
                "strategy_aux_enabled requires strategy_features_enabled=True"
            )
        if isinstance(self.strategy_node_budget, bool) or self.strategy_node_budget <= 0:
            raise ValueError(
                f"strategy_node_budget must be a positive int, got "
                f"{self.strategy_node_budget!r}"
            )
        if (
            isinstance(self.strategy_time_budget_ms, bool)
            or self.strategy_time_budget_ms < 0
        ):
            raise ValueError(
                f"strategy_time_budget_ms must be a non-negative int, got "
                f"{self.strategy_time_budget_ms!r}"
            )

    # ------------------------------------------------------------------ #
    # Canonical identity (blocker #2 fix)
    # ------------------------------------------------------------------ #
    #: The current identity version. Increment when the compatibility-dict
    #: field set changes. Version 1 = P05 (no ``score_target_transform``);
    #: version 2 = P06-P08 (adds ``score_target_transform``); version 3 = P09
    #: (adds the optional strategy architecture and ablation identity).
    #: Declared as ClassVar so the dataclass machinery does not treat it
    #: as an instance field (otherwise ``ModelV2Config(IDENTITY_VERSION=999)``
    #: would create a meaningless state).
    IDENTITY_VERSION: ClassVar[int] = 3

    def compatibility_dict(self) -> dict:
        """Return the architecture-identity subset of this config.

        Two configs with the same :meth:`compatibility_dict` produce models
        with the same architecture (same parameter shapes AND the same
        runtime behavior). This is the contract a checkpoint binds to: a
        weights sidecar saved under one config must not be loaded into a model
        built under a different config, even if the parameter SHAPES happen to
        match (e.g. ``history_heads`` 8→4 keeps the projection shapes but
        changes how the Transformer splits the hidden dim; ``score_clamp``
        32→8 changes the output clamp; ``nan_guard`` true→false changes
        whether the forward validates finiteness).

        ``strict=True`` state_dict loading only checks parameter KEYS exist —
        it cannot detect these same-shape-different-semantics drifts, so the
        config hash is the missing second identity axis alongside the feature
        schema hash.
        """
        return {
            "hidden_size": self.hidden_size,
            "history_encoder": self.history_encoder,
            "history_layers": self.history_layers,
            "history_heads": self.history_heads,
            "history_dropout": self.history_dropout,
            "role_embedding_dim": self.role_embedding_dim,
            "mlp_layers": self.mlp_layers,
            "mlp_dropout": self.mlp_dropout,
            "belief_enabled": self.belief_enabled,
            "human_prior_enabled": self.human_prior_enabled,
            "strategy_features_enabled": self.strategy_features_enabled,
            "strategy_hand_enabled": self.strategy_hand_enabled,
            "strategy_structure_enabled": self.strategy_structure_enabled,
            "strategy_control_enabled": self.strategy_control_enabled,
            "strategy_cooperation_enabled": self.strategy_cooperation_enabled,
            "strategy_risk_enabled": self.strategy_risk_enabled,
            "strategy_aux_enabled": self.strategy_aux_enabled,
            "strategy_node_budget": self.strategy_node_budget,
            "strategy_time_budget_ms": self.strategy_time_budget_ms,
            "score_clamp": self.score_clamp,
            "score_target_transform": self.score_target_transform,
            "nan_guard": self.nan_guard,
        }

    def compatibility_dict_v1(self) -> dict:
        """P05-era compatibility dict (without ``score_target_transform``).

        P05 checkpoints were saved before ``score_target_transform`` was added
        to :meth:`compatibility_dict`. This method reproduces the EXACT field
        set P05 used, so the loader can validate a P05 checkpoint's hash
        against the runtime config's v1 hash. A P05 checkpoint is only
        loadable when the runtime ``score_target_transform == "raw"`` (the P05
        default), because the P05 model's score heads were implicitly on the
        raw scale.
        """
        return {
            "hidden_size": self.hidden_size,
            "history_encoder": self.history_encoder,
            "history_layers": self.history_layers,
            "history_heads": self.history_heads,
            "history_dropout": self.history_dropout,
            "role_embedding_dim": self.role_embedding_dim,
            "mlp_layers": self.mlp_layers,
            "mlp_dropout": self.mlp_dropout,
            "belief_enabled": self.belief_enabled,
            "human_prior_enabled": self.human_prior_enabled,
            "score_clamp": self.score_clamp,
            "nan_guard": self.nan_guard,
        }

    def compatibility_dict_v2(self) -> dict:
        """P06-P08 compatibility identity, before P09 strategy fields.

        A version-2 checkpoint can migrate only into a strategy-disabled
        runtime.  The loader enforces that condition before comparing this
        hash, so old public policies remain usable without pretending they
        contain strategy parameters.
        """

        return {
            "hidden_size": self.hidden_size,
            "history_encoder": self.history_encoder,
            "history_layers": self.history_layers,
            "history_heads": self.history_heads,
            "history_dropout": self.history_dropout,
            "role_embedding_dim": self.role_embedding_dim,
            "mlp_layers": self.mlp_layers,
            "mlp_dropout": self.mlp_dropout,
            "belief_enabled": self.belief_enabled,
            "human_prior_enabled": self.human_prior_enabled,
            "score_clamp": self.score_clamp,
            "score_target_transform": self.score_target_transform,
            "nan_guard": self.nan_guard,
        }

    def stable_hash(self) -> str:
        """A SHA-256 hex digest of :meth:`compatibility_dict`.

        Stable across runs (deterministic JSON serialization with sorted
        keys). Stamp this into a checkpoint so a loader can reject a
        same-shape-different-config drift.
        """
        payload = json.dumps(self.compatibility_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stable_hash_v1(self) -> str:
        """SHA-256 of :meth:`compatibility_dict_v1` (P05 identity).

        Used by the checkpoint loader to validate a P05 checkpoint whose
        bundle lacks ``model_config_identity_version`` (or carries version 1).
        The hash must match AND the runtime ``score_target_transform`` must be
        ``"raw"`` (the P05 default) for the migration to succeed.
        """
        payload = json.dumps(self.compatibility_dict_v1(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stable_hash_v2(self) -> str:
        """SHA-256 of the P06-P08 compatibility field set."""

        payload = json.dumps(self.compatibility_dict_v2(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_model_config(cls, model_config) -> "ModelV2Config":
        """Build a :class:`ModelV2Config` from a :class:`ModelConfig`.

        This bridges the YAML-driven ``model:`` block (parsed into the frozen
        :class:`~douzero.config.schemas.ModelConfig`) and the architecture-level
        config consumed by the V2 modules. Only the fields that exist on both
        are copied; the architecture defaults fill the rest.
        """
        kwargs: dict[str, object] = {}
        for name in ("hidden_size", "history_encoder", "history_layers",
                     "history_heads", "role_embedding_dim", "belief_enabled",
                     "human_prior_enabled", "strategy_features_enabled",
                     "strategy_hand_enabled", "strategy_structure_enabled",
                     "strategy_control_enabled", "strategy_cooperation_enabled",
                     "strategy_risk_enabled", "strategy_aux_enabled",
                     "strategy_node_budget", "strategy_time_budget_ms"):
            if hasattr(model_config, name):
                kwargs[name] = getattr(model_config, name)
        return cls(**kwargs)
