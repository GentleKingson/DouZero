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

from dataclasses import dataclass

HISTORY_ENCODER_TRANSFORMER = "transformer"
HISTORY_ENCODER_LSTM = "lsm"
HISTORY_ENCODER_LSTM_CANONICAL = "lstm"
_VALID_HISTORY_ENCODERS = frozenset({HISTORY_ENCODER_TRANSFORMER, HISTORY_ENCODER_LSTM, HISTORY_ENCODER_LSTM_CANONICAL})

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
    # Score-head stability: outputs are clamped to a finite range so a wild
    # initialization cannot produce Inf that poisons the multi-objective loss.
    score_clamp: float = 32.0

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
                     "human_prior_enabled"):
            if hasattr(model_config, name):
                kwargs[name] = getattr(model_config, name)
        return cls(**kwargs)
