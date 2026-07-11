"""Versioned rule set for DouDizhu (P02).

A ``RuleSet`` is a frozen description of every configurable rule that affects
bidding, scoring, and the game phase state machine. It is the single source of
truth for rule parameters; no magic numbers for bomb multipliers, spring, or
base scores should be scattered across the environment.

Two canonical instances are provided:

- :meth:`RuleSet.legacy` — reproduces the original DouZero environment exactly:
  no bidding, no spring, ``bomb_num`` doubling (``base_score=2`` with
  ``bomb_multiplier=2`` so the effective multiplier is ``2 ** bomb_num``).
- :meth:`RuleSet.standard` — a configurable standard ruleset: 0/1/2/3 score
  bidding, all-pass redeal, spring/anti-spring x2, bomb x2, rocket x2, base
  score 1.

The legacy mode is the default everywhere; ``standard`` is opt-in via
``--ruleset standard``. Legacy behaviour is byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

# --------------------------------------------------------------------------- #
# Phase constants (used by GameEnv's state machine; P02 Slice 3)
# --------------------------------------------------------------------------- #
PHASE_DEAL = "deal"
PHASE_BIDDING = "bidding"
PHASE_REVEAL_BOTTOM = "reveal_bottom"
PHASE_PLAYING = "playing"
PHASE_TERMINAL = "terminal"

#: The three player positions, in canonical turn order.
#: ``landlord`` acts first; ``landlord_down`` is the next seat (acts immediately
#: after the landlord); ``landlord_up`` acts immediately before the landlord.
PLAYER_POSITIONS: tuple[str, ...] = ("landlord", "landlord_down", "landlord_up")

#: Bidding mode identifiers.
BIDDING_MODE_NONE = "none"
BIDDING_MODE_SCORE_0_1_2_3 = "score_0_1_2_3"

#: Supported ruleset identifiers.
RULESET_LEGACY = "legacy"
RULESET_STANDARD = "standard"


@dataclass(frozen=True)
class RuleSet:
    """Frozen description of every configurable DouDizhu rule.

    All multiplier fields are positive integers (a value of 0 disables the
    corresponding spring/anti-spring multiplier). ``max_multiplier`` of
    ``None`` means no cap.
    """

    #: ``"legacy"`` or ``"standard"``. Recorded in the checkpoint manifest as
    #: ``ruleset_id`` so an incompatible checkpoint is rejected, not silently
    #: loaded.
    ruleset_id: str

    #: ``"none"`` (legacy: no bidding phase) or ``"score_0_1_2_3"``.
    bidding_mode: str

    #: Allowed bid values. Empty tuple for legacy; ``(0, 1, 2, 3)`` for
    #: standard (0 = pass).
    bid_values: tuple[int, ...] = field(default_factory=tuple)

    #: Whether "rob landlord" (抢地主) is allowed. Reserved for a future
    #: bidding mode; P02 always sets this to ``False``.
    allow_rob: bool = False

    #: If all three players bid 0 (pass), redeal instead of assigning the
    #: landlord by default.
    all_pass_redeal: bool = False

    #: Whether the winning bid value multiplies the base score.
    bid_multiplier: bool = False

    #: Multiplier applied per bomb played (exponent base). 2 means each bomb
    #: doubles the score.
    bomb_multiplier: int = 2

    #: Multiplier applied when the rocket (king bomb) is played. 2 means the
    #: rocket doubles the score (in addition to counting as a bomb for
    #: ``bomb_count``).
    rocket_multiplier: int = 2

    #: Multiplier applied on spring (地主春天). 0 disables spring detection.
    spring_multiplier: int = 0

    #: Multiplier applied on anti-spring (农民反春). 0 disables anti-spring.
    anti_spring_multiplier: int = 0

    #: Whether "doubling" (加倍) is allowed. Reserved for future expansion;
    #: P02 always sets this to ``False``.
    allow_double: bool = False

    #: Base score before multipliers. Legacy uses 2 (so landlord wins +2,
    #: farmer loses -1); standard uses 1.
    base_score: int = 1

    #: Optional cap on the total multiplier. ``None`` = no cap.
    max_multiplier: int | None = None

    # ------------------------------------------------------------------ #
    # Canonical constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def legacy(cls) -> "RuleSet":
        """Reproduce the original DouZero environment exactly.

        No bidding, no spring, ``bomb_num`` doubling. The effective score
        multiplier is ``2 ** bomb_num`` (base_score=2, bomb_multiplier=2),
        matching ``GameEnv.compute_player_utility`` / ``update_num_wins_scores``
        in the legacy code.
        """
        return cls(
            ruleset_id=RULESET_LEGACY,
            bidding_mode=BIDDING_MODE_NONE,
            bid_values=(),
            allow_rob=False,
            all_pass_redeal=False,
            bid_multiplier=False,
            bomb_multiplier=2,
            rocket_multiplier=2,
            spring_multiplier=0,
            anti_spring_multiplier=0,
            allow_double=False,
            base_score=2,
            max_multiplier=None,
        )

    @classmethod
    def standard(cls) -> "RuleSet":
        """Standard DouDizhu rules.

        0/1/2/3 score bidding, all-pass redeal, spring/anti-spring x2,
        bomb x2, rocket x2, base score 1. The winning bid multiplies the base
        score.
        """
        return cls(
            ruleset_id=RULESET_STANDARD,
            bidding_mode=BIDDING_MODE_SCORE_0_1_2_3,
            bid_values=(0, 1, 2, 3),
            allow_rob=False,
            all_pass_redeal=True,
            bid_multiplier=True,
            bomb_multiplier=2,
            rocket_multiplier=2,
            spring_multiplier=2,
            anti_spring_multiplier=2,
            allow_double=False,
            base_score=1,
            max_multiplier=None,
        )

    # ------------------------------------------------------------------ #
    # Construction from dict / YAML
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "RuleSet":
        """Construct a RuleSet from a raw mapping, validating types and ranges.

        ``ruleset_id`` and ``bidding_mode`` are derived from the mapping if
        absent. ``bid_values`` is accepted as a list and converted to a tuple.
        Unknown keys raise ``ValueError``.
        """
        if d is None:
            return cls.legacy()
        if not isinstance(d, Mapping):
            raise TypeError(
                f"RuleSet config must be a mapping, got {type(d).__name__}"
            )

        raw = dict(d)
        valid_names = {f.name for f in fields(cls)}
        unknown = set(raw.keys()) - valid_names
        if unknown:
            raise ValueError(f"Unknown RuleSet config keys: {sorted(unknown)}")

        # Derive ruleset_id and bidding_mode if not explicitly provided.
        rid = raw.get("ruleset_id")
        if rid is None:
            rid = RULESET_LEGACY
        if rid not in (RULESET_LEGACY, RULESET_STANDARD):
            raise ValueError(
                f"RuleSet ruleset_id must be 'legacy' or 'standard', got {rid!r}"
            )

        if "bidding_mode" not in raw:
            raw["bidding_mode"] = (
                BIDDING_MODE_NONE if rid == RULESET_LEGACY
                else BIDDING_MODE_SCORE_0_1_2_3
            )

        # Convert bid_values list -> tuple.
        if "bid_values" in raw and raw["bid_values"] is not None:
            bv = raw["bid_values"]
            if not isinstance(bv, (list, tuple)):
                raise TypeError(
                    f"RuleSet bid_values must be a list/tuple, got "
                    f"{type(bv).__name__}: {bv!r}"
                )
            raw["bid_values"] = tuple(int(v) for v in bv)

        # Apply the canonical defaults for the chosen ruleset, then overlay.
        base = cls.legacy() if rid == RULESET_LEGACY else cls.standard()
        merged: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in raw:
                merged[f.name] = raw[f.name]
            else:
                merged[f.name] = getattr(base, f.name)

        cfg = cls(**merged)
        _validate_ruleset(cfg)
        return cfg

    # ------------------------------------------------------------------ #
    # Serialisation / hashing
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (bid_values as a list)."""
        d: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                val = list(val)
            d[f.name] = val
        return d

    def stable_hash(self) -> str:
        """Deterministic SHA-256 of the canonical JSON serialisation.

        Used to stamp a precise rule identity into checkpoint manifests (P16
        uses the full hash; P02 uses the short ``ruleset_id`` string for
        compatibility checks).
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_ruleset(cfg: RuleSet) -> None:
    """Validate field types and value ranges; raise on any violation."""
    # String fields.
    if cfg.ruleset_id not in (RULESET_LEGACY, RULESET_STANDARD):
        raise ValueError(
            f"ruleset_id must be 'legacy' or 'standard', got {cfg.ruleset_id!r}"
        )
    if cfg.bidding_mode not in (BIDDING_MODE_NONE, BIDDING_MODE_SCORE_0_1_2_3):
        raise ValueError(
            f"bidding_mode must be 'none' or 'score_0_1_2_3', got "
            f"{cfg.bidding_mode!r}"
        )

    # Legacy ruleset must not have bidding. Checked early so subsequent
    # bid_values validation does not produce a misleading error.
    if cfg.ruleset_id == RULESET_LEGACY and cfg.bidding_mode != BIDDING_MODE_NONE:
        raise ValueError(
            f"legacy ruleset must have bidding_mode='none', got "
            f"{cfg.bidding_mode!r}"
        )

    # bid_values: non-negative ints; 0 only allowed for standard bidding.
    for v in cfg.bid_values:
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"bid_values must be ints, got {type(v).__name__}: {v!r}")
        if v < 0:
            raise ValueError(f"bid_values must be non-negative, got {v}")
    if cfg.bidding_mode == BIDDING_MODE_SCORE_0_1_2_3:
        if not cfg.bid_values:
            raise ValueError(
                "score_0_1_2_3 bidding_mode requires non-empty bid_values"
            )

    # Multiplier fields: positive ints (spring/anti-spring may be 0 = disabled).
    for name in ("bomb_multiplier", "rocket_multiplier"):
        val = getattr(cfg, name)
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{name} must be a positive int, got {val!r}")
    for name in ("spring_multiplier", "anti_spring_multiplier"):
        val = getattr(cfg, name)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ValueError(
                f"{name} must be a non-negative int (0 disables), got {val!r}"
            )

    # base_score: positive int.
    if not isinstance(cfg.base_score, int) or isinstance(cfg.base_score, bool) or cfg.base_score < 1:
        raise ValueError(f"base_score must be a positive int, got {cfg.base_score!r}")

    # max_multiplier: None or positive int.
    if cfg.max_multiplier is not None:
        if (not isinstance(cfg.max_multiplier, int)
                or isinstance(cfg.max_multiplier, bool)
                or cfg.max_multiplier < 1):
            raise ValueError(
                f"max_multiplier must be None or a positive int, got "
                f"{cfg.max_multiplier!r}"
            )
