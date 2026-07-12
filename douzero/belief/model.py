"""The :class:`BeliefModel`: public-only joint hidden-hand posterior (P07).

The model encodes the public belief input vector (see :mod:`features`) into a
``[B, 15, 5]`` logit tensor — one log-probability distribution per rank over
count 0..4 for the canonical opponent A. A legal mask (rank cap × unseen-pool
cap × joker cap) zeroes impossible slots before any softmax/DP, so every
decoded or sampled allocation is card-conservative at the per-rank level; the
dynamic program (:mod:`dynamic_programming`) enforces the exact total.

Deployment safety
-----------------
The model accepts ONLY :class:`~douzero.belief.features.BeliefInput` (built
from a :class:`~douzero.observation.public.PublicObservation`). It never
imports :mod:`douzero.observation.privileged` and its ``forward`` signature
carries no hidden-hand argument. The leakage test asserts two states with
identical public info but different hidden allocations produce identical
logits.

The belief *predictions* (posterior counts, entropy, key-card probabilities)
may be fed to the public value model (:class:`~douzero.models_v2.model.ModelV2`
when ``belief_enabled=True``) — see :func:`belief_features_from_probs`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from .constraints import (
    NUM_BELIEF_RANKS,
    NUM_COUNT_SLOTS,
    expected_counts_from_probs,
    legal_mask,
    total_entropy_from_probs,
)
from .dynamic_programming import decode_map, sample_allocation
from .features import BELIEF_INPUT_DIM, BeliefInput

#: Finite sentinel for masked logits (mirrors :mod:`douzero.belief.losses`).
#: A finite value — not ``-inf`` — so masked softmax yields exact zeros for
#: illegal slots and the DP's arithmetic stays finite.
_MASK_LOGIT: float = -1e30

#: Dimensionality of the belief-feature vector consumed by the value model
#: (Phase D fusion). It is a FIXED named constant — not a config knob — so the
#: value model's ``belief_enabled`` flag is the sole architecture delta and
#: existing checkpoints remain loadable when belief is off:
#:   per-rank expected count (15)
#: + per-rank entropy normalized (15)
#: + per-rank max probability (15)
#: + opponent-A expected total (1)
#: + opponent-B expected total (1)
#: + total entropy (1)
BELIEF_FEATURE_DIM: int = NUM_BELIEF_RANKS * 3 + 3  # 48


@dataclass(frozen=True)
class BeliefConfig:
    """Architecture configuration for :class:`BeliefModel`.

    Kept separate from :class:`~douzero.models_v2.config.ModelV2Config` so the
    belief model can be pretrained and frozen independently of the value
    model. Defaults are small for a CPU forward/backward smoke test.
    """

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    #: Identity version; bump when the compatibility-dict field set changes.
    IDENTITY_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def compatibility_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "belief_input_dim": BELIEF_INPUT_DIM,
            "num_belief_ranks": NUM_BELIEF_RANKS,
            "num_count_slots": NUM_COUNT_SLOTS,
        }

    def stable_hash(self) -> str:
        """SHA-256 of the compatibility dict (the checkpoint identity axis)."""
        payload = json.dumps(self.compatibility_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BeliefOutput:
    """The forward output of :class:`BeliefModel` for one batch.

    ``logits`` is the raw ``[B, 15, 5]`` tensor; ``legal`` the mask;
    ``probs`` the masked softmax; ``opponent_a_total`` the per-sample target
    total. ``expected_counts`` and ``entropy`` are posterior summaries (numpy)
    for downstream consumers.
    """

    logits: torch.Tensor  # (B, 15, 5)
    legal: torch.Tensor  # (B, 15, 5) bool
    probs: torch.Tensor  # (B, 15, 5) masked softmax
    opponent_a_total: np.ndarray  # (B,) int64
    opponent_a_role: list[str]
    expected_counts: np.ndarray  # (B, 15) float64
    entropy: np.ndarray  # (B,) float64

    def detach_logits(self) -> np.ndarray:
        """Return the logits as a detached ``(B, 15, 5)`` numpy array."""
        return self.logits.detach().cpu().float().numpy()


def _build_mlp(
    in_dim: int, hidden_size: int, num_layers: int, dropout: float
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(last, hidden_size))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last = hidden_size
    return nn.Sequential(*layers)


class BeliefModel(nn.Module):
    """Joint hidden-hand belief model over the canonical opponent A.

    Parameters
    ----------
    config:
        :class:`BeliefConfig` architecture knobs.

    The model is a small MLP from ``BELIEF_INPUT_DIM`` to a flat
    ``15 * 5`` logit vector, reshaped to ``[B, 15, 5]``. The legal mask is
    applied outside the parameters (it depends on the per-decision unseen
    pool), so the same trained weights work for any public observation.
    """

    def __init__(self, config: BeliefConfig | None = None) -> None:
        super().__init__()
        self.config = config or BeliefConfig()
        self.encoder = _build_mlp(
            BELIEF_INPUT_DIM,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.dropout,
        )
        self.head = nn.Linear(self.config.hidden_size, NUM_BELIEF_RANKS * NUM_COUNT_SLOTS)
        self._output_dim = NUM_BELIEF_RANKS * NUM_COUNT_SLOTS

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _forward_logits(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        """Encode a ``(B, BELIEF_INPUT_DIM)`` matrix to ``(B, 15, 5)`` logits."""
        if feature_matrix.shape[-1] != BELIEF_INPUT_DIM:
            raise ValueError(
                f"feature matrix last dim {feature_matrix.shape[-1]} != "
                f"BELIEF_INPUT_DIM {BELIEF_INPUT_DIM}"
            )
        h = self.encoder(feature_matrix)
        flat = self.head(h)
        return flat.view(*feature_matrix.shape[:-1], NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)

    def forward(self, inputs: list[BeliefInput] | BeliefInput) -> BeliefOutput:
        """Run a forward pass for one (or a batch of) :class:`BeliefInput`.

        Returns a :class:`BeliefOutput` carrying the masked logits/probs and
        the per-sample constraint totals. Single-input callers may pass one
        :class:`BeliefInput`; it is internally promoted to a length-1 batch.
        """
        if isinstance(inputs, BeliefInput):
            inputs = [inputs]
        if len(inputs) == 0:
            raise ValueError("BeliefModel.forward received an empty input list")
        feature_matrix = torch.from_numpy(
            np.stack([inp.feature_vector for inp in inputs], axis=0).astype(np.float32)
        )
        unseen = np.stack(
            [inp.unseen_counts for inp in inputs], axis=0
        ).astype(np.int64)  # (B, 15)
        legal_np = np.stack(
            [legal_mask(inp.unseen_counts) for inp in inputs], axis=0
        )  # (B, 15, 5)
        totals = np.array(
            [inp.opponent_a_total for inp in inputs], dtype=np.int64
        )  # (B,)
        roles = [inp.opponent_a_role for inp in inputs]

        logits = self._forward_logits(feature_matrix)
        legal = torch.from_numpy(legal_np).to(logits.device).bool()
        masked = logits.masked_fill(~legal, _MASK_LOGIT)
        probs = torch.softmax(masked, dim=-1)
        probs_np = probs.detach().cpu().float().numpy()
        return BeliefOutput(
            logits=logits,
            legal=legal,
            probs=probs,
            opponent_a_total=totals,
            opponent_a_role=roles,
            expected_counts=expected_counts_from_probs(probs_np),
            entropy=total_entropy_from_probs(probs_np),
        )

    # ------------------------------------------------------------------ #
    # Constrained decoding / sampling
    # ------------------------------------------------------------------ #
    def decode_map(self, output: BeliefOutput) -> np.ndarray:
        """Return the ``(B, 15)`` MAP allocation for a :class:`BeliefOutput`.

        Each row satisfies ``sum == opponent_a_total`` and the per-rank caps.
        Raises :class:`~douzero.belief.dynamic_programming.BeliefDPError` if an
        observation is inconsistent (no legal allocation).
        """
        logp = output.detach_logits()
        # Convert masked (illegal) logits to -inf so the DP skips them. The
        # forward already applied masked_fill for probs; redo on raw logits.
        legal_np = output.legal.detach().cpu().numpy()
        logp = np.where(legal_np, logp, -np.inf)
        out = np.zeros((len(output.opponent_a_total), NUM_BELIEF_RANKS), dtype=np.int64)
        for i in range(len(output.opponent_a_total)):
            out[i] = decode_map(
                logp[i], int(output.opponent_a_total[i]),
                summary=f"role={output.opponent_a_role[i]} total={output.opponent_a_total[i]}",
            )
        return out

    def sample(
        self,
        output: BeliefOutput,
        rng: np.random.Generator,
        num_samples: int = 1,
    ) -> np.ndarray:
        """Draw ``num_samples`` allocations per batch element.

        Returns ``(B, num_samples, 15)`` int64. Each allocation satisfies the
        total + per-rank constraints exactly (forward-filter / backward-sample,
        no rejection loop).
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        logp = output.detach_logits()
        legal_np = output.legal.detach().cpu().numpy()
        logp = np.where(legal_np, logp, -np.inf)
        b = len(output.opponent_a_total)
        out = np.zeros((b, num_samples, NUM_BELIEF_RANKS), dtype=np.int64)
        for i in range(b):
            for j in range(num_samples):
                out[i, j] = sample_allocation(
                    logp[i], int(output.opponent_a_total[i]), rng=rng,
                    summary=f"role={output.opponent_a_role[i]} total={output.opponent_a_total[i]}",
                )
        return out


def belief_features_from_probs(
    probs: np.ndarray,
    opponent_a_total: np.ndarray,
    unseen_counts: np.ndarray,
) -> np.ndarray:
    """Project a belief posterior into the value-model feature vector.

    Returns a ``(B, BELIEF_FEATURE_DIM)`` float32 matrix the value model
    consumes when ``belief_enabled=True``:

    - per-rank expected count for opponent A (15),
    - per-rank normalized entropy (15),
    - per-rank max probability (15),
    - opponent-A expected total (1),
    - opponent-B expected total (1),
    - total entropy (1).

    ``unseen_counts`` is the ``(B, 15)`` public per-rank unknown-pool counts;
    opponent B's expected count per rank is ``unseen - expected_A`` (the public
    subtraction that makes the feature conservation-safe). All inputs are
    public posterior quantities — no hidden hand is read.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.shape[-2:] != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"probs must end with ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {p.shape}"
        )
    flat = p.reshape(-1, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
    counts = np.arange(NUM_COUNT_SLOTS, dtype=np.float64)
    expected_a = (flat * counts).sum(axis=-1)  # (B, 15)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(flat > 0, flat * np.log(flat), 0.0).sum(axis=-1)  # (B,15)
    max_prob = flat.max(axis=-1)  # (B, 15)
    unseen = np.asarray(unseen_counts, dtype=np.float64).reshape(-1, NUM_BELIEF_RANKS)
    expected_b = unseen - expected_a
    total_entropy = entropy.sum(axis=-1, keepdims=True)  # (B, 1)
    feat = np.concatenate(
        [
            expected_a,
            entropy,
            max_prob,
            expected_a.sum(axis=-1, keepdims=True),
            expected_b.sum(axis=-1, keepdims=True),
            total_entropy,
        ],
        axis=-1,
    ).astype(np.float32)
    if feat.shape[-1] != BELIEF_FEATURE_DIM:
        raise RuntimeError(
            f"belief feature dim mismatch: built {feat.shape[-1]} != "
            f"BELIEF_FEATURE_DIM {BELIEF_FEATURE_DIM}"
        )
    return feat
