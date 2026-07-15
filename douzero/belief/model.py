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
    style_enabled: bool = False
    style_embedding_dim: int = 32
    #: Identity version; bump when the compatibility-dict field set changes.
    IDENTITY_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not isinstance(self.style_enabled, bool):
            raise TypeError(
                f"style_enabled must be bool, got {type(self.style_enabled).__name__}"
            )
        if (
            isinstance(self.style_embedding_dim, bool)
            or not isinstance(self.style_embedding_dim, int)
            or self.style_embedding_dim <= 0
        ):
            raise ValueError(
                f"style_embedding_dim must be positive, got {self.style_embedding_dim}"
            )

    def compatibility_dict(self) -> dict:
        compatibility = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "belief_input_dim": BELIEF_INPUT_DIM,
            "num_belief_ranks": NUM_BELIEF_RANKS,
            "num_count_slots": NUM_COUNT_SLOTS,
        }
        if self.style_enabled:
            from douzero.style.features import (
                STYLE_FEATURE_VERSION,
                STYLE_LAYOUT_HASH,
            )

            compatibility.update({
                "style_enabled": True,
                "style_embedding_dim": self.style_embedding_dim,
                "style_feature_version": STYLE_FEATURE_VERSION,
                "style_layout_hash": STYLE_LAYOUT_HASH,
            })
        return compatibility

    def stable_hash(self) -> str:
        """SHA-256 of the compatibility dict (the checkpoint identity axis)."""
        payload = json.dumps(self.compatibility_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BeliefOutput:
    """The forward output of :class:`BeliefModel` for one batch.

    ``logits`` is the raw ``[B, 15, 5]`` tensor; ``legal`` the mask;
    ``factor_probs`` the **independent** per-rank masked softmax (the model's
    per-rank factor distribution, NOT conditioned on the total-count
    constraint); ``constrained_probs`` the per-rank **marginals of the
    constrained posterior** ``P(c_r=k | sum=total)`` — these are what the value
    model should consume, because their expected total equals
    ``opponent_a_total`` exactly. ``opponent_a_total`` is the per-sample target
    total. ``expected_counts`` / ``entropy`` are derived from the constrained
    marginals.
    """

    logits: torch.Tensor  # (B, 15, 5)
    legal: torch.Tensor  # (B, 15, 5) bool
    factor_probs: torch.Tensor  # (B, 15, 5) independent per-rank softmax
    constrained_probs: np.ndarray  # (B, 15, 5) constrained marginals
    opponent_a_total: np.ndarray  # (B,) int64
    opponent_a_role: list[str]
    expected_counts: np.ndarray  # (B, 15) float64 — from constrained marginals
    entropy: np.ndarray  # (B,) float64 — from constrained marginals
    # Populated only by ``forward(..., differentiable=True)``.  The NumPy
    # field above remains the compatibility surface used by exact evaluation,
    # MAP decoding, and sampling.
    constrained_probs_torch: torch.Tensor | None = None

    @property
    def probs(self) -> torch.Tensor:
        """Constrained-marginal probabilities as a torch tensor.

        Returned on CPU; callers that need a different device should recompute
        from :meth:`detach_logits` via
        :func:`~douzero.belief.dynamic_programming.constrained_marginals`.
        Kept as a property named ``probs`` so existing callers that read the
        constrained posterior get it; the independent factor distribution is
        ``factor_probs``.
        """
        return torch.from_numpy(self.constrained_probs.astype(np.float32))

    def detach_logits(self) -> np.ndarray:
        """Return the logits as a detached ``(B, 15, 5)`` numpy array."""
        return self.logits.detach().cpu().float().numpy()

    def require_differentiable_probs(self) -> torch.Tensor:
        """Return graph-bearing constrained marginals or fail explicitly."""
        if self.constrained_probs_torch is None:
            raise RuntimeError(
                "This BeliefOutput was produced by the exact evaluation path. "
                "Call BeliefModel.forward(..., differentiable=True) when a "
                "value loss must update the belief model."
            )
        return self.constrained_probs_torch


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
        self.style_encoder: nn.Module | None = None
        encoder_input_dim = BELIEF_INPUT_DIM
        if self.config.style_enabled:
            from douzero.style.encoder import StyleEncoder

            self.style_encoder = StyleEncoder(
                output_dim=self.config.style_embedding_dim,
                hidden_dim=self.config.style_embedding_dim,
            )
            encoder_input_dim += self.config.style_embedding_dim
        self.encoder = _build_mlp(
            encoder_input_dim,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.dropout,
        )
        self.head = nn.Linear(self.config.hidden_size, NUM_BELIEF_RANKS * NUM_COUNT_SLOTS)
        self._output_dim = NUM_BELIEF_RANKS * NUM_COUNT_SLOTS

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _forward_logits(
        self,
        feature_matrix: torch.Tensor,
        style_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a ``(B, BELIEF_INPUT_DIM)`` matrix to ``(B, 15, 5)`` logits."""
        if feature_matrix.shape[-1] != BELIEF_INPUT_DIM:
            raise ValueError(
                f"feature matrix last dim {feature_matrix.shape[-1]} != "
                f"BELIEF_INPUT_DIM {BELIEF_INPUT_DIM}"
            )
        if self.style_encoder is not None:
            if style_features is None:
                raise ValueError(
                    "style_features are required by a style-enabled BeliefModel"
                )
            style_features = style_features.to(
                device=feature_matrix.device, dtype=feature_matrix.dtype
            )
            encoded_style = self.style_encoder(style_features)
            feature_matrix = torch.cat([feature_matrix, encoded_style], dim=-1)
        elif style_features is not None:
            raise ValueError(
                "style_features were passed to a style-disabled BeliefModel"
            )
        h = self.encoder(feature_matrix)
        flat = self.head(h)
        return flat.view(*feature_matrix.shape[:-1], NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)

    def forward(
        self,
        inputs: list[BeliefInput] | BeliefInput,
        *,
        differentiable: bool = False,
    ) -> BeliefOutput:
        """Run a forward pass for one (or a batch of) :class:`BeliefInput`.

        Returns a :class:`BeliefOutput` carrying the masked logits, the
        independent per-rank factor distribution (``factor_probs``), and the
        **constrained marginals** (``constrained_probs``) — the per-rank
        posterior conditioned on the total-count constraint, whose expected
        total equals ``opponent_a_total`` exactly. The value model should
        consume the constrained marginals.  ``differentiable=False`` keeps the
        established exact NumPy forward-backward evaluation path.  Set it to
        ``True`` during joint training to use the equivalent float32 PyTorch
        recurrence and expose the graph-bearing tensor through
        :meth:`BeliefOutput.require_differentiable_probs`.

        Single-input callers may pass one :class:`BeliefInput`; it is
        internally promoted to a length-1 batch. Device and dtype are derived
        from the model parameters so ``model.cuda()`` / ``model.double()``
        work (the input feature vectors are cast to match).
        """
        from .dynamic_programming import constrained_marginals

        if isinstance(inputs, BeliefInput):
            inputs = [inputs]
        if len(inputs) == 0:
            raise ValueError("BeliefModel.forward received an empty input list")
        # Derive device/dtype from the model parameters so a caller that moved
        # the model to CUDA or cast to double gets matching input tensors
        # (Medium #6 fix).
        param = next(self.parameters())
        feats_np = np.stack(
            [inp.feature_vector for inp in inputs], axis=0
        ).astype(np.float32)
        feature_matrix = torch.as_tensor(
            feats_np, device=param.device, dtype=param.dtype
        )
        style_matrix = None
        if self.config.style_enabled:
            style_matrix = torch.as_tensor(
                np.stack([inp.style_features for inp in inputs], axis=0),
                device=param.device,
                dtype=param.dtype,
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

        logits = self._forward_logits(feature_matrix, style_matrix)
        legal = torch.as_tensor(legal_np, device=logits.device).bool()
        # Masked softmax and constrained partition arithmetic are intentionally
        # float32 even when the encoder runs under autocast.
        masked = logits.float().masked_fill(~legal, _MASK_LOGIT)
        factor_probs = torch.softmax(masked, dim=-1)

        # Constrained per-rank marginals P(c_r=k | sum=total). These are the
        # joint-posterior marginals (not the independent factor softmax); their
        # expected total equals opponent_a_total exactly, which is the property
        # the value-fusion features require (Blocker #3 fix).
        constrained_torch = None
        if differentiable:
            from .torch_dynamic_programming import constrained_marginals_torch

            constrained_torch = constrained_marginals_torch(
                logits,
                torch.as_tensor(totals, device=logits.device, dtype=torch.long),
                legal,
            )
            constrained = constrained_torch.detach().cpu().numpy().astype(np.float64)
        else:
            logp_np = logits.detach().cpu().float().numpy()
            logp_np = np.where(legal_np, logp_np, -np.inf)
            constrained = np.zeros(
                (len(inputs), NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), dtype=np.float64
            )
            for i in range(len(inputs)):
                constrained[i] = constrained_marginals(
                    logp_np[i], int(totals[i]),
                    summary=f"role={roles[i]} total={int(totals[i])}",
                )
        return BeliefOutput(
            logits=logits,
            legal=legal,
            factor_probs=factor_probs,
            constrained_probs=constrained,
            opponent_a_total=totals,
            opponent_a_role=roles,
            expected_counts=expected_counts_from_probs(constrained),
            entropy=total_entropy_from_probs(constrained),
            constrained_probs_torch=constrained_torch,
        )

    def forward_differentiable(
        self, inputs: list[BeliefInput] | BeliefInput
    ) -> BeliefOutput:
        """Convenience wrapper for the graph-preserving training path."""
        return self.forward(inputs, differentiable=True)

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
    *,
    assert_constrained: bool = True,
) -> np.ndarray:
    """Project the CONSTRAINED belief posterior into the value-model features.

    ``probs`` MUST be the constrained per-rank marginals
    (``P(c_r=k | sum=total)``, e.g. from
    :func:`~douzero.belief.dynamic_programming.constrained_marginals` or
    :attr:`BeliefOutput.constrained_probs`) — NOT the independent per-rank
    softmax. Only the constrained marginals have an expected total equal to
    ``opponent_a_total`` exactly, which is the conservation property the value
    fusion requires. When ``assert_constrained`` is True (default) the function
    asserts ``sum_r E[c_A_r] == opponent_a_total`` per sample and raises if the
    caller passed an unconstrained distribution.

    Returns a ``(B, BELIEF_FEATURE_DIM)`` float32 matrix the value model
    consumes when ``belief_enabled=True``:

    - per-rank expected count for opponent A (15),
    - per-rank entropy in nats (15),
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
    # Conservation: the constrained marginal's expected total must equal the
    # target. Reject an unconstrained (factor-softmax) input loudly.
    if assert_constrained:
        totals = np.asarray(opponent_a_total, dtype=np.float64).reshape(-1)
        got = expected_a.sum(axis=-1)
        if not np.allclose(got, totals, atol=1e-4):
            worst = float(np.abs(got - totals).max())
            raise ValueError(
                "belief_features_from_probs received an UNCONSTRAINED "
                "probability tensor: sum_r E[c_A_r] does not match "
                f"opponent_a_total (max abs diff {worst:.4g}). Pass the "
                "constrained marginals (BeliefOutput.constrained_probs), not "
                "the independent per-rank softmax (factor_probs)."
            )
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


def belief_features_from_torch_probs(
    probs: torch.Tensor,
    opponent_a_total: torch.Tensor | np.ndarray,
    unseen_counts: torch.Tensor | np.ndarray,
    *,
    assert_constrained: bool = True,
) -> torch.Tensor:
    """Differentiable PyTorch projection of constrained belief marginals.

    This is the graph-preserving counterpart to
    :func:`belief_features_from_probs`.  It has the same 48-field layout and
    consumes public posterior quantities only.  Arithmetic stays in float32
    so entropy and expected-count features remain finite under AMP.
    """

    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"probs must be a torch.Tensor, got {type(probs).__name__}")
    if probs.ndim != 3 or tuple(probs.shape[-2:]) != (
        NUM_BELIEF_RANKS,
        NUM_COUNT_SLOTS,
    ):
        raise ValueError(
            f"probs must have shape (B, {NUM_BELIEF_RANKS}, "
            f"{NUM_COUNT_SLOTS}), got {tuple(probs.shape)}"
        )

    p = probs.float()
    counts = torch.arange(NUM_COUNT_SLOTS, device=p.device, dtype=torch.float32)
    expected_a = (p * counts).sum(dim=-1)
    totals = torch.as_tensor(
        opponent_a_total, device=p.device, dtype=torch.float32
    ).reshape(-1)
    if totals.numel() != p.shape[0]:
        raise ValueError(
            f"opponent_a_total has {totals.numel()} values for batch "
            f"size {p.shape[0]}"
        )
    if assert_constrained:
        got = expected_a.sum(dim=-1)
        if not torch.allclose(
            got.detach(), totals.detach(), atol=2e-4, rtol=1e-5
        ):
            worst = float((got.detach() - totals.detach()).abs().max().item())
            raise ValueError(
                "belief_features_from_torch_probs received unconstrained "
                "marginals: expected counts do not match opponent_a_total "
                f"(max abs diff {worst:.4g})."
            )

    # clamp_min keeps log(0) out of the backward graph while retaining exact
    # zero probability in the multiplicative term.
    entropy = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
    max_prob = p.max(dim=-1).values
    unseen = torch.as_tensor(
        unseen_counts, device=p.device, dtype=torch.float32
    ).reshape(-1, NUM_BELIEF_RANKS)
    if unseen.shape[0] != p.shape[0]:
        raise ValueError(
            f"unseen_counts batch {unseen.shape[0]} != probs batch {p.shape[0]}"
        )
    expected_b = unseen - expected_a
    total_entropy = entropy.sum(dim=-1, keepdim=True)
    features = torch.cat(
        (
            expected_a,
            entropy,
            max_prob,
            expected_a.sum(dim=-1, keepdim=True),
            expected_b.sum(dim=-1, keepdim=True),
            total_entropy,
        ),
        dim=-1,
    )
    if features.shape[-1] != BELIEF_FEATURE_DIM:
        raise RuntimeError(
            f"belief feature dim mismatch: built {features.shape[-1]} != "
            f"BELIEF_FEATURE_DIM {BELIEF_FEATURE_DIM}"
        )
    if not bool(torch.isfinite(features.detach()).all().item()):
        raise FloatingPointError("differentiable belief features contain NaN/Inf")
    return features
