"""Exact constrained dynamic programming for the belief model (P07).

Two operations are required by AGENTS.md "Belief-model rules":

1. **MAP decode** — given the model's per-rank count log-probabilities and the
   public total ``opponent_A_cards_left``, find the single allocation
   ``c[15]`` (with ``c[r] in [0, cap_r]`` and ``sum(c) == total``) that
   maximizes ``sum_r logp[r, c[r]]``.

2. **Posterior sample** — draw an allocation exactly proportional to the
   model's constrained distribution (conditional on ``sum == total``).

Both are implemented by a bounded, exact finite dynamic program over the 15
ranks and the cumulative count ``0..total``. Complexity is
``O(15 * (total+1) * 5)``; ``total <= 20``, so this is a few thousand
operations — there is **no rejection loop** and no possibility of an infinite
hang (the cardinal rule from AGENTS.md: "decoding and sampling cannot rely on
unbounded rejection loops").

Why a DP and not greedy?
    Greedy (pick the argmax count per rank independently) can violate the
    total-sum constraint. The DP enforces it exactly, and because the rank
    log-probabilities are independent given the total, the optimal constrained
    allocation is the DP's backtrack. This is the standard "knapsack on a
    sequence of independent items with a sum constraint" formulation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constraints import NUM_BELIEF_RANKS, NUM_COUNT_SLOTS

#: Negative-infinity stand-in for a masked (impossible) logit slot. ``-1e30``
#: is used rather than ``-np.inf`` so finite-arithmetic summation in the DP
#: stays finite and ``exp()`` underflows to a clean 0.0 instead of producing
#: ``-inf - -inf = nan`` during backward sampling normalization.
_NEG_INF: float = -1e30


class BeliefDPError(ValueError):
    """Raised when the constrained DP has no feasible allocation.

    This indicates an inconsistent observation (the unseen pool and the public
    remaining-card counts do not admit a legal split). The error carries a
    short observation summary so the caller can diagnose the source (e.g. a
    feature bug, not a model bug).
    """


# --------------------------------------------------------------------------- #
# Forward DP table
# --------------------------------------------------------------------------- #
def _forward_table(
    logp: np.ndarray, total: int
) -> np.ndarray:
    """Compute the max-log-prob forward DP table.

    ``table[i, s]`` is the maximum achievable ``sum_{r<i} logp[r, c[r]]`` using
    the first ``i`` ranks with cumulative count exactly ``s``. Shape
    ``(NUM_BELIEF_RANKS + 1, total + 1)``; infeasible cells are ``-inf``.

    The recurrence is::

        table[0, 0] = 0.0 ; table[0, s>0] = -inf
        table[i+1, s] = max_{k in [0, min(cap_r, s)] with legal slot}
                            table[i, s-k] + logp[r, k]
    """
    n_ranks = NUM_BELIEF_RANKS
    # Replace any -inf (masked) entries with the finite sentinel for safe sums.
    lp = np.where(np.isneginf(logp), _NEG_INF, logp).astype(np.float64)
    neg = np.full((n_ranks + 1, total + 1), _NEG_INF, dtype=np.float64)
    neg[0, 0] = 0.0
    for r in range(n_ranks):
        row = lp[r]  # shape (5,)
        prev = neg[r]
        cur = neg[r + 1]
        # For every reachable prior cumulative count, try each legal slot k.
        for s in range(total + 1):
            base = prev[s]
            if base <= _NEG_INF:
                continue
            # k ranges over count slots; skip masked (-inf) ones cheaply.
            for k in range(NUM_COUNT_SLOTS):
                lpk = row[k]
                if lpk <= _NEG_INF:
                    continue
                ns = s + k
                if ns > total:
                    continue
                val = base + lpk
                if val > cur[ns]:
                    cur[ns] = val
    return neg


def _logsumexp(arr: np.ndarray) -> float:
    """Numerically stable logsumexp over a 1-D array (for the sampler filter)."""
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[arr > _NEG_INF]
    if finite.size == 0:
        return _NEG_INF
    m = finite.max()
    return float(m + np.log(np.exp(arr - m).clip(min=0.0).sum()))


def _forward_filter_table(
    logp: np.ndarray, total: int
) -> np.ndarray:
    """Compute the log-partition-function forward table for exact sampling.

    ``filter[i, s] = log sum over allocations of first i ranks summing to s of
    exp(sum logp)``. The sampler draws a full allocation with probability
    exactly proportional to ``exp(sum_r logp[r, c[r]])`` among all
    total-consistent allocations (forward-filter / backward-sample).
    """
    n_ranks = NUM_BELIEF_RANKS
    lp = np.where(np.isneginf(logp), _NEG_INF, logp).astype(np.float64)
    filt = np.full((n_ranks + 1, total + 1), _NEG_INF, dtype=np.float64)
    filt[0, 0] = 0.0
    for r in range(n_ranks):
        row = lp[r]
        prev = filt[r]
        cur = filt[r + 1]
        for s in range(total + 1):
            base = prev[s]
            if base <= _NEG_INF:
                continue
            for k in range(NUM_COUNT_SLOTS):
                lpk = row[k]
                if lpk <= _NEG_INF:
                    continue
                ns = s + k
                if ns > total:
                    continue
                # logsumexp accumulation across alternative (s-k) predecessors.
                combined = base + lpk
                if cur[ns] <= _NEG_INF:
                    cur[ns] = combined
                else:
                    m = max(cur[ns], combined)
                    cur[ns] = m + np.log(
                        np.exp(cur[ns] - m) + np.exp(combined - m)
                    )
    return filt


def _check_feasible(
    forward: np.ndarray, total: int, summary: str | None
) -> None:
    """Raise :class:`BeliefDPError` if no allocation reaches ``total``."""
    if forward[NUM_BELIEF_RANKS, total] <= _NEG_INF:
        raise BeliefDPError(
            "Belief DP found no feasible allocation: no per-rank count vector "
            f"sums to the opponent-A total {total}. This means the public "
            "unseen pool and the opponent remaining-card count are "
            "inconsistent. " + (f"Observation summary: {summary}" if summary else "")
        )


# --------------------------------------------------------------------------- #
# MAP decode
# --------------------------------------------------------------------------- #
def decode_map(
    logp: np.ndarray,
    total: int,
    *,
    summary: str | None = None,
) -> np.ndarray:
    """Return the maximum-a-posteriori per-rank count allocation.

    Parameters
    ----------
    logp:
        ``(15, 5)`` log-probabilities (masked slots may be ``-inf``). Only the
        relative ordering matters; values need not be normalized.
    total:
        The exact total ``sum(c)`` the allocation must satisfy (opponent A's
        public remaining-card count). Must be in ``[0, 20]``.
    summary:
        Optional human-readable observation summary attached to the
        infeasibility error.

    Returns
    -------
    numpy.ndarray
        ``(15,)`` int64 allocation ``c`` with ``sum(c) == total`` and
        ``0 <= c[r] <= cap_r``.

    Raises
    ------
    BeliefDPError
        If no legal allocation sums to ``total``.
    """
    if not isinstance(total, (int, np.integer)) or isinstance(total, bool):
        raise TypeError(f"total must be an int, got {type(total).__name__}")
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    logp_arr = np.asarray(logp, dtype=np.float64)
    if logp_arr.shape != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"logp must have shape ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {logp_arr.shape}"
        )
    if total == 0:
        # Trivial feasible case: all-zero allocation. (The forward table's
        # cell [15, 0] is always >= 0 because choosing k=0 everywhere is legal
        # whenever the all-zero slot is unmasked, which it always is.)
        return np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)

    table = _forward_table(logp_arr, total)
    _check_feasible(table, total, summary)
    # Backtrack: at each rank recover the k that achieved the optimum.
    alloc = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)
    s = total
    for r in range(NUM_BELIEF_RANKS, 0, -1):
        prev = table[r - 1]
        target = table[r, s]
        found = False
        for k in range(NUM_COUNT_SLOTS):
            ns = s - k
            if ns < 0:
                continue
            if prev[ns] <= _NEG_INF:
                continue
            lpk = logp_arr[r - 1, k]
            if np.isneginf(lpk):
                continue
            # Reconstruct with tolerance for float summation drift.
            if abs((prev[ns] + lpk) - target) <= 1e-9 * max(1.0, abs(target)):
                alloc[r - 1] = k
                s = ns
                found = True
                break
        if not found:
            # Numerical drift prevented exact backtrack. Fall back to the
            # argmax-k reconstruction (still total-consistent by construction
            # of the table): pick the k whose predecessor is finite and yields
            # the highest value.
            best_k, best_val = 0, _NEG_INF
            for k in range(NUM_COUNT_SLOTS):
                ns = s - k
                if ns < 0 or prev[ns] <= _NEG_INF or np.isneginf(logp_arr[r - 1, k]):
                    continue
                val = prev[ns] + logp_arr[r - 1, k]
                if val > best_val:
                    best_val, best_k = val, k
            alloc[r - 1] = best_k
            s = s - best_k
    return alloc


# --------------------------------------------------------------------------- #
# Posterior sampling (forward-filter / backward-sample)
# --------------------------------------------------------------------------- #
def sample_allocation(
    logp: np.ndarray,
    total: int,
    *,
    rng: np.random.Generator,
    summary: str | None = None,
) -> np.ndarray:
    """Draw one allocation exactly from the model's constrained distribution.

    Uses the standard forward-filter / backward-sample scheme: the forward
    filter table holds the log-partition function over total-consistent
    partial allocations; walking ranks backwards, each rank's count ``k`` is
    drawn with probability proportional to
    ``exp(logp[r, k]) * exp(filter[r, s-k])``. The resulting full allocation
    has probability exactly proportional to ``exp(sum_r logp[r, c[r]])`` among
    all allocations with ``sum == total`` — no rejection, no truncation bias.

    Parameters
    ----------
    logp:
        ``(15, 5)`` masked log-probabilities (illegal slots ``-inf``).
    total:
        Exact total ``sum(c)`` (opponent A's remaining cards).
    rng:
        A seeded :class:`numpy.random.Generator` for reproducibility.
    summary:
        Optional observation summary for the infeasibility error.

    Returns
    -------
    numpy.ndarray
        ``(15,)`` int64 allocation.

    Raises
    ------
    BeliefDPError
        If no legal allocation sums to ``total``.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    if not isinstance(total, (int, np.integer)) or isinstance(total, bool):
        raise TypeError(f"total must be an int, got {type(total).__name__}")
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    logp_arr = np.asarray(logp, dtype=np.float64)
    if logp_arr.shape != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"logp must have shape ({NUM_BELIEF_RANKS}, {NUM_COUNT_SLOTS}), "
            f"got {logp_arr.shape}"
        )
    if total == 0:
        return np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)

    filt = _forward_filter_table(logp_arr, total)
    _check_feasible(filt, total, summary)

    alloc = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)
    s = total
    for r in range(NUM_BELIEF_RANKS, 0, -1):
        prev = filt[r - 1]
        # Build the unnormalized probability over k for this rank at residual s.
        weights = np.full(NUM_COUNT_SLOTS, _NEG_INF, dtype=np.float64)
        for k in range(NUM_COUNT_SLOTS):
            ns = s - k
            if ns < 0:
                continue
            if prev[ns] <= _NEG_INF:
                continue
            lpk = logp_arr[r - 1, k]
            if np.isneginf(lpk):
                continue
            weights[k] = prev[ns] + lpk
        # Normalize via softmax over the finite weights.
        finite = weights[weights > _NEG_INF]
        if finite.size == 0:
            # Should not happen given feasibility, but fail loudly if it does.
            raise BeliefDPError(
                f"Belief sampler found no legal count at rank {r - 1} with "
                f"residual {s}; the observation is inconsistent."
                + (f" Summary: {summary}" if summary else "")
            )
        m = finite.max()
        probs_k = np.where(weights > _NEG_INF, np.exp(weights - m), 0.0)
        probs_k /= probs_k.sum()
        k = int(rng.choice(NUM_COUNT_SLOTS, p=probs_k))
        alloc[r - 1] = k
        s -= k
    return alloc


def sample_batch(
    logp_batch: np.ndarray,
    totals: Sequence[int],
    *,
    rng: np.random.Generator,
    summary: str | None = None,
) -> np.ndarray:
    """Sample one allocation per batch element.

    Parameters
    ----------
    logp_batch:
        ``(B, 15, 5)`` masked log-probabilities.
    totals:
        Length-``B`` sequence of target totals.
    rng:
        Seeded generator.

    Returns
    -------
    numpy.ndarray
        ``(B, 15)`` int64 allocations.
    """
    arr = np.asarray(logp_batch, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS):
        raise ValueError(
            f"logp_batch must have shape (B, {NUM_BELIEF_RANKS}, "
            f"{NUM_COUNT_SLOTS}), got {arr.shape}"
        )
    if len(totals) != arr.shape[0]:
        raise ValueError(
            f"len(totals)={len(totals)} != batch size {arr.shape[0]}"
        )
    out = np.zeros((arr.shape[0], NUM_BELIEF_RANKS), dtype=np.int64)
    for i in range(arr.shape[0]):
        out[i] = sample_allocation(arr[i], int(totals[i]), rng=rng, summary=summary)
    return out
