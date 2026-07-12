"""P06 calibration metrics (Brier, NLL, ECE, reliability bins)."""

from __future__ import annotations

import math

import pytest
import torch

from douzero.training.calibration import (
    brier_score,
    expected_calibration_error,
    nll,
    reliability_bins,
)


def test_brier_perfect_predictor_is_zero():
    p = torch.tensor([0.0, 1.0, 0.0, 1.0])
    t = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert brier_score(p, t) == pytest.approx(0.0, abs=1e-6)


def test_brier_constant_half_is_quarter():
    p = torch.tensor([0.5, 0.5, 0.5, 0.5])
    t = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert brier_score(p, t) == pytest.approx(0.25, abs=1e-6)


def test_nll_perfect_predictor_is_near_zero():
    p = torch.tensor([1e-6, 1 - 1e-6])  # clamped from 0/1
    t = torch.tensor([0.0, 1.0])
    assert nll(p, t) < 1e-4


def test_nll_confident_wrong_is_large():
    p = torch.tensor([1 - 1e-6])
    t = torch.tensor([0.0])
    assert nll(p, t) > 10.0


def test_nll_finite_at_extremes():
    p = torch.tensor([1e-7, 1 - 1e-7])
    t = torch.tensor([1.0, 0.0])
    # Perfect-with-extreme-predictions should still be finite.
    assert math.isfinite(nll(p, t))


def test_ece_single_bin_is_mean_gap():
    p = torch.tensor([0.3, 0.7])
    t = torch.tensor([0.0, 1.0])
    # mean(p) = 0.5, mean(t) = 0.5 -> ECE = 0
    assert expected_calibration_error(p, t, n_bins=1) == pytest.approx(0.0, abs=1e-6)


def test_ece_perfect_predictor_is_zero():
    """A perfectly calibrated predictor: confidence == accuracy per bin.

    With many samples at p=0.5 and half winning, the (0.4, 0.6) bin has
    accuracy 0.5 == confidence 0.5, so ECE is 0.
    """
    p = torch.tensor([0.5] * 20)
    t = torch.tensor([1.0] * 10 + [0.0] * 10)
    assert expected_calibration_error(p, t, n_bins=10) == pytest.approx(0.0, abs=1e-6)


def test_ece_overconfident_predictor_is_positive():
    # Confident but wrong on a quarter of samples.
    p = torch.tensor([0.9, 0.9, 0.9, 0.9])
    t = torch.tensor([1.0, 1.0, 1.0, 0.0])
    ece = expected_calibration_error(p, t, n_bins=5)
    assert ece > 0.0


def test_reliability_bins_count_sums_to_total():
    p = torch.tensor([0.1, 0.3, 0.55, 0.8, 0.95])
    t = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    bins = reliability_bins(p, t, n_bins=5)
    assert sum(b.count for b in bins) == 5
    # Bin edges are 0..1 in n_bins+1 steps.
    assert bins[0].low == 0.0
    assert bins[-1].high == 1.0


def test_reliability_bins_empty_bin_has_zero_count():
    p = torch.tensor([0.5, 0.55])
    t = torch.tensor([1.0, 0.0])
    bins = reliability_bins(p, t, n_bins=10)
    # The first 5 bins (0..0.5) should be empty.
    for b in bins[:5]:
        assert b.count == 0


def test_calibration_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        brier_score(torch.tensor([0.1, 0.2]), torch.tensor([1.0]))


def test_calibration_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        brier_score(torch.tensor([]), torch.tensor([]))
