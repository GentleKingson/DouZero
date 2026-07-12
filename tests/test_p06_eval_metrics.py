"""P06 evaluation metrics + DeepAgentV2 decision-policy integration."""

from __future__ import annotations

import math

import pytest
import torch

from douzero.evaluation.metrics import (
    CalibrationAggregator,
    EvaluationReport,
    RoleMetrics,
    summarize,
)
from douzero.training.decision_policy import SUPPORTED_DECISION_MODES


# --------------------------------------------------------------------------- #
# RoleMetrics
# --------------------------------------------------------------------------- #
def test_role_metrics_aggregation():
    m = RoleMetrics()
    m.wins = 4
    m.losses = 6
    m.score_sum = -12.0
    assert m.games == 10
    assert m.win_percentage == pytest.approx(0.4)
    assert m.mean_score == pytest.approx(-1.2)


def test_role_metrics_empty_is_nan():
    m = RoleMetrics()
    assert math.isnan(m.win_percentage)
    assert math.isnan(m.mean_score)


def test_role_metrics_to_dict_keys():
    m = RoleMetrics(wins=1, losses=1, score_sum=2.0)
    d = m.to_dict()
    assert set(d.keys()) == {"wins", "losses", "games", "win_percentage", "mean_score"}


# --------------------------------------------------------------------------- #
# CalibrationAggregator
# --------------------------------------------------------------------------- #
def test_calibration_aggregator_per_role_metrics():
    agg = CalibrationAggregator()
    # Add 4 landlord samples: near-perfect predictor.
    for _ in range(2):
        agg.add("landlord", 0.9, 1.0)
        agg.add("landlord", 0.1, 0.0)
    metrics = agg.metrics("landlord")
    assert metrics["count"] == 4.0
    assert metrics["brier"] < 0.05
    # ECE on 4 samples in 2 bins: gap = 0.1 per bin, weighted mean = 0.1.
    assert metrics["ece"] <= 0.15


def test_calibration_aggregator_empty_role_returns_nan():
    agg = CalibrationAggregator()
    metrics = agg.metrics("landlord_up")
    assert metrics["count"] == 0.0
    assert math.isnan(metrics["brier"])


def test_calibration_aggregator_rejects_unknown_role():
    agg = CalibrationAggregator()
    with pytest.raises(ValueError):
        agg.add("bystander", 0.5, 1.0)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def test_summarize_includes_per_role_lines():
    report = EvaluationReport(
        role_metrics={
            "landlord": RoleMetrics(wins=3, losses=2, score_sum=4.0),
            "landlord_up": RoleMetrics(wins=1, losses=4, score_sum=-6.0),
            "landlord_down": RoleMetrics(wins=1, losses=4, score_sum=-6.0),
        },
        calibration={"landlord": {"count": 5.0, "brier": 0.1, "nll": 0.3, "ece": 0.05}},
    )
    s = summarize(report)
    assert "Per-role results" in s
    assert "landlord" in s
    assert "Calibration" in s
    assert "WP=0.6000" in s


# --------------------------------------------------------------------------- #
# DeepAgentV2 decision_mode integration (P05 contract preserved + P06 modes)
# --------------------------------------------------------------------------- #
def test_deepagent_v2_accepts_all_supported_modes():
    """The DeepAgentV2 constructor accepts every supported decision mode."""
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.env.rules import RuleSet
    from douzero.evaluation.deep_agent import DeepAgentV2

    torch.manual_seed(123)
    schema = build_v2_schema()
    model = ModelV2(schema, ModelV2Config())
    ruleset = RuleSet.legacy()
    for mode in SUPPORTED_DECISION_MODES:
        agent = DeepAgentV2(
            position="landlord",
            model=model,
            ruleset=ruleset,
            decision_mode=mode,
        )
        # Aliases are canonicalized on construction.
        assert agent.decision_mode in SUPPORTED_DECISION_MODES


def test_deepagent_v2_rejects_unknown_mode():
    from douzero.models_v2.config import ModelV2Config
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema
    from douzero.env.rules import RuleSet
    from douzero.evaluation.deep_agent import DeepAgentV2

    torch.manual_seed(123)
    schema = build_v2_schema()
    model = ModelV2(schema, ModelV2Config())
    ruleset = RuleSet.legacy()
    with pytest.raises(ValueError):
        DeepAgentV2(
            position="landlord",
            model=model,
            ruleset=ruleset,
            decision_mode="greedy",
        )
