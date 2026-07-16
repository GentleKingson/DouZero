"""Nanosecond timing evidence for evaluation agents."""

from __future__ import annotations

from douzero.evaluation import agents
from douzero.evaluation.agents import TimedAgent


class _PredictingAgent:
    def act(self, infoset):
        self.last_p_win = infoset["prediction"]
        return infoset["action"]


class _PlainAgent:
    def act(self, infoset):
        return infoset


def test_timed_agent_records_one_nanosecond_delta_for_both_units(monkeypatch):
    timestamps = iter((10_000_000, 11_234_567))
    monkeypatch.setattr(agents.time, "perf_counter_ns", lambda: next(timestamps))
    timed = TimedAgent(_PredictingAgent(), "candidate", "landlord")

    action = timed.act({"action": [3], "prediction": 0.75})

    assert action == [3]
    assert timed.latencies_ns == [1_234_567]
    assert type(timed.latencies_ns[0]) is int
    assert timed.latencies_ms == [timed.latencies_ns[0] / 1_000_000.0]
    assert timed.predictions == [0.75]
    sample = timed.decision_samples[0]
    assert sample.latency_ns == timed.latencies_ns[0]
    assert sample.prediction == 0.75
    assert sample.search_called is False
    assert sample.search_timed_out is False
    assert sample.search_fallback is False


def test_timed_agent_clamps_non_monotonic_clock_and_keeps_prediction_optional(
    monkeypatch,
):
    timestamps = iter((20, 19))
    monkeypatch.setattr(agents.time, "perf_counter_ns", lambda: next(timestamps))
    timed = TimedAgent(_PlainAgent(), "baseline", "landlord_up")

    assert timed.act("pass") == "pass"
    assert timed.latencies_ns == [0]
    assert type(timed.latencies_ns[0]) is int
    assert timed.latencies_ms == [0.0]
    assert timed.predictions == []
    assert timed.decision_samples[0].prediction is None
