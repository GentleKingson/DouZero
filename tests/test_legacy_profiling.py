import torch

from douzero.dmc.profiling import legacy_profile_range


def test_legacy_profile_range_is_inert_when_disabled():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with legacy_profile_range(False, "actor.inference"):
            torch.ones(1).add_(1)
    assert "douzero.legacy.actor.inference" not in {
        event.key for event in prof.key_averages()
    }


def test_legacy_profile_range_emits_stable_name_when_enabled():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with legacy_profile_range(True, "learner.h2d"):
            torch.ones(1).add_(1)
    assert "douzero.legacy.learner.h2d" in {
        event.key for event in prof.key_averages()
    }
