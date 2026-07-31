from douzero.dmc.arguments import parser
from douzero.dmc.dmc import (
    _legacy_checkpoint_due,
    _legacy_wall_limit_reached,
)


def test_legacy_formal_runtime_controls_are_default_off():
    args = parser.parse_args([])
    assert args.checkpoint_every_updates == 0
    assert args.max_wall_time_minutes == 0.0


def test_legacy_update_checkpoint_cadence_is_exact():
    common = {
        "last_checkpoint_update": 10,
        "checkpoint_every_updates": 5,
        "now": 1.0,
        "last_checkpoint_time": 0.0,
        "save_interval_minutes": 30,
    }
    assert not _legacy_checkpoint_due(learner_updates=14, **common)
    assert _legacy_checkpoint_due(learner_updates=15, **common)


def test_legacy_wall_limit_is_bounded_and_default_off():
    assert not _legacy_wall_limit_reached(
        now=10_000.0,
        training_started=0.0,
        max_wall_time_minutes=0.0,
    )
    assert not _legacy_wall_limit_reached(
        now=59.9,
        training_started=0.0,
        max_wall_time_minutes=1.0,
    )
    assert _legacy_wall_limit_reached(
        now=60.0,
        training_started=0.0,
        max_wall_time_minutes=1.0,
    )
