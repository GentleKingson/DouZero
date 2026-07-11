"""Tests for the P02 round-4 review fixes.

Covers the three blockers from the round-4 review:

1. **Worker permanent-hang protection**: a worker stuck in an infinite loop
   must NOT block the parent forever. The parent must terminate it within a
   short ``worker_timeout`` and raise a ``RuntimeError``. The test uses a
   genuinely hanging worker (``time.sleep`` in a loop) and a short timeout
   (a few seconds), not the 300s default — otherwise the test would not
   distinguish "parent terminated the hung worker" from "parent waited
   300s".

2. **Custom ``--ruleset_config`` override contract**: a partial YAML such as
   ``{bomb_multiplier: 4}`` (no ``ruleset_id``) must be treated as a
   "standard + bomb×4" overlay, not a legacy RuleSet. A config that declares
   a ``ruleset_id`` contradicting the CLI mode must be rejected at the CLI
   boundary. A ``standard`` ID must be bound to ``score_0_1_2_3`` bidding.

3. **Unknown ``format_version`` and strict ``bidding_order`` type**: a
   ``format_version`` of 3 (or any non-2 value) must be rejected as an
   unsupported format, NOT silently treated as legacy. A ``bidding_order``
   given as a string ("012") or tuple must be rejected (only lists are
   accepted).
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import pytest


# =========================================================================== #
# Blocker 1: Worker permanent-hang protection
# =========================================================================== #
def test_evaluate_terminates_hung_worker_and_raises(tmp_path):
    """A worker that hangs forever must be terminated by the parent within
    ``worker_timeout`` seconds, and ``evaluate()`` must raise ``RuntimeError``.

    This test proves the parent does NOT block on an unbounded ``join()``. We
    point the legacy worker target at a module-level hanging function
    (``_test_hang_forever``, picklable by the spawn context), then call
    ``evaluate()`` with a short ``worker_timeout`` (3 seconds). If the bounded
    join works, the test finishes in ~3s; if it regresses to an unbounded
    join, the test would hang indefinitely (and be killed by the runner).
    """
    from unittest.mock import patch

    from douzero.evaluation import simulation

    # Prepare minimal legacy eval data (one game).
    np.random.seed(42)
    from generate_eval_data import generate
    data = [generate() for _ in range(1)]
    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)

    # Patch mp_simulate to the module-level hanging function so the spawned
    # worker loops forever and never puts to q. _test_hang_forever exists in
    # both parent and child (spawn re-imports the module), so it pickles fine.
    start = time.monotonic()
    with patch.object(simulation, "mp_simulate", simulation._test_hang_forever):
        with pytest.raises(RuntimeError, match="exceeded the"):
            simulation.evaluate(
                "random", "random", "random",
                str(pkl), num_workers=1,
                ruleset=None,
                worker_timeout=3,
            )
    elapsed = time.monotonic() - start
    # Must finish well under the test runner's own timeout. 3s budget + grace
    # for terminate/join → assert < 30s.
    assert elapsed < 30, f"evaluate() took {elapsed:.1f}s — bounded join regressed"


def test_evaluate_worker_timeout_default_is_large():
    """The default ``worker_timeout`` must be a sensible large value (3600s)
    so normal evaluations are never falsely terminated."""
    import inspect
    from douzero.evaluation.simulation import evaluate
    sig = inspect.signature(evaluate)
    assert sig.parameters["worker_timeout"].default == 3600


# =========================================================================== #
# Blocker 2: Custom ruleset_config override contract
# =========================================================================== #
def test_partial_config_overlay_uses_standard_base(tmp_path):
    """A partial YAML with no ruleset_id must merge over RuleSet.standard().

    ``{bomb_multiplier: 4}`` should produce "standard + bomb×4", NOT a legacy
    RuleSet. The resulting ruleset_id must be 'standard'.
    """
    import yaml
    from generate_eval_data import _load_ruleset_from_config
    from douzero.env.rules import RuleSet

    cfg = tmp_path / "partial.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": {"bomb_multiplier": 4}}, f)

    rs = _load_ruleset_from_config(str(cfg), expected_id="standard")
    assert rs.ruleset_id == "standard"
    assert rs.bomb_multiplier == 4
    # All other standard defaults preserved.
    assert rs.spring_multiplier == RuleSet.standard().spring_multiplier
    assert rs.bid_values == (0, 1, 2, 3)
    assert rs.bidding_mode == "score_0_1_2_3"


def test_partial_config_overlay_hash_differs_from_canonical(tmp_path):
    """The hash of a partial-overlay ruleset must differ from canonical standard."""
    import yaml
    from generate_eval_data import _load_ruleset_from_config
    from douzero.env.rules import RuleSet

    cfg = tmp_path / "partial.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": {"bomb_multiplier": 4}}, f)

    rs = _load_ruleset_from_config(str(cfg), expected_id="standard")
    assert rs.stable_hash() != RuleSet.standard().stable_hash()


def test_config_explicit_legacy_id_rejected_for_standard_cli(tmp_path):
    """--ruleset standard --ruleset_config (legacy YAML) must fail at the loader."""
    import yaml
    from generate_eval_data import _load_ruleset_from_config

    cfg = tmp_path / "legacy_rules.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": {"ruleset_id": "legacy"}}, f)

    with pytest.raises(ValueError, match="ruleset_id"):
        _load_ruleset_from_config(str(cfg), expected_id="standard")


def test_config_standard_id_with_none_bidding_rejected(tmp_path):
    """A standard ID with bidding_mode='none' must be rejected — standard is
    bound to score_0_1_2_3 bidding."""
    import yaml
    from generate_eval_data import _load_ruleset_from_config

    cfg = tmp_path / "bad_bidding.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": {"ruleset_id": "standard", "bidding_mode": "none"}}, f)

    with pytest.raises(ValueError, match="score_0_1_2_3"):
        _load_ruleset_from_config(str(cfg), expected_id="standard")


def test_ruleset_standard_id_must_bind_to_score_bidding():
    """Direct from_dict: standard ID + bidding_mode=none must raise."""
    from douzero.env.rules import RuleSet
    with pytest.raises(ValueError, match="score_0_1_2_3"):
        RuleSet.from_dict({"ruleset_id": "standard", "bidding_mode": "none"})


def test_config_default_expected_id_is_standard(tmp_path):
    """When expected_id is not passed, it defaults to 'standard'."""
    import yaml
    from generate_eval_data import _load_ruleset_from_config

    cfg = tmp_path / "default.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": {"bomb_multiplier": 3}}, f)

    rs = _load_ruleset_from_config(str(cfg))
    assert rs.ruleset_id == "standard"


def test_config_legacy_mode_passes_when_expected_legacy(tmp_path):
    """--ruleset legacy --ruleset_config (legacy overlay) must succeed."""
    import yaml
    from generate_eval_data import _load_ruleset_from_config

    cfg = tmp_path / "legacy_overlay.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": {"ruleset_id": "legacy"}}, f)

    rs = _load_ruleset_from_config(str(cfg), expected_id="legacy")
    assert rs.ruleset_id == "legacy"


def test_config_non_mapping_rejected(tmp_path):
    """A config file whose 'rules:' is a list must be rejected."""
    import yaml
    from generate_eval_data import _load_ruleset_from_config

    cfg = tmp_path / "bad.yaml"
    with open(cfg, "w") as f:
        yaml.dump({"rules": [1, 2, 3]}, f)

    with pytest.raises(ValueError, match="mapping"):
        _load_ruleset_from_config(str(cfg), expected_id="standard")


def test_cli_standard_with_legacy_config_fails_at_boundary(tmp_path):
    """End-to-end: evaluate.py --ruleset standard --ruleset_config <legacy.yaml>
    must fail immediately (not enter the standard path)."""
    import subprocess
    import sys
    import yaml as _yaml

    # A legacy-ruleset config.
    legacy_cfg = tmp_path / "legacy_rules.yaml"
    with open(legacy_cfg, "w") as f:
        _yaml.dump({"rules": {"ruleset_id": "legacy"}}, f)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--landlord", "random", "--landlord_up", "random",
         "--landlord_down", "random",
         "--eval_data", str(tmp_path / "nonexistent.pkl"),
         "--num_workers", "1",
         "--ruleset", "standard", "--ruleset_config", str(legacy_cfg)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "ruleset_id" in result.stderr or "ruleset_id" in result.stdout


# =========================================================================== #
# Blocker 3: Unknown format_version + strict bidding_order type
# =========================================================================== #
def test_unknown_format_version_3_rejected(tmp_path):
    """format_version=3 must be rejected as unsupported, NOT treated as legacy."""
    np.random.seed(42)
    data = [{"format_version": 3, "deck": list(range(54))}]
    pkl = tmp_path / "v3.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(ValueError, match="unsupported"):
        load_eval_data(str(pkl), ruleset="legacy")


def test_unknown_format_version_3_rejected_for_standard_too(tmp_path):
    """format_version=3 must be rejected even when ruleset='standard'."""
    np.random.seed(42)
    data = [{"format_version": 3}]
    pkl = tmp_path / "v3.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(ValueError, match="unsupported"):
        load_eval_data(str(pkl), ruleset="standard")


def test_explicit_format_version_none_rejected(tmp_path):
    """format_version=None (explicit) must be rejected as unknown."""
    np.random.seed(42)
    data = [{"format_version": None}]
    pkl = tmp_path / "none.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(ValueError, match="unsupported"):
        load_eval_data(str(pkl), ruleset="legacy")


def test_format_version_string_rejected(tmp_path):
    """format_version='2' (string) must be rejected — only int 2 is valid v2."""
    np.random.seed(42)
    data = [{"format_version": "2"}]
    pkl = tmp_path / "str.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(ValueError, match="unsupported"):
        load_eval_data(str(pkl), ruleset="legacy")


def test_mixed_v1_and_unknown_rejected_as_unknown(tmp_path):
    """A dataset mixing v1 and unknown-version records must be rejected.
    The unknown-version check fires before the mixed check."""
    np.random.seed(42)
    from generate_eval_data import generate
    data = [generate(), {"format_version": 3}]
    pkl = tmp_path / "mixed_unknown.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(ValueError, match="unsupported"):
        load_eval_data(str(pkl), ruleset="legacy")


def test_bidding_order_string_rejected(tmp_path):
    """bidding_order='012' (string) must be rejected — only lists are valid."""
    np.random.seed(42)
    from generate_eval_data import generate_standard
    data = [generate_standard()]
    data[0]['bidding_order'] = "012"  # string, not list
    pkl = tmp_path / "str_order.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(TypeError, match="bidding_order"):
        load_eval_data(str(pkl), ruleset="standard")


def test_bidding_order_tuple_rejected(tmp_path):
    """bidding_order=('0','1','2') (tuple) must be rejected — only lists."""
    np.random.seed(42)
    from generate_eval_data import generate_standard
    data = [generate_standard()]
    data[0]['bidding_order'] = ("0", "1", "2")  # tuple, not list
    pkl = tmp_path / "tuple_order.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    with pytest.raises(TypeError, match="bidding_order"):
        load_eval_data(str(pkl), ruleset="standard")


def test_bidding_order_valid_list_still_accepted(tmp_path):
    """A valid list bidding_order must still pass (regression guard)."""
    np.random.seed(42)
    from generate_eval_data import generate_standard
    data = [generate_standard()]
    data[0]['bidding_order'] = ["0", "1", "2"]  # valid list
    pkl = tmp_path / "valid.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    loaded = load_eval_data(str(pkl), ruleset="standard")
    assert len(loaded) == 1


# =========================================================================== #
# Regression: existing behavior still works
# =========================================================================== #
def test_normal_legacy_data_still_loads(tmp_path):
    """Legacy data (no format_version) must still load after the classifier change."""
    np.random.seed(42)
    from generate_eval_data import generate
    data = [generate() for _ in range(3)]
    pkl = tmp_path / "legacy.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    loaded = load_eval_data(str(pkl), ruleset="legacy")
    assert len(loaded) == 3


def test_normal_standard_data_still_loads(tmp_path):
    """Standard v2 data must still load after the classifier change."""
    np.random.seed(42)
    from generate_eval_data import generate_standard
    data = [generate_standard() for _ in range(3)]
    pkl = tmp_path / "standard.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    from douzero.evaluation.legacy_data_adapter import load_eval_data
    loaded = load_eval_data(str(pkl), ruleset="standard")
    assert len(loaded) == 3
