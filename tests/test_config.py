"""Tests for the P01 typed configuration system.

These pin the Slice 2 acceptance gates:
  - configs/legacy.yaml is field-for-field equal to the argparse defaults;
  - from_argparse(parse_args([])) equals the frozen LegacyConfig;
  - CLI flags override YAML values;
  - serialize round-trips;
  - the legacy default path leaves P00 baseline invariants unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from douzero.config import (
    LegacyConfig,
    from_argparse,
    load_config,
    merge,
    serialize,
    to_argparse_namespace,
)
from douzero.config.schemas import OptimizerConfig, TrainingConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_YAML = REPO_ROOT / "configs" / "legacy.yaml"


# --------------------------------------------------------------------------- #
# legacy.yaml parity with frozen defaults
# --------------------------------------------------------------------------- #
def test_legacy_yaml_matches_frozen_defaults():
    """configs/legacy.yaml must equal the frozen LegacyConfig training defaults."""
    cfg = load_config(str(LEGACY_YAML))
    assert cfg == LegacyConfig.training


def test_load_legacy_config_from_bundled_resource():
    """load_legacy_config() reads the wheel-bundled legacy.yaml and equals LegacyConfig."""
    from douzero.config import load_legacy_config

    cfg = load_legacy_config()
    assert cfg == LegacyConfig.training


def test_legacy_yaml_optimizer_matches():
    cfg = load_config(str(LEGACY_YAML))
    assert cfg.optimizer == OptimizerConfig(
        learning_rate=0.0001, alpha=0.99, momentum=0, epsilon=1e-5
    )


# --------------------------------------------------------------------------- #
# argparse <-> config round-trip
# --------------------------------------------------------------------------- #
def test_from_argparse_defaults_equal_legacy():
    """The default argparse Namespace must map to the frozen legacy config."""
    from douzero.dmc.arguments import parser

    ns = parser.parse_args([])
    cfg = from_argparse(ns)
    assert cfg == LegacyConfig.training


def test_to_argparse_namespace_round_trip():
    """Config -> Namespace -> Config must be identity (for the legacy fields)."""
    from douzero.dmc.arguments import parser

    ns = parser.parse_args([])
    cfg = from_argparse(ns)
    ns2 = to_argparse_namespace(cfg)
    cfg2 = from_argparse(ns2)
    assert cfg == cfg2


def test_namespace_has_all_legacy_dests_train_reads():
    """to_argparse_namespace must expose every dest that train(flags) reads."""
    from douzero.dmc.arguments import parser

    expected_dests = {
        "xpid", "save_interval", "objective", "actor_device_cpu", "gpu_devices",
        "num_actor_devices", "num_actors", "training_device", "load_model",
        "disable_checkpoint", "savedir", "total_frames", "exp_epsilon",
        "batch_size", "unroll_length", "num_buffers", "num_threads",
        "max_grad_norm", "learning_rate", "alpha", "momentum", "epsilon",
    }
    ns = to_argparse_namespace(LegacyConfig.training)
    missing = expected_dests - set(vars(ns).keys())
    assert not missing, f"to_argparse_namespace missing dests: {missing}"


# --------------------------------------------------------------------------- #
# CLI overrides YAML
# --------------------------------------------------------------------------- #
def test_cli_overrides_yaml():
    """A CLI value must override the YAML base."""
    base = load_config(str(LEGACY_YAML))
    assert base.batch_size == 32  # yaml default
    override_ns = argparse.Namespace(batch_size=16)
    merged = merge(base, override_ns)
    assert merged.batch_size == 16
    # Untouched fields remain the yaml/legacy values.
    assert merged.objective == "adp"


def test_cli_overrides_optimizer():
    base = load_config(str(LEGACY_YAML))
    override_ns = argparse.Namespace(learning_rate=0.05)
    merged = merge(base, override_ns)
    assert merged.optimizer.learning_rate == 0.05
    assert merged.optimizer.alpha == 0.99  # unchanged


# --------------------------------------------------------------------------- #
# serialize round-trip
# --------------------------------------------------------------------------- #
def test_serialize_round_trip():
    import copy

    cfg = load_config(str(LEGACY_YAML))
    d = serialize(cfg)
    # Rebuild from the serialized dict.
    rebuilt = TrainingConfig(
        **{k: v for k, v in d.items() if k != "optimizer"},
        optimizer=OptimizerConfig(**d["optimizer"]),
    )
    assert rebuilt == cfg


# --------------------------------------------------------------------------- #
# Unknown keys are rejected (no silent acceptance)
# --------------------------------------------------------------------------- #
def test_unknown_keys_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("xpid: douzero\nnot_a_real_field: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(str(bad))


# --------------------------------------------------------------------------- #
# P00 baseline invariants unchanged under legacy config path
# --------------------------------------------------------------------------- #
def test_legacy_config_preserves_p00_obs_widths(seed_factory):
    """The legacy config path must not change P00-frozen observation widths.

    This is the regression gate: if the config plumbing accidentally altered
    any default that feeds the env/model, the frozen x-widths (373/484) would
    drift. We rebuild the env under the legacy config and assert the widths.
    """
    from douzero.env.env import Env, get_obs

    seed_factory(4242)
    env = Env(LegacyConfig.training.objective)
    env.reset()
    obs = get_obs(env.infoset)
    assert obs["x_batch"].shape[1] in (373, 484)
    assert obs["z_batch"].shape[1:] == (5, 162)


def test_parse_args_without_config_is_legacy_path():
    """parse_args([]) with no --config must equal the raw argparse result."""
    from douzero.dmc.arguments import parse_args, parser

    via_fn = parse_args([])
    via_raw = parser.parse_args([])
    # Same attributes (the function returns the raw Namespace when no --config).
    assert vars(via_fn) == vars(via_raw)


# --------------------------------------------------------------------------- #
# Precedence regression tests (review item 6)
# --------------------------------------------------------------------------- #
def test_yaml_value_used_when_cli_not_specified(tmp_path):
    """YAML batch_size=64, CLI unspecified -> 64 (argparse default 32 NOT applied)."""
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("batch_size: 64\nobjective: adp\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml)])
    assert ns.batch_size == 64


def test_cli_overrides_yaml_explicit_value(tmp_path):
    """YAML batch_size=64, CLI --batch_size 32 -> 32.

    This MUST go through parse_args() (the real entry point), which re-parses
    with default=SUPPRESS so the explicit --batch_size is detected as an
    override. Calling merge() directly with a full-default Namespace cannot
    distinguish "user typed 32" from "argparse default is 32".
    """
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("batch_size: 64\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--batch_size", "32"])
    # parse_args returns a Namespace; the merge already happened inside.
    assert ns.batch_size == 32


def test_store_true_yaml_true_not_clobbered_by_cli_default_false(tmp_path):
    """YAML deterministic=true, CLI unspecified -> True.

    This is the critical store_true regression: argparse defaults deterministic
    to False, but parse_args (--config) must NOT inject that False and overwrite
    YAML true. Goes through parse_args (SUPPRESS path) for a faithful test.
    """
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("deterministic: true\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml)])
    assert ns.deterministic is True


def test_store_true_cli_explicit_overrides_yaml_false(tmp_path):
    """YAML load_model=false, CLI --load_model -> True."""
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("load_model: false\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--load_model"])
    assert ns.load_model is True


# --------------------------------------------------------------------------- #
# Boolean override: YAML true -> CLI false (item 5, BooleanOptionalAction)
# --------------------------------------------------------------------------- #
def test_cli_no_flag_overrides_yaml_true_to_false(tmp_path):
    """YAML deterministic=true, CLI --no-deterministic -> False (item 5).

    This is the headline item-5 scenario: a YAML ``true`` boolean that the user
    wants to flip off from the CLI. Requires BooleanOptionalAction so that
    ``--no-deterministic`` is a recognized flag.
    """
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("deterministic: true\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--no-deterministic"])
    assert ns.deterministic is False


def test_cli_no_flag_overrides_yaml_true_actor_device_cpu(tmp_path):
    """YAML actor_device_cpu=true, CLI --no-actor_device_cpu -> False."""
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("actor_device_cpu: true\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--no-actor_device_cpu"])
    assert ns.actor_device_cpu is False


def test_cli_no_flag_overrides_yaml_true_disable_checkpoint(tmp_path):
    """YAML disable_checkpoint=true, CLI --no-disable_checkpoint -> False."""
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("disable_checkpoint: true\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--no-disable_checkpoint"])
    assert ns.disable_checkpoint is False


def test_cli_no_flag_overrides_yaml_true_load_model(tmp_path):
    """YAML load_model=true, CLI --no-load_model -> False."""
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("load_model: true\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--no-load_model"])
    assert ns.load_model is False


def test_boolean_default_is_false_not_none():
    """Legacy compat: parse_args([]) must yield False, not None, for booleans.

    BooleanOptionalAction without an explicit default yields None, which would
    break callers that do ``== False`` checks. The flags declare default=False.
    """
    from douzero.dmc.arguments import parser

    ns = parser.parse_args([])
    assert ns.actor_device_cpu is False
    assert ns.load_model is False
    assert ns.disable_checkpoint is False
    assert ns.deterministic is False


def test_positive_boolean_form_still_works():
    """Legacy compat: --actor_device_cpu (positive) sets True."""
    from douzero.dmc.arguments import parser

    ns = parser.parse_args(["--actor_device_cpu", "--load_model"])
    assert ns.actor_device_cpu is True
    assert ns.load_model is True


def test_no_flag_alone_sets_false():
    """--no-deterministic on its own (no --config) sets deterministic=False."""
    from douzero.dmc.arguments import parser

    ns = parser.parse_args(["--no-deterministic"])
    assert ns.deterministic is False


# --------------------------------------------------------------------------- #
# YAML safety / type validation (review item 6)
# --------------------------------------------------------------------------- #
def test_wrong_type_rejected(tmp_path):
    """num_actors: 'five' (string) must be rejected with a clear error."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("num_actors: five\n", encoding="utf-8")
    with pytest.raises(TypeError, match="num_actors"):
        load_config(str(bad))


def test_unsafe_yaml_tag_rejected(tmp_path):
    """A malicious Python YAML tag must not execute and must fail."""
    import yaml as _yaml

    bad = tmp_path / "evil.yaml"
    # This would execute system('id') if loaded with yaml.load (unsafe). Our
    # loader uses yaml.safe_load, which raises on this tag.
    bad.write_text(
        'xpid: !!python/object/apply:os.system ["echo PWNED"]\n', encoding="utf-8"
    )
    with pytest.raises(_yaml.constructor.ConstructorError):
        load_config(str(bad))


def test_non_mapping_root_rejected(tmp_path):
    """A YAML list/scalar root must be rejected (root must be a mapping)."""
    bad = tmp_path / "list.yaml"
    bad.write_text("- xpid\n- douzero\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(str(bad))


def test_null_field_rejected(tmp_path):
    """An explicit null value must be rejected (not silently accepted)."""
    bad = tmp_path / "null.yaml"
    bad.write_text("batch_size:\n", encoding="utf-8")  # null
    with pytest.raises((TypeError, ValueError)):
        load_config(str(bad))


# --------------------------------------------------------------------------- #
# Version identifier fields (item 4): carried through config + validated
# --------------------------------------------------------------------------- #
def test_legacy_yaml_carries_version_fields():
    """configs/legacy.yaml must carry the three version fields (= legacy)."""
    cfg = load_config(str(LEGACY_YAML))
    assert cfg.feature_version == "legacy"
    assert cfg.ruleset == "legacy"
    assert cfg.model_version == "legacy"


def test_version_fields_round_trip_through_namespace():
    """Config -> Namespace -> Config must preserve the version fields."""
    from douzero.dmc.arguments import parser

    ns = parser.parse_args([])
    cfg = from_argparse(ns)
    ns2 = to_argparse_namespace(cfg)
    cfg2 = from_argparse(ns2)
    assert cfg2.feature_version == "legacy"
    assert cfg2.ruleset == "legacy"
    assert cfg2.model_version == "legacy"


def test_namespace_exposes_version_dests():
    """to_argparse_namespace must expose feature_version/ruleset/model_version."""
    ns = to_argparse_namespace(LegacyConfig.training)
    for name in ("feature_version", "ruleset", "model_version"):
        assert hasattr(ns, name), f"missing {name}"
        assert getattr(ns, name) == "legacy"


def test_yaml_version_fields_not_lost_with_config_cli(tmp_path):
    """--config + explicit CLI must NOT silently drop the version fields.

    The version fields from the YAML base must survive the merge even when an
    unrelated CLI flag (e.g. --batch_size) is present. This is the item-4
    regression gate: the fields must flow through merge() and reappear in the
    final Namespace.
    """
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text(
        "batch_size: 64\nfeature_version: legacy\nruleset: legacy\nmodel_version: legacy\n",
        encoding="utf-8",
    )
    ns = parse_args(["--config", str(yml), "--batch_size", "32"])
    assert ns.feature_version == "legacy"
    assert ns.ruleset == "legacy"
    assert ns.model_version == "legacy"
    assert ns.batch_size == 32


def test_yaml_rejects_unsupported_feature_version(tmp_path):
    """A YAML config with feature_version=v2 must be rejected (P01=legacy only)."""
    bad = tmp_path / "v2.yaml"
    bad.write_text("feature_version: v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature_version"):
        load_config(str(bad))


def test_yaml_accepts_standard_ruleset(tmp_path):
    """P02 widens the allowed ruleset set: ruleset=standard is now accepted."""
    good = tmp_path / "std.yaml"
    good.write_text("ruleset: standard\n", encoding="utf-8")
    cfg = load_config(str(good))
    assert cfg.ruleset == "standard"


def test_yaml_rejects_unsupported_ruleset(tmp_path):
    """A YAML config with an unsupported ruleset value must still be rejected."""
    bad = tmp_path / "v2.yaml"
    bad.write_text("ruleset: v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ruleset"):
        load_config(str(bad))


def test_yaml_rejects_unsupported_model_version(tmp_path):
    """A YAML config with model_version=v2 must be rejected (P01=legacy only)."""
    bad = tmp_path / "mv2.yaml"
    bad.write_text("model_version: v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model_version"):
        load_config(str(bad))


def test_cli_rejects_unsupported_feature_version():
    """argparse choices must reject --feature_version v2 at the CLI."""
    from douzero.dmc.arguments import parser

    with pytest.raises(SystemExit):
        parser.parse_args(["--feature_version", "v2"])


def test_serialize_includes_version_fields():
    """serialize() must include the version fields in the output dict."""
    cfg = load_config(str(LEGACY_YAML))
    d = serialize(cfg)
    assert d["feature_version"] == "legacy"
    assert d["ruleset"] == "legacy"
    assert d["model_version"] == "legacy"


# --------------------------------------------------------------------------- #
# Override parser option-table hygiene (BooleanOptionalAction must not
# double-register --no-no-* options)
# --------------------------------------------------------------------------- #
def test_override_parser_has_no_double_negation_options():
    """_build_override_parser must not produce --no-no-* options.

    A BooleanOptionalAction's option_strings already contains both --flag and
    --no-flag. Re-registering both makes BooleanOptionalAction derive a
    spurious --no-no-flag. The override parser must register ONLY the positive
    form so the negation is derived exactly once.
    """
    from douzero.dmc.arguments import _build_override_parser

    op = _build_override_parser()
    opts = set(op._option_string_actions.keys())
    # Positive and single-negation forms must be present.
    assert "--deterministic" in opts
    assert "--no-deterministic" in opts
    assert "--actor_device_cpu" in opts
    assert "--no-actor_device_cpu" in opts
    assert "--load_model" in opts
    assert "--no-load_model" in opts
    assert "--disable_checkpoint" in opts
    assert "--no-disable_checkpoint" in opts
    # Double-negation forms must NOT exist.
    no_no = {o for o in opts if o.startswith("--no-no-")}
    assert not no_no, f"spurious double-negation options: {sorted(no_no)}"


def test_override_parser_boolean_still_overrides_correctly(tmp_path):
    """After the fix, --no-deterministic must still correctly set False.

    Regression guard: the option-table cleanup must not break the actual
    override behavior (YAML true -> CLI false).
    """
    from douzero.dmc.arguments import parse_args

    yml = tmp_path / "run.yaml"
    yml.write_text("deterministic: true\n", encoding="utf-8")
    ns = parse_args(["--config", str(yml), "--no-deterministic"])
    assert ns.deterministic is False
    # And the positive form still works.
    ns2 = parse_args(["--config", str(yml), "--deterministic"])
    assert ns2.deterministic is True
