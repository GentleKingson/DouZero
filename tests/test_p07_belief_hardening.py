"""P07 belief hardening tests: checkpoint security + device/dtype.

Covers the reviewer blockers/non-blockers:

- Blocker #1: ``load_belief_checkpoint`` defaults to ``weights_only=True`` and
  rejects an untrusted/crafted pickle; the full manifest identity is validated
  (schema_version, model_version, checkpoint_kind, feature_version,
  belief_config_hash, ruleset identity); ``expected_ruleset`` is REQUIRED.
- Medium #6: ``BeliefModel.forward`` derives device/dtype from the model
  parameters (``model.double()`` works).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from douzero.belief import BeliefConfig, BeliefModel
from douzero.belief.checkpoint import (
    BELIEF_MANIFEST_SCHEMA_VERSION,
    load_belief_checkpoint,
    save_belief_checkpoint,
)
from douzero.env.rules import RuleSet


# --------------------------------------------------------------------------- #
# Checkpoint security (Blocker #1)
# --------------------------------------------------------------------------- #
class TestCheckpointSecurity:
    def _save(self, tmp_path, cfg=None):
        torch.manual_seed(1)
        model = BeliefModel(cfg or BeliefConfig(hidden_size=16, num_layers=1))
        path = str(tmp_path / "belief.pt")
        save_belief_checkpoint(path, model, ruleset=RuleSet.legacy())
        return path, model

    def test_required_ruleset_is_mandatory(self, tmp_path):
        path, _ = self._save(tmp_path)
        with pytest.raises(TypeError):
            load_belief_checkpoint(path)  # missing required expected_ruleset
        with pytest.raises(TypeError):
            load_belief_checkpoint(path, expected_ruleset=None)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_weights_only_rejects_crafted_pickle(self, tmp_path):
        """A non-checkpoint pickle must NOT be executed as code."""
        path = str(tmp_path / "evil.pt")
        # A crafted pickle carrying an arbitrary object. weights_only=True must
        # refuse to deserialize it (rather than running it).
        class _Nasty:
            def __reduce__(self):
                # If ever executed this would call int('999'); weights_only
                # must reject the GLOBAL before that.
                return (int, ("999",))

        with open(path, "wb") as f:
            pickle.dump({"belief_state_dict": _Nasty()}, f)
        with pytest.raises(Exception):
            load_belief_checkpoint(
                path, expected_ruleset=RuleSet.legacy(),
                allow_unsafe_pickle=False,
            )

    def test_rejects_non_bundle_top_level(self, tmp_path):
        path = str(tmp_path / "notabundle.pt")
        torch.save([1, 2, 3], path)  # a list, not a dict bundle
        with pytest.raises(ValueError):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())

    def test_rejects_missing_manifest_key(self, tmp_path):
        path = str(tmp_path / "bad.pt")
        torch.save({"belief_state_dict": {}}, path)  # no manifest
        with pytest.raises(ValueError):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())

    def test_rejects_wrong_checkpoint_kind(self, tmp_path):
        path, _ = self._save(tmp_path)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        bundle["manifest"]["checkpoint_kind"] = "public_policy"
        torch.save(bundle, path)
        with pytest.raises(ValueError, match="checkpoint_kind"):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())

    def test_rejects_wrong_schema_version(self, tmp_path):
        path, _ = self._save(tmp_path)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        bundle["manifest"]["schema_version"] = 999
        torch.save(bundle, path)
        with pytest.raises(ValueError, match="schema_version"):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())

    def test_rejects_wrong_feature_version(self, tmp_path):
        path, _ = self._save(tmp_path)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        bundle["manifest"]["feature_version"] = "v3"
        torch.save(bundle, path)
        with pytest.raises(ValueError, match="feature_version"):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())

    def test_short_source_git_sha_requires_explicit_release_strictness(self, tmp_path):
        path, _ = self._save(tmp_path)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        bundle["manifest"]["git_sha"] = "abc1234"
        torch.save(bundle, path)
        # Historical P07 checkpoints used ``rev-parse --short`` and remain
        # loadable through the compatibility path.
        load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())
        with pytest.raises(ValueError, match="full source Git SHA"):
            load_belief_checkpoint(
                path,
                expected_ruleset=RuleSet.legacy(),
                require_full_git_sha=True,
            )

    def test_save_requires_full_source_git_sha(self, tmp_path, monkeypatch):
        model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
        monkeypatch.setattr(
            "douzero.belief.checkpoint.git_sha", lambda: "unknown"
        )
        with pytest.raises(ValueError, match="full source Git SHA"):
            save_belief_checkpoint(
                str(tmp_path / "belief.pt"),
                model,
                ruleset=RuleSet.legacy(),
            )

    def test_rejects_corrupted_config_hash(self, tmp_path):
        path, _ = self._save(tmp_path)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        bundle["manifest"]["belief_config_hash"] = "0" * 64
        torch.save(bundle, path)
        with pytest.raises(ValueError, match="config"):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())

    def test_default_load_uses_weights_only_true(self, tmp_path):
        """The default (no allow_unsafe_pickle) round-trips and is safe."""
        path, model = self._save(tmp_path)
        # Must succeed with the default safe path.
        loaded = load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.equal(p1, p2)


# --------------------------------------------------------------------------- #
# Device / dtype respect (Medium #6)
# --------------------------------------------------------------------------- #
class TestDeviceDtype:
    def _input(self):
        from douzero.env.env import Env
        from douzero.observation.encode_v2 import get_obs_v2

        from douzero.belief import build_belief_input

        np.random.seed(0)
        env = Env("adp")
        env.reset()
        return build_belief_input(get_obs_v2(env.infoset).public)

    def test_double_dtype_respected(self):
        model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1)).double()
        model.eval()
        binput = self._input()
        with torch.no_grad():
            out = model([binput])
        # Logits inherit the model dtype (float64).
        assert out.logits.dtype == torch.float64
        assert bool(torch.isfinite(out.logits).all())

    def test_float_dtype_default(self):
        model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
        binput = self._input()
        with torch.no_grad():
            out = model([binput])
        assert out.logits.dtype == torch.float32
