"""P07 belief data pipeline, checkpoint round-trip, and CLI smoke tests.

These exercise the training-only data path (random self-play collection with
privileged label capture), the manifest-bearing checkpoint save/load with
identity validation, and the ``train_belief`` / ``evaluate_belief`` entry
points end-to-end on a tiny CPU workload.

All random sources are seeded; no network or downloaded weights are required.
"""

from __future__ import annotations

import numpy as np
import torch

from douzero.belief import BeliefConfig, BeliefModel
from douzero.belief.checkpoint import (
    BELIEF_MODEL_VERSION,
    load_belief_checkpoint,
    save_belief_checkpoint,
)
from douzero.belief.data import (
    BeliefDataset,
    collect_random_dataset,
    iterate_minibatches,
)
from douzero.env.rules import RuleSet


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
class TestDataCollection:
    def test_collect_yields_consistent_samples(self):
        ds = collect_random_dataset(num_episodes=4, seed=1)
        assert len(ds) > 0
        feats = ds.feature_matrix()
        targets = ds.target_tensor()
        legal = ds.legal_mask_tensor()
        unseen = ds.unseen_counts_matrix()
        # Shapes.
        from douzero.belief import BELIEF_INPUT_DIM
        from douzero.belief.constraints import NUM_BELIEF_RANKS, NUM_COUNT_SLOTS

        n = len(ds)
        assert feats.shape == (n, BELIEF_INPUT_DIM)
        assert targets.shape == (n, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        assert legal.shape == (n, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        assert unseen.shape == (n, NUM_BELIEF_RANKS)
        # Each target row is a valid one-hot (one set slot per rank).
        assert bool((targets.sum(dim=-1) == 1).all())
        # Each label allocation sums to its opponent-A hidden total.
        for i, s in enumerate(ds.samples):
            assert int(s.label.allocation.sum()) == s.binput.opponent_a_total

    def test_minibatches_cover_dataset(self):
        ds = collect_random_dataset(num_episodes=3, seed=2)
        import random

        batches = iterate_minibatches(ds, batch_size=16, shuffle=True,
                                      rng=random.Random(0))
        total = sum(int(f.shape[0]) for f, _, _ in batches)
        assert total == len(ds)

    def test_empty_dataset_accessors_raise(self):
        ds = BeliefDataset()
        with np.testing.assert_raises(Exception):
            ds.target_tensor()


# --------------------------------------------------------------------------- #
# Checkpoint round-trip + identity validation
# --------------------------------------------------------------------------- #
class TestCheckpoint:
    def test_save_load_roundtrip_preserves_config_and_weights(self, tmp_path):
        torch.manual_seed(5)
        cfg = BeliefConfig(hidden_size=24, num_layers=1, dropout=0.0)
        model = BeliefModel(cfg)
        path = str(tmp_path / "belief.pt")
        save_belief_checkpoint(path, model, ruleset=RuleSet.legacy(),
                               feature_version="v2", frames=100)
        loaded = load_belief_checkpoint(path, expected_ruleset=RuleSet.legacy())
        assert loaded.config.stable_hash() == cfg.stable_hash()
        # Weights match exactly.
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_load_rejects_config_hash_mismatch(self, tmp_path):
        torch.manual_seed(5)
        model = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
        path = str(tmp_path / "belief.pt")
        save_belief_checkpoint(path, model, ruleset=RuleSet.legacy())
        # A different runtime config must be rejected.
        other = BeliefConfig(hidden_size=48, num_layers=1)
        with __import__("pytest").raises(ValueError):
            load_belief_checkpoint(path, expected_belief_config=other,
                                   expected_ruleset=RuleSet.legacy())

    def test_load_rejects_ruleset_mismatch(self, tmp_path):
        torch.manual_seed(5)
        model = BeliefModel(BeliefConfig(hidden_size=24, num_layers=1))
        path = str(tmp_path / "belief.pt")
        save_belief_checkpoint(path, model, ruleset=RuleSet.legacy())
        with __import__("pytest").raises(ValueError):
            load_belief_checkpoint(path, expected_ruleset=RuleSet.standard())

    def test_manifest_carries_model_version_and_provenance(self, tmp_path):
        model = BeliefModel(BeliefConfig(hidden_size=16, num_layers=1))
        path = str(tmp_path / "belief.pt")
        save_belief_checkpoint(path, model, ruleset=RuleSet.legacy(), frames=7)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        m = bundle["manifest"]
        assert m["model_version"] == BELIEF_MODEL_VERSION
        assert m["checkpoint_kind"] == "belief_model"
        assert m["frames"] == 7
        assert m["ruleset_id"] == "legacy"
        assert m["belief_config_hash"]
        assert m["python_version"]


# --------------------------------------------------------------------------- #
# CLI smoke (train_belief + evaluate_belief main entry points)
# --------------------------------------------------------------------------- #
class TestCLI:
    def test_train_belief_main_runs_and_saves(self, tmp_path):
        import sys

        sys.path.insert(0, ".")
        import train_belief

        save_dir = str(tmp_path / "bp")
        rc = train_belief.main([
            "--save_dir", save_dir,
            "--num_episodes", "4",
            "--epochs", "1",
            "--batch_size", "16",
            "--hidden_size", "16",
            "--num_layers", "1",
            "--seed", "0",
        ])
        assert rc == 0
        import os

        assert os.path.isfile(os.path.join(save_dir, "belief.pt"))

    def test_evaluate_belief_main_runs(self, tmp_path):
        import evaluate_belief

        save_dir = str(tmp_path / "bp")
        import train_belief

        rc = train_belief.main([
            "--save_dir", save_dir, "--num_episodes", "4", "--epochs", "1",
            "--batch_size", "16", "--hidden_size", "16", "--num_layers", "1",
            "--seed", "0",
        ])
        assert rc == 0
        import os

        ckpt = os.path.join(save_dir, "belief.pt")
        rc2 = evaluate_belief.main([
            "--checkpoint", ckpt, "--num_episodes", "4", "--seed", "3",
        ])
        assert rc2 == 0

    def test_train_belief_help_parses(self):
        import train_belief

        args = train_belief._parse_args([
            "--save_dir", "/tmp/x", "--num_episodes", "1", "--epochs", "1",
        ])
        assert args.num_episodes == 1
        assert args.ruleset == "legacy"
