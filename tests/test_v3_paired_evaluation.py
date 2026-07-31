"""Public-only V3 checkpoint coverage for the paired evaluator."""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.evaluation.agents import BundleFactory
from douzero.evaluation.checkpoint_inputs import checkpoint_sha256
from douzero.evaluation.deep_agent_v3 import parse_v3_evaluation_config
from douzero.evaluation.scenario import BundleSpec
from douzero.observation import build_v2_schema
from douzero.v3_hybrid import (
    BELIEF_FEEDBACK_FARMERS,
    V3BeliefPolicy,
    V3HybridModel,
    V3HybridModelConfig,
    save_v3_h4_public_checkpoint,
    save_v3_hybrid_public_checkpoint,
)
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import create_pilot_learner
from douzero.v3_hybrid.public_export import export_h7_public_checkpoint
from douzero.v3_hybrid.runtime import V3_H7_CHECKPOINT_FORMAT


ROLES = ("landlord", "landlord_up", "landlord_down")


def _config(*, belief: bool = False, bidding: bool = False):
    return V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
        belief_feedback=(BELIEF_FEEDBACK_FARMERS if belief else "none"),
        bidding_enabled=bidding,
    )


def _bundle(
    path,
    config,
    belief_config=None,
    *,
    bidding_policy="rule",
    search_config=None,
):
    digest = checkpoint_sha256(path)
    model_config = {"policy": config.to_dict()}
    if belief_config is not None:
        model_config["belief"] = {
            "hidden_size": belief_config.hidden_size,
            "num_layers": belief_config.num_layers,
            "dropout": belief_config.dropout,
            "style_enabled": belief_config.style_enabled,
            "style_embedding_dim": belief_config.style_embedding_dim,
            "shared_context_dim": belief_config.shared_context_dim,
        }
    return BundleSpec(
        name="v3-public",
        backend="v3",
        checkpoints={role: str(path) for role in ROLES},
        checkpoint_sha256={role: digest for role in ROLES},
        model_config=model_config,
        bidding_policy=bidding_policy,
        search_config=search_config or {},
    )


def _infoset(role="landlord"):
    np.random.seed(20260731)
    env = Env("adp")
    env.reset()
    while env._acting_player_position != role:
        env.step(env.infoset.legal_actions[0])
    value = copy.deepcopy(env.infoset)
    value.legal_actions = value.legal_actions[:4]
    return value


def test_v3_bundle_strictly_loads_and_returns_original_legal_action(tmp_path):
    torch.manual_seed(1)
    ruleset = RuleSet.legacy()
    config = _config()
    checkpoint = tmp_path / "public-v3.ckpt"
    save_v3_hybrid_public_checkpoint(
        checkpoint, V3HybridModel(build_v2_schema(), config), ruleset=ruleset
    )
    agent = BundleFactory(ruleset).build(
        _bundle(checkpoint, config),
        "landlord",
        seed=1,
        bundle_label="candidate",
    )
    infoset = _infoset()
    action = agent.act(infoset)
    assert action in infoset.legal_actions
    assert agent.predictions and 0.0 <= agent.predictions[-1] <= 1.0


def test_v3_evaluator_is_invariant_to_privileged_hidden_hand_swap(tmp_path):
    torch.manual_seed(2)
    ruleset = RuleSet.legacy()
    config = _config()
    checkpoint = tmp_path / "public-v3.ckpt"
    save_v3_hybrid_public_checkpoint(
        checkpoint, V3HybridModel(build_v2_schema(), config), ruleset=ruleset
    )
    agent = BundleFactory(ruleset).build(
        _bundle(checkpoint, config),
        "landlord",
        seed=1,
        bundle_label="candidate",
    ).inner
    original = _infoset()
    swapped = copy.deepcopy(original)
    swapped.all_handcards["landlord_up"], swapped.all_handcards["landlord_down"] = (
        swapped.all_handcards["landlord_down"],
        swapped.all_handcards["landlord_up"],
    )
    original_action = agent.act(original)
    original_p_win = agent.last_p_win
    swapped_action = agent.act(swapped)
    assert original_action == swapped_action
    assert agent.last_p_win == pytest.approx(original_p_win)


def test_v3_coupled_belief_checkpoint_is_supported(tmp_path):
    torch.manual_seed(3)
    ruleset = RuleSet.legacy()
    config = _config(belief=True)
    belief_config = BeliefConfig(hidden_size=16, num_layers=1)
    policy = V3BeliefPolicy(
        V3HybridModel(build_v2_schema(), config),
        BeliefModel(belief_config),
        ruleset=ruleset,
    )
    checkpoint = tmp_path / "public-v3-belief.ckpt"
    save_v3_h4_public_checkpoint(checkpoint, policy)
    agent = BundleFactory(ruleset).build(
        _bundle(checkpoint, config, belief_config),
        "landlord",
        seed=1,
        bundle_label="candidate",
    )
    assert agent.act(_infoset()) in _infoset().legal_actions


def test_v3_search_uses_same_public_checkpoint_and_safely_falls_back(tmp_path):
    torch.manual_seed(4)
    ruleset = RuleSet.legacy()
    config = _config(belief=True)
    belief_config = BeliefConfig(hidden_size=16, num_layers=1)
    checkpoint = tmp_path / "public-v3-belief.ckpt"
    save_v3_h4_public_checkpoint(
        checkpoint,
        V3BeliefPolicy(
            V3HybridModel(build_v2_schema(), config),
            BeliefModel(belief_config),
            ruleset=ruleset,
        ),
    )
    bundle = _bundle(
        checkpoint,
        config,
        belief_config,
        search_config={
            "enabled": True,
            "max_nodes": 0,
            "max_rollouts": 64,
            "max_milliseconds": 100,
        },
    )
    agent = BundleFactory(ruleset).build(
        bundle, "landlord", seed=1, bundle_label="candidate"
    )
    infoset = _infoset()
    assert agent.act(infoset) in infoset.legal_actions
    assert agent.inner.last_search_log.fallback_reason == "zero budget"
    assert len(set(bundle.checkpoint_sha256.values())) == 1


def test_v3_config_and_bidding_contract_fail_closed(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match="requires a belief config"):
        parse_v3_evaluation_config({
            "policy": _config(belief=True).to_dict(),
        })
    with pytest.raises(ValueError, match="part of the public policy checkpoint"):
        BundleSpec(
            name="bad",
            backend="v3",
            checkpoints={role: str(tmp_path / "x") for role in ROLES},
            model_config={"policy": config.to_dict()},
            bidding_policy="learned",
            bidding_checkpoint=str(tmp_path / "bid"),
        )


@pytest.mark.parametrize(
    ("config_name", "expected_format"),
    [
        ("v3_role_legacy.yaml", "v3-hybrid-h1-public-policy-v1"),
        ("v3_belief_legacy.yaml", "v3-hybrid-h4-belief-public-v1"),
    ],
)
def test_h7_export_emits_only_a_strict_public_graph(
    tmp_path, config_name, expected_format
):
    root = Path(__file__).resolve().parents[1]
    formal = load_formal_config(root / "configs" / "v3_formal" / config_name)
    formal = dataclasses.replace(
        formal, runtime=dataclasses.replace(formal.runtime, device="cpu")
    )
    learner, _resolved = create_pilot_learner(formal, allow_standard=True)
    h6 = tmp_path / "h6.pt"
    learner.save_checkpoint(h6)
    h6_payload = torch.load(h6, map_location="cpu", weights_only=True)
    training = tmp_path / "h7.pt"
    torch.save(
        {
            "format": V3_H7_CHECKPOINT_FORMAT,
            "artifact_access": "privileged_training_only",
            "runtime_identity": {},
            "runtime_hash": "0" * 64,
            "h6_checkpoint": h6_payload,
            "stats": {},
            "rng_state": random.Random(1).getstate(),
            "served_version_offset": 0,
            "snapshot_step": 0,
            "long_running_state": {},
        },
        training,
    )
    public = tmp_path / "public.pt"
    export_h7_public_checkpoint(training, public, formal_config=formal)
    payload = torch.load(public, map_location="cpu", weights_only=True)
    assert payload["format"] == expected_format
    serialized_names = " ".join(str(name).lower() for name in payload)
    assert all(
        token not in serialized_names
        for token in ("oracle", "teacher", "mixer", "optimizer", "replay")
    )


def test_h7_export_rejects_unknown_training_envelope_fields(tmp_path):
    formal = load_formal_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "v3_formal"
        / "v3_role_legacy.yaml"
    )
    training = tmp_path / "bad.pt"
    torch.save({"format": V3_H7_CHECKPOINT_FORMAT, "extra": True}, training)
    with pytest.raises(ValueError, match="envelope mismatch"):
        export_h7_public_checkpoint(
            training, tmp_path / "public.pt", formal_config=formal
        )
