#!/usr/bin/env python
"""Capture the DouZero legacy behavioral baseline (P00).

Generates a fixed set of deals and records, for each:
  * observation field shapes/dtypes,
  * a canonical hash of the legal actions,
  * a hash of the model forward output under fixed initialisation,
  * the DeepAgent-selected action.

No pretrained weights are required: models are initialised from a fixed torch
seed so the output hash is reproducible offline. The resulting JSON includes
full environment metadata (python/torch/numpy/rlcard versions, git SHA, OS,
CUDA availability) so the baseline is self-describing.

Reproducibility contract (IMPORTANT):
  * The hashes this tool emits are a SAME-ENVIRONMENT determinism diagnostic.
    They are expected to be byte-identical across two runs in the *same*
    container (same python/torch/numpy, CPU). They are NOT a cross-machine
    hard gate: float model-output bytes can differ across torch builds /
    platforms while the behaviour is still equivalent.
  * The portable baseline contract (shapes, dtypes, role->width mapping, the
    fixed TYPE_15_WRONG exception set, terminal invariants) lives in the test
    suite at ``tests/test_baseline_invariants.py`` and is committed; it does
    NOT depend on float reproducibility.
  * This JSON artifact is git-ignored (host-specific). Re-generate it with
    ``--num_deals 64`` whenever you need a local diagnostic.

Usage:
    python tools/capture_baseline.py --num_deals 64
    python tools/capture_baseline.py --num_deals 2 --output artifacts/baseline/smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# Force CPU before importing torch-adjacent code: legacy models probe
# torch.cuda.is_available() directly and have no device argument.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DEFAULT_SEED = 20240611
DEFAULT_NUM_DEALS = 64
DEFAULT_OUTPUT = "artifacts/baseline/baseline.json"


def _ci_identity():
    """Return explicit PR head/merge identity when CI supplied both values."""

    head = os.environ.get("DOUZERO_CI_HEAD_SHA")
    merge = os.environ.get("DOUZERO_CI_MERGE_SHA")
    if not head and not merge:
        return None
    full_sha = lambda value: bool(
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(char in "0123456789abcdef" for char in value)
    )
    if not full_sha(head) or not full_sha(merge):
        raise ValueError(
            "DOUZERO_CI_HEAD_SHA and DOUZERO_CI_MERGE_SHA must both be full "
            "lowercase Git object IDs"
        )
    return {
        "head_sha": head,
        "merge_sha": merge,
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def _setup_imports():
    import numpy as np  # noqa: F401
    import torch  # noqa: F401

    return np, torch


def _seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _canonical_legal_actions_hash(legal_actions):
    payload = json.dumps(sorted([sorted(a) for a in legal_actions]))
    return hashlib.sha256(payload.encode()).hexdigest()


def _model_output_hash(np_module, torch_module, model, obs):
    z = torch_module.from_numpy(obs["z_batch"]).float()
    x = torch_module.from_numpy(obs["x_batch"]).float()
    with torch_module.no_grad():
        values = model(z, x, return_value=True)["values"]
    arr = values.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest(), float(arr.mean())


def _build_deepagent(torch_module, position, seed):
    """Back a DeepAgent with a deterministically-initialised synthetic ckpt."""
    import tempfile

    from douzero.dmc.models import model_dict
    from douzero.evaluation.deep_agent import DeepAgent

    torch_module.manual_seed(seed)
    model = model_dict[position]()
    ckpt_dir = tempfile.mkdtemp(prefix="douzero_baseline_")
    ckpt_path = os.path.join(ckpt_dir, f"{position}.ckpt")
    torch_module.save(model.state_dict(), ckpt_path)
    return DeepAgent(position, ckpt_path)


def _capture_one(np_module, torch_module, deepagents, seed, deal_index):
    """Run one seeded deal to terminal; snapshot obs at the opening landlord move."""
    from douzero.dmc.models import model_dict
    from douzero.env.env import Env, get_obs

    # Per-deal derived seed: deterministic and independent of iteration order.
    deal_seed = seed + deal_index
    _seed_everything(deal_seed)

    env = Env("adp")
    env.reset()
    infoset = env.infoset
    obs = get_obs(infoset)

    position = obs["position"]
    torch_module.manual_seed(seed)
    model = model_dict[position]()
    model.eval()
    out_hash, out_mean = _model_output_hash(np_module, torch_module, model, obs)

    legal_hash = _canonical_legal_actions_hash(infoset.legal_actions)
    selected = deepagents[position].act(infoset)

    # Also drive to terminal to confirm the env can complete a game and record
    # winner/bomb for sanity (these are environment facts, not strength claims).
    _seed_everything(deal_seed)
    env2 = Env("adp")
    env2.reset()
    steps = 0
    while True:
        action = env2.infoset.legal_actions[0]
        _, _, done, _ = env2.step(action)
        steps += 1
        if done or steps > 5000:
            break

    return {
        "deal_index": deal_index,
        "deal_seed": deal_seed,
        "acting_position": position,
        "num_legal_actions": len(infoset.legal_actions),
        "obs_shapes": {
            "x_batch": list(obs["x_batch"].shape),
            "z_batch": list(obs["z_batch"].shape),
            "x_no_action": list(obs["x_no_action"].shape),
            "z": list(obs["z"].shape),
        },
        "obs_dtypes": {
            "x_batch": str(obs["x_batch"].dtype),
            "z_batch": str(obs["z_batch"].dtype),
            "x_no_action": str(obs["x_no_action"].dtype),
            "z": str(obs["z"].dtype),
        },
        "legal_actions_sha256": legal_hash,
        "model_output_sha256": out_hash,
        "model_output_mean": round(out_mean, 8),
        "deepagent_selected_action": sorted(selected),
        "terminal": {
            "steps": steps,
            "winner": env2._game_winner if steps <= 5000 else None,
            "bomb_num": env2._game_bomb_num if steps <= 5000 else None,
        },
    }


def capture(num_deals: int, seed: int, output: str):
    np_module, torch_module = _setup_imports()
    from douzero._version import environment_info

    _seed_everything(seed)
    deepagents = {
        pos: _build_deepagent(torch_module, pos, seed)
        for pos in ["landlord", "landlord_up", "landlord_down"]
    }

    records = []
    for i in range(num_deals):
        records.append(_capture_one(np_module, torch_module, deepagents, seed, i))

    summary = {
        "schema_version": "p00-baseline-v1",
        "description": (
            "DouZero legacy behavioral baseline. Observations and model outputs "
            "are captured under fixed seeds with synthetic (init-only) model "
            "weights -- this freezes DETERMINISM, not playing strength."
        ),
        "environment": environment_info(),
        "ci_identity": _ci_identity(),
        "config": {
            "num_deals": num_deals,
            "base_seed": seed,
            "model_init": "fixed torch seed (no pretrained weights)",
        },
        "records": records,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # Stdout summary for humans / CI logs.
    env = summary["environment"]
    print(f"baseline -> {out_path}")
    print(
        f"  deals={num_deals} seed={seed} "
        f"python={env.get('python_version')} torch={env.get('torch_version')} "
        f"numpy={env.get('numpy_version')} git_sha={env.get('git_sha')}"
    )
    first = records[0]
    print(
        f"  sample deal0: pos={first['acting_position']} "
        f"n_legal={first['num_legal_actions']} legal_sha={first['legal_actions_sha256'][:12]}..."
    )
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_deals", type=int, default=DEFAULT_NUM_DEALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    capture(args.num_deals, args.seed, args.output)


if __name__ == "__main__":
    main()
