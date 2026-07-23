#!/usr/bin/env python3
"""Validate six P2 run pairs and emit a compact diagnostic summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.pilot import (
    P2_SEED_DERIVATION,
    P2_VARIANTS,
    build_pilot_resolved_config,
    validate_pilot_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def summarize_evidence(root: Path) -> dict:
    """Validate P2 run pairs and return their compact diagnostic summary."""

    runs = []
    source_sha = None
    image_digest = None
    source_tree = None
    accelerator_identity = None
    for variant in P2_VARIANTS:
        directory = root / variant
        before = json.loads(
            (directory / "pre-resume-summary.json").read_text(encoding="utf-8")
        )
        after = json.loads(
            (directory / "post-resume-summary.json").read_text(encoding="utf-8")
        )
        validate_pilot_summary(before)
        validate_pilot_summary(after)
        if before["variant"] != variant or after["variant"] != variant:
            raise ValueError(f"P2 evidence directory mismatch for {variant}")
        if before["status"] != "stopped" or before["resume"]["stop_signal"] != "SIGTERM":
            raise ValueError(f"{variant} is missing real SIGTERM evidence")
        if before["failure"] is not None:
            raise ValueError(f"{variant} pre-resume run failed")
        if not after["resume"]["requested"] or not after["resume"]["continued_update"]:
            raise ValueError(f"{variant} did not update after strict resume")
        if after["status"] not in {"completed", "stopped"} or after["failure"] is not None:
            raise ValueError(f"{variant} post-resume run did not complete successfully")
        formal = load_formal_config(ROOT / f"configs/v3_formal/{variant}_legacy.yaml")
        identity = formal.identity_dict()
        expected_identity = {
            "formal_config_sha256": identity["config_sha256"],
            "training_semantics_hash": identity["training_semantics_hash"],
            "seed": formal.seeds.training[0],
            "ruleset": formal.ruleset["id"],
        }
        for field, expected in expected_identity.items():
            if before[field] != expected or after[field] != expected:
                raise ValueError(f"{variant} does not match the frozen {field}")
        expected_model = identity["model_hash"]
        expected_resolved = build_pilot_resolved_config(formal).stable_hash()
        for payload in (before, after):
            if payload["metrics"]["model_hash"] != expected_model:
                raise ValueError(f"{variant} does not match the frozen model hash")
            if payload["metrics"]["resolved_config_hash"] != expected_resolved:
                raise ValueError(f"{variant} does not match the frozen resolved config")
        expected_collection = {
            "root_seed": formal.seeds.training[0],
            "worker_id": 0,
            "derivation": P2_SEED_DERIVATION,
            "epsilon": 0.01,
        }
        if before["collection"] != expected_collection:
            raise ValueError(f"{variant} does not match frozen collection settings")
        ceilings = {
            "max_seconds": formal.budgets["pilot"].wall_clock_seconds,
            "max_samples": formal.budgets["pilot"].sample_budget,
            "max_optimizer_steps": formal.budgets["pilot"].optimizer_step_budget,
            "checkpoint_every": formal.runtime.checkpoint_cadence_updates,
        }
        for payload in (before, after):
            for field, ceiling in ceilings.items():
                if payload["limits"][field] > ceiling:
                    raise ValueError(f"{variant} {field} exceeds the frozen pilot ceiling")
        expected_resume = {
            "from_samples": before["samples"],
            "from_optimizer_steps": before["optimizer_steps"],
            "from_episodes": before["resume"]["from_episodes"] + before["episodes"],
            "from_decisions": before["resume"]["from_decisions"] + before["decisions"],
            "checkpoint_sha256": before["checkpoint"]["sha256"],
        }
        for field, expected in expected_resume.items():
            if after["resume"][field] != expected:
                raise ValueError(f"{variant} resume {field} does not match pre-run evidence")
        if after["collection"] != before["collection"]:
            raise ValueError(f"{variant} collection seeds changed across resume")
        if (
            after["environment"]["container_id"]
            == before["environment"]["container_id"]
        ):
            raise ValueError(f"{variant} resume did not use a fresh container")
        if after["optimizer_steps"] <= before["optimizer_steps"]:
            raise ValueError(f"{variant} optimizer counter did not increase")
        observed_sha = after["source_git_sha"]
        observed_image = after["environment"]["image_digest"]
        observed_tree = after["environment"]["source_tree"]
        observed_accelerator = {
            field: after["environment"][field]
            for field in (
                "gpu", "driver_version", "torch_version", "cuda_runtime", "machine"
            )
        }
        source_sha = observed_sha if source_sha is None else source_sha
        image_digest = observed_image if image_digest is None else image_digest
        source_tree = observed_tree if source_tree is None else source_tree
        accelerator_identity = (
            observed_accelerator if accelerator_identity is None else accelerator_identity
        )
        if observed_sha != source_sha or before["source_git_sha"] != source_sha:
            raise ValueError("P2 runs do not share one source SHA")
        if observed_image != image_digest or before["environment"]["image_digest"] != image_digest:
            raise ValueError("P2 runs do not share one image digest")
        if observed_tree != source_tree or before["environment"]["source_tree"] != source_tree:
            raise ValueError("P2 runs do not share one source tree")
        for payload in (before, after):
            current = {
                field: payload["environment"][field]
                for field in accelerator_identity
            }
            if current != accelerator_identity:
                raise ValueError("P2 runs do not share one accelerator identity")
        runs.append({
            "variant": variant,
            "formal_config_sha256": after["formal_config_sha256"],
            "training_semantics_hash": after["training_semantics_hash"],
            "model_hash": after["metrics"]["model_hash"],
            "resolved_config_hash": after["metrics"]["resolved_config_hash"],
            "seed": after["seed"],
            "pre_resume_limits": before["limits"],
            "post_resume_limits": after["limits"],
            "wall_clock_seconds": before["wall_clock_seconds"] + after["wall_clock_seconds"],
            "episodes": before["episodes"] + after["episodes"],
            "decisions": before["decisions"] + after["decisions"],
            "samples": after["samples"],
            "optimizer_steps": after["optimizer_steps"],
            "resume_delta_samples_per_second": after["metrics"]["samples_per_second"],
            "resume_delta_optimizer_steps_per_second": after["metrics"]["optimizer_steps_per_second"],
            "skipped_long_cooperation_episodes": (
                before["metrics"]["skipped_long_cooperation_episodes"]
                + after["metrics"]["skipped_long_cooperation_episodes"]
            ),
            "checkpoint_sha256": after["checkpoint"]["sha256"],
            "sigterm": True,
            "fresh_container_resume": True,
            "continued_update": True,
            "paired_evaluation": "NOT EXECUTED",
        })
    return {
        "schema": "v3-p2-pilot-summary-v1",
        "source_git_sha": source_sha,
        "docker_image_digest": image_digest,
        "source_tree": source_tree,
        "accelerator_identity": accelerator_identity,
        "ruleset": "legacy",
        "topology": "single_process",
        "seed": 101,
        "runs": runs,
        "decision": {
            "advance_to_p4": False,
            "requires_h7_1_runtime_decision": True,
            "reason": (
                "farmer cooperation and full hybrid throughput are below the "
                "budget needed for the frozen formal matrix"
            ),
        },
        "release_candidate": "NONE",
        "release_status": "NOT READY",
        "playing_strength": "NOT MEASURED",
        "unexecuted": [
            "2,000-5,000 paired deals per variant",
            "standard full-game pilot",
            "search-on evaluation from the shared checkpoint",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate P2 SIGTERM/resume evidence and summarize it."
    )
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize_evidence(Path(args.evidence_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
