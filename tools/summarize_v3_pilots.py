#!/usr/bin/env python3
"""Validate six P2 run pairs and emit a compact diagnostic summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from douzero.v3_hybrid.pilot import P2_VARIANTS, validate_pilot_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate P2 SIGTERM/resume evidence and summarize it."
    )
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.evidence_root)
    runs = []
    source_sha = None
    image_digest = None
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
        if not after["resume"]["requested"] or not after["resume"]["continued_update"]:
            raise ValueError(f"{variant} did not update after strict resume")
        if after["optimizer_steps"] <= before["optimizer_steps"]:
            raise ValueError(f"{variant} optimizer counter did not increase")
        observed_sha = after["source_git_sha"]
        observed_image = after["environment"]["image_digest"]
        source_sha = observed_sha if source_sha is None else source_sha
        image_digest = observed_image if image_digest is None else image_digest
        if observed_sha != source_sha or before["source_git_sha"] != source_sha:
            raise ValueError("P2 runs do not share one source SHA")
        if observed_image != image_digest or before["environment"]["image_digest"] != image_digest:
            raise ValueError("P2 runs do not share one image digest")
        runs.append({
            "variant": variant,
            "formal_config_sha256": after["formal_config_sha256"],
            "training_semantics_hash": after["training_semantics_hash"],
            "model_hash": after["metrics"]["model_hash"],
            "resolved_config_hash": after["metrics"]["resolved_config_hash"],
            "seed": after["seed"],
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
    payload = {
        "schema": "v3-p2-pilot-summary-v1",
        "source_git_sha": source_sha,
        "docker_image_digest": image_digest,
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
