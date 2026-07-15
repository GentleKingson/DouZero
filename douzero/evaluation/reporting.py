"""JSON, CSV, and Markdown outputs for P15 evaluation results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .paired import PairedEvaluationResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def render_markdown(result: PairedEvaluationResult) -> str:
    """Render a compact, auditable report that never hides the scenario mode."""
    scenario = result.scenario
    metrics = result.metrics
    ci = metrics["paired_estimate_ci"]
    estimate_label = (
        "Paired WP delta"
        if metrics["paired_estimator"] == "cardplay_win_rate_delta"
        else "Paired zero-sum seat score"
    )
    win_percentage_label = (
        "Candidate WP"
        if scenario["mode"] == "cardplay_only"
        else "Candidate seat WP (descriptive)"
    )
    latency = metrics["inference_latency_ms"]
    calibration = metrics["calibration"]["overall"]
    lines = [
        f"# P15 Evaluation: {scenario['candidate']['name']} vs {scenario['baseline']['name']}",
        "",
        f"- Protocol: `{scenario['protocol']}`",
        f"- Mode: `{scenario['mode']}`",
        f"- Ruleset: `{scenario['ruleset']['ruleset_id']}`",
        f"- Deal set: `{scenario['deal_set_id']}` ({scenario['dataset_scope']})",
        f"- Seed: `{scenario['deterministic_seed']}`",
        f"- Deals / games: {metrics['sample_counts']['deals']} / "
        f"{metrics['sample_counts']['games']}",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| {win_percentage_label} | {metrics['overall_win_percentage']:.4f} |",
        f"| {estimate_label} | {ci['estimate']:+.4f} "
        f"[{ci['low']:+.4f}, {ci['high']:+.4f}] |",
        f"| Mean score | {metrics['mean_score']:+.4f} |",
        f"| Mean log score | {metrics['mean_log_score']:+.4f} |",
        f"| Mean game length | {metrics['mean_game_length']:.2f} |",
        "",
        "## Per Role",
        "",
        "| Role | Games | WP | Mean score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role, role_metrics in metrics["by_role"].items():
        if not role_metrics.get("games"):
            lines.append(f"| {role} | 0 | n/a | n/a |")
        else:
            lines.append(
                f"| {role} | {role_metrics['games']} | "
                f"{role_metrics['win_percentage']:.4f} | "
                f"{role_metrics['mean_score']:+.4f} |"
            )
    lines.extend([
        "",
        "## Rules And Systems",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Bid rate | {_format_optional(metrics['bid_rate'])} |",
        f"| Landlord acquisition | {_format_optional(metrics['landlord_acquisition_rate'])} |",
        f"| Bomb / rocket rate | {metrics['bomb_rate']:.4f} / {metrics['rocket_rate']:.4f} |",
        f"| Spring / anti-spring | {metrics['spring_rate']:.4f} / "
        f"{metrics['anti_spring_rate']:.4f} |",
        f"| Inference p50 / p95 / p99 ms | {_format_optional(latency['p50'])} / "
        f"{_format_optional(latency['p95'])} / {_format_optional(latency['p99'])} |",
        f"| Actor FPS (P15 alias; inference calls/s) | "
        f"{_format_optional(metrics['actor_fps'])} |",
        f"| Search timeout / fallback rate | "
        f"{_format_optional(metrics['search']['timeout_rate'])} / "
        f"{_format_optional(metrics['search']['fallback_rate'])} |",
        f"| p_win Brier / NLL / ECE | {_format_optional(calibration['brier'])} / "
        f"{_format_optional(calibration['nll'])} / "
        f"{_format_optional(calibration['ece'])} |",
        "",
        "Confidence intervals resample complete deals. Mirrored legs and seat "
        "rotations from one deal are clustered before bootstrap resampling.",
        "",
    ])
    gates = metrics.get("regression_gates")
    if gates is not None:
        lines.extend([
            "## Regression Gates",
            "",
            f"Overall: **{'PASS' if gates['passed'] else 'FAIL'}**",
            "",
            "| Gate | Result | Observed | Threshold |",
            "| --- | --- | --- | --- |",
        ])
        for check in gates["checks"]:
            lines.append(
                f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | "
                f"`{check['observed']}` | `{check['threshold']}` |"
            )
        lines.append("")
    return "\n".join(lines)


def _format_optional(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.4f}"


def write_report(
    result: PairedEvaluationResult, output_prefix: str | Path
) -> dict[str, str]:
    """Write all required formats and return their absolute paths."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(prefix.with_suffix(".json").resolve()),
        "csv": str(prefix.with_suffix(".csv").resolve()),
        "markdown": str(prefix.with_suffix(".md").resolve()),
    }
    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(_json_safe(result.to_dict()), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    rows = [game.to_dict() for game in result.games]
    scalar_fields = [
        "deal_id", "leg_id", "mode", "candidate_win", "candidate_score",
        "candidate_log_score", "winner_team", "winner_position", "bid_value",
        "candidate_bid_attempts", "candidate_positive_bids", "candidate_landlord",
        "bomb_count", "rocket_count", "spring", "anti_spring", "game_length",
        "redeal_count", "max_redeals_exceeded", "search_calls",
        "search_timeouts", "search_fallbacks",
        "bidding_inference_calls",
    ]
    with open(paths["csv"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=scalar_fields + [
                "assignment", "candidate_roles", "role_wins", "role_scores"
            ],
        )
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in scalar_fields}
            for field in ("assignment", "candidate_roles", "role_wins", "role_scores"):
                output[field] = json.dumps(row[field], sort_keys=True)
            writer.writerow(output)

    with open(paths["markdown"], "w", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
    return paths
