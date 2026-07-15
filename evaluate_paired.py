"""Single-command CPU entry point for the P15 paired evaluation protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from douzero.env.rules import RuleSet
from douzero.evaluation.ablation import (
    ABLATION_NAMES,
    AblationRunner,
    AblationVariant,
)
from douzero.evaluation.gates import RegressionGateConfig, evaluate_regression_gates
from douzero.evaluation.reporting import write_report
from douzero.evaluation.scenario import (
    BundleSpec,
    EvaluationScenario,
    bundle_from_dict,
)


def _deck() -> list[int]:
    return list(range(3, 15)) * 4 + [17] * 4 + [20, 30]


def generate_deals(mode: str, count: int, seed: int, ruleset: RuleSet):
    """Generate a fixed in-memory public eval set without NumPy or downloads."""
    if mode not in ("cardplay_only", "full_game"):
        raise ValueError("mode must be cardplay_only or full_game")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    deals = []
    for _ in range(count):
        deck = _deck()
        rng.shuffle(deck)
        if mode == "cardplay_only":
            deals.append({
                "landlord": sorted(deck[:20]),
                "landlord_up": sorted(deck[20:37]),
                "landlord_down": sorted(deck[37:54]),
                "three_landlord_cards": sorted(deck[17:20]),
            })
        else:
            first = str(rng.randrange(3))
            order = [str((int(first) + offset) % 3) for offset in range(3)]
            deals.append({
                "format_version": 2,
                "schema_version": 1,
                "ruleset_id": ruleset.ruleset_id,
                "ruleset_version": ruleset.ruleset_version,
                "ruleset_hash": ruleset.stable_hash(),
                "deck": deck,
                "first_bidder": first,
                "bidding_order": order,
                "bidding_script": None,
            })
    return tuple(deals)


def _load_matrix(path: str):
    if not path:
        return {}, {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("model matrix must be a JSON object")
    unknown = set(raw) - {"bundles", "ablations"}
    if unknown:
        raise ValueError(f"unknown model-matrix sections: {sorted(unknown)}")
    bundle_rows = raw.get("bundles", {})
    if not isinstance(bundle_rows, dict):
        raise TypeError("model-matrix 'bundles' must be an object")
    if any(not isinstance(config, dict) for config in bundle_rows.values()):
        raise TypeError("each model-matrix bundle must be an object")
    bundles = {
        name: bundle_from_dict({**config, "name": name})
        for name, config in bundle_rows.items()
    }
    ablations = raw.get("ablations", {})
    if not isinstance(ablations, dict):
        raise TypeError("model-matrix 'ablations' must be an object")
    for name, value in ablations.items():
        if isinstance(value, str):
            continue
        if not isinstance(value, dict):
            raise TypeError(
                f"ablation {name!r} must name a candidate bundle or be an object"
            )
        unknown = set(value) - {"candidate", "baseline"}
        if unknown or not isinstance(value.get("candidate"), str):
            raise ValueError(
                f"ablation {name!r} requires a candidate bundle name and "
                "an optional baseline bundle name"
            )
        if "baseline" in value and not isinstance(value["baseline"], str):
            raise TypeError(f"ablation {name!r} baseline must name a bundle")
    unknown_ablations = set(ablations) - set(ABLATION_NAMES)
    if unknown_ablations:
        raise ValueError(f"unknown ablations: {sorted(unknown_ablations)}")
    return bundles, ablations


def _bundle(name: str, bundles: dict[str, BundleSpec], bidding_policy: str):
    if name in bundles:
        return bundles[name]
    if name not in ("random", "rule"):
        raise ValueError(
            f"bundle {name!r} is not built in and was not found in --model-matrix"
        )
    return BundleSpec(name=name, backend=name, bidding_policy=bidding_policy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DouZero P15 paired evaluation")
    parser.add_argument("--mode", choices=("cardplay_only", "full_game"), default="cardplay_only")
    parser.add_argument("--candidate", default="rule")
    parser.add_argument("--baseline", default="random")
    parser.add_argument(
        "--candidate-bidding",
        choices=("rule", "random", "pass", "max"),
        default="rule",
    )
    parser.add_argument(
        "--baseline-bidding",
        choices=("rule", "random", "pass", "max"),
        default="rule",
    )
    parser.add_argument("--model-matrix", default="", help="JSON bundle and ablation registry")
    parser.add_argument("--eval-data", default="", help="Trusted fixed .pkl deal set")
    parser.add_argument("--dataset-scope", choices=("public", "private_holdout"), default="public")
    parser.add_argument("--num-deals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--ruleset-config", default="")
    parser.add_argument("--output", default="artifacts/evaluation/p15")
    parser.add_argument("--gates", default="", help="Predeclared regression-gate JSON")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--require-complete-ablations", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_deals < 1:
        raise ValueError("--num-deals must be positive")
    if args.dataset_scope == "private_holdout" and not args.eval_data:
        raise ValueError("private_holdout evaluation requires an explicit --eval-data path")

    if args.mode == "full_game":
        if args.ruleset_config:
            from generate_eval_data import _load_ruleset_from_config

            ruleset = _load_ruleset_from_config(args.ruleset_config)
        else:
            ruleset = RuleSet.standard()
    else:
        if args.ruleset_config:
            raise ValueError("--ruleset-config is only valid for full_game")
        ruleset = RuleSet.legacy()

    if args.eval_data:
        from douzero.evaluation.legacy_data_adapter import load_eval_data

        deals = tuple(load_eval_data(
            args.eval_data,
            ruleset="standard" if args.mode == "full_game" else "legacy",
            expected_ruleset=ruleset,
        ))
        deal_set_name = Path(args.eval_data).stem
    else:
        deals = generate_deals(args.mode, args.num_deals, args.seed, ruleset)
        deal_set_name = f"generated-seed-{args.seed}"

    bundles, ablation_names = _load_matrix(args.model_matrix)
    scenario = EvaluationScenario(
        mode=args.mode,
        ruleset=ruleset,
        candidate=_bundle(args.candidate, bundles, args.candidate_bidding),
        baseline=_bundle(args.baseline, bundles, args.baseline_bidding),
        deals=deals,
        deterministic_seed=args.seed,
        dataset_scope=args.dataset_scope,
        deal_set_name=deal_set_name,
        bootstrap_samples=args.bootstrap_samples,
    )
    variants = {}
    if args.run_ablations:
        for ablation, specification in ablation_names.items():
            if isinstance(specification, str):
                candidate_name = specification
                baseline_name = None
            else:
                candidate_name = specification["candidate"]
                baseline_name = specification.get("baseline")
            try:
                candidate = bundles[candidate_name]
                baseline = bundles[baseline_name] if baseline_name else None
            except KeyError as exc:
                raise ValueError(
                    f"ablation {ablation!r} references unknown bundle {exc.args[0]!r}"
                ) from exc
            variants[ablation] = AblationVariant(
                candidate=candidate,
                baseline=baseline,
            )
    runner = AblationRunner(
        scenario,
        variants,
        require_complete=args.require_complete_ablations,
    )
    results = runner.run(include_base=True)
    gate_config = None
    if args.gates:
        with open(args.gates, "r", encoding="utf-8") as handle:
            gate_config = RegressionGateConfig(**json.load(handle))
    gates_failed = False
    for name, result in results.items():
        if gate_config is not None:
            gate_report = evaluate_regression_gates(result.metrics, gate_config)
            result.metrics["regression_gates"] = gate_report
            gates_failed = gates_failed or not gate_report["passed"]
        prefix = args.output if name == "base" else f"{args.output}-{name}"
        paths = write_report(result, prefix)
        ci = result.metrics["paired_win_rate_delta_ci"]
        print(
            f"{name}: mode={result.scenario['mode']} deals={ci['paired_deals']} "
            f"delta={ci['estimate']:+.4f} "
            f"CI=[{ci['low']:+.4f}, {ci['high']:+.4f}] "
            f"json={paths['json']}"
        )
    return 2 if gates_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
