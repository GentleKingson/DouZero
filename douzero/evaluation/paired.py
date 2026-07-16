"""Paired card-play and seat-rotated full-game evaluation (P15)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping

from douzero._version import git_sha
from douzero.env.game import GameEnv
from douzero.env.rules import PHASE_BIDDING

from .agents import BundleFactory, RuleAgent, TimedAgent
from .protocol import (
    EVALUATION_PROTOCOL,
    MIN_PROMOTION_BOOTSTRAP_SAMPLES,
    MIN_PROMOTION_PAIRED_DEALS,
    OFFICIAL_CONFIDENCE_LEVEL,
    OFFICIAL_CI_METHOD,
    OFFICIAL_STATISTICAL_UNIT,
    OFFICIAL_PERMUTATION_HASHES,
    PROMOTION_ESTIMATOR,
    PROMOTION_MODE,
)
from .scenario import (
    ROLES,
    EvaluationScenario,
    canonical_deal_hash,
    canonical_deal_id,
    default_seat_permutations,
)
from .statistics import (
    ConfidenceInterval,
    deal_cluster_means,
    paired_bootstrap_ci,
    percentile,
)


EVALUATION_RESULT_SCHEMA_VERSION = "p15-paired-result-v2"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bundle_feature_schema(bundle: Mapping[str, Any]) -> dict[str, Any]:
    backend = str(bundle.get("backend", ""))
    if backend in {"v2", "bc"}:
        from douzero.observation.bidding import build_bidding_schema
        from douzero.observation.schema import build_v2_schema

        learned_bidding = bundle.get("bidding_policy") == "learned"
        return {
            "feature_version": "v2",
            "feature_schema_hash": build_v2_schema().stable_hash(),
            "bidding_feature_schema_hash": (
                build_bidding_schema().stable_hash() if learned_bidding else None
            ),
        }
    if backend in {"legacy", "legacy_factorized"}:
        return {
            "feature_version": "legacy",
            "feature_schema_hash": None,
            "bidding_feature_schema_hash": None,
        }
    if backend in {"random", "rule"}:
        return {
            "feature_version": "builtin-no-model-input",
            "feature_schema_hash": None,
            "bidding_feature_schema_hash": None,
        }
    raise ValueError(f"unsupported evaluation backend for schema identity: {backend!r}")


def evaluation_runtime_identity(
    scenario: Mapping[str, Any], *, ablation: str
) -> dict[str, Any]:
    """Bind evaluator code and effective scenario/schema configuration."""

    source_sha = git_sha()
    if (
        len(source_sha) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in source_sha)
    ):
        raise RuntimeError(
            "paired evaluation requires a full source Git SHA; set "
            "DOUZERO_GIT_SHA in source-less runtimes"
        )
    schema_identities = {
        side: _bundle_feature_schema(scenario.get(side, {}))
        for side in ("candidate", "baseline")
    }
    config_payload = {
        "protocol": EVALUATION_PROTOCOL,
        "ablation": ablation,
        "scenario": dict(scenario),
        "model_feature_schemas": schema_identities,
    }
    ruleset = scenario.get("ruleset", {})
    return {
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "source_git_sha": source_sha,
        "evaluation_config_hash": _canonical_sha256(config_payload),
        "ruleset_hash": (
            ruleset.get("ruleset_hash") if isinstance(ruleset, Mapping) else None
        ),
        "model_feature_schemas": schema_identities,
    }


def validate_evaluation_runtime_identity(
    result: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    expected_source_git_shas: str | Iterable[str] | None = None,
) -> None:
    """Reject missing or self-inconsistent evaluation provenance."""

    runtime = result.get("runtime_identity")
    scenario = result.get("scenario")
    ablation = result.get("ablation")
    if not isinstance(runtime, Mapping) or not isinstance(scenario, Mapping):
        raise ValueError("evaluation result is missing runtime identity or scenario")
    if result.get("protocol") != EVALUATION_PROTOCOL:
        raise ValueError("evaluation result protocol mismatch")
    if scenario.get("protocol") != EVALUATION_PROTOCOL:
        raise ValueError("evaluation scenario protocol mismatch")
    mode = scenario.get("mode")
    if mode not in {"cardplay_only", "full_game"}:
        raise ValueError("evaluation scenario mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
            f"evaluation scenario mode mismatch: expected {expected_mode!r}, got {mode!r}"
        )
    if not isinstance(ablation, str) or not ablation:
        raise ValueError("evaluation ablation identity must be a non-empty string")
    if runtime.get("schema_version") != EVALUATION_RESULT_SCHEMA_VERSION:
        raise ValueError("evaluation runtime schema version mismatch")
    source_sha = runtime.get("source_git_sha")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in source_sha)
    ):
        raise ValueError("evaluation source_git_sha must be a full Git SHA")
    if expected_source_git_shas is not None:
        approved_shas = (
            (expected_source_git_shas,)
            if isinstance(expected_source_git_shas, str)
            else tuple(expected_source_git_shas)
        )
        if not approved_shas or any(
            not isinstance(sha, str)
            or len(sha) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in sha)
            for sha in approved_shas
        ):
            raise ValueError("expected evaluator Git SHA allowlist is invalid")
        if source_sha not in approved_shas:
            raise ValueError("evaluation source_git_sha is not approved")
    schema_identities = runtime.get("model_feature_schemas")
    if not isinstance(schema_identities, Mapping):
        raise ValueError("evaluation model feature-schema identities are missing")
    expected_schema_identities: dict[str, dict[str, Any]] = {}
    for side in ("candidate", "baseline"):
        bundle = scenario.get(side)
        if not isinstance(bundle, Mapping):
            raise ValueError(f"evaluation scenario {side} bundle is invalid")
        expected_schema_identities[side] = _bundle_feature_schema(bundle)
    if dict(schema_identities) != expected_schema_identities:
        raise ValueError(
            "evaluation model feature-schema identities do not match the scenario"
        )
    config_payload = {
        "protocol": EVALUATION_PROTOCOL,
        "ablation": ablation,
        "scenario": dict(scenario),
        "model_feature_schemas": expected_schema_identities,
    }
    if runtime.get("evaluation_config_hash") != _canonical_sha256(config_payload):
        raise ValueError("evaluation_config_hash does not match the result scenario")
    ruleset = scenario.get("ruleset")
    if (
        not isinstance(ruleset, Mapping)
        or runtime.get("ruleset_hash") != ruleset.get("ruleset_hash")
    ):
        raise ValueError("evaluation runtime ruleset hash mismatch")


def _seed(base: int, *parts: object) -> int:
    token = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def _deal_id(index: int, deal: Mapping[str, Any]) -> str:
    return canonical_deal_id(index, canonical_deal_hash(deal))


@dataclass(frozen=True)
class GameRecord:
    """One mirrored leg or seat rotation, retaining its parent deal ID."""

    deal_id: str
    leg_id: str
    mode: str
    assignment: tuple[str, str, str]
    candidate_roles: tuple[str, ...]
    candidate_win: float
    candidate_score: float
    role_wins: dict[str, float]
    role_scores: dict[str, float]
    winner_team: str
    winner_position: str
    bid_value: int
    candidate_bid_attempts: int
    candidate_positive_bids: int
    candidate_landlord: int
    bomb_count: int
    rocket_count: int
    spring: bool
    anti_spring: bool
    game_length: int
    candidate_latencies_ms: tuple[float, ...] = ()
    calibration: tuple[tuple[str, float, float], ...] = ()
    # Every P17 field is appended after the two original defaulted fields so
    # pre-P17 positional construction retains its exact argument meaning.
    redeal_count: int = 0
    max_redeals_exceeded: bool = False
    search_calls: int = 0
    search_timeouts: int = 0
    search_fallbacks: int = 0
    bidding_inference_calls: int = 0
    # Forced post-cap gameplay is retained only as an audited smoke row and is
    # excluded from every formal deal-level estimate.
    formal_evaluation_eligible: bool = True
    exclusion_reason: str = ""
    # Full experiment/terminal evidence appended for positional compatibility.
    deal_hash: str = ""
    seat_to_role: dict[str, str] | None = None
    bidding_order: tuple[str, ...] = ()
    bidding_history: tuple[tuple[str, int], ...] = ()

    @property
    def candidate_log_score(self) -> float:
        if self.candidate_score == 0:
            return 0.0
        return math.copysign(math.log1p(abs(self.candidate_score)), self.candidate_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "leg_id": self.leg_id,
            "mode": self.mode,
            "assignment": list(self.assignment),
            "candidate_roles": list(self.candidate_roles),
            "candidate_win": self.candidate_win,
            "candidate_score": self.candidate_score,
            "candidate_log_score": self.candidate_log_score,
            "role_wins": dict(self.role_wins),
            "role_scores": dict(self.role_scores),
            "winner_team": self.winner_team,
            "winner_position": self.winner_position,
            "bid_value": self.bid_value,
            "candidate_bid_attempts": self.candidate_bid_attempts,
            "candidate_positive_bids": self.candidate_positive_bids,
            "candidate_landlord": self.candidate_landlord,
            "bomb_count": self.bomb_count,
            "rocket_count": self.rocket_count,
            "spring": self.spring,
            "anti_spring": self.anti_spring,
            "game_length": self.game_length,
            "redeal_count": self.redeal_count,
            "max_redeals_exceeded": self.max_redeals_exceeded,
            "candidate_latencies_ms": list(self.candidate_latencies_ms),
            "calibration": [list(sample) for sample in self.calibration],
            "search_calls": self.search_calls,
            "search_timeouts": self.search_timeouts,
            "search_fallbacks": self.search_fallbacks,
            "bidding_inference_calls": self.bidding_inference_calls,
            "formal_evaluation_eligible": self.formal_evaluation_eligible,
            "exclusion_reason": self.exclusion_reason or None,
            "deal_hash": self.deal_hash,
            "seat_to_role": copy.deepcopy(self.seat_to_role),
            "bidding_order": list(self.bidding_order),
            "bidding_history": [list(bid) for bid in self.bidding_history],
        }


@dataclass(frozen=True)
class PairedEvaluationResult:
    """JSON-ready P15 report with raw auditable game rows."""

    scenario: dict[str, Any]
    metrics: dict[str, Any]
    games: tuple[GameRecord, ...]
    ablation: str = "base"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "protocol": EVALUATION_PROTOCOL,
            "ablation": self.ablation,
            "scenario": self.scenario,
            "metrics": self.metrics,
            "games": [game.to_dict() for game in self.games],
        }
        payload["runtime_identity"] = evaluation_runtime_identity(
            self.scenario, ablation=self.ablation
        )
        return payload

    def to_promotion_evaluation(self):
        """Bridge directly to the P11 promotion gate's strict P15 contract."""
        from douzero.league.promotion import PromotionEvaluation

        if self.scenario.get("protocol") != EVALUATION_PROTOCOL:
            raise ValueError("promotion requires the official P15 protocol identity")
        if self.scenario["mode"] != PROMOTION_MODE:
            raise ValueError(
                "full_game results are descriptive and cannot be used for "
                "promotion; use a cardplay_only P15 evaluation"
            )
        if self.scenario["confidence_level"] != OFFICIAL_CONFIDENCE_LEVEL:
            raise ValueError(
                f"promotion requires confidence_level={OFFICIAL_CONFIDENCE_LEVEL}"
            )
        if (
            self.scenario.get("statistical_unit") != OFFICIAL_STATISTICAL_UNIT
            or self.scenario.get("ci_method") != OFFICIAL_CI_METHOD
            or self.scenario.get("release_protocol_id") != EVALUATION_PROTOCOL
        ):
            raise ValueError("promotion requires the closed official statistics protocol")
        if (
            self.scenario["bootstrap_samples"]
            < MIN_PROMOTION_BOOTSTRAP_SAMPLES
        ):
            raise ValueError(
                "promotion requires at least "
                f"{MIN_PROMOTION_BOOTSTRAP_SAMPLES} bootstrap samples"
            )
        if self.metrics["paired_estimate_ci"]["paired_deals"] < MIN_PROMOTION_PAIRED_DEALS:
            raise ValueError(
                "promotion requires at least "
                f"{MIN_PROMOTION_PAIRED_DEALS} paired deals"
            )
        expected_permutations = [
            list(row) for row in default_seat_permutations(PROMOTION_MODE)
        ]
        if (
            self.scenario["seat_permutations"] != expected_permutations
            or self.scenario["seat_permutation_hash"]
            != OFFICIAL_PERMUTATION_HASHES[PROMOTION_MODE]
        ):
            raise ValueError("promotion requires the official P15 seat permutations")
        if self.metrics.get("paired_estimator") != PROMOTION_ESTIMATOR:
            raise ValueError(
                f"promotion requires estimator={PROMOTION_ESTIMATOR!r}"
            )

        ci = self.metrics["paired_estimate_ci"]
        return PromotionEvaluation(
            candidate_policy_id=self.scenario["candidate"]["name"],
            incumbent_policy_id=self.scenario["baseline"]["name"],
            paired_games=ci["paired_deals"],
            estimate=ci["estimate"],
            ci_low=ci["low"],
            ci_high=ci["high"],
            evaluator_protocol=EVALUATION_PROTOCOL,
            deal_set_id=self.scenario["deal_set_id"],
            mode=PROMOTION_MODE,
            confidence_level=self.scenario["confidence_level"],
            bootstrap_samples=self.scenario["bootstrap_samples"],
            seat_permutation_hash=self.scenario["seat_permutation_hash"],
            estimator=PROMOTION_ESTIMATOR,
        )


def _candidate_samples(
    agents: dict[str, TimedAgent], candidate_roles: tuple[str, ...], winner_team: str
) -> tuple[tuple[float, ...], tuple[tuple[str, float, float], ...]]:
    latencies: list[float] = []
    calibration: list[tuple[str, float, float]] = []
    for role in candidate_roles:
        agent = agents[role]
        latencies.extend(agent.latencies_ms)
        target = float(
            winner_team == ("landlord" if role == "landlord" else "farmer")
        )
        calibration.extend((role, prediction, target) for prediction in agent.predictions)
    return tuple(latencies), tuple(calibration)


def _candidate_search_counts(
    agents: dict[str, TimedAgent], candidate_roles: tuple[str, ...]
) -> tuple[int, int, int]:
    return (
        sum(agents[role].search_calls for role in candidate_roles),
        sum(agents[role].search_timeouts for role in candidate_roles),
        sum(agents[role].search_fallbacks for role in candidate_roles),
    )


def _run_cardplay_leg(
    scenario: EvaluationScenario,
    factory: BundleFactory,
    deal: dict[str, Any],
    deal_id: str,
    deal_hash: str,
    assignment: tuple[str, str, str],
    leg_index: int,
) -> GameRecord:
    agents: dict[str, TimedAgent] = {}
    for role_index, role in enumerate(ROLES):
        label = assignment[role_index]
        bundle = scenario.candidate if label == "candidate" else scenario.baseline
        agents[role] = factory.build(
            bundle,
            role,
            seed=_seed(scenario.deterministic_seed, deal_id, "role", role),
            bundle_label=label,
        )
    env = GameEnv(agents)
    env.card_play_init(copy.deepcopy(deal))
    while not env.game_over:
        env.step()

    candidate_roles = tuple(
        role for role, label in zip(ROLES, assignment) if label == "candidate"
    )
    candidate_team = "landlord" if candidate_roles == ("landlord",) else "farmer"
    winner_team = env.get_winner()
    winner_position = next(
        role for role in ROLES if not env.info_sets[role].player_hand_cards
    )
    candidate_score = float(env.num_scores[candidate_team])
    candidate_win = float(winner_team == candidate_team)
    role_wins = {role: candidate_win for role in candidate_roles}
    role_scores = {role: candidate_score for role in candidate_roles}
    latencies, calibration = _candidate_samples(agents, candidate_roles, winner_team)
    search_calls, search_timeouts, search_fallbacks = _candidate_search_counts(
        agents, candidate_roles
    )
    return GameRecord(
        deal_id=deal_id,
        leg_id=f"cardplay-{leg_index}",
        mode=scenario.mode,
        assignment=assignment,
        candidate_roles=candidate_roles,
        candidate_win=candidate_win,
        candidate_score=candidate_score,
        role_wins=role_wins,
        role_scores=role_scores,
        winner_team=winner_team,
        winner_position=winner_position,
        bid_value=0,
        candidate_bid_attempts=0,
        candidate_positive_bids=0,
        candidate_landlord=int(candidate_team == "landlord"),
        bomb_count=env.bomb_count,
        rocket_count=env.rocket_count,
        spring=False,
        anti_spring=False,
        game_length=len(env.card_play_action_seq),
        candidate_latencies_ms=latencies,
        calibration=calibration,
        search_calls=search_calls,
        search_timeouts=search_timeouts,
        search_fallbacks=search_fallbacks,
        deal_hash=deal_hash,
    )


def _run_full_game_leg(
    scenario: EvaluationScenario,
    factory: BundleFactory,
    deal: dict[str, Any],
    deal_id: str,
    deal_hash: str,
    assignment: tuple[str, str, str],
    leg_index: int,
) -> GameRecord:
    from douzero.evaluation.legacy_data_adapter import deal_standard_deck

    bid_rng = random.Random(
        _seed(scenario.deterministic_seed, deal_id, "bidding")
    )
    deck = list(deal["deck"])
    bidding_order = list(deal["bidding_order"])
    env = GameEnv({role: RuleAgent() for role in ROLES}, ruleset=scenario.ruleset)
    candidate_bid_attempts = 0
    candidate_positive_bids = 0
    redeal_count = 0
    max_redeals_exceeded = False
    candidate_bid_latencies_ms: list[float] = []

    for _attempt in range(scenario.ruleset.max_redeals + 1):
        env.reset()
        env.card_play_init_standard(
            deal_standard_deck(deck), bidding_order=bidding_order
        )
        redeal = False
        while env.phase == PHASE_BIDDING:
            seat = env.acting_player_position
            label = assignment[int(seat)]
            bundle = scenario.candidate if label == "candidate" else scenario.baseline
            import time

            bid_started = time.perf_counter_ns()
            bid = factory.choose_bid(
                bundle,
                env.get_bidding_obs(),
                env.get_legal_bids(),
                bid_rng,
                redeal_count=redeal_count,
            )
            if label == "candidate":
                candidate_bid_latencies_ms.append(
                    (time.perf_counter_ns() - bid_started) / 1_000_000.0
                )
                candidate_bid_attempts += 1
                candidate_positive_bids += int(bid > 0)
            redeal = env.step_bidding(bid)
            if redeal:
                redeal_count += 1
                # Redeal decks depend only on the paired deal and retry index,
                # not on seat permutation or how many random bids were drawn.
                deck = list(deal["deck"])
                random.Random(
                    _seed(
                        scenario.deterministic_seed,
                        deal_id,
                        "redeal",
                        _attempt,
                    )
                ).shuffle(deck)
                break
        if not redeal:
            break
    else:
        # A deliberately all-pass policy still gets a bounded, deterministic
        # smoke result rather than hanging forever.
        env.reset()
        env.card_play_init_standard(
            deal_standard_deck(deck), bidding_order=bidding_order
        )
        env.landlord_position = bidding_order[0]
        env.bid_value = 1
        env.bidding_history = [(bidding_order[0], 1)]
        env._reveal_bottom_cards()
        max_redeals_exceeded = True

    role_to_seat = {role: seat for seat, role in env._seat_to_role.items()}
    agents: dict[str, TimedAgent] = {}
    for role in ROLES:
        seat = role_to_seat[role]
        label = assignment[int(seat)]
        bundle = scenario.candidate if label == "candidate" else scenario.baseline
        agents[role] = factory.build(
            bundle,
            role,
            seed=_seed(scenario.deterministic_seed, deal_id, "seat", seat),
            bundle_label=label,
        )
    env.players = agents
    while not env.game_over:
        env.step()

    result = env.game_result
    assert result is not None
    candidate_roles = tuple(
        role for role in ROLES if assignment[int(role_to_seat[role])] == "candidate"
    )
    # Full-game permutations contain exactly one candidate seat, so its
    # current role unambiguously determines the team-perspective result.
    candidate_role = candidate_roles[0]
    candidate_team = "landlord" if candidate_role == "landlord" else "farmer"
    candidate_win = float(result.winner_team == candidate_team)
    candidate_score = float(
        result.landlord_score if candidate_team == "landlord" else result.farmer_score
    )
    cardplay_latencies, calibration = _candidate_samples(
        agents, candidate_roles, result.winner_team
    )
    latencies = (*candidate_bid_latencies_ms, *cardplay_latencies)
    search_calls, search_timeouts, search_fallbacks = _candidate_search_counts(
        agents, candidate_roles
    )
    return GameRecord(
        deal_id=deal_id,
        leg_id=f"full-game-{leg_index}",
        mode=scenario.mode,
        assignment=assignment,
        candidate_roles=candidate_roles,
        candidate_win=candidate_win,
        candidate_score=candidate_score,
        role_wins={candidate_role: candidate_win},
        role_scores={candidate_role: candidate_score},
        winner_team=result.winner_team,
        winner_position=result.winner_position,
        bid_value=result.bid_value,
        candidate_bid_attempts=candidate_bid_attempts,
        candidate_positive_bids=candidate_positive_bids,
        candidate_landlord=int(candidate_role == "landlord"),
        bomb_count=result.bomb_count,
        rocket_count=result.rocket_count,
        spring=result.spring,
        anti_spring=result.anti_spring,
        game_length=len(env.card_play_action_seq),
        redeal_count=redeal_count,
        max_redeals_exceeded=max_redeals_exceeded,
        candidate_latencies_ms=latencies,
        calibration=calibration,
        search_calls=search_calls,
        search_timeouts=search_timeouts,
        search_fallbacks=search_fallbacks,
        bidding_inference_calls=len(candidate_bid_latencies_ms),
        formal_evaluation_eligible=not max_redeals_exceeded,
        exclusion_reason=(
            "redeal_cap_exhausted_forced_smoke_fallback"
            if max_redeals_exceeded else ""
        ),
        deal_hash=deal_hash,
        seat_to_role=dict(env._seat_to_role),
        bidding_order=tuple(env.bidding_order),
        bidding_history=tuple(
            (str(seat), int(bid)) for seat, bid in env.bidding_history
        ),
    )


def _calibration_metrics(samples: list[tuple[str, float, float]]) -> dict[str, Any]:
    def calculate(rows: list[tuple[str, float, float]]) -> dict[str, float | int]:
        if not rows:
            return {"count": 0, "brier": float("nan"), "nll": float("nan"), "ece": float("nan")}
        eps = 1e-7
        brier = sum((p - y) ** 2 for _, p, y in rows) / len(rows)
        nll = -sum(
            y * math.log(min(max(p, eps), 1 - eps))
            + (1 - y) * math.log(1 - min(max(p, eps), 1 - eps))
            for _, p, y in rows
        ) / len(rows)
        ece = 0.0
        for index in range(15):
            low, high = index / 15, (index + 1) / 15
            bucket = [
                row for row in rows
                if low <= row[1] < high or (index == 14 and row[1] == 1.0)
            ]
            if bucket:
                confidence = sum(row[1] for row in bucket) / len(bucket)
                accuracy = sum(row[2] for row in bucket) / len(bucket)
                ece += len(bucket) / len(rows) * abs(confidence - accuracy)
        return {"count": len(rows), "brier": brier, "nll": nll, "ece": ece}

    return {
        "overall": calculate(samples),
        "by_role": {
            role: calculate([sample for sample in samples if sample[0] == role])
            for role in ROLES
        },
    }


def _aggregate(scenario: EvaluationScenario, games: list[GameRecord]) -> dict[str, Any]:
    audit_games = tuple(games)
    games_by_deal: dict[str, list[GameRecord]] = {}
    for game in audit_games:
        games_by_deal.setdefault(game.deal_id, []).append(game)
    excluded_deal_ids = {
        deal_id
        for deal_id, rows in games_by_deal.items()
        if any(not row.formal_evaluation_eligible for row in rows)
    }
    # A deal is the statistical unit. If any seat rotation used a forced smoke
    # fallback, exclude every rotation of that deal from formal metrics.
    games = [
        game for game in audit_games if game.deal_id not in excluded_deal_ids
    ]

    win_by_deal = deal_cluster_means(
        (game.deal_id, game.candidate_win - 0.5) for game in games
    )
    score_by_deal = deal_cluster_means(
        (game.deal_id, game.candidate_score) for game in games
    )
    def confidence_interval(values: dict[str, float], seed: int) -> ConfidenceInterval:
        if values:
            return paired_bootstrap_ci(
                values,
                confidence_level=scenario.confidence_level,
                samples=scenario.bootstrap_samples,
                seed=seed,
            )
        return ConfidenceInterval(
            estimate=float("nan"),
            low=float("nan"),
            high=float("nan"),
            confidence_level=scenario.confidence_level,
            paired_deals=0,
            bootstrap_samples=scenario.bootstrap_samples,
        )

    win_ci = confidence_interval(win_by_deal, scenario.deterministic_seed)
    score_ci = confidence_interval(score_by_deal, scenario.deterministic_seed + 1)
    paired_estimator = (
        PROMOTION_ESTIMATOR
        if scenario.mode == PROMOTION_MODE
        else "full_game_zero_sum_seat_score"
    )
    paired_estimate_ci = win_ci if scenario.mode == PROMOTION_MODE else score_ci
    by_role: dict[str, Any] = {}
    for role_index, role in enumerate(ROLES):
        role_games = [game for game in games if role in game.role_wins]
        if not role_games:
            by_role[role] = {"games": 0}
            continue
        role_ci = paired_bootstrap_ci(
            deal_cluster_means(
                (game.deal_id, game.role_wins[role] - 0.5) for game in role_games
            ),
            confidence_level=scenario.confidence_level,
            samples=scenario.bootstrap_samples,
            seed=scenario.deterministic_seed + 10 + role_index,
        )
        by_role[role] = {
            "games": len(role_games),
            "win_percentage": sum(game.role_wins[role] for game in role_games) / len(role_games),
            "mean_score": sum(game.role_scores[role] for game in role_games) / len(role_games),
            "win_rate_delta_ci": role_ci.to_dict(),
        }

    latencies = [latency for game in games for latency in game.candidate_latencies_ms]
    inference_seconds = sum(latencies) / 1000.0
    calibration = [sample for game in games for sample in game.calibration]
    total_bids = sum(game.candidate_bid_attempts for game in games)
    total_searches = sum(game.search_calls for game in games)
    denominator = len(games)

    def mean(values) -> float:
        rows = list(values)
        return sum(rows) / len(rows) if rows else float("nan")

    inference_calls_per_second = (
        len(latencies) / inference_seconds
        if inference_seconds > 0 else float("nan")
    )
    by_bid_value = {}
    if scenario.mode == "full_game":
        for bid_value in (1, 2, 3):
            bid_games = [game for game in games if game.bid_value == bid_value]
            by_bid_value[str(bid_value)] = {
                "games": len(bid_games),
                "win_percentage": (
                    sum(game.candidate_win for game in bid_games) / len(bid_games)
                    if bid_games else float("nan")
                ),
                "mean_score": (
                    sum(game.candidate_score for game in bid_games) / len(bid_games)
                    if bid_games else float("nan")
                ),
            }
    excluded_games = [
        game for game in audit_games if game.deal_id in excluded_deal_ids
    ]
    exclusion_reasons: dict[str, int] = {}
    for game in excluded_games:
        reason = game.exclusion_reason or "formal_evaluation_ineligible"
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
    excluded_bid_attempts = sum(
        game.candidate_bid_attempts for game in excluded_games
    )
    smoke_descriptive = {
        "status": "present" if excluded_games else "not_present",
        "games": len(excluded_games),
        "bid_rate": (
            sum(game.candidate_positive_bids for game in excluded_games)
            / excluded_bid_attempts
            if excluded_bid_attempts else float("nan")
        ),
        "mean_score": mean(game.candidate_score for game in excluded_games),
        "winner_team_counts": {
            team: sum(game.winner_team == team for game in excluded_games)
            for team in ("landlord", "farmer")
        },
    }
    return {
        "sample_counts": {
            "deals": len(win_by_deal),
            "requested_deals": len(scenario.deals),
            "games": len(games),
            "games_total": len(audit_games),
            "excluded_games": len(excluded_games),
            "seat_permutations": len(scenario.seat_permutations),
            "calibration_decisions": len(calibration),
            "inference_calls": len(latencies),
            "bidding_inference_calls": sum(
                game.bidding_inference_calls for game in games
            ),
        },
        "formal_evaluation": {
            "status": (
                "contains_excluded_smoke_fallbacks"
                if excluded_games else "eligible_input"
            ),
            "eligible_deals": len(win_by_deal),
            "eligible_games": len(games),
            "excluded_deals": len(excluded_deal_ids),
            "excluded_games": len(excluded_games),
            "exclusion_reasons": exclusion_reasons,
        },
        "smoke_descriptive": smoke_descriptive,
        "overall_win_percentage": mean(game.candidate_win for game in games),
        "team_win_percentage": {
            team: (
                sum(game.candidate_win for game in team_games) / len(team_games)
                if team_games else float("nan")
            )
            for team, team_games in (
                ("landlord", [g for g in games if "landlord" in g.candidate_roles]),
                ("farmer", [g for g in games if any(r != "landlord" for r in g.candidate_roles)]),
            )
        },
        "paired_estimator": paired_estimator,
        "paired_estimate_ci": paired_estimate_ci.to_dict(),
        "paired_win_rate_delta_ci": (
            win_ci.to_dict() if scenario.mode == PROMOTION_MODE else None
        ),
        "descriptive_win_rate_delta_ci": win_ci.to_dict(),
        "mean_score": mean(game.candidate_score for game in games),
        "mean_raw_score": mean(game.candidate_score for game in games),
        "mean_log_score": mean(game.candidate_log_score for game in games),
        "zero_sum_seat_score": (
            score_ci.estimate if scenario.mode == "full_game" else float("nan")
        ),
        "paired_mean_score_ci": score_ci.to_dict(),
        "by_role": by_role,
        "bid_rate": (
            sum(game.candidate_positive_bids for game in games) / total_bids
            if total_bids else float("nan")
        ),
        "landlord_acquisition_rate": (
            sum(game.candidate_landlord for game in games) / denominator
            if scenario.mode == "full_game" and denominator else float("nan")
        ),
        "by_bid_value": by_bid_value,
        "redeals": {
            "total": sum(game.redeal_count for game in audit_games),
            "max_redeals_exceeded_games": sum(
                game.max_redeals_exceeded for game in audit_games
            ),
        },
        "bomb_rate": mean(game.bomb_count > 0 for game in games),
        "rocket_rate": mean(game.rocket_count > 0 for game in games),
        "spring_rate": mean(game.spring for game in games),
        "anti_spring_rate": mean(game.anti_spring for game in games),
        "mean_game_length": mean(game.game_length for game in games),
        "calibration": _calibration_metrics(calibration),
        "inference_latency_ms": {
            "count": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "inference_calls_per_second": inference_calls_per_second,
        # P15 exposed this key even though it counts instrumented inference
        # calls rather than actor-loop frames. Keep it as a deprecated schema
        # alias so existing result consumers do not break.
        "actor_fps": inference_calls_per_second,
        "search": {
            "calls": total_searches,
            "timeout_rate": (
                sum(game.search_timeouts for game in games) / total_searches
                if total_searches else float("nan")
            ),
            "fallback_rate": (
                sum(game.search_fallbacks for game in games) / total_searches
                if total_searches else float("nan")
            ),
        },
        "belief": {
            "status": "not_instrumented" if any(
                bundle.belief_checkpoint
                for bundle in (scenario.candidate, scenario.baseline)
            ) else "not_enabled",
            "exact_decode_rate": float("nan"),
            "conservation_violation_rate": float("nan"),
        },
    }


def evaluate_scenario(
    scenario: EvaluationScenario, *, ablation: str = "base"
) -> PairedEvaluationResult:
    """Run every deal/permutation sequentially with deterministic local RNGs."""
    factory = BundleFactory(scenario.ruleset)
    games: list[GameRecord] = []
    for deal_index, deal in enumerate(scenario.deals):
        deal_hash = canonical_deal_hash(deal)
        deal_id = _deal_id(deal_index, deal)
        for leg_index, assignment in enumerate(scenario.seat_permutations):
            if scenario.mode == "cardplay_only":
                game = _run_cardplay_leg(
                    scenario,
                    factory,
                    deal,
                    deal_id,
                    deal_hash,
                    assignment,
                    leg_index,
                )
            else:
                game = _run_full_game_leg(
                    scenario,
                    factory,
                    deal,
                    deal_id,
                    deal_hash,
                    assignment,
                    leg_index,
                )
            games.append(game)
    return PairedEvaluationResult(
        scenario=scenario.to_dict(),
        metrics=_aggregate(scenario, games),
        games=tuple(games),
        ablation=ablation,
    )
