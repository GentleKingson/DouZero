"""Paired card-play and seat-rotated full-game evaluation (P15)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any

from douzero.env.game import GameEnv
from douzero.env.rules import PHASE_BIDDING

from .agents import BundleFactory, RuleAgent, TimedAgent, choose_bid
from .scenario import EVALUATION_PROTOCOL, ROLES, EvaluationScenario
from .statistics import deal_cluster_means, paired_bootstrap_ci, percentile


def _seed(base: int, *parts: object) -> int:
    token = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def _deal_id(index: int, deal: dict[str, Any]) -> str:
    payload = json.dumps(deal, sort_keys=True, separators=(",", ":"))
    return f"{index:06d}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


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
            "candidate_latencies_ms": list(self.candidate_latencies_ms),
            "calibration": [list(sample) for sample in self.calibration],
        }


@dataclass(frozen=True)
class PairedEvaluationResult:
    """JSON-ready P15 report with raw auditable game rows."""

    scenario: dict[str, Any]
    metrics: dict[str, Any]
    games: tuple[GameRecord, ...]
    ablation: str = "base"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": EVALUATION_PROTOCOL,
            "ablation": self.ablation,
            "scenario": self.scenario,
            "metrics": self.metrics,
            "games": [game.to_dict() for game in self.games],
        }

    def to_promotion_evaluation(self):
        """Bridge directly to the P11 promotion gate's strict P15 contract."""
        from douzero.league.promotion import PromotionEvaluation

        ci = self.metrics["paired_win_rate_delta_ci"]
        return PromotionEvaluation(
            candidate_policy_id=self.scenario["candidate"]["name"],
            incumbent_policy_id=self.scenario["baseline"]["name"],
            paired_games=ci["paired_deals"],
            estimate=ci["estimate"],
            ci_low=ci["low"],
            ci_high=ci["high"],
            evaluator_protocol=EVALUATION_PROTOCOL,
            deal_set_id=self.scenario["deal_set_id"],
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


def _run_cardplay_leg(
    scenario: EvaluationScenario,
    factory: BundleFactory,
    deal: dict[str, Any],
    deal_id: str,
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
            seed=_seed(scenario.deterministic_seed, deal_id, leg_index, role),
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
    )


def _run_full_game_leg(
    scenario: EvaluationScenario,
    factory: BundleFactory,
    deal: dict[str, Any],
    deal_id: str,
    assignment: tuple[str, str, str],
    leg_index: int,
) -> GameRecord:
    from douzero.evaluation.legacy_data_adapter import deal_standard_deck

    bid_rng = random.Random(
        _seed(scenario.deterministic_seed, deal_id, leg_index, "bidding")
    )
    deck = list(deal["deck"])
    bidding_order = list(deal["bidding_order"])
    env = GameEnv({role: RuleAgent() for role in ROLES}, ruleset=scenario.ruleset)
    candidate_bid_attempts = 0
    candidate_positive_bids = 0

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
            bid = choose_bid(
                bundle, env.get_bidding_obs(), env.get_legal_bids(), bid_rng
            )
            if label == "candidate":
                candidate_bid_attempts += 1
                candidate_positive_bids += int(bid > 0)
            redeal = env.step_bidding(bid)
            if redeal:
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

    role_to_seat = {role: seat for seat, role in env._seat_to_role.items()}
    agents: dict[str, TimedAgent] = {}
    for role in ROLES:
        seat = role_to_seat[role]
        label = assignment[int(seat)]
        bundle = scenario.candidate if label == "candidate" else scenario.baseline
        agents[role] = factory.build(
            bundle,
            role,
            seed=_seed(scenario.deterministic_seed, deal_id, leg_index, role),
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
    latencies, calibration = _candidate_samples(
        agents, candidate_roles, result.winner_team
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
        candidate_latencies_ms=latencies,
        calibration=calibration,
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
    win_by_deal = deal_cluster_means(
        (game.deal_id, game.candidate_win - 0.5) for game in games
    )
    score_by_deal = deal_cluster_means(
        (game.deal_id, game.candidate_score) for game in games
    )
    win_ci = paired_bootstrap_ci(
        win_by_deal,
        confidence_level=scenario.confidence_level,
        samples=scenario.bootstrap_samples,
        seed=scenario.deterministic_seed,
    )
    score_ci = paired_bootstrap_ci(
        score_by_deal,
        confidence_level=scenario.confidence_level,
        samples=scenario.bootstrap_samples,
        seed=scenario.deterministic_seed + 1,
    )
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
    denominator = len(games)
    return {
        "sample_counts": {
            "deals": len(scenario.deals),
            "games": len(games),
            "seat_permutations": len(scenario.seat_permutations),
            "calibration_decisions": len(calibration),
            "inference_calls": len(latencies),
        },
        "overall_win_percentage": sum(game.candidate_win for game in games) / denominator,
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
        "paired_win_rate_delta_ci": win_ci.to_dict(),
        "mean_score": sum(game.candidate_score for game in games) / denominator,
        "mean_log_score": sum(game.candidate_log_score for game in games) / denominator,
        "paired_mean_score_ci": score_ci.to_dict(),
        "by_role": by_role,
        "bid_rate": (
            sum(game.candidate_positive_bids for game in games) / total_bids
            if total_bids else float("nan")
        ),
        "landlord_acquisition_rate": (
            sum(game.candidate_landlord for game in games) / denominator
            if scenario.mode == "full_game" else float("nan")
        ),
        "bomb_rate": sum(game.bomb_count > 0 for game in games) / denominator,
        "rocket_rate": sum(game.rocket_count > 0 for game in games) / denominator,
        "spring_rate": sum(game.spring for game in games) / denominator,
        "anti_spring_rate": sum(game.anti_spring for game in games) / denominator,
        "mean_game_length": sum(game.game_length for game in games) / denominator,
        "calibration": _calibration_metrics(calibration),
        "inference_latency_ms": {
            "count": len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "actor_fps": len(latencies) / inference_seconds if inference_seconds > 0 else float("nan"),
    }


def evaluate_scenario(
    scenario: EvaluationScenario, *, ablation: str = "base"
) -> PairedEvaluationResult:
    """Run every deal/permutation sequentially with deterministic local RNGs."""
    factory = BundleFactory(scenario.ruleset)
    games: list[GameRecord] = []
    for deal_index, deal in enumerate(scenario.deals):
        deal_id = _deal_id(deal_index, deal)
        for leg_index, assignment in enumerate(scenario.seat_permutations):
            if scenario.mode == "cardplay_only":
                game = _run_cardplay_leg(
                    scenario, factory, deal, deal_id, assignment, leg_index
                )
            else:
                game = _run_full_game_leg(
                    scenario, factory, deal, deal_id, assignment, leg_index
                )
            games.append(game)
    return PairedEvaluationResult(
        scenario=scenario.to_dict(),
        metrics=_aggregate(scenario, games),
        games=tuple(games),
        ablation=ablation,
    )
