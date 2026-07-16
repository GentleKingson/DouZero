"""Independent tests for deterministic P17 deal-and-trace replay."""

from __future__ import annotations

import copy
import hashlib
import random

import pytest

from douzero.env.game import GameEnv
from douzero.env.rules import PHASE_BIDDING, RuleSet
from douzero.evaluation.agents import RuleAgent
from douzero.evaluation.legacy_data_adapter import deal_standard_deck
from douzero.evaluation.replay import (
    ReplayValidationError,
    evaluation_trace_digest,
    replay_game_record,
)
from douzero.evaluation.scenario import canonical_deal_hash, canonical_deal_id
from evaluate_paired import generate_deals


ROLES = ("landlord", "landlord_up", "landlord_down")


class _RecordingRuleAgent:
    def __init__(self, trace: list[list[object]]) -> None:
        self.trace = trace
        self.rule = RuleAgent()

    def act(self, infoset):
        action = self.rule.act(infoset)
        self.trace.append([infoset.player_position, list(action)])
        return action


def _paired_seed(base: int, *parts: object) -> int:
    token = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def _finish_with_rule_agents(env: GameEnv) -> list[list[object]]:
    trace: list[list[object]] = []
    env.players = {role: _RecordingRuleAgent(trace) for role in ROLES}
    while not env.game_over:
        env.step()
    return trace


def _row(
    mode: str,
    deal: dict,
    bidding_trace: list,
    cardplay_trace: list,
) -> dict:
    deal_hash = canonical_deal_hash(deal)
    row = {
        "mode": mode,
        "deal_id": canonical_deal_id(0, deal_hash),
        "deal_hash": deal_hash,
        "bidding_trace": bidding_trace,
        "cardplay_trace": cardplay_trace,
    }
    row["trace_digest"] = evaluation_trace_digest(
        mode=mode,
        deal_hash=deal_hash,
        bidding_trace=bidding_trace,
        cardplay_trace=cardplay_trace,
    )
    return row


def _legacy_record(seed: int = 2718):
    ruleset = RuleSet.legacy()
    deal = generate_deals("cardplay_only", 1, seed, ruleset)[0]
    env = GameEnv({role: RuleAgent() for role in ROLES})
    env.card_play_init(copy.deepcopy(deal))
    trace = _finish_with_rule_agents(env)
    return deal, ruleset, env, _row("cardplay_only", deal, [], trace)


def _bid_successfully(env: GameEnv) -> list[list[object]]:
    attempt: list[list[object]] = []
    bids = (1, 0, 0)
    for bid in bids:
        assert env.phase == PHASE_BIDDING
        seat = env.acting_player_position
        attempt.append([seat, bid])
        assert env.step_bidding(bid) is False
    return attempt


def _full_game_record(seed: int = 3141):
    ruleset = RuleSet.standard()
    deal = generate_deals("full_game", 1, seed, ruleset)[0]
    env = GameEnv({role: RuleAgent() for role in ROLES}, ruleset=ruleset)
    env.card_play_init_standard(
        deal_standard_deck(deal["deck"]), bidding_order=deal["bidding_order"]
    )
    attempt = _bid_successfully(env)
    trace = _finish_with_rule_agents(env)
    return deal, ruleset, env, _row("full_game", deal, [attempt], trace)


def _full_game_record_after_redeal(seed: int = 1618):
    ruleset = RuleSet.standard()
    deal = generate_deals("full_game", 1, seed, ruleset)[0]
    deal_hash = canonical_deal_hash(deal)
    deal_id = canonical_deal_id(0, deal_hash)
    env = GameEnv({role: RuleAgent() for role in ROLES}, ruleset=ruleset)
    env.card_play_init_standard(
        deal_standard_deck(deal["deck"]), bidding_order=deal["bidding_order"]
    )

    all_pass: list[list[object]] = []
    for _ in range(3):
        seat = env.acting_player_position
        all_pass.append([seat, 0])
        redeal = env.step_bidding(0)
    assert redeal is True

    deck = list(deal["deck"])
    random.Random(_paired_seed(seed, deal_id, "redeal", 0)).shuffle(deck)
    env.reset()
    env.card_play_init_standard(
        deal_standard_deck(deck), bidding_order=deal["bidding_order"]
    )
    success = _bid_successfully(env)
    trace = _finish_with_rule_agents(env)
    return deal, ruleset, env, _row("full_game", deal, [all_pass, success], trace)


def _refresh_digest(row: dict) -> None:
    row["trace_digest"] = evaluation_trace_digest(
        mode=row["mode"],
        deal_hash=row["deal_hash"],
        bidding_trace=row["bidding_trace"],
        cardplay_trace=row["cardplay_trace"],
    )


def test_replays_legacy_trace_and_derives_terminal_facts_from_game_env():
    deal, ruleset, env, row = _legacy_record()
    row.update({
        "winner_position": "forged",
        "winner_team": "forged",
        "bomb_count": 999,
        "rocket_count": 999,
        "spring": True,
        "anti_spring": True,
        "team_scores": {"landlord": 999.0, "farmer": 999.0},
    })

    outcome = replay_game_record(
        row, deal, mode="cardplay_only", ruleset=ruleset, deterministic_seed=2718
    )

    winner_position = next(
        role for role in ROLES if not env.info_sets[role].player_hand_cards
    )
    assert outcome.winner_position == winner_position
    assert outcome.winner_team == env.get_winner()
    assert outcome.game_length == len(row["cardplay_trace"])
    assert outcome.bomb_count == env.bomb_count
    assert outcome.rocket_count == env.rocket_count
    assert outcome.team_scores == {
        "landlord": float(env.num_scores["landlord"]),
        "farmer": float(env.num_scores["farmer"]),
    }
    assert outcome.seat_to_role is None
    assert outcome.bidding_order == ()
    assert outcome.bidding_history == ()
    assert outcome.redeal_count == 0


def test_replays_full_game_and_returns_bidding_roles_and_standard_scores():
    deal, ruleset, env, row = _full_game_record()
    outcome = replay_game_record(
        row, deal, mode="full_game", ruleset=ruleset, deterministic_seed=3141
    )
    result = env.game_result
    assert result is not None
    assert outcome.winner_position == result.winner_position
    assert outcome.winner_team == result.winner_team
    assert outcome.bid_value == result.bid_value
    assert outcome.spring == result.spring
    assert outcome.anti_spring == result.anti_spring
    assert outcome.seat_to_role == env._seat_to_role
    assert outcome.bidding_order == tuple(env.bidding_order)
    assert outcome.bidding_history == tuple(env.bidding_history)
    assert outcome.team_scores == {
        "landlord": float(result.landlord_score),
        "farmer": float(result.farmer_score),
    }


def test_replays_every_redeal_with_paired_deterministic_shuffle_semantics():
    deal, ruleset, env, row = _full_game_record_after_redeal()
    outcome = replay_game_record(
        row, deal, mode="full_game", ruleset=ruleset, deterministic_seed=1618
    )
    assert outcome.redeal_count == 1
    assert outcome.winner_position == env.game_result.winner_position

    with pytest.raises(ReplayValidationError):
        replay_game_record(
            row,
            deal,
            mode="full_game",
            ruleset=ruleset,
            deterministic_seed=1619,
        )


def test_rejects_trace_digest_tampering_before_replay():
    deal, ruleset, _env, row = _legacy_record()
    row["cardplay_trace"][0][1] = []
    with pytest.raises(ReplayValidationError, match="trace_digest"):
        replay_game_record(
            row, deal, mode="cardplay_only", ruleset=ruleset, deterministic_seed=2718
        )


@pytest.mark.parametrize("mutation", ["turn", "truncated", "extra"])
def test_rejects_coordinated_illegal_or_nonterminal_cardplay_trace(mutation):
    deal, ruleset, _env, row = _legacy_record()
    if mutation == "turn":
        row["cardplay_trace"][0][0] = "landlord_up"
    elif mutation == "truncated":
        row["cardplay_trace"].pop()
    else:
        row["cardplay_trace"].append(["landlord_down", []])
    _refresh_digest(row)
    with pytest.raises(ReplayValidationError):
        replay_game_record(
            row, deal, mode="cardplay_only", ruleset=ruleset, deterministic_seed=2718
        )


def test_rejects_coordinated_illegal_bidding_trace():
    deal, ruleset, _env, row = _full_game_record()
    row["bidding_trace"][0][0][1] = 3
    _refresh_digest(row)
    with pytest.raises(ReplayValidationError, match="continues after resolution"):
        replay_game_record(
            row, deal, mode="full_game", ruleset=ruleset, deterministic_seed=3141
        )


def test_rejects_trace_bound_to_a_different_approved_deal():
    _deal, ruleset, _env, row = _legacy_record()
    other = generate_deals("cardplay_only", 1, 2719, ruleset)[0]
    with pytest.raises(ReplayValidationError, match="approved deal"):
        replay_game_record(
            row, other, mode="cardplay_only", ruleset=ruleset, deterministic_seed=2718
        )


def test_trace_digest_is_versioned_deterministic_and_order_sensitive():
    deal, _ruleset, _env, row = _legacy_record()
    digest = evaluation_trace_digest(
        mode="cardplay_only",
        deal_hash=canonical_deal_hash(deal),
        bidding_trace=(),
        cardplay_trace=tuple(
            (role, tuple(cards)) for role, cards in row["cardplay_trace"]
        ),
    )
    assert digest == row["trace_digest"]
    reordered = list(reversed(row["cardplay_trace"]))
    assert digest != evaluation_trace_digest(
        mode="cardplay_only",
        deal_hash=canonical_deal_hash(deal),
        bidding_trace=[],
        cardplay_trace=reordered,
    )
