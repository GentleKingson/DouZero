"""Deterministic Standard V2 R1 corpus and single-process reference traces."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from douzero.coach.records import CANONICAL_DECK, OpeningRecord
from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.training.standard_v2_contract import (
    STANDARD_V2_R1_CONFIG_HASH,
    STANDARD_V2_R1_CONTRACT_VERSION,
    STANDARD_V2_REFERENCE_SCHEMA_VERSION,
    stable_identity_hash,
    standard_v2_version_contract,
)


def _rotated_deck(offset: int, *, reverse: bool = False) -> tuple[int, ...]:
    cards = list(reversed(CANONICAL_DECK)) if reverse else list(CANONICAL_DECK)
    offset %= len(cards)
    return tuple(cards[offset:] + cards[:offset])


def _spring_deck() -> tuple[int, ...]:
    # Landlord receives five consecutive triples plus five single wings and
    # can legally empty all 20 cards on the opening lead.
    landlord_cards = [
        *(rank for rank in range(3, 8) for _ in range(3)),
        8,
        9,
        10,
        11,
        12,
    ]
    remaining = list(CANONICAL_DECK)
    for card in landlord_cards:
        remaining.remove(card)
    return tuple(landlord_cards[:17] + remaining + landlord_cards[17:])


def _bid(bid: int, source_policy: str) -> dict[str, Any]:
    return {"bid": bid, "source_policy": source_policy}


def build_standard_v2_corpus() -> list[dict[str, Any]]:
    """Return the complete fixed-deck R1 regression input set."""

    standard = RuleSet.standard()
    capped = replace(standard, max_redeals=1)
    specifications = (
        (
            "first_bidder_0_spring",
            _spring_deck(),
            ("0", "1", "2"),
            ((_bid(3, "learned"),),),
            standard,
            101,
        ),
        (
            "first_bidder_1_normal",
            _rotated_deck(17),
            ("1", "2", "0"),
            ((
                _bid(1, "rule"),
                _bid(2, "learned"),
                _bid(0, "epsilon_random"),
            ),),
            standard,
            202,
        ),
        (
            "first_bidder_2_normal",
            _rotated_deck(0, reverse=True),
            ("2", "0", "1"),
            ((
                _bid(0, "epsilon_random"),
                _bid(1, "rule"),
                _bid(2, "learned"),
            ),),
            standard,
            303,
        ),
        (
            "one_redeal_then_max_bid",
            _rotated_deck(31),
            ("0", "1", "2"),
            (
                (
                    _bid(0, "epsilon_random"),
                    _bid(0, "epsilon_random"),
                    _bid(0, "epsilon_random"),
                ),
                (_bid(3, "learned"),),
            ),
            standard,
            404,
        ),
        (
            "max_redeal_guard",
            _rotated_deck(9, reverse=True),
            ("1", "2", "0"),
            (
                (
                    _bid(0, "epsilon_random"),
                    _bid(0, "epsilon_random"),
                    _bid(0, "epsilon_random"),
                ),
                (
                    _bid(0, "epsilon_random"),
                    _bid(0, "epsilon_random"),
                    _bid(0, "epsilon_random"),
                ),
            ),
            capped,
            505,
        ),
    )

    corpus: list[dict[str, Any]] = []
    for scenario_id, deck, order, bid_attempts, ruleset, numpy_seed in specifications:
        opening = OpeningRecord(
            deck=deck,
            bidding_order=order,
            ruleset=ruleset.to_dict(),
            landlord_candidate=order[0],
            public_features={
                "first_bidder": order[0],
                "ruleset_hash": ruleset.stable_hash(),
            },
        )
        corpus.append({
            "scenario_id": scenario_id,
            "opening": opening.to_dict(),
            "bid_attempts": [
                [dict(instruction) for instruction in attempt]
                for attempt in bid_attempts
            ],
            "numpy_seed": numpy_seed,
            "cardplay_policy": "rocket_then_bomb_then_longest_v1",
        })
    return corpus


def _select_cardplay_action(legal_actions: list[list[int]]) -> list[int]:
    non_pass = [list(action) for action in legal_actions if action]
    if not non_pass:
        return []

    def priority(action: list[int]) -> tuple[int, int, int, tuple[int, ...]]:
        rocket = int(sorted(action) == [20, 30])
        bomb = int(len(action) == 4 and len(set(action)) == 1)
        return rocket, bomb, len(action), tuple(sorted(action))

    return max(non_pass, key=priority)


def _run_reference_scenario(specification: dict[str, Any]) -> dict[str, Any]:
    opening = OpeningRecord.from_dict(specification["opening"])
    ruleset = opening.ruleset_obj
    env = Env(objective="adp", ruleset=ruleset)
    prior_numpy_state = np.random.get_state()
    np.random.seed(int(specification["numpy_seed"]))
    try:
        env.reset(opening=opening)
        attempts: list[list[dict[str, Any]]] = []
        current_attempt: list[dict[str, Any]] = []
        attempt_index = 0
        abandoned_bids = 0
        max_redeals_exceeded = False

        while env.bidding_obs is not None:
            configured_attempts = specification["bid_attempts"]
            if attempt_index >= len(configured_attempts):
                raise RuntimeError("fixed corpus exhausted its bidding attempts")
            configured_attempt = configured_attempts[attempt_index]
            if len(current_attempt) >= len(configured_attempt):
                raise RuntimeError("fixed bidding attempt has too few actions")
            instruction = configured_attempt[len(current_attempt)]
            seat = str(env.bidding_obs["position"])
            bid_value = int(instruction["bid"])
            legal_bids = tuple(int(value) for value in env.bidding_obs["legal_bids"])
            if bid_value not in legal_bids:
                raise RuntimeError(
                    f"fixed corpus selected illegal bid {bid_value}; legal={legal_bids}"
                )
            current_attempt.append({
                "seat": seat,
                "bid": bid_value,
                "source_policy": str(instruction["source_policy"]),
                "legal_bids": list(legal_bids),
                "redeal_count": int(env._redeal_count),
            })
            _obs, _reward, done, info = env.step(None, bid_value=bid_value)
            if done and info.get("redeal"):
                attempts.append(current_attempt)
                abandoned_bids += len(current_attempt)
                current_attempt = []
                attempt_index += 1
                env.redeal()
                continue
            if info.get("max_redeals_exceeded"):
                attempts.append(current_attempt)
                abandoned_bids += len(current_attempt)
                current_attempt = []
                max_redeals_exceeded = True
            if done:
                raise RuntimeError("bidding unexpectedly terminated card play")
            if env.bidding_obs is None and current_attempt:
                attempts.append(current_attempt)

        seat_to_role = {
            str(seat): str(role)
            for seat, role in sorted(env._env._seat_to_role.items())
        }
        retained_bids = 0 if max_redeals_exceeded else len(attempts[-1])
        cardplay_trace: list[dict[str, Any]] = []
        play_decisions = 0
        for _step in range(600):
            position = str(env._acting_player_position)
            legal_actions = env.infoset.legal_actions
            action = _select_cardplay_action(legal_actions)
            if len(legal_actions) > 1:
                play_decisions += 1
            cardplay_trace.append({
                "position": position,
                "action": [int(card) for card in sorted(action)],
                "legal_action_count": len(legal_actions),
            })
            _obs, _reward, done, info = env.step(action)
            if done:
                terminal = info
                break
        else:
            raise RuntimeError("fixed Standard V2 reference game did not terminate")
    finally:
        np.random.set_state(prior_numpy_state)

    excluded = bool(max_redeals_exceeded)
    return {
        "scenario_id": specification["scenario_id"],
        "opening": specification["opening"],
        "numpy_seed": specification["numpy_seed"],
        "cardplay_policy": specification["cardplay_policy"],
        "bidding_attempts": attempts,
        "seat_to_role": seat_to_role,
        "cardplay_trace": cardplay_trace,
        "terminal": terminal,
        "excluded_from_training": excluded,
        "exclusion_reason": "redeal_cap_guard" if excluded else "",
        "counts": {
            "bidding_decisions": sum(len(attempt) for attempt in attempts),
            "bid_transitions": 0 if excluded else retained_bids,
            "abandoned_bidding_transitions": abandoned_bids,
            "cardplay_actions": len(cardplay_trace),
            "cardplay_decisions": play_decisions,
            "play_transitions": 0 if excluded else play_decisions,
            "redeals": int(env._redeal_count),
            "max_redeals_exceeded": int(max_redeals_exceeded),
        },
    }


def build_standard_v2_reference() -> dict[str, Any]:
    """Run the corpus and return a content-addressed, non-timing record."""

    scenarios = [
        _run_reference_scenario(specification)
        for specification in build_standard_v2_corpus()
    ]
    coverage = {
        "first_bidders": sorted({
            scenario["opening"]["bidding_order"][0]
            for scenario in scenarios
        }),
        "normal_auction": any(
            len(scenario["bidding_attempts"]) == 1
            and not scenario["excluded_from_training"]
            for scenario in scenarios
        ),
        "all_pass": any(
            any(all(item["bid"] == 0 for item in attempt) for attempt in scenario["bidding_attempts"])
            for scenario in scenarios
        ),
        "redeal": any(scenario["counts"]["redeals"] > 0 for scenario in scenarios),
        "max_redeal_guard": any(
            scenario["counts"]["max_redeals_exceeded"] for scenario in scenarios
        ),
        "bomb_or_rocket": any(
            scenario["terminal"]["bomb_count"] > 0
            or scenario["terminal"]["rocket_count"] > 0
            for scenario in scenarios
        ),
        "spring": any(
            scenario["terminal"]["spring"]
            for scenario in scenarios
        ),
        "anti_spring": any(
            scenario["terminal"]["anti_spring"]
            for scenario in scenarios
        ),
    }
    expected_coverage = {
        "first_bidders": ["0", "1", "2"],
        "normal_auction": True,
        "all_pass": True,
        "redeal": True,
        "max_redeal_guard": True,
        "bomb_or_rocket": True,
        "spring": True,
        "anti_spring": True,
    }
    if coverage != expected_coverage:
        raise RuntimeError(
            f"Standard V2 golden corpus coverage drifted: {coverage!r}"
        )
    payload = {
        "schema_version": STANDARD_V2_REFERENCE_SCHEMA_VERSION,
        "contract_version": STANDARD_V2_R1_CONTRACT_VERSION,
        "config_hash": STANDARD_V2_R1_CONFIG_HASH,
        "version_contract": standard_v2_version_contract(),
        "coverage": coverage,
        "scenarios": scenarios,
    }
    payload["reference_digest"] = stable_identity_hash(payload)
    return payload
