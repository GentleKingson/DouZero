"""Population episode runner that never writes opponent decisions to replay."""

from __future__ import annotations

import json
import os
import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation.bidding import BiddingObservationV2, get_bidding_obs_v2
from douzero.observation.encode_v2 import ObservationV2, get_obs_v2
from douzero.training.bidding import (
    BiddingPolicyConfig,
    BiddingTransition,
    select_bidding_action,
)
from douzero.training.v2_buffer import Episode, Transition

from .policy_pool import LoadedPolicySelector, PolicyBundle, PolicyPool

ActionSelector = Callable[[ObservationV2], int]
BiddingSelector = Callable[[BiddingObservationV2], tuple[int, str] | int]


@dataclass(frozen=True)
class MatchupRecord:
    game_index: int
    policy_ids_by_seat: dict[str, str]
    learner_controlled_seats: tuple[str, ...]
    teammate_policy_ids: dict[str, str | None]
    ruleset_id: str
    ruleset_hash: str
    winner_team: str
    score: float
    policy_bundle_hash: str
    bid_value: int = 0
    redeal_count: int = 0
    bidding_transitions: int = 0
    abandoned_bidding_transitions: int = 0


class MatchupLogger:
    """Durable JSONL logger restricted to one process and serialized threads."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()

    def append(self, record: MatchupRecord) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "MatchupLogger is single-process; use one logger process for actors"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(record), sort_keys=True) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())


class PopulationEpisodeRunner:
    """Run one fixed policy bundle and return learner-only experience."""

    def __init__(
        self,
        pool: PolicyPool,
        current_selector: ActionSelector,
        *,
        opponent_selectors: Mapping[str, LoadedPolicySelector] | None = None,
        current_bidding_selector: BiddingSelector | None = None,
        bidding_policy_config: BiddingPolicyConfig | None = None,
        ruleset: RuleSet | None = None,
        max_steps: int = 600,
        logger: MatchupLogger | None = None,
    ) -> None:
        if ruleset is not None and ruleset.ruleset_id != "standard":
            raise ValueError("population ruleset must be standard or None")
        if ruleset is not None and current_bidding_selector is None:
            raise ValueError(
                "standard population self-play requires current_bidding_selector"
            )
        self.pool = pool
        self.current_selector = current_selector
        self.opponent_selectors = dict(opponent_selectors or {})
        for policy_id, selector in self.opponent_selectors.items():
            if policy_id != selector.policy_id:
                raise ValueError(
                    f"opponent selector key {policy_id!r} does not match "
                    f"selector policy_id {selector.policy_id!r}"
                )
            pool.validate_loaded_selector(selector)
        self.ruleset = ruleset
        self.current_bidding_selector = current_bidding_selector
        self.bidding_policy_config = bidding_policy_config or BiddingPolicyConfig()
        self.max_steps = max_steps
        self.logger = logger

    def run(
        self,
        game_index: int,
        *,
        opening=None,
        policy_version_at_start: str = "",
        policy_step_at_start: int = -1,
    ) -> tuple[Episode, MatchupRecord]:
        """Run one game, optionally from a validated P12 training opening."""

        bundle = self.pool.sample_bundle(game_index)
        expected_bundle_hash = bundle.bundle_hash
        rng = random.Random(self.pool.config.seed + game_index * 97_409)
        env = Env(objective=self.pool.current.objective, ruleset=self.ruleset)
        env.reset(opening=opening)
        episode = Episode(
            policy_ids_by_seat=dict(bundle.policy_ids_by_seat),
            learner_controlled_seats=bundle.learner_controlled_seats,
            policy_version_at_start=policy_version_at_start,
            policy_step_at_start=policy_step_at_start,
        )
        # In standard mode the bundle is assigned to physical seats before a
        # landlord exists. These canonical labels are only bundle slots; after
        # bidding we remap the same fixed policies to actual roles.
        neutral_to_bundle_seat = {
            "0": "landlord",
            "1": "landlord_up",
            "2": "landlord_down",
        }
        for _ in range(self.max_steps):
            bundle.assert_unchanged(expected_bundle_hash)
            if self.ruleset is not None and env.bidding_obs is not None:
                bidding_obs = get_bidding_obs_v2(
                    env.bidding_obs,
                    ruleset=self.ruleset,
                    redeal_count=env._redeal_count,
                )
                neutral_seat = bidding_obs.current_seat
                bundle_seat = neutral_to_bundle_seat[neutral_seat]
                policy_id = bundle.policy_ids_by_seat[bundle_seat]
                source_policy = "learned"
                if policy_id == self.pool.current.policy_id:
                    selected = self.current_bidding_selector(bidding_obs)
                    if isinstance(selected, tuple):
                        bid, source_policy = int(selected[0]), str(selected[1])
                    else:
                        bid = int(selected)
                elif policy_id == "builtin-random":
                    bid = int(rng.choice(bidding_obs.legal_bids))
                    source_policy = "random"
                elif policy_id == "builtin-rule":
                    bid, source_policy = select_bidding_action(
                        bidding_obs, BiddingPolicyConfig(policy="rule"), rng
                    )
                else:
                    try:
                        selector = self.opponent_selectors[policy_id]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"no bidding selector loaded for policy {policy_id!r}"
                        ) from exc
                    bid = selector.bid(bidding_obs)
                if bid not in bidding_obs.legal_bids:
                    raise ValueError(
                        f"policy {policy_id!r} returned illegal bid {bid}"
                    )
                if bundle_seat in bundle.learner_controlled_seats:
                    episode.bidding_transitions.append(BiddingTransition(
                        obs=bidding_obs,
                        bid_action=bid,
                        policy_version=policy_version_at_start,
                        source_policy=source_policy,
                    ))
                _obs, _reward, done, info = env.step(None, bid_value=bid)
                if done and info.get("redeal"):
                    episode.abandoned_bidding_transitions += len(
                        episode.bidding_transitions
                    )
                    episode.bidding_transitions.clear()
                    episode.redeal_count = int(info["redeal_count"])
                    env.redeal()
                    continue
                if done:
                    raise RuntimeError("bidding ended without terminal card play")
                if env.bidding_obs is not None:
                    continue
                role_to_neutral = {
                    role: neutral for neutral, role in env._env._seat_to_role.items()
                }
                actual_assignments = {
                    role: bundle.policy_ids_by_seat[
                        neutral_to_bundle_seat[neutral]
                    ]
                    for role, neutral in role_to_neutral.items()
                }
                learner_roles = tuple(
                    role
                    for role, neutral in role_to_neutral.items()
                    if neutral_to_bundle_seat[neutral]
                    in bundle.learner_controlled_seats
                )
                episode.policy_ids_by_seat = actual_assignments
                episode.learner_controlled_seats = learner_roles
                continue
            position = env._acting_player_position
            infoset = env.infoset
            policy_ids = episode.policy_ids_by_seat or dict(bundle.policy_ids_by_seat)
            learner_seats = episode.learner_controlled_seats or bundle.learner_controlled_seats
            policy_id = policy_ids[position]
            if len(infoset.legal_actions) == 1:
                action_index = 0
                action = infoset.legal_actions[0]
            else:
                obs = get_obs_v2(
                    infoset, ruleset=self.ruleset or RuleSet.legacy()
                )
                if policy_id == self.pool.current.policy_id:
                    action_index = int(self.current_selector(obs))
                elif policy_id == "builtin-random":
                    valid = [
                        i for i, valid in enumerate(obs.actions.action_mask) if valid
                    ]
                    action_index = rng.choice(valid)
                elif policy_id == "builtin-rule":
                    # Deterministic public heuristic: prefer shedding more
                    # cards, then higher total rank. Legality remains entirely
                    # controlled by the environment-provided action list.
                    action_index = max(
                        range(len(obs.actions.legal_actions)),
                        key=lambda index: (
                            len(obs.actions.legal_actions[index]),
                            sum(obs.actions.legal_actions[index]),
                            tuple(obs.actions.legal_actions[index]),
                        ),
                    )
                else:
                    try:
                        selector = self.opponent_selectors[policy_id]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"no action selector loaded for policy {policy_id!r}"
                        ) from exc
                    action_index = int(selector(obs))
                if not 0 <= action_index < len(infoset.legal_actions):
                    raise ValueError(
                        f"policy {policy_id!r} returned illegal action index {action_index}"
                    )
                action = infoset.legal_actions[action_index]
                if position in learner_seats:
                    teammate_policy_id = None
                    if position != "landlord":
                        teammate = (
                            "landlord_down"
                            if position == "landlord_up"
                            else "landlord_up"
                        )
                        teammate_policy_id = policy_ids[teammate]
                    episode.transitions.append(Transition(
                        obs=obs,
                        action_index=action_index,
                        position=position,
                        trace_index=len(episode.action_trace),
                        policy_id=policy_id,
                        teammate_policy_id=teammate_policy_id,
                    ))
            episode.action_trace.append((position, tuple(sorted(action))))
            _obs, _reward, done, info = env.step(action)
            if done:
                episode.terminal_result = info or {}
                break
        else:
            raise RuntimeError(f"population episode exceeded max_steps={self.max_steps}")

        record = self._record(bundle, episode)
        if self.logger is not None:
            self.logger.append(record)
        return episode, record

    def _record(self, bundle: PolicyBundle, episode: Episode) -> MatchupRecord:
        terminal = episode.terminal_result
        team_targets = terminal.get("team_targets", {})
        landlord_target = team_targets.get("landlord", {})
        return MatchupRecord(
            game_index=bundle.game_index,
            policy_ids_by_seat=(
                dict(episode.policy_ids_by_seat)
                if episode.policy_ids_by_seat
                else dict(bundle.policy_ids_by_seat)
            ),
            learner_controlled_seats=(
                episode.learner_controlled_seats
                or bundle.learner_controlled_seats
            ),
            teammate_policy_ids={
                seat: (
                    None
                    if seat == "landlord"
                    else episode.policy_ids_by_seat[
                        "landlord_down" if seat == "landlord_up" else "landlord_up"
                    ]
                )
                for seat in (
                    episode.learner_controlled_seats
                    or bundle.learner_controlled_seats
                )
            },
            ruleset_id=(self.ruleset.ruleset_id if self.ruleset else "legacy"),
            ruleset_hash=self.pool.runtime_ruleset_hash,
            winner_team=str(terminal.get("winner_team", "")),
            score=float(landlord_target.get("target_score", 0.0)),
            policy_bundle_hash=bundle.bundle_hash,
            bid_value=int(terminal.get("bid_value", 0)),
            redeal_count=episode.redeal_count,
            bidding_transitions=len(episode.bidding_transitions),
            abandoned_bidding_transitions=(
                episode.abandoned_bidding_transitions
            ),
        )
