"""Population episode runner that never writes opponent decisions to replay."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

from douzero.env.env import Env
from douzero.env.rules import RuleSet
from douzero.observation.encode_v2 import ObservationV2, get_obs_v2
from douzero.training.v2_buffer import Episode, Transition

from .policy_pool import PolicyBundle, PolicyPool

ActionSelector = Callable[[ObservationV2], int]


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


class MatchupLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def append(self, record: MatchupRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            json.dump(asdict(record), handle, sort_keys=True)
            handle.write("\n")


class PopulationEpisodeRunner:
    """Run one fixed policy bundle and return learner-only experience."""

    def __init__(
        self,
        pool: PolicyPool,
        current_selector: ActionSelector,
        *,
        opponent_selectors: Mapping[str, ActionSelector] | None = None,
        ruleset: RuleSet | None = None,
        max_steps: int = 600,
        logger: MatchupLogger | None = None,
    ) -> None:
        if ruleset is not None:
            raise NotImplementedError(
                "PopulationEpisodeRunner currently supports legacy card-play "
                "mode only; a standard-rules bidding driver is not yet wired."
            )
        self.pool = pool
        self.current_selector = current_selector
        self.opponent_selectors = dict(opponent_selectors or {})
        self.ruleset = ruleset
        self.max_steps = max_steps
        self.logger = logger

    def run(self, game_index: int) -> tuple[Episode, MatchupRecord]:
        bundle = self.pool.sample_bundle(game_index)
        expected_bundle_hash = bundle.bundle_hash
        rng = random.Random(self.pool.config.seed + game_index * 97_409)
        env = Env(objective=self.pool.current.objective, ruleset=self.ruleset)
        env.reset()
        episode = Episode(
            policy_ids_by_seat=dict(bundle.policy_ids_by_seat),
            learner_controlled_seats=bundle.learner_controlled_seats,
        )
        for _ in range(self.max_steps):
            bundle.assert_unchanged(expected_bundle_hash)
            position = env._acting_player_position
            infoset = env.infoset
            policy_id = bundle.policy_ids_by_seat[position]
            if len(infoset.legal_actions) == 1:
                action_index = 0
                action = infoset.legal_actions[0]
            else:
                obs = get_obs_v2(infoset, ruleset=RuleSet.legacy())
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
                if position in bundle.learner_controlled_seats:
                    episode.transitions.append(Transition(
                        obs=obs,
                        action_index=action_index,
                        position=position,
                        trace_index=len(episode.action_trace),
                        policy_id=policy_id,
                        teammate_policy_id=bundle.teammate_policy_id(position),
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
            policy_ids_by_seat=dict(bundle.policy_ids_by_seat),
            learner_controlled_seats=bundle.learner_controlled_seats,
            teammate_policy_ids={
                seat: bundle.teammate_policy_id(seat)
                for seat in bundle.learner_controlled_seats
            },
            ruleset_id="legacy",
            ruleset_hash=self.pool.runtime_ruleset_hash,
            winner_team=str(terminal.get("winner_team", "")),
            score=float(landlord_target.get("target_score", 0.0)),
            policy_bundle_hash=bundle.bundle_hash,
        )
