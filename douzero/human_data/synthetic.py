"""Deterministic synthetic human-game generator (P08).

When no real ``<HUMAN_DATA_PATH>`` is available, this generator produces
canonical :class:`~douzero.human_data.schema.HumanGameRecord` objects by playing
random self-play games on the legacy card-play environment. Its ONLY purpose is
to exercise the data pipeline (ingest → validate → split → BC-sample → train)
end-to-end on CPU with no network and no downloaded weights.

Real playing-strength is **not measured** here. Random play produces no signal
beyond "the pipeline runs and the loss can decrease on a trivial target"; any
strength claim requires authorized human data and is recorded as "未测" (not
measured) per AGENTS.md.

The generator records the TRUE deal (privileged) and the random action
sequence, plus a ``final_result`` derived from the env's terminal state. The
record is immediately replayable by :mod:`douzero.human_data.validate`.
"""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np

from .schema import (
    ACTION_ROLES,
    HumanGameRecord,
)
from .identifiers import make_internal_game_id

#: Ruleset identity for synthetic records. The legacy card-play env
#: (``Env("adp")``, ruleset=None = legacy) is the only collection mode in P08.
_SYNTHETIC_RULESET_ID: str = "legacy"
_SYNTHETIC_RULESET_VERSION: str = "legacy-v1"


def _legacy_ruleset_hash() -> str:
    """Return the canonical legacy RuleSet hash (computed once)."""
    from douzero.env.rules import RuleSet

    return RuleSet.legacy().stable_hash()


def _deal_card_play_data(np_rng: np.random.Generator) -> dict[str, list[int]]:
    """Deal a legacy 20/17/17+3 layout, mirroring :meth:`Env.reset`."""
    deck: list[int] = []
    for rank in range(3, 15):
        deck.extend([rank] * 4)
    deck.extend([17] * 4)
    deck.extend([20, 30])
    order = deck.copy()
    np_rng.shuffle(order)
    return {
        "landlord": sorted(order[:20]),
        "landlord_up": sorted(order[20:37]),
        "landlord_down": sorted(order[37:54]),
        "three_landlord_cards": sorted(order[17:20]),
    }


def _play_random_game(
    np_rng: np.random.Generator,
    *,
    max_steps: int,
) -> tuple[dict[str, list[int]], list[tuple[str, tuple[int, ...]]], dict]:
    """Play one random self-play legacy game and return (deal, actions, result).

    Uses :class:`~douzero.env.env.Env` so the exact same legality/turn/terminal
    machinery as RL training drives the record. Each action is recorded with
    its acting role and sorted card tuple.
    """
    from douzero.env.env import Env

    env = Env("adp")
    env.reset()
    deal_snapshot = {
        "landlord": sorted(env.infoset.player_hand_cards),
        # The other two roles' initial hands are not on the acting infoset; we
        # capture them from the env's internal info_sets before any step.
    }
    # Capture the full deal from the env's internal state (all three roles).
    full_deal = {
        "landlord": sorted(env._env.info_sets["landlord"].player_hand_cards),
        "landlord_up": sorted(env._env.info_sets["landlord_up"].player_hand_cards),
        "landlord_down": sorted(env._env.info_sets["landlord_down"].player_hand_cards),
        "three_landlord_cards": sorted(env._env.three_landlord_cards),
    }
    del deal_snapshot

    actions: list[tuple[str, tuple[int, ...]]] = []
    steps = 0
    terminal_info: dict = {}
    while True:
        if steps >= max_steps:
            raise RuntimeError(
                f"synthetic random game exceeded {max_steps} steps; possible "
                "infinite loop in move generation."
            )
        steps += 1
        infoset = env.infoset
        legal = list(infoset.legal_actions)
        if not legal:
            break
        # Prefer non-empty moves so the game progresses; fall back to pass.
        nonempty = [a for a in legal if len(a) > 0]
        pool = nonempty if nonempty else legal
        action = list(pool[int(np_rng.integers(len(pool)))])
        actions.append((infoset.player_position, tuple(sorted(action))))
        _obs, _reward, done, info = env.step(action)
        if done:
            terminal_info = info or {}
            break

    # Derive the winner from the env's terminal utility dict. The legacy env
    # stamps ``team_targets`` onto info via _attach_team_perspective_labels; the
    # raw winner is on env._env (GameEnv).
    winner_position = _derive_winner_position(env, actions)
    winner_team = "landlord" if winner_position == "landlord" else "farmer"
    result = {
        "winner_team": winner_team,
        "winner_position": winner_position,
        "bomb_count": int(env._env.bomb_count) if hasattr(env._env, "bomb_count") else 0,
        "rocket_count": int(env._env.rocket_count) if hasattr(env._env, "rocket_count") else 0,
        "bomb_num": int(env._env.bomb_num),
    }
    return full_deal, actions, result


def _derive_winner_position(env, actions: list[tuple[str, tuple[int, ...]]]) -> str:
    """Return the role that emptied its hand (the winner), or '' if unknown."""
    genv = env._env
    for pos in ACTION_ROLES:
        if len(genv.info_sets[pos].player_hand_cards) == 0:
            return pos
    # Fallback: the last non-pass actor.
    for pos, cards in reversed(actions):
        if len(cards) > 0:
            return pos
    return ""


def generate_synthetic_record(
    game_id: str,
    *,
    seed: int,
    max_steps: int = 600,
    player_skill_weight: dict[str, float] | None = None,
    source_name: str = "synthetic",
) -> HumanGameRecord:
    """Generate one canonical record from a seeded random self-play game."""
    np_rng = np.random.default_rng(seed)
    # Env.reset uses np.random.shuffle, so seed numpy's global RNG too.
    np.random.seed(seed)
    deal, actions, result = _play_random_game(np_rng, max_steps=max_steps)
    return HumanGameRecord(
        game_id=make_internal_game_id(game_id),
        ruleset_id=_SYNTHETIC_RULESET_ID,
        ruleset_version=_SYNTHETIC_RULESET_VERSION,
        ruleset_hash=_legacy_ruleset_hash(),
        seats=tuple(ACTION_ROLES),
        initial_hands={
            role: tuple(cards) for role, cards in deal.items()
        },
        bottom_cards=tuple(deal["three_landlord_cards"]),
        bidding_history=(),
        action_history=tuple(actions),
        final_result=result,
        player_skill_weight=dict(player_skill_weight or {}),
        source_metadata={"source": source_name, "license": "synthetic"},
    )


def generate_synthetic_records(
    *,
    num_games: int,
    base_seed: int,
    max_steps: int = 600,
    game_id_prefix: str = "synthetic",
) -> Iterator[HumanGameRecord]:
    """Yield ``num_games`` deterministic synthetic records.

    Each game's seed is derived deterministically from ``base_seed`` and its
    index, so the same call reproduces the same records. Games are yielded in
    index order.
    """
    if num_games < 0:
        raise ValueError(f"num_games must be non-negative, got {num_games}")
    rng = random.Random(base_seed)
    for i in range(num_games):
        # Distinct, deterministic per-game seed.
        game_seed = rng.randrange(1 << 30)
        game_id = f"{game_id_prefix}-{i:06d}-{base_seed}"
        yield generate_synthetic_record(
            game_id,
            seed=game_seed,
            max_steps=max_steps,
        )
