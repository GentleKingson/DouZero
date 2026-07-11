"""Legacy adapter: reconstruct the legacy observation tensors from V2 (P03).

The legacy encoder (``douzero.env.env.get_obs``) produces role-specific
``x_batch`` / ``z_batch`` / ``x_no_action`` / ``z`` tensors with per-role field
orderings and widths (landlord 319/373, farmers 430/484). P03 requires a legacy
adapter that reconstructs these from an :class:`ObservationV2` ALONE — no extra
infoset/history arguments (item 7) — so a legacy model can consume a V2
observation without any model-side change.

Item 7 compliance: :func:`legacy_observation_from_v2` takes only the V2
observation. The raw public action sequence it needs to rebuild ``z`` is stored
on :attr:`ObservationV2.card_play_action_seq` (an immutable tuple of tuples),
so nothing beyond the V2 container is required.

Field-order references (must match ``douzero/env/env.py`` exactly):

- landlord ``x_no_action``: my(54), other(54), last(54), up_played(54),
  down_played(54), up_left(17), down_left(17), bomb(15)  -> 319.
- landlord_up ``x_no_action``: my(54), other(54), landlord_played(54),
  teammate(down)_played(54), last(54), last_landlord(54), last_teammate(54),
  landlord_left(20), teammate_left(17), bomb(15)  -> 430.
- landlord_down ``x_no_action``: my(54), other(54), landlord_played(54),
  teammate(up)_played(54), last(54), last_landlord(54), last_teammate(54),
  landlord_left(20), teammate_left(17), bomb(15)  -> 430.
- ``x_batch`` = ``x_no_action`` features tiled across N rows + my_action(54).
- ``z`` = last 15 moves reshaped to (5, 162); ``z_batch`` = z tiled across N.
"""

from __future__ import annotations

import numpy as np

from douzero.env.env import (
    _action_seq_list2array,
    _cards2array,
    _get_one_hot_array,
    _get_one_hot_bomb,
    _process_action_seq,
)

from .encode_v2 import ObservationV2

#: Legacy landlord x_no_action width (see module docstring).
LANDLORD_X_NO_ACTION_WIDTH: int = 319
#: Legacy landlord x_batch width (x_no_action + 54 action).
LANDLORD_X_BATCH_WIDTH: int = 373
#: Legacy farmer x_no_action width.
FARMER_X_NO_ACTION_WIDTH: int = 430
#: Legacy farmer x_batch width.
FARMER_X_BATCH_WIDTH: int = 484


def _cards_left_onehot_legacy(count: int, max_cards: int) -> np.ndarray:
    """Legacy one-hot for a card count, using the legacy helper directly."""
    return _get_one_hot_array(count, max_cards)


def _build_landlord_x_no_action(public) -> np.ndarray:
    """Reconstruct the landlord ``x_no_action`` vector from public state."""
    my = _cards2array(list(public.my_handcards))
    other = _cards2array(list(public.other_handcards))
    last = _cards2array(list(public.last_move))
    up_played = _cards2array(list(public.played_cards.get("landlord_up", ())))
    down_played = _cards2array(list(public.played_cards.get("landlord_down", ())))
    up_left = _cards_left_onehot_legacy(
        public.num_cards_left.get("landlord_up", 0), 17)
    down_left = _cards_left_onehot_legacy(
        public.num_cards_left.get("landlord_down", 0), 17)
    bomb = _get_one_hot_bomb(public.bomb_count)
    return np.hstack((my, other, last, up_played, down_played,
                      up_left, down_left, bomb)).astype(np.int8)


def _build_farmer_x_no_action(public) -> np.ndarray:
    """Reconstruct a farmer ``x_no_action`` vector from public state."""
    my = _cards2array(list(public.my_handcards))
    other = _cards2array(list(public.other_handcards))
    landlord_played = _cards2array(list(public.played_cards.get("landlord", ())))
    last = _cards2array(list(public.last_move))
    last_landlord = _cards2array(list(public.last_move_dict.get("landlord", ())))
    landlord_left = _cards_left_onehot_legacy(
        public.num_cards_left.get("landlord", 0), 20)
    bomb = _get_one_hot_bomb(public.bomb_count)

    if public.acting_role == "landlord_up":
        teammate_played = _cards2array(
            list(public.played_cards.get("landlord_down", ())))
        last_teammate = _cards2array(
            list(public.last_move_dict.get("landlord_down", ())))
        teammate_left = _cards_left_onehot_legacy(
            public.num_cards_left.get("landlord_down", 0), 17)
    else:  # landlord_down
        teammate_played = _cards2array(
            list(public.played_cards.get("landlord_up", ())))
        last_teammate = _cards2array(
            list(public.last_move_dict.get("landlord_up", ())))
        teammate_left = _cards_left_onehot_legacy(
            public.num_cards_left.get("landlord_up", 0), 17)

    return np.hstack((my, other, landlord_played, teammate_played, last,
                      last_landlord, last_teammate, landlord_left,
                      teammate_left, bomb)).astype(np.int8)


def legacy_observation_from_v2(obs: ObservationV2) -> dict:
    """Build the legacy obs dict (x_batch/z_batch/x_no_action/z) from V2 ONLY.

    Item 7: this function takes ONLY the :class:`ObservationV2` — no extra
    infoset or action-sequence argument. The raw public action sequence needed
    to rebuild ``z`` is read from :attr:`obs.card_play_action_seq` (stored on
    the V2 container at encode time).

    The returned dict matches the legacy ``get_obs`` contract exactly: keys
    ``position``, ``x_batch`` (float32), ``z_batch`` (float32),
    ``legal_actions``, ``x_no_action`` (int8), ``z`` (int8). The legal-action
    order and dtypes are preserved so a legacy model consumes it unchanged.
    """
    public = obs.public
    n = len(public.legal_actions)

    if public.acting_role == "landlord":
        x_no_action = _build_landlord_x_no_action(public)
    else:
        x_no_action = _build_farmer_x_no_action(public)

    # Tile the state features across N action rows and append the per-action
    # card vector (matching the legacy encoder's trailing my_action block).
    my_action_block = np.zeros((max(n, 0), 54), dtype=np.int8)
    for j, action in enumerate(public.legal_actions):
        my_action_block[j, :] = _cards2array(list(action))
    if n > 0:
        x_batch = np.hstack([
            np.repeat(x_no_action[np.newaxis, :], n, axis=0),
            my_action_block,
        ]).astype(np.float32)
    else:
        x_batch = np.zeros((0, x_no_action.shape[0] + 54), dtype=np.float32)

    # z reconstruction from the V2-stored raw action sequence (item 7).
    action_seq = [list(a) for a in obs.card_play_action_seq]
    z = _action_seq_list2array(_process_action_seq(action_seq))
    z_batch = (
        np.repeat(z[np.newaxis, :, :], n, axis=0).astype(np.float32)
        if n > 0
        else np.zeros((0, 5, 162), dtype=np.float32)
    )
    z_int = z.astype(np.int8)

    return {
        "position": public.acting_role,
        "x_batch": x_batch,
        "z_batch": z_batch,
        "legal_actions": [list(a) for a in public.legal_actions],
        "x_no_action": x_no_action,
        "z": z_int,
    }
