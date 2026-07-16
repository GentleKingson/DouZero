from collections import Counter
import numpy as np

from douzero.env.game import GameEnv, IllegalActionError, IllegalPhaseError
from douzero.env.rules import (
    PHASE_BIDDING,
    PHASE_PLAYING,
    PHASE_TERMINAL,
    RuleSet,
)

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, objective, ruleset=None):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.

        ``ruleset`` is ``None`` (legacy, default) or a :class:`RuleSet`
        instance. When a standard ruleset is passed, ``reset`` enters the
        bidding phase and ``step`` dispatches by phase. Legacy behaviour is
        unchanged when ``ruleset`` is ``None``.

        In standard mode, the bidding phase uses neutral seat labels
        ("0", "1", "2"). The caller drives bidding by passing ``bid_value``
        to ``step``. The environment does NOT run any bidding policy
        internally — the bidding decision (random, SL, RL) is the caller's
        responsibility (see ``douzero/evaluation/simulation.py`` for the
        random bidding agent used in evaluation).
        """
        self.objective = objective
        self.ruleset = ruleset

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players, ruleset=ruleset)

        self.infoset = None
        # Standard-mode: the current bidding observation (None in legacy).
        self.bidding_obs = None
        # Standard-mode: redeal flag (set when all bidders pass).
        self.need_redeal = False
        # Standard-mode: redeal counter (bounded by ruleset.max_redeals).
        self._redeal_count = 0

    def _current_bidding_observation(self):
        """Return the public bidding state with session-level redeal metadata."""
        obs = self._env.get_bidding_obs()
        obs["redeal_count"] = self._redeal_count
        return obs

    def reset(self, opening=None, bidding_order=None):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.

        ``opening`` is an optional training-only P12 ``OpeningRecord``. The
        default remains the original NumPy shuffle. Passing a record injects
        only its validated deal into the environment; the full deck is never
        copied into an observation or infoset.
        """
        self._env.reset()
        self.need_redeal = False
        self._redeal_count = 0

        if opening is None:
            # Original legacy behavior: use the process-global NumPy RNG.
            _deck = deck.copy()
            np.random.shuffle(_deck)
            card_play_data = None
            # Callers may supply a reproducible training schedule.
        else:
            from douzero.coach.records import OpeningRecord

            if not isinstance(opening, OpeningRecord):
                raise TypeError("opening must be an OpeningRecord")
            expected_id = "standard" if self.ruleset is not None else "legacy"
            if opening.ruleset_obj.ruleset_id != expected_id:
                raise ValueError(
                    f"opening ruleset {opening.ruleset_obj.ruleset_id!r} does not "
                    f"match environment mode {expected_id!r}"
                )
            if (
                self.ruleset is not None
                and opening.ruleset_obj.stable_hash() != self.ruleset.stable_hash()
            ):
                raise ValueError("opening RuleSet hash does not match the environment")
            _deck = list(opening.deck)
            card_play_data = opening.to_card_play_data()
            opening_order = list(opening.bidding_order)
            if bidding_order is not None and list(bidding_order) != opening_order:
                raise ValueError("bidding_order conflicts with the supplied opening")
            bidding_order = opening_order

        if self.ruleset is not None:
            # Standard mode: deal 17+17+17 + 3 bottom cards, enter bidding.
            # Seats are neutral ("0", "1", "2") during bidding.
            if card_play_data is None:
                card_play_data = {'landlord': _deck[:17],
                                  'landlord_up': _deck[17:34],
                                  'landlord_down': _deck[34:51],
                                  'three_landlord_cards': _deck[51:54],
                                  }
            for key in card_play_data:
                card_play_data[key].sort()
            self._env.card_play_init_standard(
                card_play_data, bidding_order=bidding_order
            )
            self.bidding_obs = self._current_bidding_observation()
            self.infoset = None
            return self.bidding_obs

        # Legacy mode (unchanged).
        if card_play_data is None:
            card_play_data = {'landlord': _deck[:20],
                              'landlord_up': _deck[20:37],
                              'landlord_down': _deck[37:54],
                              'three_landlord_cards': _deck[17:20],
                              }
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def redeal(self):
        """Re-deal a new deck for the same game session (after all-pass redeal).

        Unlike ``reset()``, this does NOT clear ``_redeal_count`` — the
        redeal guard accumulates across redeals within the same game. Use
        ``reset()`` to start a completely new game.
        """
        bidding_order = list(self._env.bidding_order)
        self._env.reset()
        self.need_redeal = False
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:17],
                          'landlord_up': _deck[17:34],
                          'landlord_down': _deck[34:51],
                          'three_landlord_cards': _deck[51:54],
                          }
        for key in card_play_data:
            card_play_data[key].sort()
        self._env.card_play_init_standard(
            card_play_data, bidding_order=bidding_order
        )
        self.bidding_obs = self._current_bidding_observation()
        self.infoset = None
        return self.bidding_obs

    def step(self, action, bid_value=None):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.

        In standard bidding mode, ``bid_value`` (an int in
        ``ruleset.bid_values``) is used instead of ``action``; ``action``
        may be ``None`` during the bidding phase.
        """
        # Standard bidding phase: dispatch to step_bidding.
        if self.ruleset is not None and self._env.phase == PHASE_BIDDING:
            return self._step_bidding(bid_value)

        # Playing phase (both legacy and standard).
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        info = {}
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
            # Standard mode: return structured terminal info.
            if self.ruleset is not None and self._env.game_result is not None:
                info = self._env.game_result.to_dict()
            # P06: attach team-perspective multi-objective labels to the
            # terminal info dict in BOTH legacy and standard modes. These
            # labels are derived from the public terminal result (winner +
            # bomb count + bid value), never from hidden hands, and are
            # consumed by the V2 trainer and the calibration harness. The
            # legacy ``reward`` field above is unchanged; the new keys are
            # purely additive.
            info = self._attach_team_perspective_labels(info)
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, info

    def _step_bidding(self, bid_value):
        """Process a bid in the BIDDING phase.

        Returns ``(obs, reward, done, info)``. If all bidders pass and
        ``all_pass_redeal`` is set, the environment signals a redeal:
        ``done=True`` with ``info={'redeal': True}``. The caller should call
        ``redeal()`` (NOT ``reset()``) to re-deal for the next attempt.
        ``reset()`` starts a new game and clears the redeal count;
        ``redeal()`` preserves it. The redeal count is bounded by
        ``ruleset.max_redeals``; if exceeded, the landlord is assigned to the
        first bidder with the minimum bid.
        """
        if bid_value is None:
            raise IllegalActionError(
                "bid_value must be provided during the bidding phase"
            )
        redeal = self._env.step_bidding(bid_value)
        if redeal:
            self._redeal_count += 1
            if self._redeal_count > self._env.ruleset.max_redeals:
                # Exceeded max redeals: force-assign landlord to first bidder.
                self._env.landlord_position = self._env.bidding_order[0]
                self._env.bid_value = 1
                self._env._reveal_bottom_cards()
                self.bidding_obs = None
                self.infoset = self._game_infoset
                return get_obs(self.infoset), 0.0, False, {
                    'bidding_complete': True,
                    'max_redeals_exceeded': True,
                }
            self.need_redeal = True
            return None, 0.0, True, {'redeal': True, 'redeal_count': self._redeal_count}

        if self._env.phase == PHASE_BIDDING:
            # More bids to go.
            self.bidding_obs = self._current_bidding_observation()
            return self.bidding_obs, 0.0, False, {}

        # Bidding complete; transitioned to PLAYING.
        self.bidding_obs = None
        self.infoset = self._game_infoset
        return get_obs(self.infoset), 0.0, False, {'bidding_complete': True}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.

        In standard mode, the reward is derived from the GameResult's
        landlord_score (from the landlord's perspective). The actor loop
        negates this for farmer positions, unchanged.
        """
        # Standard mode: use the structured GameResult.
        if self.ruleset is not None and self._env.game_result is not None:
            return float(self._env.game_result.landlord_score)

        # Legacy mode (unchanged).
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    def _legacy_terminal_result_dict(self):
        """Build a GameResult-compatible dict for a legacy-mode terminal state.

        P02's :class:`GameResult` is only produced in standard mode. For the
        legacy path (still the default) we construct the minimal
        team-perspective terminal dict the P06 label helpers consume. The
        convention matches ``douzero/env/scoring.py``'s documented legacy
        scoring (``landlord_score = ±2 * 2**bomb_num``,
        ``farmer_score = ∓1 * 2**bomb_num``, bomb_num includes the rocket),
        so score conservation (``landlord_score + 2*farmer_score == 0``)
        holds.

        For ``objective='wp'`` we use base 2 / 1 (landlord plays for two);
        for ``objective='logadp'`` we use ``±2*(bomb_num+1)`` /
        ``∓(bomb_num+1)`` mirroring the legacy reward magnitude.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        landlord_won = winner == 'landlord'
        sign = 1 if landlord_won else -1
        if self.objective == 'adp':
            magnitude = float(2 ** bomb_num)
        elif self.objective == 'logadp':
            magnitude = float(bomb_num + 1)
        else:  # 'wp'
            magnitude = 1.0
        return {
            "winner_team": winner,
            "winner_position": winner if landlord_won else (
                # The legacy game engine does not record which farmer emptied
                # first here; the team-perspective label does not depend on
                # it. We record the winning team only.
                "farmer"
            ),
            "bid_value": 0,
            "bomb_count": int(bomb_num),
            "rocket_count": 0,  # legacy bomb_num already includes the rocket
            "spring": False,
            "anti_spring": False,
            "multiplier_breakdown": {"bombs_and_rocket": int(bomb_num)},
            "total_multiplier": int(2 ** bomb_num) if self.objective == "adp" else 1,
            # Landlord plays for two (sign × 2 × magnitude); each farmer plays
            # for one (sign-flipped × magnitude). Conservation holds.
            "landlord_score": int(sign * 2 * magnitude),
            "farmer_score": int(-sign * magnitude),
            "ruleset_id": "legacy",
            "ruleset_version": "legacy-v1",
            "ruleset_hash": "",
        }

    def _attach_team_perspective_labels(self, info):
        """Attach P06 team-perspective multi-objective labels to terminal info.

        Additive only: the existing ``info`` (which is the full GameResult
        dict in standard mode, or an empty dict in legacy mode) is preserved.
        The new ``team_targets`` key maps each position to its
        ``target_win`` / ``target_score`` / ``target_log_score`` triple, and
        ``terminal_result`` carries the GameResult-like dict the labels were
        derived from (so the trainer/calibration harness can re-derive
        per-position metrics without re-running the env).
        """
        from douzero.training.labels import team_targets

        if not isinstance(info, dict):
            info = {} if info is None else dict(info)
        if "winner_team" not in info:
            # Legacy mode: synthesize the minimal terminal result dict.
            info.update(self._legacy_terminal_result_dict())
        result_dict = {
            k: info[k] for k in (
                "winner_team", "landlord_score", "farmer_score"
            ) if k in info
        }
        per_position = {
            pos: team_targets(result_dict, pos)
            for pos in ("landlord", "landlord_up", "landlord_down")
        }
        info["team_targets"] = per_position
        info["terminal_result"] = result_dict
        return info

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.

    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'landlord_up':
        return _get_obs_landlord_up(infoset)
    elif infoset.player_position == 'landlord_down':
        return _get_obs_landlord_down(infoset)
    else:
        raise ValueError('')


# --------------------------------------------------------------------------- #
# Factorized observation encoder (P04)
# --------------------------------------------------------------------------- #
# get_obs_factorized produces the SAME shared-state vector and per-action
# vectors as the legacy encoders above, but NEVER tiles the shared state or
# history across the N legal-action rows. It returns:
#
#   z_single       : (1, 5, 162) float32   — the shared history, encoded once
#   x_state_single : (1, D_state) float32  — the shared state, encoded once
#   x_action       : (N, 54) float32       — per-action card vectors
#   legal_actions  : list                  — the N legal actions
#
# where D_state is 319 (landlord) / 430 (farmers), matching x_no_action. The
# factorized DeepAgent (backend='legacy_factorized') consumes these directly
# via model.forward_factorized(z_single, x_state_single, x_action), avoiding:
#   * the NumPy np.repeat tiling of the shared state/history,
#   * the CPU tensor allocation for the tiled (N, ...) batches,
#   * the CPU->GPU transfer of the tiled batches.
#
# The shared-state construction below is byte-for-byte identical to the
# legacy x_no_action (same field order, same helpers); only the tiling is
# removed. z is identical to the legacy z. Parity is pinned by
# tests/test_factorized_parity.py (get_obs_factorized vs legacy get_obs).
def get_obs_factorized(infoset):
    """Return the factorized (split) observation, never tiling shared state.

    Args:
        infoset: the game infoset with ``player_position`` set.

    Returns:
        dict with keys ``position``, ``z_single`` (1,5,162) float32,
        ``x_state_single`` (1, D_state) float32, ``x_action`` (N, 54)
        float32, ``legal_actions``. The shared blocks carry a leading 1 so
        they feed the model's singleton-input forward directly.
    """
    if infoset.player_position == 'landlord':
        return _get_obs_factorized_landlord(infoset)
    elif infoset.player_position == 'landlord_up':
        return _get_obs_factorized_landlord_up(infoset)
    elif infoset.player_position == 'landlord_down':
        return _get_obs_factorized_landlord_down(infoset)
    else:
        raise ValueError(
            f"Unknown player_position {infoset.player_position!r}; expected "
            f"'landlord', 'landlord_up', or 'landlord_down'."
        )


def _build_shared_state_landlord(infoset):
    """Build the landlord shared-state (x_no_action) vector once.

    Field order matches _get_obs_landlord exactly:
    my(54), other(54), last(54), up_played(54), down_played(54),
    up_left(17), down_left(17), bomb(15) -> 319.
    """
    my_handcards = _cards2array(infoset.player_hand_cards)
    other_handcards = _cards2array(infoset.other_hand_cards)
    last_action = _cards2array(infoset.last_move)
    landlord_up_played_cards = _cards2array(infoset.played_cards['landlord_up'])
    landlord_down_played_cards = _cards2array(infoset.played_cards['landlord_down'])
    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    bomb_num = _get_one_hot_bomb(infoset.bomb_num)
    return np.hstack((my_handcards,
                      other_handcards,
                      last_action,
                      landlord_up_played_cards,
                      landlord_down_played_cards,
                      landlord_up_num_cards_left,
                      landlord_down_num_cards_left,
                      bomb_num))


def _build_shared_state_landlord_up(infoset):
    """Build the landlord_up shared-state (x_no_action) vector once.

    Field order matches _get_obs_landlord_up exactly:
    my(54), other(54), landlord_played(54), teammate(down)_played(54),
    last(54), last_landlord(54), last_teammate(54), landlord_left(20),
    teammate_left(17), bomb(15) -> 430.
    """
    my_handcards = _cards2array(infoset.player_hand_cards)
    other_handcards = _cards2array(infoset.other_hand_cards)
    last_action = _cards2array(infoset.last_move)
    landlord_played_cards = _cards2array(infoset.played_cards['landlord'])
    teammate_played_cards = _cards2array(infoset.played_cards['landlord_down'])
    last_landlord_action = _cards2array(infoset.last_move_dict['landlord'])
    last_teammate_action = _cards2array(infoset.last_move_dict['landlord_down'])
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    bomb_num = _get_one_hot_bomb(infoset.bomb_num)
    return np.hstack((my_handcards,
                      other_handcards,
                      landlord_played_cards,
                      teammate_played_cards,
                      last_action,
                      last_landlord_action,
                      last_teammate_action,
                      landlord_num_cards_left,
                      teammate_num_cards_left,
                      bomb_num))


def _build_shared_state_landlord_down(infoset):
    """Build the landlord_down shared-state (x_no_action) vector once.

    Field order matches _get_obs_landlord_down exactly:
    my(54), other(54), landlord_played(54), teammate(up)_played(54),
    last(54), last_landlord(54), last_teammate(54), landlord_left(20),
    teammate_left(17), bomb(15) -> 430.
    """
    my_handcards = _cards2array(infoset.player_hand_cards)
    other_handcards = _cards2array(infoset.other_hand_cards)
    last_action = _cards2array(infoset.last_move)
    landlord_played_cards = _cards2array(infoset.played_cards['landlord'])
    teammate_played_cards = _cards2array(infoset.played_cards['landlord_up'])
    last_landlord_action = _cards2array(infoset.last_move_dict['landlord'])
    last_teammate_action = _cards2array(infoset.last_move_dict['landlord_up'])
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    bomb_num = _get_one_hot_bomb(infoset.bomb_num)
    return np.hstack((my_handcards,
                      other_handcards,
                      landlord_played_cards,
                      teammate_played_cards,
                      last_action,
                      last_landlord_action,
                      last_teammate_action,
                      landlord_num_cards_left,
                      teammate_num_cards_left,
                      bomb_num))


def _build_action_matrix(infoset):
    """Build the (N, 54) per-action card-vector matrix (never tiled)."""
    num_legal_actions = len(infoset.legal_actions)
    x_action = np.zeros((num_legal_actions, 54), dtype=np.float32)
    for j, action in enumerate(infoset.legal_actions):
        x_action[j, :] = _cards2array(action)
    return x_action


def _get_obs_factorized_landlord(infoset):
    x_state = _build_shared_state_landlord(infoset).astype(np.float32)
    x_action = _build_action_matrix(infoset)
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    # Leading 1 so the singleton feeds the model's factorized forward directly.
    return {
        'position': 'landlord',
        'z_single': z[np.newaxis, :, :].astype(np.float32),
        'x_state_single': x_state[np.newaxis, :],
        'x_action': x_action,
        'legal_actions': infoset.legal_actions,
        # Untiled singletons, for parity/debugging (matching legacy obs keys).
        'x_no_action': x_state.astype(np.int8),
        'z': z.astype(np.int8),
    }


def _get_obs_factorized_landlord_up(infoset):
    x_state = _build_shared_state_landlord_up(infoset).astype(np.float32)
    x_action = _build_action_matrix(infoset)
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    return {
        'position': 'landlord_up',
        'z_single': z[np.newaxis, :, :].astype(np.float32),
        'x_state_single': x_state[np.newaxis, :],
        'x_action': x_action,
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_state.astype(np.int8),
        'z': z.astype(np.int8),
    }


def _get_obs_factorized_landlord_down(infoset):
    x_state = _build_shared_state_landlord_down(infoset).astype(np.float32)
    x_action = _build_action_matrix(infoset)
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    return {
        'position': 'landlord_down',
        'z_single': z[np.newaxis, :, :].astype(np.float32),
        'x_state_single': x_state[np.newaxis, :],
        'x_action': x_action,
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_state.astype(np.int8),
        'z': z.astype(np.int8),
    }

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_up(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_down(infoset):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs
