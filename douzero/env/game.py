from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener
from .rules import (
    PHASE_BIDDING,
    PHASE_DEAL,
    PHASE_PLAYING,
    PHASE_REVEAL_BOTTOM,
    PHASE_TERMINAL,
    PLAYER_POSITIONS,
    RuleSet,
)

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

bombs = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]

class GameEnv(object):

    def __init__(self, players, ruleset=None):

        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.players = players

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.num_wins = {'landlord': 0,
                         'farmer': 0}

        self.num_scores = {'landlord': 0,
                           'farmer': 0}

        self.info_sets = {'landlord': InfoSet('landlord'),
                         'landlord_up': InfoSet('landlord_up'),
                         'landlord_down': InfoSet('landlord_down')}

        self.bomb_num = 0
        self.last_pid = 'landlord'

        # --- P02 standard-mode fields (unused in legacy; ruleset=None) --- #
        self.ruleset = ruleset
        self.phase = PHASE_DEAL if ruleset is not None else None
        # Per-position count of non-pass actions (for spring detection).
        self.action_counts = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        # Separate bomb / rocket counts (legacy bomb_num conflates both).
        self.bomb_count = 0
        self.rocket_count = 0
        # Bidding state.
        self.bidding_history = []          # [(position, bid_value), ...]
        self.bidding_order = []            # ordered list of positions
        self._bidding_index = 0            # next bidder index
        self.landlord_position = None      # determined after bidding
        self.bid_value = 0
        self.bottom_cards_revealed = []    # the 3 bottom cards (entity identity)
        self.three_landlord_cards_initial = None  # copy before landlord plays
        self.game_result = None            # GameResult (set at terminal)

    def card_play_init(self, card_play_data):
        self.info_sets['landlord'].player_hand_cards = \
            card_play_data['landlord']
        self.info_sets['landlord_up'].player_hand_cards = \
            card_play_data['landlord_up']
        self.info_sets['landlord_down'].player_hand_cards = \
            card_play_data['landlord_down']
        self.three_landlord_cards = card_play_data['three_landlord_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        if len(self.info_sets['landlord'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_up'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_down'].player_hand_cards) == 0:
            # if one of the three players discards his hand,
            # then game is over.
            self.compute_player_utility()
            self.update_num_wins_scores()

            self.game_over = True

    def compute_player_utility(self):

        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            self.player_utility_dict = {'landlord': 2,
                                        'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -2,
                                        'farmer': 1}

    def update_num_wins_scores(self):
        for pos, utility in self.player_utility_dict.items():
            base_score = 2 if pos == 'landlord' else 1
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                self.num_scores[pos] += base_score * (2 ** self.bomb_num)
            else:
                self.num_scores[pos] -= base_score * (2 ** self.bomb_num)

    def get_winner(self):
        return self.winner

    def get_bomb_num(self):
        return self.bomb_num

    def step(self):
        action = self.players[self.acting_player_position].act(
            self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if len(action) > 0:
            self.last_pid = self.acting_player_position
            # P02: track per-position valid (non-pass) action count for
            # spring detection. This does not affect legacy behaviour.
            self.action_counts[self.acting_player_position] += 1

        if action in bombs:
            self.bomb_num += 1
            # P02: separate bomb vs rocket counts for standard scoring.
            # Legacy bomb_num conflates both; this is additive only.
            if action == [20, 30]:
                self.rocket_count += 1
            else:
                self.bomb_count += 1

        self.last_move_dict[
            self.acting_player_position] = action.copy()

        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_position] += action

        # Track which bottom cards have been played by the landlord. In legacy
        # mode the landlord is always position 'landlord'; in standard mode
        # the landlord is self.landlord_position (set after bidding).
        landlord_pos = self.landlord_position if self.ruleset is not None else 'landlord'
        if self.acting_player_position == landlord_pos and \
                len(action) > 0 and \
                len(self.three_landlord_cards) > 0:
            for card in action:
                if len(self.three_landlord_cards) > 0:
                    if card in self.three_landlord_cards:
                        self.three_landlord_cards.remove(card)
                else:
                    break

        # P02: standard mode uses its own terminal check (produces GameResult);
        # legacy mode (ruleset=None) uses the original game_done() unchanged.
        if self.ruleset is not None:
            self.game_done_standard()
        else:
            self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]

        return last_move

    def get_last_two_moves(self):
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'

        else:
            if self.acting_player_position == 'landlord':
                self.acting_player_position = 'landlord_down'

            elif self.acting_player_position == 'landlord_down':
                self.acting_player_position = 'landlord_up'

            else:
                self.acting_player_position = 'landlord'

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action != []:
            for card in action:
                self.info_sets[
                    self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        mg = MovesGener(
            self.info_sets[self.acting_player_position].player_hand_cards)

        action_sequence = self.card_play_action_seq

        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == md.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = ms.filter_type_13_4_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_14_4_22:
            all_moves = mg.gen_type_14_4_22()
            moves = ms.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [md.TYPE_0_PASS,
                                   md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def reset(self):
        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.info_sets = {'landlord': InfoSet('landlord'),
                         'landlord_up': InfoSet('landlord_up'),
                         'landlord_down': InfoSet('landlord_down')}

        self.bomb_num = 0
        self.last_pid = 'landlord'

        # P02 standard-mode fields reset.
        self.phase = PHASE_DEAL if self.ruleset is not None else None
        self.action_counts = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
        self.bomb_count = 0
        self.rocket_count = 0
        self.bidding_history = []
        self.bidding_order = []
        self._bidding_index = 0
        self.landlord_position = None
        self.bid_value = 0
        self.bottom_cards_revealed = []
        self.three_landlord_cards_initial = None
        self.game_result = None

    def get_infoset(self):
        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].bomb_num = self.bomb_num

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_two_moves = self.get_last_two_moves()

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in ['landlord', 'landlord_up', 'landlord_down']}

        self.info_sets[self.acting_player_position].other_hand_cards = []
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards
        self.info_sets[self.acting_player_position].three_landlord_cards = \
            self.three_landlord_cards
        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in ['landlord', 'landlord_up', 'landlord_down']}

        return deepcopy(self.info_sets[self.acting_player_position])

    # ------------------------------------------------------------------ #
    # P02 standard-mode methods (only called when ruleset is not None).
    # Legacy path (ruleset=None) never touches these.
    # ------------------------------------------------------------------ #
    def card_play_init_standard(self, card_play_data, bidding_order=None):
        """Initialise for standard mode: deal 17+17+17 + 3 bottom cards.

        Unlike ``card_play_init``, the 3 bottom cards are NOT added to any
        player's hand yet; they are revealed after the landlord is determined
        by bidding. ``bidding_order`` defaults to the canonical turn order.
        """
        if bidding_order is None:
            bidding_order = list(PLAYER_POSITIONS)
        self.bidding_order = list(bidding_order)
        self._bidding_index = 0

        # Each player gets 17 cards (no bottom cards added to landlord yet).
        self.info_sets['landlord'].player_hand_cards = \
            sorted(card_play_data['landlord'])
        self.info_sets['landlord_up'].player_hand_cards = \
            sorted(card_play_data['landlord_up'])
        self.info_sets['landlord_down'].player_hand_cards = \
            sorted(card_play_data['landlord_down'])

        # Bottom cards are stored but not revealed yet.
        self.three_landlord_cards = sorted(card_play_data['three_landlord_cards'])
        self.three_landlord_cards_initial = list(self.three_landlord_cards)
        self.bottom_cards_revealed = []

        # The first bidder acts first.
        self.acting_player_position = self.bidding_order[0]
        self.phase = PHASE_BIDDING

    def step_bidding(self, bid_value):
        """Process one bid in the BIDDING phase.

        Returns ``True`` if the game should redeal (all pass +
        ``all_pass_redeal``), ``False`` otherwise. After the last bid, if not
        a redeal, the landlord is determined, bottom cards are revealed, and
        the phase transitions to PLAYING.
        """
        if self.phase != PHASE_BIDDING:
            raise IllegalPhaseError(
                f"step_bidding called in phase {self.phase!r}; "
                f"expected {PHASE_BIDDING!r}"
            )
        if bid_value not in self.ruleset.bid_values:
            raise IllegalActionError(
                f"Bid {bid_value!r} is not in allowed bid_values "
                f"{self.ruleset.bid_values}"
            )

        pos = self.acting_player_position
        self.bidding_history.append((pos, bid_value))
        self._bidding_index += 1

        # Check if bidding is complete (all positions have bid).
        if self._bidding_index >= len(self.bidding_order):
            return self._resolve_bidding()

        # Advance to the next bidder.
        self.acting_player_position = self.bidding_order[self._bidding_index]
        return False

    def _resolve_bidding(self):
        """Determine the landlord after all bids are in.

        Returns ``True`` if a redeal is required (all pass). Otherwise sets
        ``landlord_position``, ``bid_value``, reveals bottom cards, and
        transitions to PLAYING.
        """
        bids = [(pos, val) for pos, val in self.bidding_history]
        max_bid = max(val for _, val in bids)

        if max_bid == 0:
            # All pass.
            if self.ruleset.all_pass_redeal:
                return True
            # No redeal: assign landlord to the first bidder by default.
            self.landlord_position = self.bidding_order[0]
            self.bid_value = 1  # minimum
        else:
            # Highest bidder wins; ties broken by bidding order (first wins).
            for pos, val in bids:
                if val == max_bid:
                    self.landlord_position = pos
                    self.bid_value = val
                    break

        self._reveal_bottom_cards()
        return False

    def _reveal_bottom_cards(self):
        """Add the 3 bottom cards to the landlord's hand and transition to PLAYING."""
        self.phase = PHASE_REVEAL_BOTTOM
        bottom = list(self.three_landlord_cards)
        self.bottom_cards_revealed = list(bottom)
        landlord_hand = self.info_sets[self.landlord_position].player_hand_cards
        landlord_hand.extend(bottom)
        landlord_hand.sort()
        self.three_landlord_cards = list(bottom)  # keep for tracking

        # The landlord acts first in the playing phase.
        self.acting_player_position = self.landlord_position
        self.last_pid = self.landlord_position
        self.phase = PHASE_PLAYING
        self.game_infoset = self.get_infoset()

    def game_done_standard(self):
        """Terminal check for standard mode: produces a GameResult."""
        from douzero.env.scoring import compute_game_result

        # Determine the winner (first to empty hand).
        winner_position = None
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            if len(self.info_sets[pos].player_hand_cards) == 0:
                winner_position = pos
                break
        if winner_position is None:
            return  # not terminal

        result = compute_game_result(
            played_cards=self.played_cards,
            action_counts=dict(self.action_counts),
            winner_position=winner_position,
            bomb_count=self.bomb_count,
            rocket_count=self.rocket_count,
            bid_value=self.bid_value,
            ruleset=self.ruleset,
        )
        self.game_result = result

        # Backward-compatible utility dict (legacy interface).
        if result.winner_team == 'landlord':
            self.player_utility_dict = {'landlord': 1, 'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -1, 'farmer': 1}
        self.winner = winner_position
        self.game_over = True
        self.phase = PHASE_TERMINAL

        # Update num_wins / num_scores (legacy interface, for simulation.py).
        if result.winner_team == 'landlord':
            self.num_wins['landlord'] += 1
        else:
            self.num_wins['farmer'] += 1
        self.num_scores['landlord'] += result.landlord_score
        self.num_scores['farmer'] += result.farmer_score

    def get_bidding_obs(self):
        """Return the bidding-phase observation for the current bidder.

        Contains only public bidding history and the bidder's own hand.
        No other player's hand or the bottom cards are included.
        """
        if self.phase != PHASE_BIDDING:
            raise IllegalPhaseError(
                f"get_bidding_obs called in phase {self.phase!r}; "
                f"expected {PHASE_BIDDING!r}"
            )
        pos = self.acting_player_position
        return {
            'phase': 'bidding',
            'position': pos,
            'my_handcards': sorted(self.info_sets[pos].player_hand_cards),
            'bidding_history': list(self.bidding_history),
            'bidding_order': list(self.bidding_order),
            'bid_values': list(self.ruleset.bid_values),
            'num_cards_left': {
                p: len(self.info_sets[p].player_hand_cards)
                for p in ['landlord', 'landlord_up', 'landlord_down']
            },
        }


class IllegalPhaseError(Exception):
    """Raised when an action is attempted in the wrong game phase."""


class IllegalActionError(Exception):
    """Raised when an illegal bid or action is submitted."""


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int 
        self.num_cards_left_dict = None
        # The three landload cards. A list.
        self.three_landlord_cards = None
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player 
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the postions
        self.last_move_dict = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict. 
        self.all_handcards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
        # The number of bombs played so far
        self.bomb_num = None
