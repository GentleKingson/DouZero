from douzero.env.utils import MIN_SINGLE_CARDS, MIN_PAIRS, MIN_TRIPLES
import collections
import itertools

class MovesGener(object):
    """
    This is for generating the possible combinations
    """
    def __init__(self, cards_list, budget_check=None):
        self.cards_list = cards_list
        # Optional cooperative deadline hook used by bounded strategy search.
        # The normal legality path passes None and preserves legacy output.
        self._budget_check = budget_check
        self.cards_dict = collections.defaultdict(int)

        for i in self.cards_list:
            self._check_budget()
            self.cards_dict[i] += 1

        self.single_card_moves = []
        self.gen_type_1_single()
        self.pair_moves = []
        self.gen_type_2_pair()
        self.triple_cards_moves = []
        self.gen_type_3_triple()
        self.bomb_moves = []
        self.gen_type_4_bomb()
        self.final_bomb_moves = []
        self.gen_type_5_king_bomb()

    def _check_budget(self):
        if self._budget_check is not None:
            self._budget_check()

    def _gen_serial_moves(self, cards, min_serial, repeat=1, repeat_num=0):
        if repeat_num < min_serial:  # at least repeat_num is min_serial
            repeat_num = 0

        single_cards = sorted(list(set(cards)))
        seq_records = list()
        moves = list()

        start = i = 0
        longest = 1
        while i < len(single_cards):
            self._check_budget()
            if i + 1 < len(single_cards) and single_cards[i + 1] - single_cards[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        for seq in seq_records:
            self._check_budget()
            if seq[1] < min_serial:
                continue
            start, longest = seq[0], seq[1]
            longest_list = single_cards[start: start + longest]

            if repeat_num == 0:  # No limitation on how many sequences
                steps = min_serial
                while steps <= longest:
                    self._check_budget()
                    index = 0
                    while steps + index <= longest:
                        self._check_budget()
                        target_moves = sorted(longest_list[index: index + steps] * repeat)
                        moves.append(target_moves)
                        index += 1
                    steps += 1

            else:  # repeat_num > 0
                if longest < repeat_num:
                    continue
                index = 0
                while index + repeat_num <= longest:
                    self._check_budget()
                    target_moves = sorted(longest_list[index: index + repeat_num] * repeat)
                    moves.append(target_moves)
                    index += 1

        return moves

    def gen_type_1_single(self):
        self.single_card_moves = []
        for i in set(self.cards_list):
            self._check_budget()
            self.single_card_moves.append([i])
        return self.single_card_moves

    def gen_type_2_pair(self):
        self.pair_moves = []
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v >= 2:
                self.pair_moves.append([k, k])
        return self.pair_moves

    def gen_type_3_triple(self):
        self.triple_cards_moves = []
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v >= 3:
                self.triple_cards_moves.append([k, k, k])
        return self.triple_cards_moves

    def gen_type_4_bomb(self):
        self.bomb_moves = []
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v == 4:
                self.bomb_moves.append([k, k, k, k])
        return self.bomb_moves

    def gen_type_5_king_bomb(self):
        self._check_budget()
        self.final_bomb_moves = []
        if 20 in self.cards_list and 30 in self.cards_list:
            self.final_bomb_moves.append([20, 30])
        return self.final_bomb_moves

    def gen_type_6_3_1(self):
        result = []
        for t in self.single_card_moves:
            self._check_budget()
            for i in self.triple_cards_moves:
                self._check_budget()
                if t[0] != i[0]:
                    result.append(t+i)
        return result

    def gen_type_7_3_2(self):
        result = list()
        for t in self.pair_moves:
            self._check_budget()
            for i in self.triple_cards_moves:
                self._check_budget()
                if t[0] != i[0]:
                    result.append(t+i)
        return result

    def gen_type_8_serial_single(self, repeat_num=0):
        return self._gen_serial_moves(self.cards_list, MIN_SINGLE_CARDS, repeat=1, repeat_num=repeat_num)

    def gen_type_9_serial_pair(self, repeat_num=0):
        single_pairs = list()
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v >= 2:
                single_pairs.append(k)

        return self._gen_serial_moves(single_pairs, MIN_PAIRS, repeat=2, repeat_num=repeat_num)

    def gen_type_10_serial_triple(self, repeat_num=0):
        single_triples = list()
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v >= 3:
                single_triples.append(k)

        return self._gen_serial_moves(single_triples, MIN_TRIPLES, repeat=3, repeat_num=repeat_num)

    def gen_type_11_serial_3_1(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_1_moves = list()

        for s3 in serial_3_moves:  # s3 is like [3,3,3,4,4,4]
            self._check_budget()
            s3_set = set(s3)
            new_cards = [i for i in self.cards_list if i not in s3_set]

            # Get any s3_len items from cards
            for selected in itertools.combinations(new_cards, len(s3_set)):
                self._check_budget()
                serial_3_1_moves.append(s3 + list(selected))

        return list(k for k, _ in itertools.groupby(serial_3_1_moves))

    def gen_type_12_serial_3_2(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_2_moves = list()
        pair_set = sorted([k for k, v in self.cards_dict.items() if v >= 2])

        for s3 in serial_3_moves:
            self._check_budget()
            s3_set = set(s3)
            pair_candidates = [i for i in pair_set if i not in s3_set]

            # Get any s3_len items from cards
            for selected in itertools.combinations(pair_candidates, len(s3_set)):
                self._check_budget()
                serial_3_2_moves.append(sorted(s3 + list(selected) * 2))

        return serial_3_2_moves

    def gen_type_13_4_2(self):
        four_cards = list()
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v == 4:
                four_cards.append(k)

        result = list()
        for fc in four_cards:
            self._check_budget()
            cards_list = [k for k in self.cards_list if k != fc]
            for selected in itertools.combinations(cards_list, 2):
                self._check_budget()
                result.append([fc]*4 + list(selected))
        return list(k for k, _ in itertools.groupby(result))

    def gen_type_14_4_22(self):
        four_cards = list()
        for k, v in self.cards_dict.items():
            self._check_budget()
            if v == 4:
                four_cards.append(k)

        result = list()
        for fc in four_cards:
            self._check_budget()
            cards_list = [k for k, v in self.cards_dict.items() if k != fc and v>=2]
            for selected in itertools.combinations(cards_list, 2):
                self._check_budget()
                result.append(
                    [fc] * 4
                    + [selected[0], selected[0], selected[1], selected[1]]
                )
        return result

    # generate all possible moves from given cards
    def gen_moves(self):
        moves = []
        self._check_budget()
        moves.extend(self.gen_type_1_single())
        self._check_budget()
        moves.extend(self.gen_type_2_pair())
        self._check_budget()
        moves.extend(self.gen_type_3_triple())
        self._check_budget()
        moves.extend(self.gen_type_4_bomb())
        self._check_budget()
        moves.extend(self.gen_type_5_king_bomb())
        self._check_budget()
        moves.extend(self.gen_type_6_3_1())
        self._check_budget()
        moves.extend(self.gen_type_7_3_2())
        self._check_budget()
        moves.extend(self.gen_type_8_serial_single())
        self._check_budget()
        moves.extend(self.gen_type_9_serial_pair())
        self._check_budget()
        moves.extend(self.gen_type_10_serial_triple())
        self._check_budget()
        moves.extend(self.gen_type_11_serial_3_1())
        self._check_budget()
        moves.extend(self.gen_type_12_serial_3_2())
        self._check_budget()
        moves.extend(self.gen_type_13_4_2())
        self._check_budget()
        moves.extend(self.gen_type_14_4_22())
        self._check_budget()
        return moves
