import argparse
import pickle
import numpy as np

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=10000, type=int)
    parser.add_argument('--ruleset', default='legacy', type=str,
                        choices=['legacy', 'standard'],
                        help='Deal format: legacy (4-key card_play_data) or '
                             'standard (full deck + first_bidder + ruleset_id)')
    return parser

def generate():
    """Generate a legacy deal: landlord gets 20, up 17, down 17, bottom 3."""
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {'landlord': _deck[:20],
                      'landlord_up': _deck[20:37],
                      'landlord_down': _deck[37:54],
                      'three_landlord_cards': _deck[17:20],
                      }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


def generate_standard():
    """Generate a standard deal: full deck order + first_bidder + ruleset_id.

    The deal is stored as the complete 54-card deck order (not pre-sliced
    into hands) so that the evaluation pipeline can deal 17+17+17+3 and
    run bidding. Uses neutral seat labels ("0", "1", "2") for first_bidder
    and bidding_order, matching the standard state machine's BIDDING phase.
    """
    from douzero.env.rules import RuleSet

    rs = RuleSet.standard()
    _deck = deck.copy()
    np.random.shuffle(_deck)
    # Randomly choose the first bidder seat (0, 1, or 2).
    first = np.random.randint(0, 3)
    order = [str((first + i) % 3) for i in range(3)]
    return {
        'format_version': 2,
        'schema_version': 1,
        'ruleset_id': rs.ruleset_id,
        'ruleset_version': rs.ruleset_version,
        'ruleset_hash': rs.stable_hash(),
        'deck': list(_deck),
        'first_bidder': str(first),
        'bidding_order': order,
        'bidding_script': None,
    }


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'

    print("output_pickle:", output_pickle)
    print("ruleset:", flags.ruleset)
    print("generating data...")

    data = []
    if flags.ruleset == 'standard':
        for _ in range(flags.num_games):
            data.append(generate_standard())
    else:
        for _ in range(flags.num_games):
            data.append(generate())

    print("saving pickle file...")
    with open(output_pickle,'wb') as g:
        pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)




