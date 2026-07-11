import multiprocessing as mp
import pickle
import random

from douzero.env.game import GameEnv
from douzero.env.rules import RuleSet

def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict, q):

    players = load_card_play_models(card_play_model_path_dict)

    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()

    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']
         ))

def _random_bidding(env, rng):
    """Run a random bidding sequence and return True if the game should redeal.

    Each bidder bids a random value from the ruleset's bid_values. If all
    pass and ``all_pass_redeal`` is set, redeal (caller loops). Otherwise the
    landlord is determined and the game transitions to PLAYING.
    """
    for _ in range(len(env.bidding_order)):
        bid = rng.choice(env.ruleset.bid_values)
        redeal = env.step_bidding(bid)
        if redeal:
            return True
    return False

def mp_simulate_standard(card_play_data_list, card_play_model_path_dict, q, seed=0):
    """Standard-mode multiprocessing simulation.

    Each deal is a v2-format dict with a full deck. The worker deals
    17+17+17+3, runs random bidding (with redeal on all-pass), then plays
    to terminal using the loaded agents.
    """
    from douzero.evaluation.legacy_data_adapter import deal_standard_deck

    players = load_card_play_models(card_play_model_path_dict)
    ruleset = RuleSet.standard()
    env = GameEnv(players, ruleset=ruleset)
    rng = random.Random(seed)

    for idx, deal in enumerate(card_play_data_list):
        # Deal from the full deck.
        card_play_data = deal_standard_deck(deal['deck'])
        bidding_order = deal.get('bidding_order', ['landlord', 'landlord_down', 'landlord_up'])

        # Keep retrying until bidding produces a landlord (handles redeal).
        for _attempt in range(100):
            env.reset()
            env.card_play_init_standard(card_play_data, bidding_order=bidding_order)
            if not _random_bidding(env, rng):
                break
            # Redeal: reshuffle the same deck for a new attempt.
            rng.shuffle(card_play_data['landlord'])
            # Actually need a fresh shuffle of the full deck.
            deck_copy = list(deal['deck'])
            rng.shuffle(deck_copy)
            card_play_data = deal_standard_deck(deck_copy)
        else:
            # Could not resolve bidding after 100 attempts; skip this deal.
            continue

        # Play to terminal.
        env.game_infoset = env.get_infoset()
        while not env.game_over:
            env.step()
        # reset() is called at the top of the loop; no need here.

    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']
         ))

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers, ruleset=None):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for idx, card_paly_data in enumerate(card_play_data_list_each_worker):
        if ruleset == 'standard':
            p = ctx.Process(
                target=mp_simulate_standard,
                args=(card_paly_data, card_play_model_path_dict, q, idx))
        else:
            p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    if num_total_wins == 0:
        print('No games completed.')
        return
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins))
