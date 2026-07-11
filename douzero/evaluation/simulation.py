import hashlib
import multiprocessing as mp
import random
import traceback

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
    """Legacy-mode worker: replay fixed deals, no bidding."""

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


def _derive_game_seed(eval_seed, game_index, deck, first_bidder):
    """Derive a deterministic per-game seed from the eval seed, game index,
    deck hash, and first bidder seat.

    This ensures that the same game (same deck, same index, same first bidder)
    always produces the same bidding sequence regardless of worker count or
    scheduling. The seed is NOT derived from PID or worker index.
    """
    deck_hash = hashlib.sha256(str(deck).encode()).hexdigest()[:8]
    token = f"douzero-eval|{eval_seed}|{game_index}|{deck_hash}|{first_bidder}"
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFFFFFF


def _random_bidding(env, rng):
    """Run a random bidding sequence and return True if the game should redeal.

    Each bidder bids a random LEGAL value (from ``env.get_legal_bids()``). If
    all pass and ``all_pass_redeal`` is set, redeal (caller loops). Otherwise
    the landlord is determined and the game transitions to PLAYING. If a
    maximum bid (3) is played, bidding ends immediately and this function
    detects the phase change and stops early.

    This is a bidding POLICY that lives in the evaluation/agent layer. The
    GameEnv itself only exposes ``get_legal_bids`` and ``step_bidding``; it
    never runs a bidding policy internally. Future SL/RL bidding agents
    replace this function while keeping the same GameEnv interface.
    """
    from douzero.env.rules import PHASE_BIDDING
    for _ in range(len(env.bidding_order)):
        if env.phase != PHASE_BIDDING:
            # Bidding ended early (e.g., a max bid of 3 was played).
            return False
        legal_bids = env.get_legal_bids()
        bid = rng.choice(legal_bids)
        redeal = env.step_bidding(bid)
        if redeal:
            return True
    return False


def mp_simulate_standard(card_play_data_list, card_play_model_path_dict, q,
                         eval_seed=0, global_indices=None, ruleset=None):
    """Standard-mode multiprocessing simulation.

    Each deal is a v2-format dict with a full deck, first_bidder, and
    bidding_order (using neutral seat labels). The worker deals 17+17+17+3,
    runs random bidding (with redeal on all-pass), then plays to terminal.

    The per-game bidding seed is derived from ``eval_seed``, the game's GLOBAL
    index, the deck, and the first bidder seat — so the same game always
    produces the same bidding sequence regardless of which worker processes it.

    ``ruleset`` is the active RuleSet (from config or RuleSet.standard()).
    """
    from douzero.evaluation.legacy_data_adapter import deal_standard_deck

    if ruleset is None:
        ruleset = RuleSet.standard()

    try:
        players = load_card_play_models(card_play_model_path_dict)
        env = GameEnv(players, ruleset=ruleset)

        for idx, deal in enumerate(card_play_data_list):
            game_index = global_indices[idx] if global_indices is not None else idx
            first_bidder = deal.get('first_bidder', '0')
            game_seed = _derive_game_seed(eval_seed, game_index, deal['deck'], first_bidder)
            rng = random.Random(game_seed)
            # Seed the global random module so RandomAgent (which uses
            # random.choice) is also deterministic per-game.
            random.seed(game_seed)

            # Deal from the full deck.
            card_play_data = deal_standard_deck(deal['deck'])
            # Use the deal's bidding_order (neutral seats), not a hardcoded default.
            bidding_order = deal.get('bidding_order', ['0', '1', '2'])

            # Keep retrying until bidding produces a landlord (handles redeal).
            # Bounded by ruleset.max_redeals.
            for _attempt in range(ruleset.max_redeals + 1):
                env.reset()
                env.card_play_init_standard(card_play_data, bidding_order=bidding_order)
                if not _random_bidding(env, rng):
                    break
                # Redeal: reshuffle the full deck deterministically using the
                # same game-local RNG (not a global or worker RNG).
                deck_copy = list(deal['deck'])
                rng.shuffle(deck_copy)
                card_play_data = deal_standard_deck(deck_copy)
            else:
                # Exceeded max_redeals: force-assign landlord to first bidder.
                env.reset()
                env.card_play_init_standard(card_play_data, bidding_order=bidding_order)
                env.landlord_position = env.bidding_order[0]
                env.bid_value = 1
                env._reveal_bottom_cards()

            # Play to terminal.
            env.game_infoset = env.get_infoset()
            while not env.game_over:
                env.step()

        q.put((env.num_wins['landlord'],
               env.num_wins['farmer'],
               env.num_scores['landlord'],
               env.num_scores['farmer']
             ))
    except Exception:
        # Send the traceback to the parent so it can report a precise error
        # instead of hanging on q.get() forever.
        q.put(('error', traceback.format_exc()))


def data_allocation_per_worker(card_play_data_list, num_workers):
    """Split data round-robin, tracking the global game index for each game.

    Returns a list of (worker_data, global_indices) tuples so each worker
    knows its games' original global indices for deterministic seed
    derivation. This ensures the same game always gets the same seed
    regardless of worker count.
    """
    worker_data = [[] for _ in range(num_workers)]
    worker_indices = [[] for _ in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        w = idx % num_workers
        worker_data[w].append(data)
        worker_indices[w].append(idx)
    return list(zip(worker_data, worker_indices))


def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers,
             ruleset=None, eval_seed=0, ruleset_obj=None):
    """Run evaluation. Uses the adapter to validate data format BEFORE spawning
    workers, so format/schema/hash/deck mismatches fail with a precise error
    instead of crashing inside a child process.

    ``ruleset_obj`` is an optional RuleSet instance built from config. When
    provided, it is passed to the workers and used as the active rule set
    (overriding the hardcoded RuleSet.standard()).
    """
    from douzero.evaluation.legacy_data_adapter import load_eval_data

    # Validate data format up front (before any model loading or worker spawn).
    ruleset_name = ruleset or "legacy"
    active_ruleset = ruleset_obj if ruleset_obj is not None else (
        RuleSet.standard() if ruleset == 'standard' else RuleSet.legacy()
    )
    card_play_data_list = load_eval_data(
        eval_data, ruleset=ruleset_name, expected_ruleset=active_ruleset
    )

    worker_assignments = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for worker_data, global_indices in worker_assignments:
        if ruleset == 'standard':
            p = ctx.Process(
                target=mp_simulate_standard,
                args=(worker_data, card_play_model_path_dict, q,
                      eval_seed, global_indices, active_ruleset))
        else:
            p = ctx.Process(
                target=mp_simulate,
                args=(worker_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    # Collect results; detect child-process errors so we don't hang.
    for i in range(num_workers):
        result = q.get()
        if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
            raise RuntimeError(
                f"Worker {i} crashed with the following traceback:\n{result[1]}"
            )
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
