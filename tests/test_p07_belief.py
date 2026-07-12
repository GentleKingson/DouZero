"""P07 belief-model tests.

Covers the AGENTS.md "Belief-model rules" test matrix:

- per-rank allocations cannot exceed unseen counts (legal mask + sampled
  allocations respect the cap),
- joker counts are at most one each,
- each opponent's total equals the public remaining-card count (DP total
  constraint),
- the two opponent hands sum exactly to the unseen pool (A + B = pool),
- known public bottom cards are excluded from the farmer's unknown pool,
- decoding and sampling never hang / always produce a legal allocation,
- the public belief input is invariant under hidden re-allocation (leakage),
- the masked cross-entropy loss is finite, disables at zero weights, and flows
  gradient,
- the label builder rejects inconsistent (privileged vs public) inputs.

The headline acceptance criterion is the 1000-random-state conservation sweep:
for every sampled allocation, A's hand + B's hand equals the unknown pool
exactly, with no negative count and the correct per-opponent total —
irrespective of the (randomly initialized) model's output.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from douzero.belief import (
    BELIEF_FEATURE_DIM,
    BELIEF_INPUT_DIM,
    BELIEF_RANKS,
    BeliefConfig,
    BeliefDPError,
    BeliefModel,
    canonical_opponent,
    canonical_opponent_b,
    belief_features_from_probs,
    belief_loss,
    belief_metrics,
    build_belief_input,
    build_belief_label,
    decode_map,
    expected_counts_from_probs,
    legal_mask,
    per_rank_counts,
    sample_allocation,
    target_allocation_tensor,
    unseen_counts_per_rank,
)
from douzero.belief.constraints import (
    JOKER_MAX_COUNT,
    NUMERIC_MAX_COUNT,
    NUM_BELIEF_RANKS,
    NUM_COUNT_SLOTS,
    _max_count_for_rank_index,
    is_joker_rank,
)
from douzero.observation.cards import DECK, JOKERS, NUMERIC_RANKS
from douzero.observation.public import compute_belief_unknown_pool, compute_unseen_pool


# --------------------------------------------------------------------------- #
# Test state collection via random self-play
# --------------------------------------------------------------------------- #
def _play_random_episodes(num_episodes: int, seed: int = 20240707):
    """Play random self-play games and yield each decision's infoset.

    Returns a list of infosets (one per non-trivial decision). Uses the legacy
    card-play env (no bidding) so the focus stays on the belief pool math.
    """
    from douzero.env.env import Env

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    out = []
    for _ in range(num_episodes):
        env = Env("adp")
        env.reset()
        steps = 0
        while True:
            assert steps < 1000, "random episode did not terminate"
            steps += 1
            infoset = env.infoset
            legal = list(infoset.legal_actions)
            if not legal:
                break
            # Prefer a non-empty action to make progress; pass only when forced.
            nonempty = [a for a in legal if len(a) > 0]
            pool = nonempty if nonempty else legal
            action = list(pool[int(rng.integers(len(pool)))])
            # Record the decision BEFORE stepping (infoset reflects the state).
            if len(legal) > 1:
                out.append(infoset)
            _obs, _r, done, _info = env.step(action)
            if done:
                break
    return out


def _build_input_and_label(infoset):
    """Build the public belief input + the privileged label for an infoset."""
    from douzero.observation.encode_v2 import get_obs_v2

    obs = get_obs_v2(infoset)
    binput = build_belief_input(obs.public)
    label = build_belief_label(
        acting_role=infoset.player_position,
        all_handcards=infoset.all_handcards,
        unseen_counts=binput.unseen_counts,
        num_cards_left=infoset.num_cards_left_dict,
        bottom_unplayed=infoset.three_landlord_cards,
    )
    return binput, label


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
class TestConstraints:
    def test_belief_ranks_are_15_categories(self):
        assert len(BELIEF_RANKS) == 15
        assert BELIEF_RANKS[:13] == tuple(NUMERIC_RANKS)
        assert BELIEF_RANKS[13:] == tuple(JOKERS)

    def test_canonical_opponents_are_next_and_previous(self):
        # landlord -> next is landlord_down, previous is landlord_up
        assert canonical_opponent("landlord") == "landlord_down"
        assert canonical_opponent_b("landlord") == "landlord_up"
        # landlord_up -> next is landlord, previous is landlord_down
        assert canonical_opponent("landlord_up") == "landlord"
        assert canonical_opponent_b("landlord_up") == "landlord_down"

    def test_legal_mask_caps_numeric_at_four_and_joker_at_one(self):
        # A full unseen pool: numeric ranks can be 0..4, jokers 0..1.
        pool = list(DECK)
        unseen = unseen_counts_per_rank(pool)
        mask = legal_mask(unseen)
        assert mask.shape == (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        for r in range(NUM_BELIEF_RANKS):
            cap = NUMERIC_MAX_COUNT if not is_joker_rank(r) else JOKER_MAX_COUNT
            for k in range(NUM_COUNT_SLOTS):
                assert mask[r, k] == (k <= cap), (r, k)

    def test_legal_mask_bounds_by_unseen_count(self):
        # Only two 3s unseen: count slots 0,1,2 legal (<=2), 3,4 illegal.
        unseen = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)
        unseen[0] = 2  # rank 3
        mask = legal_mask(unseen)
        assert mask[0].tolist() == [True, True, True, False, False]

    def test_legal_mask_rejects_negative_unseen(self):
        unseen = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)
        unseen[0] = -1
        with pytest.raises(ValueError):
            legal_mask(unseen)

    def test_per_rank_counts_round_trip(self):
        hand = [3, 3, 3, 17, 17, 20, 30]
        counts = per_rank_counts(hand)
        assert counts[BELIEF_RANKS.index(3)] == 3
        assert counts[BELIEF_RANKS.index(17)] == 2
        assert counts[BELIEF_RANKS.index(20)] == 1
        assert counts[BELIEF_RANKS.index(30)] == 1
        assert counts.sum() == len(hand)

    def test_expected_counts_from_probs(self):
        # A uniform-over-5 distribution has expected count 2.0.
        probs = np.full((1, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), 0.2)
        ec = expected_counts_from_probs(probs)
        assert ec.shape == (1, NUM_BELIEF_RANKS)
        np.testing.assert_allclose(ec, 2.0)


# --------------------------------------------------------------------------- #
# Dynamic programming: MAP and sampling
# --------------------------------------------------------------------------- #
class TestDynamicProgramming:
    def test_map_recovers_uniform_argmax_under_total_constraint(self):
        # Rank 0 strongly prefers count 2; rank 1 strongly prefers count 2;
        # total must be 3 -> DP must pick (2,1) or (1,2). With rank0 slightly
        # higher, it picks rank0=2, rank1=1.
        logp = np.full((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), -5.0)
        logp[0, 2] = 0.0
        logp[1, 2] = -0.1
        logp[1, 1] = -0.2
        alloc = decode_map(logp, total=3)
        assert alloc.sum() == 3
        assert alloc[0] == 2
        assert alloc[1] == 1
        assert alloc[2:].sum() == 0

    def test_map_respects_joker_cap(self):
        # Joker slots (indices 13, 14) can only take 0 or 1; force total over
        # them. Put all mass on count 1 for both jokers; total among jokers = 2.
        logp = np.full((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), -5.0)
        logp[13, 1] = 0.0
        logp[14, 1] = 0.0
        # The legal mask is applied by the caller; here slots 2..4 for jokers
        # are illegal. Mask them to -inf to emulate legal_mask.
        for j in (13, 14):
            for k in (2, 3, 4):
                logp[j, k] = -np.inf
        alloc = decode_map(logp, total=2)
        assert alloc[13] == 1
        assert alloc[14] == 1
        assert alloc.sum() == 2

    def test_map_total_zero_returns_all_zero(self):
        logp = np.zeros((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS))
        alloc = decode_map(logp, total=0)
        assert alloc.sum() == 0

    def test_map_raises_on_infeasible_total(self):
        # All slots illegal except count 0 everywhere -> cannot reach total 3.
        logp = np.full((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), -np.inf)
        logp[:, 0] = 0.0
        with pytest.raises(BeliefDPError):
            decode_map(logp, total=3)

    def test_sample_is_total_consistent_and_stochastic(self):
        rng = np.random.default_rng(7)
        # Only ranks 0 and 1 may carry counts; all other ranks locked to 0.
        logp = np.full((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), -np.inf)
        for r in range(2, NUM_BELIEF_RANKS):
            logp[r, 0] = 0.0  # forced zero
        logp[0, 0] = 0.0
        logp[0, 1] = 0.0
        logp[0, 2] = -0.5
        logp[1, 0] = 0.0
        logp[1, 1] = 0.0
        logp[1, 2] = -0.5
        total = 3
        seen = set()
        for _ in range(40):
            a = sample_allocation(logp, total=total, rng=rng)
            assert a.sum() == total
            assert a[2:].sum() == 0
            assert 1 <= a[0] <= 2 and 1 <= a[1] <= 2
            seen.add((int(a[0]), int(a[1])))
        # Both (1,2) and (2,1) should appear (stochastic, not greedy).
        assert (2, 1) in seen and (1, 2) in seen

    def test_sample_distribution_matches_unnormalized_weights(self):
        # Two ranks (0, 1) share total=1; ranks 2..14 forced to 0. With known
        # per-rank slot weights the constrained marginal of (rank0=1) must
        # match the normalized weight.
        rng = np.random.default_rng(123)
        logp = np.full((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), -np.inf)
        for r in range(2, NUM_BELIEF_RANKS):
            logp[r, 0] = 0.0  # forced zero
        # Rank 0: P(0)=0.25, P(1)=0.75. Rank 1: P(0)=1, P(1)=1 (uninformative).
        logp[0, 0] = np.log(0.25)
        logp[0, 1] = np.log(0.75)
        logp[1, 0] = 0.0
        logp[1, 1] = 0.0
        n = 4000
        rank0_ones = 0
        for _ in range(n):
            a = sample_allocation(logp, total=1, rng=rng)
            assert a.sum() == 1
            rank0_ones += int(a[0])
        frac = rank0_ones / n
        # P(rank0=1) = 0.75 / (0.75 + 0.25) = 0.75.
        assert 0.70 < frac < 0.80


# --------------------------------------------------------------------------- #
# Constrained marginals (Blocker #3): the per-rank posterior conditioned on
# the total-count constraint, used by the value-fusion features.
# --------------------------------------------------------------------------- #
class TestConstrainedMarginals:
    def _logp(self):
        # Two ranks share the total; others forced to 0.
        logp = np.full((NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), -np.inf)
        for r in range(2, NUM_BELIEF_RANKS):
            logp[r, 0] = 0.0
        logp[0, 0] = np.log(0.4)
        logp[0, 1] = np.log(0.6)
        logp[1, 0] = np.log(0.5)
        logp[1, 1] = np.log(0.5)
        return logp

    def test_expected_total_equals_target_exactly(self):
        from douzero.belief import constrained_marginals

        for total in (0, 1, 2):
            marg = constrained_marginals(self._logp(), total=total)
            counts = np.arange(NUM_COUNT_SLOTS, dtype=np.float64)
            expected_total = float((marg * counts).sum())
            assert abs(expected_total - total) < 1e-9, (total, expected_total)

    def test_different_total_produces_different_posterior(self):
        """Same logits/unseen, different total => different marginals."""
        from douzero.belief import constrained_marginals

        logp = self._logp()
        m1 = constrained_marginals(logp, total=1)
        m2 = constrained_marginals(logp, total=2)
        assert not np.allclose(m1, m2)
        # Both must have expected totals matching their targets.
        counts = np.arange(NUM_COUNT_SLOTS, dtype=np.float64)
        assert abs((m1 * counts).sum() - 1) < 1e-9
        assert abs((m2 * counts).sum() - 2) < 1e-9

    def test_rows_sum_to_one_over_legal_slots(self):
        from douzero.belief import constrained_marginals

        marg = constrained_marginals(self._logp(), total=1)
        assert marg.shape == (NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        for r in range(NUM_BELIEF_RANKS):
            assert abs(marg[r].sum() - 1.0) < 1e-9

    def test_total_zero_forces_all_zero_counts(self):
        from douzero.belief import constrained_marginals

        marg = constrained_marginals(self._logp(), total=0)
        # Every rank's mass is on count 0 (the only feasible per-rank value).
        assert np.allclose(marg[:, 0], 1.0)

    def test_belief_features_reject_unconstrained_factor_probs(self):
        """belief_features_from_probs must reject an independent softmax."""
        # Independent per-rank softmax: expected total generally != target.
        factor = np.full((1, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS), 0.5)
        factor[0, 0, :2] = 0.5  # only slots 0,1 legal for rank 0
        factor[0, 0, 2:] = 0.0
        unseen = np.zeros((1, NUM_BELIEF_RANKS), dtype=np.int64)
        unseen[0, 0] = 1
        with pytest.raises(ValueError):
            belief_features_from_probs(
                factor, np.array([1]), unseen, assert_constrained=True,
            )

    def test_belief_features_accept_constrained_marginals(self):
        from douzero.belief import constrained_marginals

        marg = constrained_marginals(self._logp(), total=1)[None, ...]
        unseen = np.ones((1, NUM_BELIEF_RANKS), dtype=np.int64)
        feat = belief_features_from_probs(marg, np.array([1]), unseen)
        assert feat.shape == (1, BELIEF_FEATURE_DIM)
        assert np.all(np.isfinite(feat))


# --------------------------------------------------------------------------- #
# Headline: 1000-state conservation sweep on real game states
# --------------------------------------------------------------------------- #
class TestConservationSweep:
    def test_1000_random_states_sampled_allocations_conserved(self, seed_factory):
        """Every sampled allocation must be card-conservative (AGENTS.md)."""
        seed_factory(20240707)
        infosets = _play_random_episodes(num_episodes=60)
        assert len(infosets) >= 200, len(infosets)
        torch.manual_seed(7)
        model = BeliefModel(BeliefConfig(hidden_size=64, num_layers=1))
        model.eval()
        rng = np.random.default_rng(99)

        checked = 0
        for infoset in infosets:
            binput, label = _build_input_and_label(infoset)
            with torch.no_grad():
                out = model([binput])
            # MAP decode conservation.
            map_alloc = model.decode_map(out)[0]
            assert map_alloc.sum() == binput.opponent_a_total
            assert np.all(map_alloc >= 0)
            assert np.all(map_alloc <= binput.unseen_counts)
            # Sampled conservation (3 samples each).
            samples = model.sample(out, rng=rng, num_samples=3)[0]
            for s in samples:
                assert s.sum() == binput.opponent_a_total
                assert np.all(s >= 0)
                assert np.all(s <= binput.unseen_counts)
                # Opponent B = unseen - A; must be non-negative and total-consistent.
                b = binput.unseen_counts - s
                assert np.all(b >= 0)
                assert b.sum() == binput.opponent_b_total
                # A + B must reconstruct the full unseen pool.
                recon = s + b
                np.testing.assert_array_equal(recon, binput.unseen_counts)
            # Constrained-marginal expected total must equal the target exactly
            # (Blocker #3): the value-fusion features are conservation-safe.
            exp_total = float(out.expected_counts[0].sum())
            assert abs(exp_total - binput.opponent_a_total) < 1e-6, (
                exp_total, binput.opponent_a_total
            )
            checked += 1
            if checked >= 1000:
                break
        assert checked >= 1000, f"only {checked} decision points checked"

    def test_joker_counts_never_exceed_one_in_samples(self, seed_factory):
        seed_factory(20240707)
        infosets = _play_random_episodes(num_episodes=40)
        torch.manual_seed(3)
        model = BeliefModel(BeliefConfig(hidden_size=32, num_layers=1))
        model.eval()
        rng = np.random.default_rng(5)
        joker_indices = (13, 14)
        for infoset in infosets:
            binput, _label = _build_input_and_label(infoset)
            with torch.no_grad():
                out = model([binput])
            samples = model.sample(out, rng=rng, num_samples=2)[0]
            for s in samples:
                for j in joker_indices:
                    assert s[j] <= 1, (j, int(s[j]))


# --------------------------------------------------------------------------- #
# Imperfect-information boundary (leakage)
# --------------------------------------------------------------------------- #
class TestLeakageBoundary:
    def test_belief_input_invariant_under_hidden_reallocation(
        self, fixed_card_play_data
    ):
        """Same landlord public info, swapped farmer cards -> identical input.

        From the landlord's perspective the public footprint is unchanged when
        two cards are swapped between the farmers, so the belief input vector
        must be byte-identical. This is the imperfect-information invariant.
        """
        import copy

        from douzero.env.game import GameEnv

        class _NoopAgent:
            def act(self, infoset):
                return infoset.legal_actions[0]

        data_a = copy.deepcopy(fixed_card_play_data)
        data_b = copy.deepcopy(fixed_card_play_data)
        # Swap a card between the two farmers.
        up_b = list(data_b["landlord_up"])
        down_b = list(data_b["landlord_down"])
        if up_b[0] != down_b[0]:
            up_b[0], down_b[0] = down_b[0], up_b[0]
            data_b["landlord_up"] = sorted(up_b)
            data_b["landlord_down"] = sorted(down_b)

        def build(data):
            players = {p: _NoopAgent() for p in
                       ["landlord", "landlord_up", "landlord_down"]}
            env = GameEnv(players)
            env.card_play_init(data)
            return env.game_infoset

        from douzero.observation.encode_v2 import get_obs_v2

        obs_a = get_obs_v2(build(data_a))
        obs_b = get_obs_v2(build(data_b))
        inp_a = build_belief_input(obs_a.public)
        inp_b = build_belief_input(obs_b.public)
        np.testing.assert_array_equal(
            inp_a.feature_vector, inp_b.feature_vector
        )
        # But the TRUE allocations differ (the label must change).
        iset_a = build(data_a)
        iset_b = build(data_b)
        lab_a = build_belief_label(
            acting_role="landlord",
            all_handcards=iset_a.all_handcards,
            unseen_counts=inp_a.unseen_counts,
            num_cards_left=iset_a.num_cards_left_dict,
            bottom_unplayed=iset_a.three_landlord_cards,
        )
        lab_b = build_belief_label(
            acting_role="landlord",
            all_handcards=iset_b.all_handcards,
            unseen_counts=inp_b.unseen_counts,
            num_cards_left=iset_b.num_cards_left_dict,
            bottom_unplayed=iset_b.three_landlord_cards,
        )
        assert not np.array_equal(lab_a.allocation, lab_b.allocation)

    def test_belief_input_excludes_farmer_bottom_cards(self):
        """A farmer's belief pool excludes the unplayed public bottom cards."""
        # Hand-construct a CONSISTENT farmer observation: landlord_up holds 17
        # cards (none of which are the bottom cards 3,4,5), landlord has 20,
        # landlord_down has 17, bottom unplayed = [3,4,5].
        from douzero.observation.public import build_public_observation

        my_hand = [6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 12, 13, 14, 17, 20, 30]
        assert len(my_hand) == 17
        played = {"landlord": [], "landlord_up": [], "landlord_down": []}
        public = build_public_observation(
            acting_role="landlord_up",
            my_handcards=my_hand,
            other_handcards=compute_unseen_pool(my_hand, played, [3, 4, 5]),
            played_cards=played,
            last_move=[],
            last_move_dict={},
            three_landlord_cards=[3, 4, 5],
            three_landlord_cards_revealed=[3, 4, 5],
            num_cards_left={
                "landlord": 20, "landlord_up": 17, "landlord_down": 17
            },
            legal_actions=[()],
        )
        binput = build_belief_input(public)
        # The three bottom-card ranks are known landlord property -> excluded
        # from the farmer's belief pool (4 copies each, minus 1 bottom = 3).
        assert binput.unseen_counts[BELIEF_RANKS.index(3)] == 3
        assert binput.unseen_counts[BELIEF_RANKS.index(4)] == 3
        assert binput.unseen_counts[BELIEF_RANKS.index(5)] == 3
        # Opponent A (landlord) hidden total excludes the 3 bottom cards.
        assert binput.opponent_a_total == 17  # 20 - 3 bottom
        # Pool total equals the two opponents' *hidden* totals.
        assert int(binput.unseen_counts.sum()) == (
            binput.opponent_a_total + binput.opponent_b_total
        )


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
class TestLabels:
    def test_label_matches_true_opponent_hand(self, fixed_card_play_data):
        from douzero.env.game import GameEnv

        class _NoopAgent:
            def act(self, infoset):
                return infoset.legal_actions[0]

        players = {p: _NoopAgent() for p in
                   ["landlord", "landlord_up", "landlord_down"]}
        env = GameEnv(players)
        env.card_play_init(fixed_card_play_data)
        infoset = env.game_infoset
        from douzero.observation.encode_v2 import get_obs_v2

        obs = get_obs_v2(infoset)
        binput = build_belief_input(obs.public)
        label = build_belief_label(
            acting_role=infoset.player_position,
            all_handcards=infoset.all_handcards,
            unseen_counts=binput.unseen_counts,
            num_cards_left=infoset.num_cards_left_dict,
            bottom_unplayed=infoset.three_landlord_cards,
        )
        opp_a = canonical_opponent(infoset.player_position)
        true_counts = per_rank_counts(infoset.all_handcards[opp_a])
        np.testing.assert_array_equal(label.allocation, true_counts)
        assert label.opponent_a_total == len(infoset.all_handcards[opp_a])
        # Opponent B by subtraction matches B's true hand too.
        opp_b = canonical_opponent_b(infoset.player_position)
        true_b = per_rank_counts(infoset.all_handcards[opp_b])
        np.testing.assert_array_equal(binput.unseen_counts - label.allocation, true_b)

    def test_label_rejects_inconsistent_total(self):
        unseen = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64)
        unseen[0] = 4
        # True hand has 4 cards but num_cards_left says 3 -> inconsistent.
        with pytest.raises(ValueError):
            build_belief_label(
                acting_role="landlord",
                all_handcards={"landlord_down": [3, 3, 3, 3],
                               "landlord_up": [], "landlord": []},
                unseen_counts=unseen,
                num_cards_left={"landlord_down": 3, "landlord_up": 0,
                                "landlord": 0},
                bottom_unplayed=[],
            )

    def test_target_allocation_tensor_shape(self):
        a1 = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64); a1[0] = 2
        a2 = np.zeros(NUM_BELIEF_RANKS, dtype=np.int64); a2[1] = 1
        t = target_allocation_tensor([a1, a2])
        assert t.shape == (2, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        assert t[0, 0, 2] == 1.0
        assert t[1, 1, 1] == 1.0
        # One slot is set per rank per sample (incl. count-0 ranks).
        assert t.sum() == 2 * NUM_BELIEF_RANKS


# --------------------------------------------------------------------------- #
# Loss + metrics
# --------------------------------------------------------------------------- #
class TestBeliefLoss:
    def _make_batch(self, B=4):
        torch.manual_seed(0)
        logits = torch.randn(
            B, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS, requires_grad=True
        )
        # Legal mask: numeric 0..4, jokers 0..1, all unseen = 4.
        legal = torch.zeros(B, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS, dtype=torch.bool)
        for r in range(NUM_BELIEF_RANKS):
            cap = _max_count_for_rank_index(r)
            legal[:, r, :cap + 1] = True
        target = torch.zeros(B, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        for b in range(B):
            for r in range(NUM_BELIEF_RANKS):
                cap = _max_count_for_rank_index(r)
                k = int(torch.randint(0, cap + 1, (1,)).item())
                target[b, r, k] = 1.0
        return logits, legal, target

    def test_loss_is_finite_and_has_grad(self):
        logits, legal, target = self._make_batch()
        comps = belief_loss(logits, target, legal)
        assert torch.isfinite(comps.total).all()
        assert comps.cross_entropy > 0
        comps.total.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_loss_zero_reg_means_pure_cross_entropy(self):
        logits, legal, target = self._make_batch()
        a = belief_loss(logits, target, legal)
        b = belief_loss(logits, target, legal,
                        lambda_count_reg=0.0, lambda_entropy_reg=0.0)
        assert abs(a.cross_entropy - b.cross_entropy) < 1e-6

    def test_loss_count_reg_adds_finite_term(self):
        logits, legal, target = self._make_batch()
        comps = belief_loss(logits, target, legal, lambda_count_reg=0.5)
        assert torch.isfinite(comps.total).all()
        assert comps.count_reg >= 0.0

    def test_loss_rejects_shape_mismatch(self):
        logits, legal, target = self._make_batch()
        bad = target[:, :, :4]
        with pytest.raises(ValueError):
            belief_loss(logits, bad, legal)

    def test_metrics_on_perfect_prediction(self):
        # Construct probs that exactly match the target -> perfect metrics.
        B = 3
        target_alloc = np.zeros((B, NUM_BELIEF_RANKS), dtype=np.int64)
        target_alloc[:, 0] = 2
        target_onehot = torch.zeros(B, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        legal = torch.ones(B, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS, dtype=torch.bool)
        for j in (13, 14):
            legal[:, j, 2:] = False
        for b in range(B):
            for r in range(NUM_BELIEF_RANKS):
                target_onehot[b, r, target_alloc[b, r]] = 1.0
        probs = target_onehot.numpy()
        m = belief_metrics(probs, target_alloc, legal.numpy())
        assert m["rank_accuracy"] == 1.0
        assert m["exact_match"] == 1.0
        assert m["count_mae"] == 0.0


# --------------------------------------------------------------------------- #
# Model forward + value-fusion feature projection
# --------------------------------------------------------------------------- #
class TestBeliefModelForward:
    def test_forward_shapes_and_masked_normalization(self, seed_factory):
        seed_factory(7)
        infosets = _play_random_episodes(num_episodes=5)
        model = BeliefModel(BeliefConfig(hidden_size=32, num_layers=1))
        model.eval()
        binput = build_belief_input(
            _get_obs_public(infosets[0]))
        with torch.no_grad():
            out = model([binput])
        assert out.logits.shape == (1, NUM_BELIEF_RANKS, NUM_COUNT_SLOTS)
        assert out.legal.shape == out.logits.shape
        # Probs over legal slots sum to 1; illegal slots are ~0.
        probs = out.probs.numpy()[0]
        for r in range(NUM_BELIEF_RANKS):
            legal_slots = out.legal.numpy()[0, r]
            np.testing.assert_allclose(probs[r, legal_slots].sum(), 1.0, atol=1e-5)
            assert probs[r, ~legal_slots].sum() < 1e-6
        # Expected counts respect the legal cap.
        cap = np.array([_max_count_for_rank_index(r) for r in range(NUM_BELIEF_RANKS)])
        assert np.all(out.expected_counts[0] <= cap + 1e-6)

    def test_belief_features_shape_and_finite(self, seed_factory):
        seed_factory(7)
        infosets = _play_random_episodes(num_episodes=5)
        model = BeliefModel(BeliefConfig(hidden_size=32, num_layers=1))
        model.eval()
        binput = build_belief_input(_get_obs_public(infosets[0]))
        with torch.no_grad():
            out = model([binput])
        feat = belief_features_from_probs(
            out.probs.numpy(),
            out.opponent_a_total,
            np.stack([binput.unseen_counts]),
        )
        assert feat.shape == (1, BELIEF_FEATURE_DIM)
        assert np.all(np.isfinite(feat))

    def test_belief_input_dim_constant_matches_feature_vector(self, seed_factory):
        seed_factory(7)
        infosets = _play_random_episodes(num_episodes=3)
        binput = build_belief_input(_get_obs_public(infosets[0]))
        assert binput.feature_vector.shape == (BELIEF_INPUT_DIM,)


def _get_obs_public(infoset):
    from douzero.observation.encode_v2 import get_obs_v2

    return get_obs_v2(infoset).public


# --------------------------------------------------------------------------- #
# Belief training smoke: one optimizer step changes parameters
# --------------------------------------------------------------------------- #
class TestBeliefTrainingSmoke:
    def test_one_optimizer_step_changes_parameters(self, seed_factory):
        seed_factory(11)
        infosets = _play_random_episodes(num_episodes=8)
        # Build a small labelled dataset of (input, label).
        data = [_build_input_and_label(iset) for iset in infosets[:16]]
        model = BeliefModel(BeliefConfig(hidden_size=32, num_layers=1))
        opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]

        feats = torch.from_numpy(
            np.stack([d[0].feature_vector for d in data]).astype(np.float32)
        )
        out = model._forward_logits(feats)
        legal = torch.from_numpy(
            np.stack([legal_mask(d[0].unseen_counts) for d in data])
        ).bool()
        target = torch.from_numpy(
            target_allocation_tensor([d[1].allocation for d in data])
        )
        comps = belief_loss(out, target, legal)
        comps.total.backward()
        opt.step()

        after = [p.detach().clone() for p in model.parameters()]
        assert any(not torch.equal(b, a) for b, a in zip(before, after))
