# Belief Sampling Search and Endgame Solver

P13 adds optional, bounded inference search for Model V2. It is disabled by
default, so legacy, factorized, and ordinary V2 action selection are unchanged.

## Information Boundary

`BeliefSearch.select()` accepts an `ObservationV2` with a public payload. It
does not accept an infoset, `all_handcards`, `PrivilegedObservation`, or a true
opponent allocation. The P07 belief model distributes the public unknown-card
pool with its constrained dynamic-programming sampler. Public unplayed bottom
cards are then restored to the landlord's sampled hand.

Each sample creates an immutable `SearchGameState`. Applying an action returns
a new state, so rollout and solver branches never mutate the live environment
or share mutable hand, history, scoring, or pass state.

## Decision Flow

1. Model V2 scores every legal action once.
2. `candidate.select_top_k()` keeps a small, stable top-k set.
3. The P07 sampler draws `belief_samples` exact, card-conserving allocations.
4. Each candidate is applied independently to each sample.
5. Positions at or below `endgame_cards_threshold` use exact team minimax.
   Larger positions use a deterministic fixed-depth fast-policy rollout.
6. Candidate values report mean win probability, expected team score, and an
   optional lower-confidence penalty.

Farmers share one utility in minimax. Terminal scores call the canonical rule
engine, including bids, bombs, rocket, spring, anti-spring, and multiplier
caps. The transposition identity contains all hands, actor, move-to-beat,
leader, pass count, action counts, bid, bombs, rocket, and ruleset hash.
The solver cache additionally scopes every entry by the root team because
win probability, score, and min/max direction are perspective-dependent.
Nonterminal score estimates use the same canonical multiplier helper as
terminal scoring, including the landlord/farmer 2:1 score units, bids,
independent bomb/rocket multipliers, and caps. They intentionally do not guess
future spring or anti-spring outcomes.

## Budgets and Fallback

`max_nodes`, `max_rollouts`, and `max_milliseconds` are hard cooperative
limits. Search-only belief inference starts after the wall-clock budget, and
move generation also checks the deadline. Root candidates are already legal,
so sampled roots apply them without an unbudgeted second legality expansion.
Exhausting any limit returns the base-policy action, rather than a partially
searched result. A zero budget therefore gives the same action as search-off
mode.

The structured `SearchLog` records the base and searched actions, aggregate
candidate values, sample/node/rollout counts, elapsed time, timeout status, and
fallback reason. Timeout fallback retains the number of belief samples already
generated, so audit counters remain internally consistent. A fixed seed makes
sampling and the complete decision stable.

## Configuration

```yaml
search:
  enabled: false
  top_k: 3
  belief_samples: 8
  rollout_depth: 12
  endgame_cards_threshold: 12
  max_nodes: 20000
  max_rollouts: 64
  max_milliseconds: 100
  risk_penalty: 0.0
  selection_mode: win_then_score
  seed: 0
```

Enabling search in `DeepAgentV2` requires a P07 `BeliefModel`, even when the
value model itself does not fuse belief features. This fail-closed rule ensures
there is no alternative route for true hidden hands.

`GameEnv.get_infoset()` carries the public bid, independent bomb and rocket
counts, non-pass action counts, bidding history/order, phase, and current
multiplier. Consequently `DeepAgentV2.act(infoset)` preserves standard and
custom scoring state without requiring callers to reconstruct keyword args.

## Benchmark

Run the synthetic CPU latency benchmark with:

```bash
python benchmarks/bench_search.py --iterations 20
```

It reports measured p50/p95 latency and mean expanded nodes. No production
latency or playing-strength claim is made without running a representative
model, hardware, and evaluation set.
