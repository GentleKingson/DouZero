# P4 Formal Ablation Decision Record

Status: FROZEN BEFORE DEVELOPMENT EVALUATION

Base SHA: `8030f19127077481f5be08db17c42818f66ed236`

Release candidate: NONE

Release status: NOT READY

Playing strength: NOT MEASURED

## Fixed Protocol

- The runnable matrix is exactly the configurations in `configs/v3_formal/`.
- Every standard-rules control enables the same learned-bidding objective and
  separate bid action space. Standard evidence without bidding is rejected
  rather than silently running legacy card play under a standard label.
- Runtime topology is frozen per executable capability combination. Legacy
  A1 and the standard Oracle, belief, and cooperation ablations use
  `single_process`; Model V2, the other V3 ablations, and both full-hybrid
  variants use `async_single_gpu`. The three standard partial sidecar
  ablations remain single-process because H7.1 intentionally does not define
  a partial sidecar-plus-bidding async transport.
- Runtime workload identity also freezes the family-owned profile, actor and
  game counts, cycle cadence, batch and replay sizes, checkpoint cadence, and
  Legacy A1 unroll length. The strict P4 dispatcher rejects a seed or resume
  mode outside that frozen contract before launching a trainer.
- Training seeds are `101`, `202`, and `303`.
- Development uses the frozen 14,400-second wall ceiling, 5,000,000-sample
  ceiling, 50,000 optimizer-step ceiling, and 20,000 paired deals.
- Promotion uses the frozen 28,800-second wall ceiling, 20,000,000-sample
  ceiling, 200,000 optimizer-step ceiling, and 100,000 paired deals.
- A training run ends when its first frozen training ceiling is reached.
- Every run uses checkpointing, a real SIGTERM, fresh-container resume, and at
  least one post-resume parameter update.
- Search-off and search-on evaluate the same full-hybrid public checkpoint.
- Human BC is excluded because no authorized dataset identity is declared.
- All comparisons use the same committed source, Docker image, GPU, and
  ruleset-specific fixed deal set.

## Development Shortlist

The shortlist rule is frozen before any promotion holdout is read:

1. Keep Legacy A1.
2. Keep Model V2.
3. Among V3 ablations, keep the wall-clock-efficient variant whose overall
   WP and ADP are both non-negative and which has no significant role
   regression.
4. Keep full-hybrid search-off.
5. Keep full-hybrid search-on using the identical training checkpoint.
6. Exclude a variant from promotion when training is unstable, its overall
   confidence interval is negative, any role is catastrophically worse, or
   its throughput cost violates the frozen budget.

Ties are resolved in this order: higher lower confidence bound for overall
WP, higher lower confidence bound for overall ADP, higher learner samples/s,
then lexicographic variant name. Promotion results never affect selection.

## Promotion Rule

A candidate is promotable only when both overall WP and ADP 95% confidence
interval lower bounds are positive, no role has a significant regression,
checkpoint/resume and public-only reload pass, and search benefit covers its
whole-population p99 latency cost. Failed and stopped runs remain in evidence.

This record contains no benchmark or playing-strength result.
