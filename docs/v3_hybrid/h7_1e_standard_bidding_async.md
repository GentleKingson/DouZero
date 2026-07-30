# H7.1e Standard Bidding Async

H7.1e extends the existing bounded H7 request coordinator to standard-rules
full games. It does not create a second inference protocol. Each shared request
has a versioned kind: card play or bidding. Card play keeps variable legal
actions and `dmc_q`; bidding uses the independent public 0/1/2/3 action schema
and the existing V3 bidding head.

The committed configuration is
`configs/v3_hybrid_h7_1e_standard_bidding.yaml`. The capability remains
isolated from belief, Oracle, cooperation, strategy, style, league, and
curriculum transports. Unsupported combinations fail during typed runtime
validation, before CUDA initialization, worker spawn, replay allocation, or
checkpoint I/O.

## Contracts

- Ruleset: `standard-v1`; environment scoring and legal bids are authoritative.
- Bid seats: neutral physical seats `0`, `1`, and `2` until the auction closes.
- First bidder: deterministic rotation by global episode ID by default.
- Request protocol:
  `v2-shared-slots-v3-standard-separate-bid-cardplay-actions-v1`.
- Replay protocol:
  `v3-public-cardplay-plus-separate-bidding-replay-redeal-cap-excluded-v2`.
- Replay: card-play and bidding rows use separate bounded buffers.
- Commit: abandoned all-pass auction rows never receive a later deal's labels.
- Guard: a redeal-cap fallback game is excluded from all training replay.
- Cadence: `bidding_eligible_steps` advances after each successful public
  optimizer update; `bidding_update_interval` and its counter are checkpointed.
- Resume: replay is flushed at the checkpoint boundary; model, optimizer,
  loss schedules, RNG, policy version, bidding cadence, and cumulative counters
  restore strictly.
- Deployment: the public policy package may contain the bidding head but never
  optimizer, replay, terminal labels, or hidden hands.

The actor constructs `BiddingObservationV2` from the encoder's explicit public
allow-list. Learned bids are masked by the environment-provided legal bid mask.
Warm-start rule bids remain explicit imitation samples. A bid is never routed
through the card-action encoder, and bidding never changes the environment's
score calculation.

## Evidence Gate

Current-head evidence must include CUDA parameter update, real SIGTERM,
fresh-container resume, fault injection, two-hour soak, and three matched
repetitions for `single_process`, `async_4x4`, and `async_8x4`. Bidding samples,
optimizer cadence, head VRAM, bid/card decision counts, abandoned auctions,
redeals, policy lag, memory plateau, and clean shutdown are reported separately.

Release candidate: NONE

Release status: NOT READY

Playing strength: NOT MEASURED
