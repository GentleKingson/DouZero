# AGENTS.md

> Repository-level instructions for OpenAI Codex and other coding agents.
>
> Scope: the entire repository unless a more deeply nested `AGENTS.md` or
> `AGENTS.override.md` provides narrower instructions.
>
> Keep this root file focused and below 32 KiB. Put subsystem-specific details
> in nested instruction files rather than expanding this file indefinitely.

## Project mission

This repository is based on **DouZero**, a PyTorch reinforcement-learning
framework for three-player DouDizhu (斗地主). The enhancement program aims to
improve real-game strength while preserving a reproducible legacy baseline.

Optimize for the following priorities, in order:

1. Correct game rules and legal actions.
2. No imperfect-information leakage.
3. Reproducibility and backward compatibility.
4. Measurable playing-strength improvements.
5. Training and inference efficiency.
6. Maintainability, documentation, and license compliance.

Do not trade a higher reported win rate for invalid rules, hidden-card leakage,
biased evaluation, or an unreproducible experiment.

## Instruction and task handling

- Read this file, relevant nested instruction files, the user task, and the
  affected source files before editing.
- Direct user instructions override this file. Narrower nested instructions
  override this file for files in their subtree.
- Inspect `git status` before changes. Do not discard, reset, overwrite, or
  reformat unrelated user work.
- Do not create or switch branches unless the user asks.
- Do not commit or push unless the user asks. When commits are requested, keep
  them atomic and do not amend existing commits without explicit permission.
- For ambiguous behavior, preserve the legacy behavior and place new behavior
  behind an explicit configuration option or versioned interface.
- Do not stop at a plan when the task requests implementation. Implement,
  validate, and report the result.
- Never claim that a command, benchmark, training run, or evaluation passed
  unless it was actually executed.

## Repository map

Understand these paths before changing related behavior:

- `train.py`: training entry point.
- `evaluate.py`: card-play evaluation entry point.
- `generate_eval_data.py`: deterministic/random evaluation deal generation.
- `douzero/env/game.py`: game state, turn order, legal play integration,
  terminal detection, and scoring bookkeeping.
- `douzero/env/env.py`: RL environment wrapper and observation encoding.
- `douzero/env/move_generator.py`: candidate move generation.
- `douzero/env/move_detector.py`: move-type classification.
- `douzero/env/move_selector.py`: response filtering against the previous move.
- `douzero/dmc/models.py`: legacy role-specific value models.
- `douzero/dmc/dmc.py`: learner loop, loss, checkpointing, and actor sync.
- `douzero/dmc/utils.py`: actor loop, shared buffers, batching, optimizers.
- `douzero/dmc/env_utils.py`: tensor/device environment adapter.
- `douzero/dmc/arguments.py`: legacy CLI configuration.
- `douzero/evaluation/deep_agent.py`: checkpoint loading and deterministic
  legal-action selection.
- `douzero/evaluation/simulation.py`: multiprocessing evaluation.
- `baselines/`: baseline model locations; do not modify or delete weights.
- `docs/`, `configs/`, `tests/`, `benchmarks/`, and enhanced modules may be
  introduced by the modernization plan. Read their local documentation first.

If an enhancement roadmap exists under `docs/`, treat it as the product
roadmap. This file remains the execution and quality policy.

## Supported modes and compatibility

The project may contain several modes:

- `legacy`: original DouZero rules, observations, models, checkpoints, and CLI.
- `legacy_factorized`: numerically equivalent legacy model with state/history
  encoded once per decision.
- `v2` or `enhanced`: versioned observations, configurable rules, shared
  encoders, multi-head values, belief modeling, and optional search.

Compatibility requirements:

- Do not silently change the default legacy semantics.
- Existing role checkpoints for `landlord`, `landlord_up`, and
  `landlord_down` must either load unchanged or fail with a precise,
  actionable compatibility error.
- New checkpoints must record enough metadata to reject incompatible feature,
  rule, or model schemas.
- Never use permissive partial loading to hide broad key or shape mismatches.
- If a migration is required, add a dedicated migration tool and tests.
- Legacy evaluation data must remain readable or have an explicit converter.

## Environment setup

Use the active Python environment. Do not replace the user's PyTorch or CUDA
installation unless explicitly requested.

Typical editable installation:

```bash
python -m pip install -e .
```

The upstream package declares `torch` and `rlcard`. Add production dependencies
only when the implementation materially requires them. Explain each new
dependency and prefer the standard library or existing dependencies.

The original project supports Python 3.6+, but enhanced code may raise the
minimum version only through an explicit, documented compatibility decision.
Do not accidentally introduce syntax unsupported by the declared version.

Prefer Linux for GPU training. On Windows, assume actor-side CUDA
multiprocessing is unsupported unless the repository has added and tested a
different implementation; use CPU actors or WSL as appropriate.

## Standard work sequence

For every nontrivial task:

1. Inspect:
   - `git status --short`
   - relevant source files and tests
   - current configuration and checkpoint interfaces
   - existing benchmark or baseline artifacts
2. State the compatibility surface:
   - rules
   - observations/features
   - model/checkpoint
   - actor/learner protocol
   - evaluation/data format
3. Make the smallest coherent change.
4. Add or update tests with the implementation.
5. Run targeted checks first, then the broadest affordable checks.
6. Inspect the final diff for unrelated changes and hidden-information leaks.
7. Report files changed, tests run, tests not run, compatibility impact, and
   remaining risks.

Do not perform repository-wide formatting as part of a focused feature or bug
fix.

## Validation commands

Choose commands that match the change. Do not run long training jobs or large
evaluations unless the user requests them.

### Fast repository checks

```bash
python -m compileall -q douzero train.py evaluate.py generate_eval_data.py
python train.py --help
python evaluate.py --help
python generate_eval_data.py --help
git diff --check
```

### Unit and integration tests

When `pytest` tests are present:

```bash
python -m pytest -q
```

For a focused change, run the targeted test module first, then the full suite
when feasible. Tests must be deterministic, offline, and independent of
downloaded pretrained weights unless explicitly marked as optional.

### Small evaluation smoke test

Use a temporary location and a small number of games:

```bash
tmp_dir="$(mktemp -d)"
python generate_eval_data.py --output "$tmp_dir/eval_smoke" --num_games 12
python evaluate.py \
  --landlord random \
  --landlord_up random \
  --landlord_down random \
  --eval_data "$tmp_dir/eval_smoke.pkl" \
  --num_workers 1
```

If an enhanced evaluation CLI replaces this flow, retain an equivalent
CPU-only deterministic smoke test.

### Training changes

Do not use the full asynchronous training entry point as the only test.
Training-related changes must add a bounded test that exercises at least one
actor rollout or synthetic batch, one forward/backward step, optimizer update,
checkpoint save/load, and clean shutdown.

A full or timed training smoke may be run only after the bounded test exists.
Use tiny buffers, one actor, one learner thread, CPU mode, a temporary output
directory, and an explicit timeout.

### GPU checks

GPU tests are optional when no GPU is available. Record them as not run rather
than simulating success. CPU correctness remains mandatory.

## DouDizhu domain invariants

### Cards and roles

- A standard deck has 54 cards: four copies of ranks 3 through A and 2, plus
  one small joker and one big joker.
- Preserve canonical rank ordering and canonical sorted action
  representations.
- The card-play turn order is:
  `landlord -> landlord_down -> landlord_up -> landlord`.
- `landlord_up` acts immediately before the landlord.
- `landlord_down` acts immediately after the landlord.
- The two farmers share team utility but retain different positional roles.
- No hand count, unseen-card count, or rank multiplicity may become negative.
- At terminal state, exactly one team wins.

### Legal actions

- `move_generator`, `move_detector`, and `move_selector` are the rule source of
  truth for card-play legality unless a versioned rule engine explicitly
  supersedes them.
- A model, heuristic, human prior, belief model, or search procedure may rank
  legal actions but must not manufacture an illegal action.
- Do not remove technically legal actions because they look strategically bad.
  Strategic discouragement belongs in features, priors, losses, or value
  estimates, not the legality layer.
- Passing is legal only when the rules and current trick permit it.
- Add table-driven tests for every changed move type and boundary rank.

### Bidding and scoring

If bidding or complete scoring is implemented:

- Model the game as an explicit phase/state machine.
- Keep bidding mode, bid values, re-deal behavior, bomb/rocket multipliers,
  spring/anti-spring, doubling, base score, and multiplier caps in a versioned
  `RuleSet` or equivalent configuration.
- Do not hard-code one commercial platform's rules as universal DouDizhu.
- Public bidding history and revealed bottom cards are public information.
- Terminal score accounting must be internally consistent and tested for all
  multiplier combinations.
- Preserve a legacy card-play-only mode for old data and checkpoints.

## Imperfect-information boundary

This is the most important safety and correctness rule.

### Public model inputs may include

- acting role and relative seat
- the acting player's hand
- legal actions
- public bidding history and final bid
- revealed bottom cards and their public ownership
- played cards and complete public action history
- each player's number of remaining cards
- public multiplier, bomb, and phase state
- derived features computed only from the above

### Privileged training-only data may include

- exact hidden hands
- `all_handcards`
- target hidden-card labels
- perfect-information teacher inputs
- future trajectory labels and terminal outcomes

Requirements:

- Represent public and privileged data with separate, explicitly named types or
  dictionaries.
- Production `act()` and exported models must accept public data only.
- Do not pass privileged fields and rely on a convention that the model ignores
  them.
- Belief models infer hidden cards from public evidence; they never read the
  true allocation during inference.
- Search samples hidden states from the belief distribution; it never clones
  the environment's true hidden hands.
- Add leakage tests: two states with identical public information but different
  hidden allocations must produce identical public observations.
- Production modules should not import privileged teacher/data modules unless
  there is a narrowly documented tooling reason.

## Observation and feature rules

- Version every nontrivial observation schema.
- Derive dimensions from a schema or named constants. Do not scatter magic
  widths such as role-specific flattened feature sizes across files.
- Preserve dtype intentionally. Card/count tensors may be compact in buffers,
  but model arithmetic must use an appropriate floating dtype.
- Keep state/history tensors separate from legal-action tensors.
- Encode shared state and history once per decision; do not duplicate and
  recompute them for every legal action without a measured reason.
- Variable legal-action counts require an explicit batch representation and
  mask.
- Masked actions or history tokens must not affect valid outputs.
- Public bottom-card information must not remain in the generic unknown-card
  pool.
- Feature changes require tests, documentation, and checkpoint compatibility
  handling.

## Model rules

- Keep the legacy model available while introducing enhanced architectures.
- Enhanced models should expose clear components such as state/history encoder,
  action encoder, role conditioning, state-action fusion, value heads, belief
  heads, and auxiliary heads.
- Share general card and action knowledge where appropriate, while preserving
  landlord and farmer positional differences through role embeddings, adapters,
  or heads.
- Always apply legal-action masks before selection.
- Forward passes and losses must remain finite; test NaN/Inf handling.
- Add tests for:
  - all three roles
  - one and many legal actions
  - variable history lengths
  - masked padding invariance
  - forward and backward passes
  - save/load equivalence
  - CPU and optional GPU
- Do not claim that a deeper or larger model is stronger without controlled
  evaluation.

## Rewards, targets, and action selection

Use one documented sign convention throughout enhanced code:

- `target_win` is from the current acting player's team perspective.
- `target_score` is the final signed score from that same perspective.
- A farmer win is positive for both farmer roles.
- Avoid scattered implicit negation; centralize perspective conversion and
  test landlord/farmer symmetry.

For multi-head value models:

- Train win probability with a stable classification loss.
- Train conditional win/loss score heads only on the applicable samples.
- Prefer Huber, log-score, distributional, or otherwise robust handling for
  large multiplier tails.
- Derive expected score consistently from calibrated win probability and
  conditional scores.
- Keep deployment decision modes explicit, for example pure win rate, pure
  expected score, or lexicographic threshold selection.
- Threshold logic must work correctly for negative values.
- Evaluation mode is deterministic unless stochastic evaluation is explicitly
  requested.

## Belief-model rules

A hidden-hand belief implementation must respect exact card constraints:

- per-rank allocations cannot exceed unseen counts
- joker counts are at most one each
- each opponent's total equals the public remaining-card count
- the two opponent hands sum exactly to the unseen pool
- known public bottom cards are assigned correctly
- decoding and sampling cannot rely on unbounded rejection loops

Prefer constrained dynamic programming or another exact finite method for MAP
decoding and posterior sampling. Test thousands of random states for card
conservation.

Belief labels come from privileged training data only. Belief predictions may
be supplied to the public value model as posterior means, probabilities,
entropy, or legal samples.

## Human-game data and strategy priors

- Use only lawfully obtained and authorized data. Do not add account
  automation, scraping, anti-detection, or service-term bypass code.
- Do not store personal identifiers or credentials.
- Canonicalize and validate every recorded game by replaying it through the
  rule engine. Quarantine invalid games; do not silently repair them.
- Split data by complete game, and where appropriate by player and time, to
  prevent leakage.
- Do not train only on won games.
- Human behavior cloning must score the current legal-action list; do not
  replace the variable action representation with a brittle global class list.
- Human heuristics such as preserving bombs, avoiding teammate suppression,
  minimizing remaining turns, and controlling high cards should be measurable
  features, auxiliary targets, or uncertainty-gated priors. They are not hard
  legality rules.
- All strategy-prior contributions need an ablation switch.

## Actor, learner, and multiprocessing rules

- Every actor receives a deterministic seed derived from the run seed and actor
  identity.
- Do not overwrite actor model parameters in place while another thread or
  process may be executing a forward pass.
- Publish versioned, complete policy snapshots and switch at a safe boundary,
  preferably between games.
- Record the policy version that generated each trajectory.
- Keep queue, buffer, and process shutdown paths bounded and testable.
- Avoid hidden global mutable state shared across runs.
- Keep CPU actor mode functional.
- For multi-GPU learning, prefer one process per GPU with Distributed Data
  Parallel over ad hoc shared-model threading.
- AMP and `torch.compile` are optional optimizations, not correctness
  requirements. Add them behind flags and benchmark them.
- Numerically sensitive belief, normalization, probability, and loss operations
  may remain float32 under mixed precision.

## Search and endgame solver rules

Optional search must be safe and bounded:

- Start from model-ranked legal candidates.
- Sample hidden states from the public belief model.
- Clone state without sharing mutable game objects.
- Use hard node, rollout, and time budgets.
- On timeout or failure, return the base model's legal action.
- With budget zero, behavior must equal the base policy.
- Exact endgame solvers must include hands, acting player, current trick/pass
  state, and scoring state in transposition keys.
- Small handcrafted endgames require comparison against exhaustive solutions.

Search is disabled by default unless latency and strength tradeoffs have been
measured.

## Evaluation requirements

Do not accept training loss or self-play return as evidence of improvement.

Strength comparisons should use:

- fixed deal sets
- the same ruleset and feature version
- paired role-swapped card-play tests
- seat rotation for full bidding games
- landlord, landlord-up, and landlord-down results separately
- win percentage and score-based metrics
- bidding, bomb, rocket, spring, and game-length diagnostics
- calibrated win-probability metrics where available
- paired bootstrap confidence intervals
- inference latency and throughput on identified hardware
- a diverse opponent pool, not only the current self-play mirror
- ablations for each major enhancement

Do not tune hyperparameters on the final holdout evaluation set. Do not report a
strength gain without recording model identifiers, code revision, ruleset,
deal-set identifier, seeds, sample count, and confidence interval.

Evaluation data generation must be seedable and must not overwrite existing
datasets by default.

## Performance work

- Profile before optimizing.
- Benchmark before and after on the same machine, device, batch shape, legal
  action distribution, and warm-up policy.
- Report median and tail latency where relevant.
- Separate first-run compilation cost from steady-state performance.
- Do not sacrifice correctness, deterministic evaluation, or legacy
  compatibility for an unmeasured optimization.
- Avoid repeated state/history encoding for every candidate action.
- Do not commit generated profiler traces, checkpoints, datasets, or benchmark
  dumps unless explicitly requested.

## Coding conventions

Match existing code in touched legacy files and use modern conventions in new
modules:

- Python source and identifiers in English.
- Four-space indentation.
- Descriptive names; avoid unexplained single-letter variables outside compact
  mathematical code.
- Type annotations for new public functions, data structures, and module
  boundaries.
- Docstrings for public APIs and non-obvious algorithms.
- Small pure functions for card/rule transformations.
- Dataclasses or typed structures for versioned configuration and model output.
- Named constants instead of repeated card/rank/role strings.
- Explicit device and dtype handling.
- `torch.inference_mode()` for deployment inference.
- No bare `except`.
- Error messages should include the invalid phase, role, shape, or schema
  version needed to diagnose the problem.
- Comments should explain why, invariants, or mathematical intent—not restate
  the code.
- Keep formatting-only edits separate from behavioral changes.

If `ruff`, `black`, `mypy`, or another formatter/linter is configured in the
repository, use the repository configuration. Do not introduce a competing
style tool casually.

## Tests by change type

### Rules or move generation

Test legal and illegal examples, pass behavior, boundary ranks, bombs, rocket,
turn order, hand conservation, terminal detection, and legacy snapshots.

### Observation changes

Test schema shapes/dtypes, serialization, public/privileged separation,
different hidden allocations with identical public output, bottom-card
ownership, masks, and legacy adapter parity.

### Model changes

Test three roles, variable actions/history, padding invariance, finite outputs,
gradient flow, deterministic inference, checkpoint round-trip, and legacy
loading behavior.

### Loss or reward changes

Test sign conventions, landlord/farmer symmetry, terminal labels, conditional
masks with empty subsets, extreme multipliers, and finite gradients.

### Actor/learner changes

Test one bounded update, policy-version stability within a game, snapshot
atomicity, queue cleanup, checkpoint resume, and CPU execution.

### Evaluation changes

Test deal reproducibility, seat rotation, paired aggregation, confidence
interval calculations on synthetic known data, and correct role-level metrics.

### Data-pipeline changes

Test replay validation, duplicate handling, invalid-game quarantine, split
isolation, legal human-action indexing, and absence of personal information.

## Dependencies, licenses, and third-party code

The upstream repository is Apache-2.0. Preserve copyright and license notices.

Before copying or adapting third-party code:

1. Identify its exact license.
2. Determine compatibility with this repository's intended distribution.
3. Record the source and decision in third-party documentation.
4. Prefer an independent implementation from papers and public behavior when
   the source is GPL, has no clear license, or is otherwise incompatible.

Do not copy code from GPL-licensed or unlicensed DouZero derivatives into an
Apache-licensed distribution without an explicit project-level licensing
decision. Algorithmic ideas may be implemented independently.

Do not commit:

- external pretrained weights
- private or proprietary game records
- credentials, tokens, or cookies
- large generated datasets
- platform automation or evasion tooling

## Checkpoints, artifacts, and experiment records

- Write tests and smoke artifacts to temporary directories.
- Never overwrite `baselines/` or an existing checkpoint directory.
- Use atomic checkpoint writes where practical.
- New checkpoint manifests should include:
  - format/model/feature version
  - ruleset identifier or hash
  - code revision
  - full effective configuration
  - role support
  - public versus privileged access class
  - training frames/steps and policy version
- Resume must restore optimizer and training counters consistently.
- A failed or partial checkpoint must not be registered in a policy league.
- Keep machine-readable experiment results alongside a short human-readable
  summary.

## Documentation

Update documentation whenever behavior, configuration, file format, CLI, model
schema, or evaluation protocol changes.

Documentation must distinguish:

- legacy behavior
- enhanced behavior
- default versus optional features
- measured results versus proposed results
- card-play-only versus end-to-end bidding evaluation
- public inference inputs versus privileged training labels
- CPU, single-GPU, and multi-GPU support

Provide migration and rollback instructions for breaking changes.

## Completion report

At the end of a coding task, report:

1. What changed and why.
2. Important files touched.
3. Commands actually run and their results.
4. Checks not run and the concrete reason.
5. Backward-compatibility impact.
6. Data, rule, checkpoint, or deployment migration requirements.
7. Performance or strength measurements, or explicitly `not measured`.
8. Remaining risks or follow-up work.

Do not bury failures. A partial but accurate result is preferable to a polished
but unsupported claim.

## Definition of done

A change is complete only when:

- the implementation matches the requested scope
- legal-action and game-state invariants hold
- public inference contains no privileged information
- legacy behavior is preserved or explicitly versioned
- targeted tests pass
- the broadest affordable checks pass
- documentation and configuration are updated
- the final diff contains no unrelated changes
- all performance and strength claims are backed by recorded measurements
