"""Run the bounded H7 V3+ADMC async single-GPU topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from douzero.belief.model import BeliefConfig, BeliefModel
from douzero.env.rules import RuleSet
from douzero.observation.schema import build_v2_schema
from douzero.training.long_running import (
    CheckpointSeries,
    LongRunningConfig,
    LongRunningState,
    LongRunningTrainer,
)
from douzero.v3_hybrid import V3HybridModel
from douzero.v3_hybrid.formal_config import load_formal_config
from douzero.v3_hybrid.h7_smoke import build_v3_h7_smoke_config
from douzero.v3_hybrid.integration_config import load_v3_hybrid_config
from douzero.v3_hybrid.pilot import (
    build_pilot_resolved_config,
    create_pilot_learner,
)
from douzero.v3_hybrid.runtime import (
    V3_H71A_REPLAY_PROTOCOL,
    V3_H71A_REQUEST_PROTOCOL,
    V3_H71A_SNAPSHOT_SEMANTICS,
    V3_H71B_REPLAY_PROTOCOL,
    V3_H71B_REQUEST_PROTOCOL,
    V3_H71C_REPLAY_PROTOCOL,
    V3_H71C_REQUEST_PROTOCOL,
    V3_H71D_REPLAY_PROTOCOL,
    V3_H71D_REQUEST_PROTOCOL,
    V3_H71E_REPLAY_PROTOCOL,
    V3_H71E_REQUEST_PROTOCOL,
    V3AsyncSingleGPUTrainer,
    V3H7RuntimeConfig,
    V3SingleProcessTrainer,
    resolve_v3_h7_seed_contract,
    validate_v3_h7_formal_initialization,
    validate_v3_h7_runtime_config,
)
from douzero.v3_hybrid.support_matrix import (
    TOPOLOGY_ASYNC_SINGLE_GPU,
    TOPOLOGY_SINGLE_PROCESS,
)
from douzero.v3_hybrid.training.h6_learner import V3H6Learner


def _resolve_checkpoint(path: str) -> Path:
    source = Path(path)
    if source.name.endswith("-latest.json"):
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload = CheckpointSeries._validate_manifest(payload)
        source = source.parent / payload["latest"]
    if not source.is_file():
        raise FileNotFoundError(f"H7 resume checkpoint does not exist: {source}")
    return source


def _oracle_update_limit(learner, oracle_enabled: bool) -> int:
    if not oracle_enabled:
        return 0
    schedule = learner.base.base.base.config.schedule
    return (
        schedule.warmup_updates
        + schedule.guided_updates
        + schedule.finetune_updates
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--config", type=Path)
    config.add_argument(
        "--formal-config",
        type=Path,
        help="Use a frozen P1 formal config, including H7.1 sidecars.",
    )
    config.add_argument(
        "--smoke-config",
        action="store_true",
        help="Use the explicit tiny CUDA test identity; never a strength run.",
    )
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--games-per-actor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--belief-sidecar-capacity", type=int, default=4096)
    parser.add_argument("--oracle-sidecar-capacity", type=int, default=4096)
    parser.add_argument("--cooperation-sidecar-capacity", type=int, default=4096)
    parser.add_argument("--cooperation-episode-capacity", type=int, default=1024)
    parser.add_argument("--bidding-replay-capacity", type=int, default=4096)
    parser.add_argument("--bidding-batch-size", type=int, default=16)
    parser.add_argument("--bidding-update-interval", type=int, default=1)
    parser.add_argument(
        "--bidding-policy",
        choices=("random", "rule", "max", "pass", "learned"),
        default="learned",
    )
    parser.add_argument(
        "--bidding-warm-start-policy",
        choices=("random", "rule", "max", "pass"),
        default="rule",
    )
    parser.add_argument("--bidding-learned-probability", type=float, default=1.0)
    parser.add_argument(
        "--first-bidder-mode",
        choices=("rotate", "seeded_random"),
        default="rotate",
    )
    parser.add_argument("--target-microbatch", type=int, default=4)
    parser.add_argument("--max-policy-lag", type=int, default=128)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--episodes-per-cycle", type=int, default=4)
    parser.add_argument("--optimizer-steps-per-cycle", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--max-wall-time-minutes", type=float, default=0.0)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-every-cycles", type=int, default=1)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--topology",
        choices=(TOPOLOGY_SINGLE_PROCESS, TOPOLOGY_ASYNC_SINGLE_GPU),
        default=TOPOLOGY_ASYNC_SINGLE_GPU,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    formal = (
        None
        if args.formal_config is None
        else load_formal_config(args.formal_config)
    )
    if formal is not None:
        validate_v3_h7_formal_initialization(formal.initialization.kind)
    resolved = (
        build_v3_h7_smoke_config()
        if args.smoke_config
        else (
            load_v3_hybrid_config(args.config)
            if formal is None
            else build_pilot_resolved_config(formal)
        )
    )
    belief_enabled = resolved.learner.features.belief
    oracle_enabled = resolved.learner.features.oracle
    cooperation_enabled = resolved.learner.features.cooperation
    public_aux_enabled = (
        resolved.learner.features.strategy
        or resolved.learner.features.style
    )
    bidding_enabled = resolved.learner.features.bidding
    environment_seed, action_seed, seed_derivation = (
        resolve_v3_h7_seed_contract(
            formal_training_seeds=(
                None if formal is None else formal.seeds.training
            ),
            formal_derivation=(
                None if formal is None else formal.seeds.derivation
            ),
            requested_environment_seed=args.seed,
            requested_action_seed=args.action_seed,
        )
    )
    runtime_config = V3H7RuntimeConfig(
        topology=args.topology,
        num_actors=args.num_actors,
        games_per_actor=args.games_per_actor,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        belief_sidecar_capacity=args.belief_sidecar_capacity,
        oracle_sidecar_capacity=args.oracle_sidecar_capacity,
        cooperation_sidecar_capacity=args.cooperation_sidecar_capacity,
        cooperation_episode_capacity=args.cooperation_episode_capacity,
        bidding_replay_capacity=args.bidding_replay_capacity,
        bidding_batch_size=args.bidding_batch_size,
        bidding_update_interval=args.bidding_update_interval,
        bidding_policy=args.bidding_policy,
        bidding_warm_start_policy=args.bidding_warm_start_policy,
        bidding_learned_probability=args.bidding_learned_probability,
        first_bidder_mode=args.first_bidder_mode,
        target_microbatch=args.target_microbatch,
        max_policy_lag=args.max_policy_lag,
        environment_seed=environment_seed,
        environment_seed_derivation=seed_derivation,
        action_seed=action_seed,
        epsilon=args.epsilon,
        belief_runtime_enabled=belief_enabled,
        oracle_runtime_enabled=oracle_enabled,
        cooperation_runtime_enabled=cooperation_enabled,
        public_aux_runtime_enabled=public_aux_enabled,
        bidding_runtime_enabled=bidding_enabled,
        request_protocol=(
            V3_H71A_REQUEST_PROTOCOL
            if belief_enabled
            else (
                V3_H71B_REQUEST_PROTOCOL
                if oracle_enabled
                else (
                    V3_H71C_REQUEST_PROTOCOL
                    if cooperation_enabled
                    else (
                        V3_H71D_REQUEST_PROTOCOL
                        if public_aux_enabled
                        else (
                            V3_H71E_REQUEST_PROTOCOL
                            if bidding_enabled
                            else V3H7RuntimeConfig.request_protocol
                        )
                    )
                )
            )
        ),
        replay_protocol=(
            V3_H71A_REPLAY_PROTOCOL
            if belief_enabled
            else (
                V3_H71B_REPLAY_PROTOCOL
                if oracle_enabled
                else (
                    V3_H71C_REPLAY_PROTOCOL
                    if cooperation_enabled
                    else (
                        V3_H71D_REPLAY_PROTOCOL
                        if public_aux_enabled
                        else (
                            V3_H71E_REPLAY_PROTOCOL
                            if bidding_enabled
                            else V3H7RuntimeConfig.replay_protocol
                        )
                    )
                )
            )
        ),
        snapshot_semantics=(
            V3_H71A_SNAPSHOT_SEMANTICS
            if belief_enabled
            else V3H7RuntimeConfig.snapshot_semantics
        ),
    )
    validate_v3_h7_runtime_config(resolved, runtime_config)
    if not torch.cuda.is_available():
        raise RuntimeError("H7 async runtime requires CUDA")
    if formal is None:
        model = V3HybridModel(build_v2_schema(), resolved.model)
        belief_model = (
            BeliefModel(BeliefConfig()) if belief_enabled else None
        )
        learner = V3H6Learner(
            model,
            ruleset=(
                RuleSet.standard() if bidding_enabled else RuleSet.legacy()
            ),
            config=resolved,
            belief_model=belief_model,
        )
    else:
        learner, learner_resolved = create_pilot_learner(formal)
        if learner_resolved != resolved:
            raise RuntimeError("H7 formal config resolution is not stable")
        model = learner.model
    trainer_type = (
        V3SingleProcessTrainer
        if args.topology == TOPOLOGY_SINGLE_PROCESS
        else V3AsyncSingleGPUTrainer
    )
    trainer = trainer_type(learner, resolved, runtime_config)
    state = None
    checkpoint_series = CheckpointSeries(
        args.checkpoint_path, args.keep_last_checkpoints
    )
    if args.resume:
        source = _resolve_checkpoint(args.resume)
        state = LongRunningState.from_dict(
            trainer.load_training_checkpoint(source)
        )
        checkpoint_series = CheckpointSeries.from_checkpoint(
            source, state, args.keep_last_checkpoints
        )
    long_config = LongRunningConfig(
        episodes_per_cycle=args.episodes_per_cycle,
        optimizer_steps_per_cycle=args.optimizer_steps_per_cycle,
        max_total_optimizer_steps=_oracle_update_limit(
            learner, oracle_enabled
        ),
        max_cycles=args.max_cycles,
        max_wall_time_minutes=args.max_wall_time_minutes,
        checkpoint_every_cycles=args.checkpoint_every_cycles,
        keep_last_checkpoints=args.keep_last_checkpoints,
        save_on_interrupt=True,
        v2_training_mode=args.topology,
        num_actors=args.num_actors,
        games_per_actor=args.games_per_actor,
        replay_schema_version=4 if bidding_enabled else 3,
        compact_bidding_replay_schema_version=1 if bidding_enabled else 0,
        snapshot_publication_semantics=runtime_config.snapshot_semantics,
        request_ordering_semantics=runtime_config.request_protocol,
        actor_rng_resume_semantics="restart-from-stable-task-and-domain-seeds-v1",
    )
    print(json.dumps({
        "event": "h7_start",
        "config_hash": resolved.stable_hash(),
        "runtime_hash": runtime_config.stable_hash(),
        "model_hash": model.config.stable_hash(),
        "playing_strength": "not measured",
    }, sort_keys=True), flush=True)
    runner = LongRunningTrainer(
        trainer,
        long_config,
        checkpoint_series,
        state=state,
        collect_records=True,
    )
    final_state, reason, records = runner.run()
    print(json.dumps({
        "event": "h7_stop",
        "reason": reason,
        "state": vars(final_state),
        "last_record": records[-1] if records else {},
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
