"""Run the bounded H7 V3+ADMC async single-GPU topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from douzero.env.rules import RuleSet
from douzero.observation.schema import build_v2_schema
from douzero.training.long_running import (
    CheckpointSeries,
    LongRunningConfig,
    LongRunningState,
    LongRunningTrainer,
)
from douzero.v3_hybrid import (
    ADMC_SAFE_HYBRID,
    AdaptiveDMCConfig,
    V3H2LearnerConfig,
    V3HybridLossComposerConfig,
    V3HybridModel,
    V3HybridModelConfig,
)
from douzero.v3_hybrid.integration_config import (
    V3H6FeatureFlags,
    V3H6LearnerConfig,
    V3H6ResolvedConfig,
    V3H6TopologyConfig,
    load_v3_hybrid_config,
)
from douzero.v3_hybrid.runtime import (
    V3AsyncSingleGPUTrainer,
    V3H7RuntimeConfig,
)
from douzero.v3_hybrid.training.h3_learner import V3H3LearnerConfig
from douzero.v3_hybrid.training.h4_learner import V3H4LearnerConfig
from douzero.v3_hybrid.training.h5_learner import V3H5LearnerConfig
from douzero.v3_hybrid.training.h6_learner import V3H6Learner


def _smoke_config() -> V3H6ResolvedConfig:
    model = V3HybridModelConfig(
        hidden_size=16,
        history_layers=1,
        history_heads=4,
        shared_fusion_layers=1,
        landlord_adapter_layers=1,
        farmer_adapter_layers=1,
    )
    public = V3H2LearnerConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        device="cuda",
        adaptive_dmc=AdaptiveDMCConfig(mode=ADMC_SAFE_HYBRID),
    )
    base = V3H5LearnerConfig(
        base=V3H4LearnerConfig(base=V3H3LearnerConfig(public=public))
    )
    learner = V3H6LearnerConfig(
        base=base,
        losses=V3HybridLossComposerConfig(lambda_dmc=1.0),
        features=V3H6FeatureFlags(adaptive_dmc=True),
        topology=V3H6TopologyConfig(ruleset="legacy"),
    )
    return V3H6ResolvedConfig(model=model, learner=learner)


def _resolve_checkpoint(path: str) -> Path:
    source = Path(path)
    if source.name.endswith("-latest.json"):
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload = CheckpointSeries._validate_manifest(payload)
        source = source.parent / payload["latest"]
    if not source.is_file():
        raise FileNotFoundError(f"H7 resume checkpoint does not exist: {source}")
    return source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--config", type=Path)
    config.add_argument(
        "--smoke-config",
        action="store_true",
        help="Use the explicit tiny CUDA test identity; never a strength run.",
    )
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--games-per-actor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--target-microbatch", type=int, default=4)
    parser.add_argument("--max-policy-lag", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action-seed", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--episodes-per-cycle", type=int, default=4)
    parser.add_argument("--optimizer-steps-per-cycle", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--max-wall-time-minutes", type=float, default=0.0)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-every-cycles", type=int, default=1)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--resume", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("H7 async runtime requires CUDA")
    resolved = _smoke_config() if args.smoke_config else load_v3_hybrid_config(args.config)
    runtime_config = V3H7RuntimeConfig(
        num_actors=args.num_actors,
        games_per_actor=args.games_per_actor,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_microbatch=args.target_microbatch,
        max_policy_lag=args.max_policy_lag,
        environment_seed=args.seed,
        action_seed=args.action_seed,
        epsilon=args.epsilon,
    )
    model = V3HybridModel(build_v2_schema(), resolved.model)
    learner = V3H6Learner(
        model, ruleset=RuleSet.legacy(), config=resolved
    )
    trainer = V3AsyncSingleGPUTrainer(learner, resolved, runtime_config)
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
        max_cycles=args.max_cycles,
        max_wall_time_minutes=args.max_wall_time_minutes,
        checkpoint_every_cycles=args.checkpoint_every_cycles,
        keep_last_checkpoints=args.keep_last_checkpoints,
        save_on_interrupt=True,
        v2_training_mode="async_single_gpu",
        num_actors=args.num_actors,
        games_per_actor=args.games_per_actor,
        replay_schema_version=3,
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
