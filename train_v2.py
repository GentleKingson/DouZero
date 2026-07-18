"""V2 multi-objective training entry point (P06).

This is the single-process trainer CLI for the P06 acceptance criterion:
"极短训练能完成一次优化且参数变化" (a very short training run completes one
optimizer step and changes the parameters). It loads a YAML config
(``configs/enhanced.yaml`` by convention), builds a
:class:`~douzero.models_v2.model.ModelV2`, and runs
:class:`~douzero.training.v2_trainer.V2Trainer`. Its default remains that
one-shot flow; ``--long_running`` explicitly enables resumable training cycles.

The legacy :file:`train.py` path is unchanged. This entry point exists so
the multi-objective loss + decision-policy + team-perspective labels can be
exercised without touching the legacy multiprocessing actor/learner.

CPU remains the default. P14 adds optional one-process-per-GPU DDP via
``torchrun`` while keeping the single-process path simple.

Precedence (P06 r1 fix)
-----------------------
``explicit CLI flag > YAML value > V2 built-in default``.

The argparse parser uses ``default=argparse.SUPPRESS`` for every trainer
knob so that an ABSENT CLI flag does NOT shadow the YAML value (the classic
"argparse default clobbers YAML" bug). Only flags the user actually typed
appear in the parsed namespace. This makes a YAML ``lambda_win: 0`` (or
``batch_size: 64``, etc.) actually take effect even when the CLI does not
repeat it.

Usage
-----
::

    python train_v2.py --config configs/enhanced.yaml --episodes 8 \\
        --optimizer_steps 2 --seed 0

CPU smoke (no GPU required):

::

    python train_v2.py --episodes 4 --optimizer_steps 1 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from douzero.config import load_config
from douzero.runtime import maybe_set_global_deterministic, set_global_seed


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with SUPPRESS defaults for trainer knobs.

    Every trainer knob uses ``default=argparse.SUPPRESS`` so an absent flag
    does NOT shadow the YAML value. ``--config`` and the boolean
    ``--deterministic`` are the only exceptions (they have real defaults).
    """
    parser = argparse.ArgumentParser(
        description="DouZero V2 multi-objective trainer (CPU or P14 DDP)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a YAML training config (e.g. configs/enhanced.yaml). "
        "When empty, the V2 trainer uses its built-in defaults.",
    )
    # Trainer knobs: SUPPRESS so absent flags do not clobber YAML.
    parser.add_argument("--episodes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--optimizer_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--buffer_capacity", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--exp_epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rmsprop_alpha", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rmsprop_momentum", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rmsprop_epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_grad_norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_steps_per_episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"],
                        default=argparse.SUPPRESS)
    parser.add_argument("--ddp_enabled", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--ddp_backend", choices=["auto", "nccl", "gloo"],
                        default=argparse.SUPPRESS)
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--amp_enabled", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"],
                        default=argparse.SUPPRESS)
    parser.add_argument("--amp_fallback_on_nonfinite",
                        action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable full deterministic mode (torch.use_deterministic_algorithms). "
        "Absent: defer to the YAML 'deterministic' value (default false).",
    )
    parser.add_argument(
        "--belief_checkpoint",
        default=argparse.SUPPRESS,
        help="Path to a pretrained belief checkpoint (.pt) for belief-enabled "
        "value models (model.belief_enabled=true). Loaded with "
        "load_belief_checkpoint (ruleset + feature identity validated) and "
        "passed to V2Trainer as the belief model. REQUIRED when the "
        "value model has belief_enabled=true; ignored otherwise.",
    )
    parser.add_argument(
        "--belief_training_mode",
        choices=["frozen", "joint", "alternating"],
        default=argparse.SUPPRESS,
        help="Belief/value optimization mode (default: frozen).",
    )
    parser.add_argument(
        "--belief_supervised_weight", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--belief_alternating_interval", type=int, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--belief_supervised_batch_size", type=int, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--belief_supervised_episodes",
        type=int,
        default=argparse.SUPPRESS,
        help="Synthetic labelled episodes for the supervised joint/alternating term.",
    )
    parser.add_argument(
        "--first_bidder_mode",
        choices=["rotate", "seeded_random"],
        default=argparse.SUPPRESS,
        help="Reproducible opening-bidder schedule for standard training.",
    )
    parser.add_argument(
        "--bidding_policy",
        choices=["random", "rule", "max", "pass", "learned"],
        default=argparse.SUPPRESS,
        help="Standard-mode bidding policy (CLI overrides bidding.policy).",
    )
    parser.add_argument(
        "--bidding_warm_start_policy",
        choices=["random", "rule", "max", "pass"],
        default=argparse.SUPPRESS,
        help="Fallback policy while learned bidding is being phased in.",
    )
    parser.add_argument(
        "--bidding_learned_probability",
        type=float,
        default=argparse.SUPPRESS,
        help="Probability of using the learned head when policy=learned.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=argparse.SUPPRESS,
        help="Strict resumable trainer checkpoint to restore before collection.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=argparse.SUPPRESS,
        help="Atomically write a resumable trainer checkpoint after training.",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=argparse.SUPPRESS,
        help="Atomically write sanitized machine-readable training metrics.",
    )
    parser.add_argument(
        "--long_running",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly enable collect/optimize/checkpoint cycles. Absent: "
        "run the legacy one-shot V2 flow.",
    )
    parser.add_argument("--episodes_per_cycle", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--optimizer_steps_per_cycle", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_cycles", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_total_episodes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_total_optimizer_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--max_wall_time_minutes",
        type=float,
        default=argparse.SUPPRESS,
        help="Cumulative wall-time budget across checkpoint resume boundaries.",
    )
    parser.add_argument("--checkpoint_every_cycles", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_every_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_every_minutes", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--keep_last_checkpoints", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--save_on_interrupt", action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--eval_every_cycles", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--eval_command", type=str, default=argparse.SUPPRESS,
        help="Existing evaluator command; {checkpoint} and {cycle} placeholders "
        "are expanded without a shell.",
    )
    parser.add_argument(
        "--eval_fail_fast", action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    return parser


def _load_yaml_config(path: str):
    """Load a TrainingConfig from YAML, or return None when path is empty."""
    if not path:
        return None
    return load_config(path)


def _assert_v2_identity(cfg) -> None:
    """Reject a YAML config whose identity does not match the V2 trainer.

    P06 r2 fix: the r0/r1 entry unconditionally built a ModelV2 even when
    the YAML declared feature_version=legacy / model_version=legacy, so
    ``--config configs/legacy.yaml`` would silently run a V2 model under a
    legacy identity. This gate fails FAST — before any model/env is built
    — with a precise error.

    Both legacy card-play and the standard full-game bidding state machine are
    supported. Standard mode requires the bidding config, model head, and loss
    identity to agree; legacy mode rejects those opt-in fields so its parameter
    graph remains unchanged.
    """
    if cfg is None:
        return
    if cfg.feature_version != "v2":
        raise ValueError(
            f"train_v2.py requires feature_version='v2', got "
            f"{cfg.feature_version!r}. The V2 trainer builds a ModelV2 that "
            f"consumes the V2 observation schema; a legacy feature_version "
            f"would silently pair a V2 model with legacy observations. Use "
            f"configs/enhanced.yaml or set feature_version: v2."
        )
    if cfg.model_version != "v2":
        raise ValueError(
            f"train_v2.py requires model_version='v2', got {cfg.model_version!r}."
        )
    # P06 r6: defense-in-depth. The loader already cross-validates
    # model_version == model.version, but this gate is the last line of
    # defense before a V2 model is constructed.
    if cfg.model.version != "v2":
        raise ValueError(
            f"train_v2.py requires model.version='v2', got "
            f"{cfg.model.version!r}. The nested model.version must match "
            f"model_version (enforced by the loader)."
        )
    if cfg.ruleset not in {"legacy", "standard"}:
        raise ValueError(f"unsupported V2 training ruleset {cfg.ruleset!r}")
    if cfg.ruleset == "standard":
        if not cfg.bidding.enabled:
            raise ValueError(
                "ruleset='standard' requires bidding.enabled=true; standard "
                "training cannot bypass the public auction state machine."
            )
        if not cfg.model.bidding_enabled:
            raise ValueError(
                "ruleset='standard' requires model.bidding_enabled=true."
            )
    elif cfg.bidding.enabled or cfg.model.bidding_enabled:
        raise ValueError(
            "legacy V2 training requires bidding.enabled=false and "
            "model.bidding_enabled=false so the legacy graph is unchanged."
        )
    if cfg.objective != "adp":
        raise NotImplementedError(
            f"train_v2.py (P06) only supports objective='adp' (the multi-objective "
            f"label path derives team-perspective ADP scores); got {cfg.objective!r}."
        )


#: Legacy multiprocess fields the P06 V2 trainer does NOT consume. Each entry
#: maps the field name to its TrainingConfig default; a non-default value in a
#: YAML config triggers a visible warning (P06 r3 fix: these were silently
#: ignored in r0-r2). P14's high-throughput trainer will consume them.
_UNSUPPORTED_LEGACY_FIELDS: dict[str, object] = {
    "xpid": "douzero",
    "save_interval": 30,
    "savedir": "douzero_checkpoints",
    "actor_device_cpu": False,
    "gpu_devices": "0",
    "num_actor_devices": 1,
    "num_actors": 5,
    "training_device": "0",
    "load_model": False,
    "disable_checkpoint": False,
    "total_frames": 100000000000,
    "unroll_length": 100,
    "num_buffers": 50,
    "num_threads": 4,
}


def _warn_unsupported_legacy_fields(cfg) -> None:
    """Print a visible warning when a YAML config sets a legacy multiprocess
    field to a non-default value the P06 V2 trainer will silently ignore.

    P06 r3 fix: the enhanced.yaml carries these fields for TrainingConfig
    schema compatibility, but the single-process trainer does not consume
    them. Rather than silently accepting a user's tuned ``num_actors: 16``
    (which would have NO effect), we print a stderr warning naming each
    ignored field so the user knows the V2 trainer path does not use it.
    """
    if cfg is None:
        return
    ignored = []
    for field_name, default_val in _UNSUPPORTED_LEGACY_FIELDS.items():
        actual = getattr(cfg, field_name, default_val)
        if actual != default_val:
            ignored.append(f"  {field_name}: {actual!r} (default {default_val!r})")
    if ignored:
        import sys

        print(
            "[train_v2] WARNING: the following legacy multiprocess fields are "
            "set in the config but NOT consumed by the P06 single-process V2 "
            "trainer. They will be consumed by P14's high-throughput trainer.\n"
            + "\n".join(ignored),
            file=sys.stderr,
        )


def _resolve_ruleset(cfg):
    """Use ``None`` for byte-compatible legacy play and an explicit standard rule."""
    from douzero.env.rules import RuleSet

    if cfg is None:
        return None
    if cfg.ruleset == "standard":
        return RuleSet.standard()
    return None


def _build_loss_config(cfg):
    """Build the trainer's LossConfig honouring explicit YAML weights.

    P06 r1 fix: a YAML ``lambda_win: 0`` is preserved as 0 (it disables the
    term per the LossConfig contract). The previous r0 code converted 0
    back to 1.0, silently breaking the "λ=0 disables" contract. When no
    YAML config is provided we fall back to the V2 multi-objective default
    (``LossConfig()``).

    P08 Blocker 1: ``lambda_bc`` is now threaded through so the YAML
    ``loss.lambda_bc`` actually drives the BC auxiliary term.
    """
    from douzero.training import LossConfig

    if cfg is None:
        return LossConfig()
    return LossConfig(
        lambda_win=cfg.loss.lambda_win,
        lambda_score=cfg.loss.lambda_score,
        lambda_uncertainty=cfg.loss.lambda_uncertainty,
        lambda_bc=cfg.loss.lambda_bc,
        lambda_bid_policy=cfg.loss.lambda_bid_policy,
        lambda_bid_win=cfg.loss.lambda_bid_win,
        lambda_bid_score=cfg.loss.lambda_bid_score,
        lambda_bid_regret=cfg.loss.lambda_bid_regret,
        lambda_min_turns=cfg.loss.lambda_min_turns,
        lambda_regain_initiative=cfg.loss.lambda_regain_initiative,
        lambda_teammate_finish=cfg.loss.lambda_teammate_finish,
        lambda_spring=cfg.loss.lambda_spring,
        lambda_structure=cfg.loss.lambda_structure,
        score_delta=cfg.loss.score_delta,
        score_target_transform=cfg.loss.score_target_transform,
        score_clamp=cfg.loss.score_clamp,
    )


def _build_decision_config(cfg):
    from douzero.training import DecisionConfig

    if cfg is None:
        return DecisionConfig()
    return DecisionConfig(
        mode=cfg.decision_policy.mode,
        abs_tol=cfg.decision_policy.abs_tol,
        rel_tol=cfg.decision_policy.rel_tol,
        risk_penalty=cfg.decision_policy.risk_penalty,
        prior_alpha=cfg.decision_policy.prior_alpha,
    )


def _build_model_cfg(cfg):
    """Build a ModelV2Config from the YAML config (P06 r2/r5).

    The score_clamp and score_target_transform come from the ``loss:``
    block (not a ``model:`` block) because they are training-identity
    fields that bind the model's output semantics to what the loss
    supervised. The remaining architecture knobs come from the ``model:``
    block when present (P06 r5).
    """
    from douzero.models_v2.config import ModelV2Config

    if cfg is None:
        return ModelV2Config()
    # If a model: block is present, bridge it through from_model_config
    # so YAML architecture knobs (hidden_size, history_encoder, etc.)
    # actually drive model construction. Then overlay the loss-identity
    # fields (score_clamp, score_target_transform).
    return ModelV2Config.from_training_config(cfg)


def _build_curriculum(cfg):
    """Build the opt-in P12 training sampler without touching evaluation."""

    if cfg is None or not cfg.curriculum.enabled:
        return None, None, "current", 0
    from douzero.coach import (
        CurriculumAuditLogger,
        CurriculumSchedule,
        CoachLabelStore,
        OpeningSampler,
        load_coach_checkpoint,
    )
    from douzero.env.rules import RuleSet

    cc = cfg.curriculum
    ruleset = RuleSet.standard() if cfg.ruleset == "standard" else RuleSet.legacy()
    coach = None
    coach_manifest = None
    if cc.mode != "true_random":
        coach, coach_manifest = load_coach_checkpoint(
            cc.coach_checkpoint,
            expected_ruleset_hash=ruleset.stable_hash(),
            expected_policy_version=cc.policy_version,
            current_policy_step=cc.policy_step,
            max_age_steps=cc.max_coach_age_steps,
        )
    schedule = CurriculumSchedule(
        early_until=cc.early_until,
        mid_until=cc.mid_until,
        min_true_random_ratio=cc.min_true_random_ratio,
        early={
            "true_random": cc.early_true_random,
            "balanced": cc.early_balanced,
            "hard_for_role": cc.early_hard_for_role,
        },
        middle={
            "true_random": cc.middle_true_random,
            "balanced": cc.middle_balanced,
            "hard_for_role": cc.middle_hard_for_role,
        },
        late={
            "true_random": cc.late_true_random,
            "balanced": cc.late_balanced,
            "hard_for_role": cc.late_hard_for_role,
        },
    )
    audit_logger = (
        CurriculumAuditLogger(cc.audit_log_path) if cc.audit_log_path else None
    )
    label_store = CoachLabelStore(cc.labels_path) if cc.labels_path else None
    sampler = OpeningSampler(
        ruleset=ruleset,
        policy_version=cc.policy_version,
        coach=coach,
        coach_policy_step=(
            coach_manifest["policy_step"] if coach_manifest is not None else None
        ),
        max_coach_age_steps=(
            cc.max_coach_age_steps if coach_manifest is not None else None
        ),
        mode=cc.mode,
        schedule=schedule,
        hard_role=cc.hard_role,
        candidate_pool_size=cc.candidate_pool_size,
        seed=cc.seed,
        logger=audit_logger,
    )
    return sampler, label_store, cc.policy_version, cc.policy_step


def _validate_ddp_features(
    cfg,
    model_cfg,
    loss_cfg,
    distributed,
    *,
    belief_training_mode: str = "frozen",
    resume_checkpoint: str = "",
    checkpoint_path: str = "",
) -> None:
    """Reject unsupported DDP requests before process-group initialization.

    ``distributed`` accepts either the requested boolean or an initialized
    :class:`DistributedContext`.  The CLI deliberately calls this helper with
    the boolean request so unsupported graphs and checkpoint I/O fail before
    ``torch.distributed.init_process_group`` can create any runtime state.
    """
    enabled = (
        bool(distributed)
        if isinstance(distributed, bool)
        else bool(distributed.enabled)
    )
    if not enabled:
        return
    if bool(getattr(model_cfg, "bidding_enabled", False)) or (
        cfg is not None and getattr(cfg, "ruleset", "legacy") == "standard"
    ):
        raise NotImplementedError(
            "DDP does not yet support standard learned-bidding training: the "
            "bid and card-play paths use different parameter sets under the "
            "current static graph. Run this configuration single-process."
        )
    if belief_training_mode != "frozen":
        raise NotImplementedError(
            "DDP does not yet synchronize joint/alternating BeliefModel "
            "gradients. Use belief_training_mode=frozen or run "
            "single-process."
        )
    if resume_checkpoint or checkpoint_path:
        raise NotImplementedError(
            "trainer checkpoint save/resume is currently single-process only; "
            "remove --resume_checkpoint/--checkpoint_path for DDP"
        )
    if cfg is not None and cfg.curriculum.enabled:
        raise NotImplementedError(
            "DDP does not support curriculum/coach-label training yet: its "
            "audit and label stores require a single writer. Disable "
            "curriculum or run train_v2.py without DDP."
        )
    if loss_cfg.lambda_bc > 0:
        raise NotImplementedError(
            "DDP does not support RL+BC auxiliary training yet: BC validation "
            "and quarantine require rank-zero coordination. Set lambda_bc=0 "
            "or run train_v2.py without DDP."
        )
    if model_cfg.human_prior_enabled:
        raise ValueError(
            "DDP requires every enabled trainable head to contribute to the "
            "loss: model.human_prior_enabled=true but loss.lambda_bc=0. "
            "Disable the prior head for DDP."
        )
    strategy_weight = sum(
        float(getattr(loss_cfg, name))
        for name in (
            "lambda_min_turns",
            "lambda_regain_initiative",
            "lambda_teammate_finish",
            "lambda_spring",
            "lambda_structure",
        )
    )
    if model_cfg.strategy_aux_enabled and strategy_weight == 0:
        raise ValueError(
            "DDP requires every enabled trainable head to contribute to the "
            "loss: model.strategy_aux_enabled=true but all strategy auxiliary "
            "loss weights are zero. Disable the auxiliary heads or enable an "
            "auxiliary loss."
        )


def _build_training_metrics(
    stats,
    *,
    training_wall_seconds: float,
    device_type: str,
    peak_memory_bytes: int | None,
    peak_reserved_memory_bytes: int | None,
    amp_enabled: bool,
    amp_dtype: str,
    amp_fallback_on_nonfinite: bool,
    compile_enabled: bool,
    ddp_enabled: bool,
    world_size: int,
    parameters_changed: bool | None,
) -> dict[str, object]:
    """Build finite, sanitized P17 throughput and memory diagnostics."""
    elapsed = max(float(training_wall_seconds), 1.0e-12)
    cardplay = int(stats.transitions_collected)
    bidding = int(stats.bidding_transitions_collected)
    decisions = cardplay + bidding

    def per_second(count: int) -> float:
        return round(float(count) / elapsed, 6)

    def mib(value: int | None) -> float | None:
        if value is None:
            return None
        return round(float(value) / (1024.0 * 1024.0), 3)

    return {
        "schema_version": "p17-gpu-run-v1",
        "status": "passed",
        "device_type": str(device_type),
        "training_wall_seconds": round(elapsed, 6),
        "counts": {
            "episodes": int(stats.episodes_completed),
            "cardplay_transitions": cardplay,
            "bidding_decisions": bidding,
            "total_decisions": decisions,
            "learner_steps": int(stats.optimizer_steps),
            "redeals": int(stats.redeals),
            "max_redeals_exceeded": int(stats.max_redeals_exceeded),
            "belief_supervised_steps": int(stats.belief_supervised_steps),
        },
        "metrics": {
            "peak_memory_mib": mib(peak_memory_bytes),
            "peak_reserved_memory_mib": mib(peak_reserved_memory_bytes),
            "cardplay_transitions_per_second": per_second(cardplay),
            "bidding_decisions_per_second": per_second(bidding),
            # Samples are replay transitions; decisions include card play and bids.
            "samples_per_second": per_second(decisions),
            "decisions_per_second": per_second(decisions),
            "learner_steps_per_second": per_second(int(stats.optimizer_steps)),
        },
        "amp": {
            "enabled": bool(amp_enabled),
            "dtype": str(amp_dtype),
            "fallback_on_nonfinite": bool(amp_fallback_on_nonfinite),
            "fallback_count": int(stats.amp_fallbacks),
            "fallback_exercised": bool(stats.amp_fallbacks),
        },
        "compile": {"enabled": bool(compile_enabled)},
        "distributed": {
            "enabled": bool(ddp_enabled),
            "world_size": int(world_size),
        },
        "parameter_update_observed": parameters_changed,
        "privacy": "sanitized_no_host_or_device_identifiers",
    }


def _write_metrics_atomic(path: str, payload: dict[str, object]) -> None:
    """Write a JSON metric artifact without exposing command-line paths."""
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


class _ResumeCheckpoint:
    __slots__ = ("checkpoint", "series_base", "manifest", "total_wall_seconds")

    def __init__(
        self,
        checkpoint: str,
        series_base: str = "",
        manifest: str = "",
        total_wall_seconds: float | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.series_base = series_base
        self.manifest = manifest
        self.total_wall_seconds = total_wall_seconds


def _resolve_resume_checkpoint(path: str) -> _ResumeCheckpoint:
    """Resolve only a valid ``*-latest.json`` manifest, including an orphan."""
    import json
    from pathlib import Path

    source = Path(path)
    if not source.name.endswith("-latest.json"):
        try:
            import torch

            bundle = torch.load(source, map_location="cpu", weights_only=True)
            state_payload = (
                bundle.get("long_running_state")
                if isinstance(bundle, dict) else None
            )
            if state_payload is None:
                return _ResumeCheckpoint(str(source))
            from douzero.training.long_running import (
                CheckpointSeries,
                LongRunningState,
            )

            state = LongRunningState.from_dict(state_payload)
            series = CheckpointSeries.from_checkpoint(source, state, 1)
            if series.latest_manifest.exists():
                resolved = _resolve_resume_checkpoint(str(series.latest_manifest))
                resolved_sequence = series.checkpoint_sequence(
                    resolved.checkpoint, run_id=state.run_id
                )
                if state.checkpoint_sequence > resolved_sequence:
                    raise ValueError(
                        "resume checkpoint is ahead of the checkpoint series manifest"
                    )
                return resolved
            return _ResumeCheckpoint(
                str(source), str(series.base), total_wall_seconds=state.total_wall_seconds
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            raise
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _ResumeCheckpoint(str(source))
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        return _ResumeCheckpoint(str(source))

    from douzero.training.long_running import CheckpointSeries, LongRunningState

    payload = CheckpointSeries._validate_manifest(payload)
    latest = source.parent / payload["latest"]
    if not latest.is_file():
        raise FileNotFoundError(f"latest checkpoint does not exist: {latest}")
    prefix = source.name[:-len("-latest.json")]
    suffix = latest.suffix or ".pt"
    series = CheckpointSeries(str(source.with_name(f"{prefix}{suffix}")), 1)
    expected = series.cycle_path(
        payload["run_id"],
        payload["checkpoint_sequence"],
        payload["cycle"],
        payload["total_optimizer_steps"],
    )
    if latest.name != expected.name:
        raise ValueError("latest manifest checkpoint name does not match its counters")

    import torch

    latest_bundle = torch.load(latest, map_location="cpu", weights_only=True)
    latest_state_payload = (
        latest_bundle.get("long_running_state")
        if isinstance(latest_bundle, dict) else None
    )
    latest_state = LongRunningState.from_dict(latest_state_payload)
    if payload["total_wall_seconds"] < latest_state.total_wall_seconds:
        raise ValueError("latest manifest wall time is older than checkpoint state")
    latest_state.total_wall_seconds = payload["total_wall_seconds"]
    if payload != series._manifest_payload(latest_state, latest):
        raise ValueError("latest manifest does not match checkpoint state")

    indexed = series._indexed_checkpoints(payload["run_id"])
    candidates = [
        candidate for sequence, candidate in indexed.items()
        if sequence > payload["checkpoint_sequence"]
    ]
    if candidates:
        if len(candidates) != 1:
            raise ValueError("checkpoint series contains multiple orphan sequences")
        candidate = candidates[0]
        bundle = torch.load(candidate, map_location="cpu", weights_only=True)
        state_payload = bundle.get("long_running_state") if isinstance(bundle, dict) else None
        state = LongRunningState.from_dict(state_payload)
        if state.run_id != payload["run_id"] or (
            state.checkpoint_sequence != payload["checkpoint_sequence"] + 1
        ):
            raise ValueError("orphan checkpoint is not the next series sequence")
        expected_orphan = series.cycle_path(
            state.run_id,
            state.checkpoint_sequence,
            state.cycle,
            state.total_optimizer_steps,
        )
        if candidate.name != expected_orphan.name:
            raise ValueError("orphan checkpoint filename does not match its state")
        latest = candidate
        payload["total_wall_seconds"] = max(
            payload["total_wall_seconds"], state.total_wall_seconds
        )
    return _ResumeCheckpoint(
        str(latest),
        str(series.base),
        str(source),
        total_wall_seconds=payload["total_wall_seconds"],
    )


def _select_long_running_checkpoint_path(
    requested: str, resume: _ResumeCheckpoint
) -> str:
    """Keep manifest resume on its original series unless no resume exists."""
    if resume.series_base:
        if requested and Path(requested).resolve() != Path(resume.series_base).resolve():
            raise ValueError(
                "--checkpoint_path does not match the resumed manifest series"
            )
        return resume.series_base
    if requested:
        return requested
    if resume.checkpoint:
        return ""
    return "douzero_checkpoints/v2-long-running.pt"


def main() -> None:
    args = _build_parser().parse_args()

    long_option_names = {
        "episodes_per_cycle", "optimizer_steps_per_cycle", "max_cycles",
        "max_total_episodes", "max_total_optimizer_steps",
        "max_wall_time_minutes", "checkpoint_every_cycles",
        "checkpoint_every_steps", "checkpoint_every_minutes",
        "keep_last_checkpoints", "save_on_interrupt", "eval_every_cycles",
        "eval_command", "eval_fail_fast",
    }
    if not args.long_running and any(hasattr(args, name) for name in long_option_names):
        raise ValueError("long-running options require explicit --long_running")

    yaml_cfg = _load_yaml_config(args.config)

    # P06 r2: fail FAST on an identity mismatch. The V2 trainer builds a
    # ModelV2 + V2 observation schema; a legacy/standard identity would
    # silently pair them with the wrong observation path.
    _assert_v2_identity(yaml_cfg)

    # P06 r3: warn about legacy multiprocess fields the V2 trainer does not
    # consume, so a user who tunes num_actors / gpu_devices / etc. in the
    # YAML is alerted that the single-process trainer ignores them.
    _warn_unsupported_legacy_fields(yaml_cfg)

    # Build the seed/deterministic resolver. ``resolve(cli_dest, yaml_value,
    # default_value)`` implements CLI > YAML > built-in-default precedence.
    # argparse SUPPRESS defaults mean absent CLI flags do NOT appear in
    # vars(args), so the YAML value wins when the user did not pass the flag.
    from douzero.training import TrainerConfig

    defaults = TrainerConfig()

    def resolve(cli_dest, yaml_value, default_value):
        if hasattr(args, cli_dest):
            return getattr(args, cli_dest)
        if yaml_value is not None:
            return yaml_value
        return default_value

    opt = yaml_cfg.optimizer if yaml_cfg is not None else None
    seed = resolve("seed", yaml_cfg.seed if yaml_cfg else None, defaults.seed)
    deterministic = resolve(
        "deterministic",
        yaml_cfg.deterministic if yaml_cfg else None,
        False,
    )
    ddp_enabled = resolve(
        "ddp_enabled", yaml_cfg.ddp_enabled if yaml_cfg else None, False
    )
    ddp_backend = resolve(
        "ddp_backend", yaml_cfg.ddp_backend if yaml_cfg else None, "auto"
    )
    loss_cfg = _build_loss_config(yaml_cfg)
    model_cfg = _build_model_cfg(yaml_cfg)
    belief_training_mode = resolve(
        "belief_training_mode",
        yaml_cfg.belief_training_mode if yaml_cfg else None,
        defaults.belief_training_mode,
    )
    if args.long_running and ddp_enabled:
        raise NotImplementedError("long-running V2 training is single-process only")
    resume_argument = getattr(args, "resume_checkpoint", "")
    resume = (
        _resolve_resume_checkpoint(resume_argument)
        if resume_argument and args.long_running
        else _ResumeCheckpoint(resume_argument)
    )
    resume_checkpoint = resume.checkpoint
    output_checkpoint = getattr(args, "checkpoint_path", "")
    if args.long_running:
        output_checkpoint = _select_long_running_checkpoint_path(
            output_checkpoint, resume
        )
    # This must remain before initialize_distributed. Unsupported feature
    # combinations and checkpoint I/O are configuration errors, not reasons to
    # create a process group and only then abort each rank.
    _validate_ddp_features(
        yaml_cfg,
        model_cfg,
        loss_cfg,
        bool(ddp_enabled),
        belief_training_mode=belief_training_mode,
        resume_checkpoint=resume_checkpoint,
        checkpoint_path=output_checkpoint,
    )

    import torch

    from douzero.runtime.distributed import initialize_distributed

    distributed = initialize_distributed(
        enabled=ddp_enabled, backend=ddp_backend
    )
    if distributed.enabled:
        import atexit

        atexit.register(distributed.close)
    requested_device = resolve("device", None, "cpu")
    if distributed.enabled:
        learner_device = distributed.device
    elif requested_device == "auto":
        learner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        learner_device = torch.device("cuda")
    else:
        learner_device = torch.device("cpu")
    rank_seed = seed + distributed.rank if seed != 0 else 0
    set_global_seed(rank_seed)
    maybe_set_global_deterministic(deterministic)

    # Build the V2 model from the schema + config (honouring score_clamp so
    # the head clamp matches the loss target clamp exactly).
    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    # P06 r3: respect the project's seed=0 → no-op contract. set_global_seed(0)
    # is a no-op (Python/NumPy/Torch all unseeded); we must NOT separately
    # torch.manual_seed(0) or the model init would be seeded while the deal
    # shuffle is not — a mixed, contradictory reproducibility state. Only
    # seed Torch when the user explicitly requested a non-zero seed.
    if rank_seed != 0:
        torch.manual_seed(rank_seed)
    schema = build_v2_schema()
    model = ModelV2(schema, model_cfg).to(learner_device)
    compile_enabled = resolve(
        "compile_model", yaml_cfg.compile_model if yaml_cfg else None, False
    )
    if compile_enabled:
        if belief_training_mode != "frozen":
            raise NotImplementedError(
                "compile_model has not been validated with joint/alternating "
                "belief training; use frozen mode or eager execution."
            )
        if model_cfg.bidding_enabled:
            raise NotImplementedError(
                "compile_model is not yet supported for learned bidding: "
                "torch.compile captures ModelV2.forward, while the auction "
                "uses the separate forward_bidding contract."
            )
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model requires torch.compile")
        model = torch.compile(model, dynamic=True)
    core_model = model
    amp_enabled = resolve(
        "amp_enabled", yaml_cfg.amp_enabled if yaml_cfg else None, False
    )
    model = distributed.wrap(model)
    if distributed.enabled:
        # DDP intentionally exposes only Module's API. The trainer also needs
        # immutable model identity/config helpers, so forward them explicitly.
        model.config = core_model.config
        model.schema = core_model.schema
        model.strategy_feature_config = core_model.strategy_feature_config

    from douzero.training import V2Trainer

    trainer_cfg = TrainerConfig(
        seed=rank_seed,
        rng_seed=rank_seed,
        # CLI dest "episodes" maps to TrainerConfig.max_episodes (TrainingConfig
        # has no episodes field; it comes from CLI or the TrainerConfig default).
        max_episodes=resolve("episodes", None, defaults.max_episodes),
        optimizer_steps=resolve("optimizer_steps", None, defaults.optimizer_steps),
        batch_size=resolve("batch_size", yaml_cfg.batch_size if yaml_cfg else None, defaults.batch_size),
        # buffer_capacity / max_steps_per_episode are TrainerConfig-only
        # (TrainingConfig has no such field), so they come from CLI or the
        # TrainerConfig default only.
        buffer_capacity=resolve("buffer_capacity", None, defaults.buffer_capacity),
        exp_epsilon=resolve("exp_epsilon", yaml_cfg.exp_epsilon if yaml_cfg else None, defaults.exp_epsilon),
        learning_rate=resolve(
            "learning_rate",
            opt.learning_rate if opt else None,
            defaults.learning_rate,
        ),
        rmsprop_alpha=resolve("rmsprop_alpha", opt.alpha if opt else None, defaults.rmsprop_alpha),
        rmsprop_momentum=resolve(
            "rmsprop_momentum", opt.momentum if opt else None, defaults.rmsprop_momentum
        ),
        rmsprop_epsilon=resolve(
            "rmsprop_epsilon", opt.epsilon if opt else None, defaults.rmsprop_epsilon
        ),
        max_grad_norm=resolve(
            "max_grad_norm", yaml_cfg.max_grad_norm if yaml_cfg else None, defaults.max_grad_norm
        ),
        max_steps_per_episode=resolve("max_steps_per_episode", None, defaults.max_steps_per_episode),
        device=str(learner_device),
        amp_enabled=amp_enabled,
        amp_dtype=resolve(
            "amp_dtype", yaml_cfg.amp_dtype if yaml_cfg else None, "float16"
        ),
        amp_fallback_on_nonfinite=resolve(
            "amp_fallback_on_nonfinite",
            yaml_cfg.amp_fallback_on_nonfinite if yaml_cfg else None,
            True,
        ),
        belief_training_mode=belief_training_mode,
        belief_supervised_weight=resolve(
            "belief_supervised_weight",
            yaml_cfg.belief_supervised_weight if yaml_cfg else None,
            defaults.belief_supervised_weight,
        ),
        belief_alternating_interval=resolve(
            "belief_alternating_interval",
            yaml_cfg.belief_alternating_interval if yaml_cfg else None,
            defaults.belief_alternating_interval,
        ),
        belief_supervised_batch_size=resolve(
            "belief_supervised_batch_size",
            yaml_cfg.belief_supervised_batch_size if yaml_cfg else None,
            defaults.belief_supervised_batch_size,
        ),
        first_bidder_mode=resolve(
            "first_bidder_mode",
            yaml_cfg.first_bidder_mode if yaml_cfg else None,
            defaults.first_bidder_mode,
        ),
    )
    if distributed.enabled and trainer_cfg.belief_training_mode != "frozen":
        raise NotImplementedError(
            "DDP does not yet synchronize BeliefModel gradients; use "
            "belief_training_mode=frozen or run single-process."
        )

    ruleset = _resolve_ruleset(yaml_cfg)
    decision_cfg = _build_decision_config(yaml_cfg)
    bidding_policy_cfg = None
    if ruleset is not None:
        from douzero.training import BiddingPolicyConfig

        yaml_bidding = yaml_cfg.bidding
        bidding_policy_cfg = BiddingPolicyConfig(
            policy=resolve(
                "bidding_policy", yaml_bidding.policy, yaml_bidding.policy
            ),
            warm_start_policy=resolve(
                "bidding_warm_start_policy",
                yaml_bidding.warm_start_policy,
                yaml_bidding.warm_start_policy,
            ),
            learned_probability=resolve(
                "bidding_learned_probability",
                yaml_bidding.learned_probability,
                yaml_bidding.learned_probability,
            ),
        )
    opening_sampler, coach_label_store, policy_version, policy_step = (
        _build_curriculum(yaml_cfg)
    )

    # P17: load the belief model when belief fusion is enabled. Frozen mode
    # keeps the P07 feature-source behavior; joint/alternating mode makes the
    # same public-only encoder trainable inside V2Trainer.
    # The checkpoint is validated (ruleset + feature version + architecture
    # hash) via load_belief_checkpoint. Without this a belief-enabled value
    # model fails closed at forward.
    belief_model = None
    if getattr(model_cfg, "belief_enabled", False):
        belief_ckpt = getattr(args, "belief_checkpoint", None)
        if not belief_ckpt:
            raise ValueError(
                "The value model has belief_enabled=true but no "
                "--belief_checkpoint was supplied. A belief-enabled value "
                "model requires a pretrained BeliefModel initialization. Run "
                "train_belief.py with the matching --ruleset first, then pass its "
                "checkpoint via --belief_checkpoint."
            )
        from douzero.belief.checkpoint import load_belief_checkpoint

        belief_model = load_belief_checkpoint(
            belief_ckpt,
            expected_ruleset=ruleset or __import__(
                "douzero.env.rules", fromlist=["RuleSet"]
            ).RuleSet.legacy(),
            expected_feature_version="v2",
            require_full_git_sha=True,
        )

    belief_supervised_episodes = resolve(
        "belief_supervised_episodes",
        yaml_cfg.belief_supervised_episodes if yaml_cfg else None,
        0,
    )
    belief_supervised_samples = None
    if belief_supervised_episodes > 0:
        if trainer_cfg.belief_training_mode == "frozen":
            raise ValueError(
                "belief_supervised_episodes is only valid in joint/alternating mode"
            )
        if trainer_cfg.belief_supervised_weight <= 0:
            raise ValueError(
                "belief_supervised_episodes requires belief_supervised_weight > 0"
            )
        from douzero.belief.data import collect_random_dataset

        belief_dataset = collect_random_dataset(
            belief_supervised_episodes,
            seed=rank_seed,
            max_steps_per_episode=trainer_cfg.max_steps_per_episode,
            ruleset=ruleset,
        )
        belief_supervised_samples = belief_dataset.samples
        if not belief_supervised_samples:
            raise RuntimeError("supervised belief collection produced no samples")
    elif trainer_cfg.belief_supervised_weight > 0:
        raise ValueError(
            "belief_supervised_weight > 0 requires --belief_supervised_episodes "
            "(or the YAML belief_supervised_episodes field) in the CLI path"
        )

    # P08 Blocker 1: build the BC auxiliary samples + schedule when the YAML
    # enables RL+BC. This wires bc.data_path -> BCSamples -> V2Trainer's
    # bc_aux_samples, and bc.schedule -> BCSchedule, so the YAML config
    # ACTUALLY drives the BC auxiliary term end-to-end (not just the
    # programmatic interface tested in isolation).
    import sys  # noqa: F811 — local import keeps the BC block self-contained

    bc_aux_samples = None
    bc_schedule = None
    bc_cfg = getattr(yaml_cfg, "bc", None) if yaml_cfg is not None else None
    # The model must have a prior head when lambda_bc > 0; if the YAML sets
    # lambda_bc > 0 but forgot human_prior_enabled, fail fast with a precise
    # error rather than letting the trainer reject it later.
    if loss_cfg.lambda_bc > 0:
        if not getattr(model_cfg, "human_prior_enabled", False):
            raise ValueError(
                "loss.lambda_bc > 0 requires model.human_prior_enabled=true. "
                "The BC auxiliary loss trains the prior head, which the model "
                "does not have under the current config."
            )
        from douzero.training.bc_loss import BCSchedule

        if bc_cfg is None:
            raise ValueError(
                "loss.lambda_bc > 0 requires a bc: config block "
                "(bc.data_path at minimum)."
            )
        if not bc_cfg.data_path:
            raise ValueError(
                "loss.lambda_bc > 0 requires bc.data_path (a validated "
                "canonical JSONL of human games). Run ingest_human_games.py "
                "+ validate_human_games.py first."
            )
        # Load + validate + sample the human data ONCE at startup (Blocker 2:
        # the ruleset identity is verified inside build_bc_samples).
        from douzero.human_data.sample import build_bc_samples_with_report
        from douzero.human_data.schema import read_verified_jsonl
        from douzero.human_data.validate import validate_record
        from douzero.human_data.weights import WeightConfig, apply_sample_weights

        # Blocker 1: load + validate + sample with NO silent drops. Records
        # that fail replay validation are collected into validation_quarantine
        # and written to a bc_quarantine.jsonl alongside the checkpoint, with
        # the game_id + reason + error. They are NEVER swallowed by a filter
        # generator (the earlier `r for r in ... if validate_record(r).ok`
        # silently dropped them with no trace).
        bc_records = list(read_verified_jsonl(bc_cfg.data_path))
        valid_records = []
        validation_quarantine: list[str] = []
        import json as _json

        for record in bc_records:
            result = validate_record(record)
            if result.ok:
                valid_records.append(record)
            else:
                validation_quarantine.append(
                    _json.dumps(
                        {
                            "game_id": record.game_id,
                            "stage": "replay_validation",
                            "reason": result.reason,
                            "error": result.error,
                        },
                        sort_keys=True, ensure_ascii=False,
                    )
                )
        bc_report = build_bc_samples_with_report(valid_records)
        # Merge the sampling-stage quarantine with the validation-stage one.
        for game_id, error in bc_report.quarantined:
            validation_quarantine.append(
                _json.dumps(
                    {
                        "game_id": game_id,
                        "stage": "bc_sampling",
                        "reason": "BCSampleError",
                        "error": error,
                    },
                    sort_keys=True, ensure_ascii=False,
                )
            )
        if validation_quarantine:
            q_path = "bc_quarantine.jsonl"
            with open(q_path, "w", encoding="utf-8") as fh:
                for line in validation_quarantine:
                    fh.write(line)
                    fh.write("\n")
            print(
                f"[train_v2] BC quarantine: {len(validation_quarantine)} "
                f"records failed validation/sampling -> {q_path} "
                f"(run validate_human_games.py to quarantine upstream).",
                file=sys.stderr,
            )
        if not bc_report.samples:
            raise ValueError(
                "configured BC dataset yielded no BC samples "
                f"({len(validation_quarantine)} quarantined)."
            )
        bc_aux_samples = apply_sample_weights(
            bc_report.samples,
            config=WeightConfig(
                skill_weight_clip=bc_cfg.skill_weight_clip
            ),
        )
        bc_schedule = BCSchedule(
            base_lambda=loss_cfg.lambda_bc,
            schedule=bc_cfg.schedule,
            schedule_steps=bc_cfg.schedule_steps,
            schedule_floor=bc_cfg.schedule_floor,
        )
        print(
            f"[train_v2] BC aux: {len(bc_aux_samples)} samples from "
            f"configured dataset, schedule={bc_cfg.schedule} "
            f"lambda_bc={loss_cfg.lambda_bc}",
            file=sys.stderr,
        )

    trainer = V2Trainer(
        model,
        ruleset=ruleset,
        loss_config=loss_cfg,
        decision_config=decision_cfg,
        config=trainer_cfg,
        belief_model=belief_model,
        belief_supervised_samples=belief_supervised_samples,
        bc_aux_samples=bc_aux_samples,
        bc_schedule=bc_schedule,
        bc_temperature=(bc_cfg.temperature if bc_cfg is not None else 1.0),
        bc_label_smoothing=(bc_cfg.label_smoothing if bc_cfg is not None else 0.0),
        opening_sampler=opening_sampler,
        coach_label_store=coach_label_store,
        policy_version=policy_version,
        policy_step=policy_step,
        bidding_policy_config=bidding_policy_cfg,
        distributed_context=distributed,
    )

    resume_identity = None
    if resume_checkpoint:
        resume_identity = trainer.load_training_checkpoint(resume_checkpoint)
        if distributed.is_rank_zero:
            print(
                f"[train_v2] resumed checkpoint={resume_checkpoint!r} "
                f"identity={resume_identity}"
            )

    if distributed.is_rank_zero:
        print(
        f"[train_v2] model={type(model).__name__} "
        f"params={sum(p.numel() for p in model.parameters())} "
        f"score_clamp={model_cfg.score_clamp} "
        f"loss_cfg={loss_cfg.to_dict()} "
        f"decision={decision_cfg.to_dict()} "
        f"trainer=batch_size={trainer_cfg.batch_size} "
        f"lr={trainer_cfg.learning_rate} "
        f"epsilon={trainer_cfg.exp_epsilon} "
        f"amp={trainer_cfg.amp_enabled}:{trainer_cfg.amp_dtype} "
        f"belief_mode={trainer_cfg.belief_training_mode} "
        f"ruleset={trainer.ruleset.ruleset_id if trainer.ruleset else 'legacy'}"
        )
    metrics_path = getattr(args, "metrics_path", "")
    if learner_device.type == "cuda":
        torch.cuda.synchronize(learner_device)
        torch.cuda.reset_peak_memory_stats(learner_device)
    import time

    training_started = time.perf_counter()
    stop_reason = "one_shot_complete"
    try:
        if args.long_running:
            import json

            from douzero.training.long_running import (
                CheckpointSeries,
                LongRunningConfig,
                LongRunningState,
                LongRunningTrainer,
                RunMetricsWriter,
                command_evaluator,
            )

            long_cfg = LongRunningConfig(
                episodes_per_cycle=getattr(
                    args, "episodes_per_cycle", trainer_cfg.max_episodes
                ),
                optimizer_steps_per_cycle=getattr(
                    args, "optimizer_steps_per_cycle", trainer_cfg.optimizer_steps
                ),
                max_cycles=getattr(args, "max_cycles", 0),
                max_total_episodes=getattr(args, "max_total_episodes", 0),
                max_total_optimizer_steps=getattr(
                    args, "max_total_optimizer_steps", 0
                ),
                max_wall_time_minutes=getattr(args, "max_wall_time_minutes", 0.0),
                checkpoint_every_cycles=getattr(
                    args, "checkpoint_every_cycles", 1
                ),
                checkpoint_every_steps=getattr(args, "checkpoint_every_steps", 0),
                checkpoint_every_minutes=getattr(
                    args, "checkpoint_every_minutes", 0.0
                ),
                keep_last_checkpoints=getattr(args, "keep_last_checkpoints", 3),
                save_on_interrupt=getattr(args, "save_on_interrupt", True),
                eval_every_cycles=getattr(args, "eval_every_cycles", 0),
                eval_fail_fast=getattr(args, "eval_fail_fast", True),
            )
            eval_command = getattr(args, "eval_command", "")
            if long_cfg.eval_every_cycles and not eval_command:
                raise ValueError("--eval_every_cycles requires --eval_command")
            evaluator = command_evaluator(eval_command) if eval_command else None
            state = None
            if resume_checkpoint:
                state_payload = resume_identity.get("long_running_state")
                if state_payload is None:
                    raise ValueError(
                        "--long_running resume requires a cycle-boundary long-running checkpoint"
                    )
                state = LongRunningState.from_dict(state_payload)
                if resume.total_wall_seconds is not None:
                    if resume.total_wall_seconds < state.total_wall_seconds:
                        raise ValueError(
                            "resume manifest wall time is older than checkpoint state"
                        )
                    state.total_wall_seconds = resume.total_wall_seconds
                state.resume_source = resume_checkpoint
                derived_series = CheckpointSeries.from_checkpoint(
                    resume_checkpoint, state, long_cfg.keep_last_checkpoints
                )
                if output_checkpoint and (
                    Path(output_checkpoint).resolve() != derived_series.base.resolve()
                ):
                    raise ValueError(
                        "--checkpoint_path does not match the resumed checkpoint series"
                    )
                output_checkpoint = str(derived_series.base)

            metrics_writer = None

            def emit_cycle(record: dict) -> None:
                if distributed.is_rank_zero:
                    print("[train_v2] cycle_metric=" + json.dumps(record, sort_keys=True))
                    if metrics_writer is not None:
                        metrics_writer.write_cycle(record)

            peak_memory = (
                lambda: int(torch.cuda.max_memory_allocated(learner_device))
                if learner_device.type == "cuda" else None
            )
            runner = LongRunningTrainer(
                trainer,
                long_cfg,
                CheckpointSeries(output_checkpoint, long_cfg.keep_last_checkpoints),
                state=state,
                evaluator=evaluator,
                metric_sink=emit_cycle,
                peak_memory=peak_memory,
            )
            if metrics_path and distributed.is_rank_zero:
                metrics_writer = RunMetricsWriter(
                    metrics_path,
                    run_id=runner.state.run_id,
                    resume=bool(resume_checkpoint),
                )
            try:
                final_state, stop_reason, _records = runner.run()
            except BaseException as exc:
                if metrics_writer is not None:
                    metrics_writer.finalize(
                        status="failed",
                        stop_reason=(
                            runner.last_stop_reason or runner.stop.reason
                        ),
                        state=runner.state,
                        error=type(exc).__name__,
                    )
                raise
            stats = trainer.stats
            if metrics_writer is not None:
                metrics_writer.finalize(
                    status="stopped",
                    stop_reason=stop_reason,
                    state=final_state,
                )
        else:
            stats = trainer.train()
        if learner_device.type == "cuda":
            torch.cuda.synchronize(learner_device)
        training_wall_seconds = time.perf_counter() - training_started
        peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(learner_device))
            if learner_device.type == "cuda"
            else None
        )
        peak_reserved_memory_bytes = (
            int(torch.cuda.max_memory_reserved(learner_device))
            if learner_device.type == "cuda"
            else None
        )
        if output_checkpoint and distributed.is_rank_zero and not args.long_running:
            identity = trainer.save_training_checkpoint(output_checkpoint)
            print(
                f"[train_v2] saved checkpoint={output_checkpoint!r} "
                f"identity={identity}"
            )
        if metrics_path and distributed.is_rank_zero and not args.long_running:
            changed = getattr(trainer, "stats_last_run_changed", None)
            if not isinstance(changed, bool):
                changed = None
            _write_metrics_atomic(
                metrics_path,
                _build_training_metrics(
                    stats,
                    training_wall_seconds=training_wall_seconds,
                    device_type=learner_device.type,
                    peak_memory_bytes=peak_memory_bytes,
                    peak_reserved_memory_bytes=peak_reserved_memory_bytes,
                    amp_enabled=trainer_cfg.amp_enabled,
                    amp_dtype=trainer_cfg.amp_dtype,
                    amp_fallback_on_nonfinite=(
                        trainer_cfg.amp_fallback_on_nonfinite
                    ),
                    compile_enabled=compile_enabled,
                    ddp_enabled=distributed.enabled,
                    world_size=distributed.world_size,
                    parameters_changed=changed,
                ),
            )
    finally:
        distributed.close()
    if distributed.is_rank_zero:
        print(
        f"[train_v2] episodes_completed={stats.episodes_completed} "
        f"transitions={stats.transitions_collected} "
        f"bidding_transitions={stats.bidding_transitions_collected} "
        f"redeals={stats.redeals} "
        f"max_redeals_exceeded={stats.max_redeals_exceeded} "
        f"optimizer_steps={stats.optimizer_steps} "
        f"stop_reason={stop_reason} "
        f"parameters_changed={getattr(trainer, 'stats_last_run_changed', 'unknown')} "
        f"last_loss={stats.last_loss} "
        f"grad_norm={stats.grad_norm_last_step:.4f} "
        f"p_win_mean={stats.p_win_mean:.4f}"
        f" opening_strategies={stats.opening_strategy_counts}"
        f" opening_predicted_win_mean={stats.opening_predicted_win_mean:.4f}"
        )


if __name__ == "__main__":
    main()
