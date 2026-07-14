"""V2 multi-objective training entry point (P06).

This is the single-process trainer CLI for the P06 acceptance criterion:
"极短训练能完成一次优化且参数变化" (a very short training run completes one
optimizer step and changes the parameters). It loads a YAML config
(``configs/enhanced.yaml`` by convention), builds a
:class:`~douzero.models_v2.model.ModelV2`, and runs
:class:`~douzero.training.v2_trainer.V2Trainer`.

The legacy :file:`train.py` path is unchanged. This entry point exists so
the multi-objective loss + decision-policy + team-perspective labels can be
exercised without touching the legacy multiprocessing actor/learner.

CPU-only (P06). GPU + multiprocessing is P14.

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
import os

from douzero.config import load_config
from douzero.runtime import maybe_set_global_deterministic, set_global_seed


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with SUPPRESS defaults for trainer knobs.

    Every trainer knob uses ``default=argparse.SUPPRESS`` so an absent flag
    does NOT shadow the YAML value. ``--config`` and the boolean
    ``--deterministic`` are the only exceptions (they have real defaults).
    """
    parser = argparse.ArgumentParser(
        description="DouZero V2 multi-objective trainer (P06, CPU-only)",
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
        "passed to V2Trainer as a frozen feature source. REQUIRED when the "
        "value model has belief_enabled=true; ignored otherwise.",
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

    The V2 trainer (P06) only supports the legacy card-play-only ruleset
    and the ADP objective; the standard ruleset requires a bidding driver
    that is part of P11's league work, and non-ADP objectives are not wired
    through the multi-objective label path.
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
    if cfg.ruleset != "legacy":
        raise NotImplementedError(
            f"train_v2.py (P06) only supports ruleset='legacy' (the card-play-"
            f"only env); got {cfg.ruleset!r}. Standard mode requires a bidding "
            f"driver that is part of P11's league work."
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
    """Resolve the ruleset for the trainer.

    P06's V2Trainer only accepts ``ruleset=None`` (the legacy card-play-only
    env), because ``Env`` treats ANY non-None ``RuleSet`` as standard mode
    (entering the bidding phase), and the trainer has no bidding driver.
    A YAML ``ruleset: standard`` is surfaced to the trainer as
    ``RuleSet.standard()`` so the trainer's gate raises a precise error
    rather than silently mis-driving bidding.
    """
    from douzero.env.rules import RuleSet

    if cfg is None:
        return None
    if cfg.ruleset == "standard":
        # The trainer will reject this at construction with a precise error.
        return RuleSet.standard()
    # legacy or any other value: return None so Env runs in legacy card-play
    # mode (no bidding). The trainer accepts None.
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


def main() -> None:
    args = _build_parser().parse_args()

    # CPU-only (P06). Force CUDA invisible so the model never accidentally
    # lands on a GPU the trainer does not move tensors to.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    set_global_seed(seed)
    maybe_set_global_deterministic(deterministic)

    # Build the V2 model from the schema + config (honouring score_clamp so
    # the head clamp matches the loss target clamp exactly).
    import torch

    from douzero.models_v2.model import ModelV2
    from douzero.observation.schema import build_v2_schema

    # P06 r3: respect the project's seed=0 → no-op contract. set_global_seed(0)
    # is a no-op (Python/NumPy/Torch all unseeded); we must NOT separately
    # torch.manual_seed(0) or the model init would be seeded while the deal
    # shuffle is not — a mixed, contradictory reproducibility state. Only
    # seed Torch when the user explicitly requested a non-zero seed.
    if seed != 0:
        torch.manual_seed(seed)
    schema = build_v2_schema()
    model_cfg = _build_model_cfg(yaml_cfg)
    model = ModelV2(schema, model_cfg)

    from douzero.training import V2Trainer

    trainer_cfg = TrainerConfig(
        seed=seed,
        rng_seed=seed,
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
    )

    ruleset = _resolve_ruleset(yaml_cfg)
    loss_cfg = _build_loss_config(yaml_cfg)
    decision_cfg = _build_decision_config(yaml_cfg)

    # P07: load a frozen belief model when the value model is belief-enabled.
    # The checkpoint is validated (ruleset + feature version + architecture
    # hash) via load_belief_checkpoint; the trainer freezes it and computes the
    # constrained posterior features at both the collection and optimizer call
    # sites. Without this a belief_enabled value model fails closed at forward.
    belief_model = None
    if getattr(model_cfg, "belief_enabled", False):
        belief_ckpt = getattr(args, "belief_checkpoint", None)
        if not belief_ckpt:
            raise ValueError(
                "The value model has belief_enabled=true but no "
                "--belief_checkpoint was supplied. A belief-enabled value "
                "model can only be trained with a frozen pretrained "
                "BeliefModel. Run train_belief.py first, then pass its "
                "checkpoint via --belief_checkpoint."
            )
        from douzero.belief.checkpoint import load_belief_checkpoint

        belief_model = load_belief_checkpoint(
            belief_ckpt,
            expected_ruleset=ruleset or __import__(
                "douzero.env.rules", fromlist=["RuleSet"]
            ).RuleSet.legacy(),
            expected_feature_version="v2",
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
        from douzero.human_data.schema import read_jsonl
        from douzero.human_data.validate import validate_record
        from douzero.human_data.weights import WeightConfig, apply_sample_weights

        # Blocker 1: load + validate + sample with NO silent drops. Records
        # that fail replay validation are collected into validation_quarantine
        # and written to a bc_quarantine.jsonl alongside the checkpoint, with
        # the game_id + reason + error. They are NEVER swallowed by a filter
        # generator (the earlier `r for r in ... if validate_record(r).ok`
        # silently dropped them with no trace).
        bc_records = list(read_jsonl(bc_cfg.data_path))
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
                f"bc.data_path {bc_cfg.data_path!r} yielded no BC samples "
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
            f"{bc_cfg.data_path!r}, schedule={bc_cfg.schedule} "
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
        bc_aux_samples=bc_aux_samples,
        bc_schedule=bc_schedule,
        bc_temperature=(bc_cfg.temperature if bc_cfg is not None else 1.0),
        bc_label_smoothing=(bc_cfg.label_smoothing if bc_cfg is not None else 0.0),
    )

    print(
        f"[train_v2] model={type(model).__name__} "
        f"params={sum(p.numel() for p in model.parameters())} "
        f"score_clamp={model_cfg.score_clamp} "
        f"loss_cfg={loss_cfg.to_dict()} "
        f"decision={decision_cfg.to_dict()} "
        f"trainer=batch_size={trainer_cfg.batch_size} "
        f"lr={trainer_cfg.learning_rate} "
        f"epsilon={trainer_cfg.exp_epsilon} "
        f"ruleset={trainer.ruleset.ruleset_id if trainer.ruleset else 'legacy'}"
    )
    stats = trainer.train()
    print(
        f"[train_v2] episodes_completed={stats.episodes_completed} "
        f"transitions={stats.transitions_collected} "
        f"optimizer_steps={stats.optimizer_steps} "
        f"parameters_changed={getattr(trainer, 'stats_last_run_changed', 'unknown')} "
        f"last_loss={stats.last_loss} "
        f"grad_norm={stats.grad_norm_last_step:.4f} "
        f"p_win_mean={stats.p_win_mean:.4f}"
    )


if __name__ == "__main__":
    main()
