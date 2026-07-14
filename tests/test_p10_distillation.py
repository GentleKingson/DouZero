"""P10 privileged-teacher/public-student safety and training tests."""

from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from douzero.checkpoint import CheckpointCompatibilityError, load_v2_checkpoint
from douzero.config import load_config
from douzero.distillation import (
    DistillationDataset,
    DistillationLossConfig,
    DistillationSample,
    TeacherCache,
    TeacherCacheIdentity,
    TeacherModel,
    TeacherModelConfig,
    TeacherOutput,
    align_teacher_output,
    build_public_example_input,
    export_public_student,
    load_offline_dataset,
    save_offline_dataset,
    state_dict_sha256,
    teacher_observation_hash,
    distillation_loss,
)
from douzero.distillation.distill_student import StudentDistiller, StudentTrainConfig
from douzero.distillation.teacher_model import forward_public_model
from douzero.distillation.train_teacher import (
    TeacherTrainConfig,
    TeacherTrainer,
    load_teacher_checkpoint,
    save_teacher_checkpoint,
    teacher_supervised_loss,
)
from douzero.env.rules import RuleSet
from douzero.models_v2.config import ModelV2Config
from douzero.models_v2.model import ModelV2
from douzero.models_v2.output import ModelOutput
from douzero.observation.encode_v2 import get_obs_v2
from douzero.observation.privileged import PrivilegedObservation
from douzero.observation.schema import build_v2_schema


def _tiny_model(
    *,
    prior: bool = False,
    history_heads: int = 4,
    score_clamp: float = 32.0,
    score_target_transform: str = "raw",
) -> ModelV2:
    config = ModelV2Config(
        hidden_size=16,
        history_encoder="transformer",
        history_layers=1,
        history_heads=history_heads,
        role_embedding_dim=4,
        mlp_layers=1,
        human_prior_enabled=prior,
        score_clamp=score_clamp,
        score_target_transform=score_target_transform,
    )
    return ModelV2(build_v2_schema(max_history_len=12), config)


@pytest.fixture
def p10_sample(seeded_env):
    infoset = copy.deepcopy(seeded_env.infoset)
    # Keep the synthetic convergence tests bounded while preserving a subset
    # of actions that the rule engine actually declared legal.
    infoset.legal_actions = infoset.legal_actions[:4]
    obs = get_obs_v2(
        infoset,
        schema=build_v2_schema(max_history_len=12),
        ruleset=RuleSet.legacy(),
    )
    privileged = PrivilegedObservation(
        all_handcards=infoset.all_handcards,
        acting_role=infoset.player_position,
    )
    return DistillationSample(
        public_observation=obs,
        privileged_observation=privileged,
        action_index=0,
        target_win=1.0,
        target_score=2.0,
        sample_id="synthetic-0",
    ).tensorize()


def _swapped_hidden(privileged: PrivilegedObservation) -> PrivilegedObservation:
    hands = {role: list(cards) for role, cards in privileged.all_handcards.items()}
    left = hands["landlord_up"]
    right = hands["landlord_down"]
    pair = next(
        (i, j) for i, a in enumerate(left) for j, b in enumerate(right) if a != b
    )
    left[pair[0]], right[pair[1]] = right[pair[1]], left[pair[0]]
    return PrivilegedObservation(
        all_handcards=hands,
        acting_role=privileged.acting_role,
    )


def test_config_defaults_distillation_fully_off():
    cfg = load_config("configs/enhanced.yaml")
    assert cfg.distillation.enabled is False
    assert cfg.distillation.teacher_checkpoint == ""


def test_production_modules_do_not_import_privileged_and_student_signature_is_public():
    root = Path(__file__).parents[1]
    for relative in (
        "douzero/models_v2/model.py",
        "douzero/models_v2/batch.py",
        "douzero/evaluation/deep_agent.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        assert "douzero.observation.privileged" not in imports
    parameters = inspect.signature(ModelV2.forward).parameters
    assert "hidden_hands" not in parameters
    assert "all_handcards" not in parameters
    assert "privileged_observation" not in parameters


def test_teacher_requires_privileged_and_student_example_has_no_hidden_fields(p10_sample):
    teacher = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16))
    with pytest.raises(TypeError, match="PrivilegedObservation"):
        teacher(p10_sample.public_inputs, object(), action_keys=p10_sample.action_keys)

    first = teacher(
        p10_sample.public_inputs,
        p10_sample.privileged_observation,
        action_keys=p10_sample.action_keys,
    )
    changed = teacher(
        p10_sample.public_inputs,
        _swapped_hidden(p10_sample.privileged_observation),
        action_keys=p10_sample.action_keys,
    )
    assert first.action_keys == p10_sample.action_keys
    assert not torch.equal(first.action_logits, changed.action_logits)

    example = build_public_example_input(p10_sample.public_inputs)
    serialized_keys = " ".join(example).lower()
    assert "hidden" not in serialized_keys
    assert "all_handcards" not in serialized_keys
    assert "privileged" not in serialized_keys
    # Public inference remains usable without retaining a privileged variable.
    student = _tiny_model().eval()
    del changed
    with torch.inference_mode():
        output = forward_public_model(student, p10_sample.public_inputs)
    assert output.num_actions == len(p10_sample.action_keys)


def test_canonical_action_alignment_reorders_teacher_rows():
    keys = ((3,), (4,), (5, 5))
    values = torch.tensor([[10.0], [20.0], [30.0]])
    teacher = TeacherOutput(
        action_keys=(keys[2], keys[0], keys[1]),
        win_logit=values,
        p_win=torch.sigmoid(values),
        score_if_win=values + 0.25,
        score_if_loss=values - 0.25,
        expected_score=values + 1,
        action_logits=values + 2,
        action_mask=torch.ones(3, dtype=torch.bool),
    )
    aligned = align_teacher_output(teacher, keys)
    assert aligned.win_logit.squeeze(-1).tolist() == [20.0, 30.0, 10.0]
    with pytest.raises(ValueError, match="key mismatch"):
        align_teacher_output(teacher, ((3,), (4,), (6,)))


def test_offline_dataset_round_trip_and_teacher_hash_includes_hidden(tmp_path, p10_sample):
    dataset = DistillationDataset([p10_sample])
    path = tmp_path / "offline.pt"
    ruleset = RuleSet.legacy()
    save_offline_dataset(
        path,
        dataset,
        feature_schema_hash=p10_sample.public_inputs.feature_schema_hash,
        ruleset_hash=ruleset.stable_hash(),
        producer_model_sha="policy-sha",
    )
    loaded = load_offline_dataset(
        path,
        expected_feature_schema_hash=p10_sample.public_inputs.feature_schema_hash,
        expected_ruleset_hash=ruleset.stable_hash(),
    )
    assert len(loaded) == 1
    assert loaded[0].action_keys == p10_sample.action_keys
    altered = replace(
        p10_sample,
        privileged_observation=_swapped_hidden(p10_sample.privileged_observation),
    )
    assert teacher_observation_hash(altered) != teacher_observation_hash(p10_sample)
    with pytest.raises(ValueError, match="ruleset_hash"):
        load_offline_dataset(
            path,
            expected_feature_schema_hash=p10_sample.public_inputs.feature_schema_hash,
            expected_ruleset_hash="wrong",
        )


def test_teacher_cache_rejects_identity_drift(tmp_path, p10_sample):
    teacher = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16)).eval()
    with torch.inference_mode():
        output = teacher(
            p10_sample.public_inputs,
            p10_sample.privileged_observation,
            action_keys=p10_sample.action_keys,
        )
    path = tmp_path / "teacher-cache.json"
    identity = TeacherCacheIdentity(
        feature_schema_hash=p10_sample.public_inputs.feature_schema_hash,
        ruleset_hash=RuleSet.legacy().stable_hash(),
        teacher_model_sha=state_dict_sha256(teacher),
        teacher_config_hash=teacher.config_hash(),
    )
    cache = TeacherCache(path, identity)
    cache.put(p10_sample, output)
    cache.save()
    assert TeacherCache(path, identity).get(p10_sample) is not None
    with pytest.raises(ValueError, match="identity mismatch"):
        TeacherCache(
            path,
            TeacherCacheIdentity(
                feature_schema_hash=identity.feature_schema_hash,
                ruleset_hash=identity.ruleset_hash,
                teacher_model_sha="different-model",
                teacher_config_hash=identity.teacher_config_hash,
            ),
        )


@pytest.mark.parametrize(
    "changed_teacher",
    [
        lambda: TeacherModel(
            _tiny_model(), TeacherModelConfig(hidden_size=16, score_delta_clamp=8.0)
        ),
        lambda: TeacherModel(
            _tiny_model(history_heads=2), TeacherModelConfig(hidden_size=16)
        ),
    ],
)
def test_teacher_cache_binds_teacher_and_public_config(
    tmp_path, p10_sample, changed_teacher
):
    original = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16))
    changed = changed_teacher()
    changed.load_state_dict(original.state_dict(), strict=True)
    assert state_dict_sha256(changed) == state_dict_sha256(original)
    assert changed.config_hash() != original.config_hash()

    path = tmp_path / "config-bound-cache.json"
    common = {
        "feature_schema_hash": p10_sample.public_inputs.feature_schema_hash,
        "ruleset_hash": RuleSet.legacy().stable_hash(),
        "teacher_model_sha": state_dict_sha256(original),
    }
    TeacherCache(
        path,
        TeacherCacheIdentity(
            **common, teacher_config_hash=original.config_hash()
        ),
    ).save()
    with pytest.raises(ValueError, match="teacher_config_hash"):
        TeacherCache(
            path,
            TeacherCacheIdentity(
                **common, teacher_config_hash=changed.config_hash()
            ),
        )


def test_teacher_checkpoint_is_privileged_and_not_loadable_as_public(tmp_path):
    teacher = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16))
    path = tmp_path / "teacher.pt"
    manifest = save_teacher_checkpoint(path, teacher, ruleset=RuleSet.legacy())
    assert manifest.model_access == "privileged"
    assert manifest.checkpoint_kind == "privileged_teacher"

    fresh = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16))
    loaded = load_teacher_checkpoint(path, fresh, expected_ruleset=RuleSet.legacy())
    assert loaded.model_access == "privileged"
    with pytest.raises(CheckpointCompatibilityError):
        load_v2_checkpoint(
            str(path),
            expected_schema_hash=fresh.schema.stable_hash(),
            expected_model_config_hash=fresh.public_model.config.stable_hash(),
            expected_ruleset=RuleSet.legacy(),
        )
    with pytest.raises(TypeError, match="public ModelV2"):
        export_public_student(
            str(tmp_path / "bad.ckpt"), teacher, ruleset=RuleSet.legacy()
        )


def test_public_student_export_carries_public_access(tmp_path):
    student = _tiny_model()
    manifest = export_public_student(
        str(tmp_path / "student.ckpt"), student, ruleset=RuleSet.legacy()
    )
    assert manifest.model_access == "public"
    assert manifest.checkpoint_kind == "public_policy"


def test_teacher_overfits_tiny_synthetic_dataset(p10_sample):
    teacher = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16))
    trainer = TeacherTrainer(
        teacher,
        TeacherTrainConfig(learning_rate=0.01, max_grad_norm=20.0),
    )
    initial = trainer.train_epoch([p10_sample])
    last = initial
    for _ in range(14):
        last = trainer.train_epoch([p10_sample])
    assert last < initial


def test_teacher_terminal_score_does_not_update_non_applicable_head(p10_sample):
    teacher = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16))
    loss = teacher_supervised_loss(teacher, p10_sample, TeacherTrainConfig())
    loss.backward()
    non_applicable = teacher.public_model.heads.score_loss_head.weight.grad
    assert non_applicable is None or torch.count_nonzero(non_applicable) == 0
    privileged_output = teacher.privileged_action_head[-1].weight.grad
    assert privileged_output is not None
    assert torch.count_nonzero(privileged_output[2]) == 0


def test_student_distillation_loss_decreases_and_disabled_path_needs_no_teacher(p10_sample):
    teacher = TeacherModel(_tiny_model(), TeacherModelConfig(hidden_size=16)).eval()
    student = _tiny_model()
    config = DistillationLossConfig(
        enabled=True,
        temperature=2.0,
        top_k=3,
        lambda_supervised_score=0.1,
    )
    distiller = StudentDistiller(
        student,
        teacher=teacher,
        loss_config=config,
        train_config=StudentTrainConfig(learning_rate=0.01, max_grad_norm=20.0),
    )
    initial = distiller.train_epoch([p10_sample])
    last = initial
    for _ in range(14):
        last = distiller.train_epoch([p10_sample])
    assert last < initial

    public_only = StudentDistiller(
        _tiny_model(),
        loss_config=DistillationLossConfig(enabled=False),
    )
    loss = public_only.loss_for_sample(p10_sample)
    assert torch.isfinite(loss.total)
    with pytest.raises(ValueError, match="teacher was supplied"):
        StudentDistiller(
            _tiny_model(),
            teacher=teacher,
            loss_config=DistillationLossConfig(enabled=False),
        )


def test_shared_model_config_applies_score_training_semantics():
    cfg = load_config("configs/enhanced.yaml")
    cfg = replace(
        cfg,
        loss=replace(
            cfg.loss,
            score_clamp=0.5,
            score_target_transform="signed_log",
        ),
    )
    model_cfg = ModelV2Config.from_training_config(cfg)
    assert model_cfg.score_clamp == 0.5
    assert model_cfg.score_target_transform == "signed_log"


def test_supervised_score_uses_transform_clamp_and_only_applicable_head():
    win_logit = torch.zeros((1, 1), requires_grad=True)
    score_if_win = torch.zeros((1, 1), requires_grad=True)
    score_if_loss = torch.full((1, 1), 7.0, requires_grad=True)
    p_win = torch.sigmoid(win_logit)
    output = ModelOutput(
        win_logit=win_logit,
        score_if_win=score_if_win,
        score_if_loss=score_if_loss,
        p_win=p_win,
        score_mean=p_win.detach() * score_if_win + (1.0 - p_win.detach()) * score_if_loss,
        action_mask=torch.ones(1, dtype=torch.bool),
    )
    components = distillation_loss(
        output,
        ((3,),),
        action_index=0,
        target_win=1.0,
        target_score=100.0,
        teacher=None,
        config=DistillationLossConfig(
            enabled=False,
            lambda_kl=0.0,
            lambda_rank=0.0,
            lambda_teacher_win=0.0,
            lambda_teacher_score=0.0,
            lambda_supervised_win=0.0,
            lambda_supervised_score=1.0,
            score_delta=1.0,
            score_target_transform="signed_log",
            score_clamp=0.5,
        ),
    )
    assert components.supervised_score == pytest.approx(0.125)
    components.total.backward()
    assert score_if_win.grad is not None
    assert torch.count_nonzero(score_if_win.grad) == 1
    assert score_if_loss.grad is None or torch.count_nonzero(score_if_loss.grad) == 0


def test_student_epoch_uses_bounded_minibatches(p10_sample):
    distiller = StudentDistiller(
        _tiny_model(),
        loss_config=DistillationLossConfig(enabled=False),
        train_config=StudentTrainConfig(batch_size=2),
    )
    loss = distiller.train_epoch([p10_sample] * 5)
    assert torch.isfinite(torch.tensor(loss))
    assert distiller.optimizer_steps_last_epoch == 3
    assert distiller.max_batch_size_last_epoch == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("action_index", True, "action_index must be an int"),
        ("action_index", 0.0, "action_index must be an int"),
        ("target_win", 2.0, "target_win"),
        ("target_win", True, "target_win"),
        ("target_score", float("nan"), "target_score"),
        ("target_score", float("inf"), "target_score"),
        ("target_score", "2", "target_score"),
    ],
)
def test_offline_sample_rejects_invalid_labels(p10_sample, field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(p10_sample, **{field: value})


def test_offline_deserialization_does_not_coerce_invalid_labels(tmp_path, p10_sample):
    path = tmp_path / "invalid-label.pt"
    save_offline_dataset(
        path,
        DistillationDataset([p10_sample]),
        feature_schema_hash=p10_sample.public_inputs.feature_schema_hash,
        ruleset_hash=RuleSet.legacy().stable_hash(),
        producer_model_sha="policy-sha",
    )
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    bundle["samples"][0]["action_index"] = True
    torch.save(bundle, path)
    with pytest.raises(TypeError, match="action_index must be an int"):
        load_offline_dataset(
            path,
            expected_feature_schema_hash=p10_sample.public_inputs.feature_schema_hash,
            expected_ruleset_hash=RuleSet.legacy().stable_hash(),
        )
