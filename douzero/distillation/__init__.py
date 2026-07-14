"""P10 privileged-teacher/public-student distillation (training only)."""

from .cache import TeacherCache, TeacherCacheIdentity
from .dataset import (
    DistillationDataset,
    DistillationSample,
    OfflineDistillationSample,
    load_offline_dataset,
    save_offline_dataset,
    teacher_observation_hash,
)
from .export import build_public_example_input, export_public_student
from .losses import (
    DistillationLossComponents,
    DistillationLossConfig,
    align_teacher_output,
    distillation_loss,
)
from .teacher_model import (
    TeacherModel,
    TeacherModelConfig,
    TeacherOutput,
    canonical_action_key,
    canonical_action_keys,
    state_dict_sha256,
)

__all__ = [
    "DistillationDataset",
    "DistillationLossComponents",
    "DistillationLossConfig",
    "DistillationSample",
    "OfflineDistillationSample",
    "TeacherCache",
    "TeacherCacheIdentity",
    "TeacherModel",
    "TeacherModelConfig",
    "TeacherOutput",
    "align_teacher_output",
    "build_public_example_input",
    "canonical_action_key",
    "canonical_action_keys",
    "distillation_loss",
    "export_public_student",
    "load_offline_dataset",
    "save_offline_dataset",
    "state_dict_sha256",
    "teacher_observation_hash",
]
