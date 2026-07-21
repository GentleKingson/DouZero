"""Isolated GPU-native model experiments; never Legacy checkpoint compatible."""

from .config import GPUV3Config
from .checkpoint import load_gpu_v3_checkpoint, save_gpu_v3_checkpoint
from .identity import GPU_V3_CHECKPOINT_KIND, GPU_V3_FEATURE_VERSION, GPU_V3_MODEL_VERSION
from .models import IndependentRoleDualTower, SharedTrunkRoleHeads
from .distillation import build_legacy_student_inputs, legacy_teacher_distillation_loss

__all__ = [
    "GPUV3Config",
    "GPU_V3_CHECKPOINT_KIND",
    "GPU_V3_FEATURE_VERSION",
    "GPU_V3_MODEL_VERSION",
    "IndependentRoleDualTower",
    "SharedTrunkRoleHeads",
    "build_legacy_student_inputs",
    "legacy_teacher_distillation_loss",
    "load_gpu_v3_checkpoint",
    "save_gpu_v3_checkpoint",
]
