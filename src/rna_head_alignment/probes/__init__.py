"""Frozen downstream probe framework for Stage 4B/5 head-to-head distillation."""

from __future__ import annotations

from rna_head_alignment.probes.task_config import (
    FEATURE_MODES,
    GATED_FEATURE_MODES,
    STUDENT_FEATURE_MODES,
    TASK_TO_ROUTE_DEPTH,
    TASK_TO_TYPE,
    default_teacher_cache_dir,
    downstream_data_dir,
    validate_feature_mode,
    validate_task,
)

__all__ = [
    "FEATURE_MODES",
    "GATED_FEATURE_MODES",
    "STUDENT_FEATURE_MODES",
    "TASK_TO_ROUTE_DEPTH",
    "TASK_TO_TYPE",
    "default_teacher_cache_dir",
    "downstream_data_dir",
    "validate_feature_mode",
    "validate_task",
]
