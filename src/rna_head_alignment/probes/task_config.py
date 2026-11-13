"""Task and feature-mode configuration for frozen downstream probes."""

from __future__ import annotations

# Route depth used for each downstream task in the Stage 4B head-to-head model.
# L6 = sequence-level, L8 = chemical reactivity, L12 = structure-level.
TASK_TO_ROUTE_DEPTH: dict[str, int] = {
    "splice_donor_seqcls": 6,
    "splice_acceptor_seqcls": 6,
    "rbp_binding_seqcls": 6,
    "ncRNA_family_seqcls": 6,
    "chemical_reactivity_token": 8,
    "secondary_structure_pair": 12,
}

TASK_TO_TYPE: dict[str, str] = {
    "splice_donor_seqcls": "sequence_classification",
    "splice_acceptor_seqcls": "sequence_classification",
    "rbp_binding_seqcls": "sequence_classification",
    "ncRNA_family_seqcls": "sequence_classification",
    "chemical_reactivity_token": "token_regression",
    "secondary_structure_pair": "pairwise",
}

FEATURE_MODES: tuple[str, ...] = (
    "vanilla_hidden",
    "distilled_hidden",
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_aligned_gated",
    "distilled_hidden_teacher",
    "distilled_hidden_random_heads",
)

# Modes that use the learned softmax gates over per-head attention maps.
GATED_FEATURE_MODES: frozenset[str] = frozenset({
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_aligned_gated",
    "distilled_hidden_teacher",
    "distilled_hidden_random_heads",
})

# Modes that require the full Stage 4B head-to-head student (adapters).
STUDENT_FEATURE_MODES: frozenset[str] = frozenset({
    "distilled_hidden",
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_aligned_gated",
    "distilled_hidden_random_heads",
})

# Mode that needs the pre-computed exact-teacher cache for downstream sequences.
TEACHER_CACHE_MODE: str = "distilled_hidden_teacher"

# Default downstream task directory layout.
def downstream_data_dir(task: str) -> str:
    return f"data/downstream_stage5/{task}"


def default_teacher_cache_dir(task: str) -> str:
    return f"outputs/downstream_teacher_head_cache/{task}"


def validate_task(task: str) -> None:
    if task not in TASK_TO_ROUTE_DEPTH:
        raise ValueError(
            f"Unknown downstream task {task!r}. "
            f"Supported: {list(TASK_TO_ROUTE_DEPTH)}"
        )


def validate_feature_mode(mode: str) -> None:
    if mode not in FEATURE_MODES:
        raise ValueError(
            f"Unknown feature mode {mode!r}. "
            f"Supported: {list(FEATURE_MODES)}"
        )
