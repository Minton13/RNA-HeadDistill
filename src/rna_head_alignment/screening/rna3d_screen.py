from __future__ import annotations

from rna_head_alignment.data.task_loaders import TaskSample

from .pairwise_screen import score_pairwise_heads


def score_rna3d_heads(model: str, model_type: str, task: str, samples: list[TaskSample], attentions: list, primary: str = "auprc") -> list[dict]:
    return score_pairwise_heads(model, model_type, task, samples, attentions, primary=primary)
