"""Stage 4B data-driven head-pair search.

Matches native RiNALMo-micro attention heads to the preserved teacher target
heads by comparing the attention-like map induced by each student head against
each cached teacher attention map.  Similarity is scored with MSE, cosine,
Pearson and Spearman; per route the final pairing is either a maximum-weight
bipartite assignment or a greedy top-1 selection.

The manual seed pairs from ``teacher_bank_selected.yaml`` are kept as a
fallback; the data-driven manifest is written to the configured output path.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from .head_manifest import PAIR_MANIFEST_COLUMNS, STUDENT_MODEL, load_manifest
from .sequence_dataset import align_tokens_to_sequence

DEFAULT_METRICS = ("mse", "cosine", "pearson", "spearman")

# Higher values use more memory but give more stable Spearman estimates.
DEFAULT_SPEARMAN_SAMPLE_SIZE = 2048


def _resolve_candidate_heads(
    candidate_student_heads: str | list[int] | None,
    num_heads: int,
) -> list[int]:
    """Normalize ``candidate_student_heads`` to a list of 0-based head indices."""
    if candidate_student_heads is None or candidate_student_heads == "all":
        return list(range(num_heads))
    if isinstance(candidate_student_heads, int):
        heads = [candidate_student_heads]
    else:
        heads = sorted({int(h) for h in candidate_student_heads})
    invalid = [h for h in heads if not (0 <= h < num_heads)]
    if invalid:
        raise ValueError(
            f"candidate student heads out of range 0..{num_heads - 1}: {invalid}"
        )
    return heads


def _load_slim_shard(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_student_head_outputs(
    student_backbone,
    sequences: list[str],
    lengths: list[int],
    route_depth: int,
) -> torch.Tensor:
    """Run the student backbone up to ``route_depth`` and return every native
    head output at that layer.

    Parameters
    ----------
    student_backbone:
        :class:`RiNALMoStudentBackbone` instance.
    sequences:
        Raw nucleotide sequences (no special tokens).
    lengths:
        Original nucleotide lengths; used only to choose the collation length.
    route_depth:
        1-based route layer (6, 8 or 12).

    Returns
    -------
    ``[B, H, L, head_dim]`` float tensor of all native head outputs at the
    route layer, where ``H`` is the total number of attention heads in the
    student model.
    """
    tokenizer = student_backbone.tokenizer()
    max_len = int(max(lengths)) if lengths else 0
    if max_len == 0:
        raise ValueError("empty sequence list or zero lengths")

    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    for seq in sequences:
        ids, _seq_len = align_tokens_to_sequence(tokenizer, seq, max_len)
        input_ids_list.append(ids)
        attention_mask_list.append(torch.ones_like(ids))

    pad_id = 0
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )

    device = next(student_backbone.model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    exits = student_backbone(
        input_ids, attention_mask, return_all_heads=True
    )
    route = int(route_depth)
    if route not in exits:
        raise KeyError(f"route {route} not in backbone exits {list(exits.keys())}")
    return exits[route]["all_head_outputs"]


def _student_cosine_map(head_outputs: torch.Tensor) -> torch.Tensor:
    """Convert per-head value-mixed outputs to a pairwise cosine map.

    Parameters
    ----------
    head_outputs:
        ``[B, H, L, d]`` native head outputs.

    Returns
    -------
        ``[B, H, L, L]`` cosine similarity map for each head.
    """
    # head_outputs: [B, H, L, d]
    norms = head_outputs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # [B, H, L, 1] * [B, H, 1, L] -> denominator [B, H, L, L]
    denom = norms * norms.transpose(-2, -1)
    return (head_outputs @ head_outputs.transpose(-2, -1)) / denom


def score_head_pairs(
    slim_shard: dict[str, Any],
    student_outputs: torch.Tensor,
    route_depth: int,
    metrics: list[str] | tuple[str, ...] = DEFAULT_METRICS,
    candidate_student_heads: str | list[int] | None = "all",
    spearman_sample_size: int = DEFAULT_SPEARMAN_SAMPLE_SIZE,
    rng: torch.Generator | None = None,
) -> pd.DataFrame:
    """Score every teacher-head / student-head pair for one route.

    The student head outputs ``[B, H, L, d]`` are converted to a pairwise
    cosine similarity map ``[B, H, L, L]`` and compared against the cached
    teacher attention map ``[B, T, L, L]``.  Metrics are computed over the
    valid (non-padded) token positions only.

    Parameters
    ----------
    slim_shard:
        Slim shard dict containing ``teacher_heads_L{route_depth}``,
        ``teacher_head_ids_L{route_depth}`` and ``lengths``.
    student_outputs:
        ``[B, H, L, head_dim]`` all-native-head outputs for the route.
    route_depth:
        1-based route layer (6, 8 or 12).
    metrics:
        Subset of ``{"mse", "cosine", "pearson", "spearman"}``.
    candidate_student_heads:
        ``"all"`` or a list of 0-based student head indices to score.  Other
        heads are omitted from the returned DataFrame.
    spearman_sample_size:
        Number of valid positions sampled per sample for Spearman (kept small
        because exact ranking over very large L^2 regions is expensive).
    rng:
        :class:`torch.Generator` for reproducible Spearman sampling.

    Returns
    -------
    DataFrame with one row per (teacher_head, student_head) and columns
    ``teacher_idx``, ``teacher_head_key``, ``student_head``, plus one column
    per requested metric.
    """
    metrics = tuple(m.lower() for m in metrics)
    route_key = f"L{int(route_depth)}"
    teacher_tensor_key = f"teacher_heads_{route_key}"
    teacher_ids_key = f"teacher_head_ids_{route_key}"

    if teacher_tensor_key not in slim_shard or teacher_ids_key not in slim_shard:
        raise KeyError(
            f"slim shard missing route {route_key} "
            f"(need {teacher_tensor_key}, {teacher_ids_key})"
        )

    A = slim_shard[teacher_tensor_key].to(torch.float32)  # [B, T, L_slim, L_slim]
    teacher_head_keys = list(slim_shard[teacher_ids_key])
    lengths = slim_shard["lengths"]  # [B]
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long)

    S = student_outputs.to(torch.float32)  # [B, H, L_stud, d]
    B, H_all, L_stud, _ = S.shape
    _, T, L_slim, _ = A.shape

    # Compare on the common spatial size.
    L = min(L_stud, L_slim)
    A = A[:, :, :L, :L].to(S.device)
    S = S[:, :, :L, :]
    lengths = lengths.clamp(max=L).to(S.device)

    # Valid 2-D mask [B, L, L] based on sequence lengths.
    arange = torch.arange(L, device=S.device)
    len_mask = arange.unsqueeze(0) < lengths.unsqueeze(-1)  # [B, L]
    valid_2d = len_mask.unsqueeze(2) & len_mask.unsqueeze(1)  # [B, L, L]

    M_cos = _student_cosine_map(S)  # [B, H_all, L, L]

    candidate_heads = _resolve_candidate_heads(candidate_student_heads, H_all)
    H = len(candidate_heads)
    head_idx_map = {h: i for i, h in enumerate(candidate_heads)}

    # Accumulators for exact metrics (all valid positions).
    sum_M = torch.zeros(H, device=S.device)
    sum_A = torch.zeros(T, device=S.device)
    sum_MA = torch.zeros(T, H, device=S.device)
    sum_M2 = torch.zeros(H, device=S.device)
    sum_A2 = torch.zeros(T, device=S.device)
    count = torch.zeros(T, H, device=S.device)

    # Spearman accumulators (sample-based, averaged per sample).
    do_spearman = "spearman" in metrics
    sum_spearman = torch.zeros(T, H, device=S.device)
    count_spearman = torch.zeros(T, H, device=S.device)

    if rng is None:
        rng = torch.Generator(device=S.device)

    for b in range(B):
        mask_b = valid_2d[b]  # [L, L]
        idx = mask_b.nonzero(as_tuple=True)
        N = idx[0].numel()
        if N == 0:
            continue

        # [H_all, N] values for every native head.
        M_vals_all = M_cos[b, :, idx[0], idx[1]]
        A_vals = A[b, :, idx[0], idx[1]]  # [T, N]

        # Restrict to candidate heads.
        M_vals = M_vals_all[candidate_heads, :]  # [H, N]

        # ---- Exact metrics over all valid positions. ----
        sum_M += M_vals.sum(dim=1)
        sum_A += A_vals.sum(dim=1)
        sum_MA += A_vals @ M_vals.T
        sum_M2 += (M_vals ** 2).sum(dim=1)
        sum_A2 += (A_vals ** 2).sum(dim=1)
        count += float(N)

        if not do_spearman or N < 2:
            continue

        # ---- Spearman on a per-sample random sample. ----
        K = min(int(spearman_sample_size), int(N))
        perm = torch.randperm(N, generator=rng, device=S.device)[:K]
        M_samp = M_vals[:, perm]
        A_samp = A_vals[:, perm]

        # Stable ranking via double argsort; float32 for mean/variance.
        ranks_M = torch.argsort(torch.argsort(M_samp, dim=-1), dim=-1).float()
        ranks_A = torch.argsort(torch.argsort(A_samp, dim=-1), dim=-1).float()

        mean_M = ranks_M.mean(dim=-1)
        mean_A = ranks_A.mean(dim=-1)
        cov = (ranks_A @ ranks_M.T) / K - mean_A.unsqueeze(1) * mean_M.unsqueeze(0)
        var_M = (ranks_M ** 2).mean(dim=-1) - mean_M ** 2
        var_A = (ranks_A ** 2).mean(dim=-1) - mean_A ** 2
        denom = torch.sqrt(var_A.unsqueeze(1) * var_M.unsqueeze(0)).clamp(min=1e-8)
        spearman_b = cov / denom
        spearman_b = torch.nan_to_num(spearman_b, nan=0.0)
        sum_spearman += spearman_b
        count_spearman += 1.0

    eps = 1e-12
    count_safe = count.clamp(min=eps)

    # Build metric tensors [T, H].
    metric_tensors: dict[str, torch.Tensor] = {}
    if "mse" in metrics:
        mse = (sum_M2.unsqueeze(0) + sum_A2.unsqueeze(1) - 2.0 * sum_MA) / count_safe
        metric_tensors["mse"] = mse.clamp(min=0.0)
    if "cosine" in metrics:
        cosine = sum_MA / (
            torch.sqrt(sum_M2.unsqueeze(0)) * torch.sqrt(sum_A2.unsqueeze(1))
        ).clamp(min=eps)
        metric_tensors["cosine"] = torch.nan_to_num(cosine, nan=0.0)
    if "pearson" in metrics:
        mean_M = sum_M.unsqueeze(0) / count_safe  # [1, H]
        mean_A = sum_A.unsqueeze(1) / count_safe  # [T, 1]
        cov = sum_MA / count_safe - mean_A * mean_M
        var_M = sum_M2.unsqueeze(0) / count_safe - mean_M ** 2
        var_A = sum_A2.unsqueeze(1) / count_safe - mean_A ** 2
        pearson = cov / torch.sqrt(var_A * var_M).clamp(min=eps)
        metric_tensors["pearson"] = torch.nan_to_num(pearson, nan=0.0)
    if do_spearman:
        spearman = sum_spearman / count_spearman.clamp(min=eps)
        metric_tensors["spearman"] = torch.nan_to_num(spearman, nan=0.0)

    rows: list[dict[str, Any]] = []
    for t, teacher_key in enumerate(teacher_head_keys):
        for h in candidate_heads:
            row: dict[str, Any] = {
                "teacher_idx": t,
                "teacher_head_key": teacher_key,
                "student_head": h,
            }
            for metric in metrics:
                row[metric] = float(metric_tensors[metric][t, head_idx_map[h]])
            rows.append(row)
    return pd.DataFrame(rows)


def _normalize_score_matrix(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Min-max normalize a score vector/matrix to [0, 1]."""
    vmin, vmax = float(scores.min()), float(scores.max())
    if math.isclose(vmax, vmin):
        return np.full_like(scores, 0.5)
    normed = (scores - vmin) / (vmax - vmin)
    if not higher_is_better:
        normed = 1.0 - normed
    return normed


def solve_pairing(
    scores_df: pd.DataFrame,
    method: str = "bipartite",
) -> pd.DataFrame:
    """Select one student head per teacher head for a single route.

    Each requested metric is min-max normalized to a higher-is-better score in
    ``[0, 1]`` (MSE is inverted) and the mean of the normalized scores is the
    combined objective.  ``method="bipartite"`` solves a maximum-weight
    bipartite assignment with :func:`scipy.optimize.linear_sum_assignment`;
    ``method="top1"`` greedily picks the highest-scoring student head for each
    teacher head (collisions allowed).

    Parameters
    ----------
    scores_df:
        Output of :func:`score_head_pairs`.
    method:
        ``"bipartite"`` or ``"top1"``.

    Returns
    -------
    DataFrame with one row per teacher head and columns ``teacher_idx``,
    ``teacher_head_key``, ``student_head``, ``matching_metric``,
    ``matching_score``, ``pairing_method``.
    """
    from scipy.optimize import linear_sum_assignment

    known_metrics = [m for m in DEFAULT_METRICS if m in scores_df.columns]
    if not known_metrics:
        raise ValueError("no recognizable metric columns in scores_df")

    # Add a normalized score column for every metric and compute the combined score.
    combined = np.zeros(len(scores_df))
    scores_df = scores_df.copy()
    for metric in known_metrics:
        vals = scores_df[metric].to_numpy(dtype=np.float64)
        higher_is_better = metric != "mse"
        normed = _normalize_score_matrix(vals, higher_is_better)
        scores_df[f"{metric}_norm"] = normed
        combined += normed
    combined /= len(known_metrics)
    scores_df["combined_score"] = combined

    teacher_idxs = sorted(scores_df["teacher_idx"].unique())
    student_heads = sorted(scores_df["student_head"].unique())
    T, H = len(teacher_idxs), len(student_heads)

    score_matrix = scores_df.pivot(
        index="teacher_idx", columns="student_head", values="combined_score"
    ).loc[teacher_idxs, student_heads].to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    if method == "bipartite":
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        for r, c in zip(row_ind, col_ind):
            sub = scores_df[
                (scores_df["teacher_idx"] == teacher_idxs[r])
                & (scores_df["student_head"] == student_heads[c])
            ]
            rows.append(
                {
                    "teacher_idx": teacher_idxs[r],
                    "teacher_head_key": sub["teacher_head_key"].iloc[0],
                    "student_head": student_heads[c],
                    "matching_metric": "combined",
                    "matching_score": float(score_matrix[r, c]),
                    "pairing_method": "bipartite",
                }
            )
    elif method == "top1":
        for r in range(T):
            c = int(np.argmax(score_matrix[r]))
            sub = scores_df[
                (scores_df["teacher_idx"] == teacher_idxs[r])
                & (scores_df["student_head"] == student_heads[c])
            ]
            rows.append(
                {
                    "teacher_idx": teacher_idxs[r],
                    "teacher_head_key": sub["teacher_head_key"].iloc[0],
                    "student_head": student_heads[c],
                    "matching_metric": "combined",
                    "matching_score": float(score_matrix[r, c]),
                    "pairing_method": "top1",
                }
            )
    else:
        raise ValueError(f"unknown pairing method {method!r}")

    return pd.DataFrame(rows)


def _select_shard_samples(
    shard: dict[str, Any],
    max_samples: int,
) -> list[int]:
    """Choose up to ``max_samples`` indices from a slim shard.

    Stratified sampling across the length-sorted shard avoids a short-sequence
    bias when ``max_samples`` is smaller than the shard size.
    """
    n = len(shard["seq_ids"])
    if n <= max_samples:
        return list(range(n))
    # Evenly spaced indices (deterministic).
    return np.unique(
        np.linspace(0, n - 1, max_samples, dtype=int)
    ).tolist()


def build_head_pair_manifest(
    target_manifest: str | Path | pd.DataFrame,
    seed_pair_manifest: str | Path | pd.DataFrame,
    cache_root: str | Path,
    shards: list[int],
    student_model_dir: str | Path,
    route_depths: tuple[int, ...] | list[int] = (6, 8, 12),
    metrics: list[str] | tuple[str, ...] = DEFAULT_METRICS,
    pairing: str = "bipartite",
    max_samples_per_shard: int = 256,
    candidate_student_heads: str | list[int] | None = "all",
    batch_size: int = 8,
    max_len: int = 512,
    precision: str = "bf16",
    device: str = "auto",
    seed: int = 42,
    spearman_sample_size: int = DEFAULT_SPEARMAN_SAMPLE_SIZE,
    fallback_to_seed: bool = True,
    progress: bool = True,
    student_config_overrides: dict | None = None,
    student_init_checkpoint: str | Path | None = None,
) -> pd.DataFrame:
    """Run the data-driven head-pair search and build a manifest.

    Parameters
    ----------
    target_manifest:
        Path or DataFrame of the target teacher-head manifest.
    seed_pair_manifest:
        Path or DataFrame of the manual seed pair manifest (used as fallback).
    cache_root:
        Directory containing ``shard_{idx:05d}.pt`` slim shards.
    shards:
        Shard indices to use for the search (4-16 recommended).
    student_model_dir:
        Local rinalmo-micro checkpoint directory.
    route_depths:
        1-based route layers to solve (default 6, 8, 12).
    metrics:
        Metrics to compute and combine.
    pairing:
        ``"bipartite"`` or ``"top1"``.
    max_samples_per_shard:
        Maximum sequences drawn from each shard.
    candidate_student_heads:
        ``"all"`` or a list of 0-based student head indices.
    batch_size:
        Forward-batch size for the student backbone.
    max_len:
        Maximum nucleotide length per sequence.
    precision:
        ``"bf16"``, ``"fp16"`` or ``"fp32"``.
    device:
        ``"auto"``, ``"cuda"`` or ``"cpu"``.
    seed:
        RNG seed for deterministic Spearman sampling and shard stratification.
    spearman_sample_size:
        Per-sample Spearman sample size.
    fallback_to_seed:
        If ``True`` and a route produces no data-driven pairs, use the seed
        pairs for that route instead.
    progress:
        Print per-shard progress to stdout.

    Returns
    -------
    DataFrame with :data:`PAIR_MANIFEST_COLUMNS`.
    """
    from .student_backbone import RiNALMoStudentBackbone
    from rna_head_alignment.io.paths import get_device

    target = (
        load_manifest(target_manifest)
        if isinstance(target_manifest, (str, Path))
        else target_manifest.copy()
    )
    seed_manifest = (
        load_manifest(seed_pair_manifest)
        if isinstance(seed_pair_manifest, (str, Path))
        else seed_pair_manifest.copy()
    )
    cache_root = Path(cache_root)
    resolved_device = get_device(device)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision.lower(), torch.float32)

    # Build backbone.  We do not need the manual paired-head subset here, so use
    # the default routes and heads.  For the from-scratch small student, pass
    # ``student_config_overrides`` (random init) plus ``student_init_checkpoint``
    # (S0 MLM weights) to evaluate *trained* native heads.
    student = RiNALMoStudentBackbone(
        model_dir=student_model_dir,
        routes=tuple(route_depths),
        config_overrides=student_config_overrides,
    )
    if student_init_checkpoint is not None:
        ckpt = torch.load(student_init_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {
            k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
        }
        if not backbone_state:
            raise RuntimeError(
                f"no backbone.* keys in student init checkpoint {student_init_checkpoint}"
            )
        missing, _unexpected = student.load_state_dict(backbone_state, strict=False)
        if len(missing) > 0.05 * sum(1 for _ in student.parameters()):
            raise RuntimeError(
                f"student init checkpoint mismatch: {len(missing)} missing keys"
            )
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    student = student.to(resolved_device).to(dtype)

    candidate_heads = _resolve_candidate_heads(
        candidate_student_heads, student.num_heads
    )

    rng = torch.Generator(device=resolved_device).manual_seed(seed)

    # Per-route score accumulators (weighted by batch size).
    score_acc: dict[int, dict[str, np.ndarray]] = {}
    weight_acc: dict[int, np.ndarray] = {}
    teacher_keys_by_route: dict[int, list[str]] = {}

    for shard_idx in shards:
        shard_path = cache_root / f"shard_{shard_idx:05d}.pt"
        if not shard_path.exists():
            if progress:
                print(f"[pair_search] shard {shard_idx:05d}: missing, skipping")
            continue

        shard = _load_slim_shard(shard_path)
        sample_indices = _select_shard_samples(shard, max_samples_per_shard)
        if progress:
            print(
                f"[pair_search] shard {shard_idx:05d}: {len(sample_indices)} samples",
                flush=True,
            )

        sequences = [shard["sequences"][i] for i in sample_indices]
        shard_lengths = shard["lengths"][sample_indices]
        if not torch.is_tensor(shard_lengths):
            shard_lengths = torch.tensor(shard_lengths, dtype=torch.long)

        # Process in batches.  Tokenize once per batch, run the student once for
        # all routes, then score each route from the captured hook outputs.
        tokenizer = student.tokenizer()
        for start in range(0, len(sample_indices), batch_size):
            end = min(start + batch_size, len(sample_indices))
            batch_seqs = sequences[start:end]
            batch_lengths = shard_lengths[start:end].tolist()
            batch_input_lengths = [min(int(l), max_len) for l in batch_lengths]

            ids_list: list[torch.Tensor] = []
            mask_list: list[torch.Tensor] = []
            for seq in batch_seqs:
                ids, _ = align_tokens_to_sequence(tokenizer, seq, max_len)
                ids_list.append(ids)
                mask_list.append(torch.ones_like(ids))
            pad_id = 0
            input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
            attention_mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
            input_ids = input_ids.to(resolved_device)
            attention_mask = attention_mask.to(resolved_device)

            with torch.no_grad():
                exits = student(
                    input_ids, attention_mask, return_all_heads=True
                )

            for route_depth in route_depths:
                route_key = f"L{route_depth}"
                teacher_tensor_key = f"teacher_heads_{route_key}"
                teacher_ids_key = f"teacher_head_ids_{route_key}"
                if teacher_tensor_key not in shard or teacher_ids_key not in shard:
                    continue

                teacher_keys_by_route[route_depth] = list(shard[teacher_ids_key])
                T = len(teacher_keys_by_route[route_depth])
                H = len(candidate_heads)

                S_route = exits[route_depth]["all_head_outputs"].to(torch.float32)
                A_route = shard[teacher_tensor_key][
                    sample_indices[start:end]
                ].to(torch.float32)

                mini_slim = {
                    teacher_tensor_key: A_route,
                    teacher_ids_key: shard[teacher_ids_key],
                    "lengths": torch.tensor(batch_input_lengths, dtype=torch.long),
                }

                scores_df = score_head_pairs(
                    mini_slim,
                    S_route,
                    route_depth,
                    metrics=metrics,
                    candidate_student_heads=candidate_heads,
                    spearman_sample_size=spearman_sample_size,
                    rng=rng,
                )

                if route_depth not in score_acc:
                    score_acc[route_depth] = {
                        m: np.zeros((T, H), dtype=np.float64) for m in metrics
                    }
                    weight_acc[route_depth] = np.zeros((T, H), dtype=np.float64)

                weight = float(end - start)
                for metric in metrics:
                    mat = scores_df.pivot(
                        index="teacher_idx", columns="student_head", values=metric
                    ).loc[range(T), candidate_heads].to_numpy(dtype=np.float64)
                    score_acc[route_depth][metric] += weight * mat
                weight_acc[route_depth] += weight

            # Free the captured hooks and batch tensors.
            del exits, S_route, A_route, input_ids, attention_mask
            if resolved_device.startswith("cuda"):
                torch.cuda.empty_cache()

        del shard
        gc.collect()

    # Finalize scores and solve per route.
    route_pair_rows: list[dict[str, Any]] = []
    for route_depth in route_depths:
        if route_depth not in score_acc:
            if fallback_to_seed:
                route_pair_rows.extend(
                    seed_manifest[seed_manifest["route_depth"] == route_depth].to_dict("records")
                )
            continue

        T, H = weight_acc[route_depth].shape
        final_scores: dict[str, np.ndarray] = {}
        for metric in metrics:
            final_scores[metric] = score_acc[route_depth][metric] / np.maximum(
                weight_acc[route_depth], 1e-12
            )

        rows = []
        for t in range(T):
            for h_idx, h in enumerate(candidate_heads):
                row = {
                    "teacher_idx": t,
                    "teacher_head_key": teacher_keys_by_route[route_depth][t],
                    "student_head": h,
                }
                for metric in metrics:
                    row[metric] = float(final_scores[metric][t, h_idx])
                rows.append(row)
        scores_df = pd.DataFrame(rows)

        solved = solve_pairing(scores_df, method=pairing)

        # Merge with target manifest.
        target_route = target[target["route_depth"] == route_depth].copy()
        solved = solved.copy()
        solved["head_key"] = solved["teacher_head_key"]
        merged = target_route.merge(
            solved[["head_key", "student_head", "matching_metric", "matching_score", "pairing_method"]],
            on="head_key",
            how="left",
        )

        if merged["student_head"].isna().any() and fallback_to_seed:
            # Fill missing from seed pairs for this route.
            seed_route = seed_manifest[seed_manifest["route_depth"] == route_depth].copy()
            seed_lookup = {
                row["head_key"]: row
                for _, row in seed_route.iterrows()
                if pd.notna(row.get("head_key"))
            }
            for col in ["student_head", "matching_metric", "matching_score", "pairing_method"]:
                merged[col] = merged.apply(
                    lambda row: (
                        seed_lookup.get(row["head_key"], {}).get(col, row[col])
                        if pd.isna(row[col]) and row["head_key"] in seed_lookup
                        else row[col]
                    ),
                    axis=1,
                )

        for _, row in merged.iterrows():
            if pd.isna(row.get("student_head")):
                continue
            route_pair_rows.append(
                {
                    "task": row["task"],
                    "route_depth": route_depth,
                    "teacher_model": row["teacher_model"],
                    "teacher_layer": row["teacher_layer"],
                    "teacher_head": row["teacher_head"],
                    "teacher_cache_channel": row["cache_channel"],
                    "student_model": STUDENT_MODEL,
                    "student_layer": route_depth,
                    "student_head": int(row["student_head"]),
                    "matching_metric": row.get("matching_metric", "combined"),
                    "matching_score": (
                        float(row["matching_score"])
                        if pd.notna(row.get("matching_score"))
                        else float("nan")
                    ),
                    "pairing_method": row.get("pairing_method", pairing),
                    "keep": True,
                    "adapter_key": f"L{route_depth}_H{int(row['student_head'])}",
                }
            )

    df = pd.DataFrame(route_pair_rows, columns=PAIR_MANIFEST_COLUMNS)
    # If the whole data-driven run produced nothing, fall back to the full seed manifest.
    if df.empty and fallback_to_seed:
        return seed_manifest[PAIR_MANIFEST_COLUMNS].copy()
    return df
