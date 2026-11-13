"""Frozen feature extractor for downstream probes.

Implements the seven probe feature modes P0-P6:

* P0 vanilla_hidden: vanilla RiNALMo-micro hidden state H_d.
* P1 distilled_hidden: Stage 4B release hidden state H_d (no adapters).
* P2 distilled_hidden_native: H_d + attention-weighted aggregation of the paired
  native student head outputs.
* P3 distilled_aligned: Z_d = sum_h alpha_h (A_d,h @ H_d) using the Stage 4B
  aligned adapter attention maps.
* P4 distilled_hidden_aligned: H_d + Z_d (primary mode).
* P4g distilled_hidden_aligned_gated: H_d + g * W(Z_d) with a zero-initialized
  scalar gate ``g`` — a stabilized variant of P4 that starts exactly in the
  P1 hidden-only regime and fades the aligned branch in only when it lowers
  the task loss.
* P5 distilled_hidden_teacher: H_d + Z_d computed from cached exact teacher
  attention maps.
* P6 distilled_hidden_random_heads: H_d + Z_d computed by running the trained
  adapters on a different (deterministic) native student head, negative control.

All non-gated/non-projection outputs are projected back to the backbone hidden
size so the same probe heads can be used across modes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone


def _native_head_maps(
    head_outputs: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Self-attention maps from paired native head outputs ``[B, K, L, D]``."""
    _B, K, L, D = head_outputs.shape
    scores = torch.einsum("bkld,bked->bkle", head_outputs, head_outputs) / math.sqrt(D)
    if key_padding_mask is not None:
        scores = scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs


class TeacherCacheIndex:
    """Lazy index over a downstream exact-teacher attention cache.

    The cache is expected to contain ``shard_*.pt`` files each with at least
    ``seq_ids`` and either ``teacher_heads_L{depth}`` or ``attention``
    ``[B, K, L, L]`` tensors.  Only the files needed for a batch are loaded into
    memory.

    A precomputed ``index.json`` (seq_id -> shard filename) is used when
    available to avoid loading every shard at startup.  If missing, the index
    is built once and persisted for subsequent runs.
    """

    INDEX_NAME = "index.json"

    def __init__(self, cache_dir: str | Path, route_depth: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.route_depth = route_depth
        self.route_key = f"L{route_depth}"
        self.seq_to_file: dict[str, Path] = {}
        self._load_index()

    def _load_index(self) -> None:
        index_path = self.cache_dir / self.INDEX_NAME
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.seq_to_file = {sid: self.cache_dir / fname for sid, fname in raw.items()}
            return
        self._build_and_save_index(index_path)

    def _build_and_save_index(self, index_path: Path) -> None:
        if not self.cache_dir.exists():
            return
        seq_to_file: dict[str, str] = {}
        for path in sorted(self.cache_dir.glob("shard_*.pt")):
            try:
                shard = torch.load(path, map_location="cpu", weights_only=False)
                seq_ids = list(shard.get("seq_ids", []))
                for sid in seq_ids:
                    seq_to_file[sid] = path.name
                del shard
            except Exception:
                continue
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(seq_to_file, fh)
        self.seq_to_file = {sid: self.cache_dir / fname for sid, fname in seq_to_file.items()}

    def __len__(self) -> int:
        return len(self.seq_to_file)

    def lookup(self, seq_ids: list[str]) -> torch.Tensor:
        """Return attention tensor ``[B, K, L, L]`` in the order of ``seq_ids``."""
        if not self.seq_to_file:
            raise FileNotFoundError(
                f"Teacher cache is empty or missing: {self.cache_dir}"
            )

        file_to_positions: dict[Path, list[tuple[int, str]]] = {}
        missing: list[str] = []
        for pos, sid in enumerate(seq_ids):
            path = self.seq_to_file.get(sid)
            if path is None:
                missing.append(sid)
                continue
            file_to_positions.setdefault(path, []).append((pos, sid))

        if missing:
            raise FileNotFoundError(
                f"Teacher cache missing {len(missing)} seq_ids (e.g. {missing[:3]}). "
                f"Run build_downstream_teacher_cache first."
            )

        out_tensors: list[torch.Tensor | None] = [None] * len(seq_ids)
        for path, positions in file_to_positions.items():
            shard = self._load_shard(path)
            attn_key = f"teacher_heads_{self.route_key}"
            if attn_key not in shard:
                attn_key = "attention"
            if attn_key not in shard:
                raise KeyError(f"No attention tensor found in {path}")
            attn = shard[attn_key]  # [B, K, L, L]
            local_id_to_row = {
                sid: i for i, sid in enumerate(shard.get("seq_ids", []))
            }
            for global_pos, sid in positions:
                local_row = local_id_to_row[sid]
                out_tensors[global_pos] = attn[local_row]

        tensors = [t for t in out_tensors if t is not None]
        sizes = {t.shape[-1] for t in tensors}
        if len(sizes) > 1:
            # Rows come from length-bucketed shards ([K, B_i, B_i]); pad each
            # map to the largest bucket in this batch with zeros so they can
            # be stacked. Callers crop/pad to the true padded length anyway,
            # and the padded region is zero attention.
            target = max(sizes)
            tensors = [
                torch.nn.functional.pad(t, (0, target - t.shape[-1], 0, target - t.shape[-2]))
                if t.shape[-1] != target
                else t
                for t in tensors
            ]
        return torch.stack(tensors, dim=0)

    def _load_shard(self, path: Path) -> dict[str, Any]:
        """Load a shard into a small LRU cache (max 8 shards).

        ``mmap=True`` keeps tensor payloads lazily backed by the file, so a
        batch lookup only reads the rows it actually indexes instead of
        deserializing multi-GB shards over NFS.
        """
        if not hasattr(self, "_shard_cache"):
            self._shard_cache: dict[str, dict[str, Any]] = {}
        key = str(path)
        if key in self._shard_cache:
            return self._shard_cache[key]
        shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self._shard_cache[key] = shard
        while len(self._shard_cache) > 8:
            self._shard_cache.pop(next(iter(self._shard_cache)))
        return shard


class ProbeFeatureExtractor(nn.Module):
    """Extract a frozen per-token feature vector for a downstream probe.

    Parameters
    ----------
    model:
        Either a :class:`RiNALMoStudentBackbone` (P0/P1) or a full
        :class:`RiNALMoHeadToHeadStudent` (P2-P6).  All of its parameters are
        kept frozen; only the internal task-specific gates and optional
        projection are trainable.
    feature_mode:
        One of the P0-P6 mode names.
    task:
        Downstream task name; used to select the route depth and to key the
        learned gates.
    route_depth:
        1-based route layer (6/8/12) relevant to this task.
    teacher_cache_dir:
        Required for ``distilled_hidden_teacher``.
    """

    def __init__(
        self,
        model: RiNALMoStudentBackbone | RiNALMoHeadToHeadStudent,
        feature_mode: str,
        task: str,
        route_depth: int,
        teacher_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if isinstance(model, RiNALMoHeadToHeadStudent):
            self.student = model
            self.backbone = model.backbone
        elif isinstance(model, RiNALMoStudentBackbone):
            self.student = None
            self.backbone = model
        else:
            raise TypeError(
                f"model must be RiNALMoStudentBackbone or RiNALMoHeadToHeadStudent, "
                f"got {type(model)}"
            )

        if route_depth not in ROUTE_DEPTHS:
            raise ValueError(f"route_depth must be one of {ROUTE_DEPTHS}")

        self.feature_mode = feature_mode
        self.task = task
        self.route_depth = route_depth
        self.route_key = f"L{route_depth}"
        self.hidden_size = int(self.backbone.hidden_size)
        self.head_dim = int(self.backbone.head_dim)
        self.num_heads = int(self.backbone.num_heads)
        self.use_gates = feature_mode in {
            "distilled_hidden_native",
            "distilled_aligned",
            "distilled_hidden_aligned",
            "distilled_hidden_aligned_gated",
            "distilled_hidden_teacher",
            "distilled_hidden_random_heads",
        }
        self.use_projection = feature_mode in {
            "distilled_hidden_native",
            "distilled_hidden_aligned",
            "distilled_hidden_teacher",
            "distilled_hidden_random_heads",
        }
        self.use_student = feature_mode in {
            "distilled_hidden",
            "distilled_hidden_native",
            "distilled_aligned",
            "distilled_hidden_aligned",
            "distilled_hidden_aligned_gated",
            "distilled_hidden_random_heads",
        }
        self.use_residual_gate = feature_mode == "distilled_hidden_aligned_gated"

        if self.use_student and self.student is None:
            raise ValueError(
                f"feature_mode={feature_mode} requires a RiNALMoHeadToHeadStudent"
            )

        self.teacher_cache: TeacherCacheIndex | None = None
        if feature_mode == "distilled_hidden_teacher":
            self.teacher_cache = TeacherCacheIndex(
                teacher_cache_dir or f"outputs/downstream_teacher_head_cache/{task}",
                route_depth,
            )

        # Lazily created on first forward so the head count K is known.
        self._head_count: int | None = None
        self.gates: nn.Parameter | None = None
        self.projection: nn.Linear | None = None
        # P4g gated-residual branch (zero-init scalar -> starts as P1).
        self.residual_gate: nn.Parameter | None = None
        self.attended_proj: nn.Linear | None = None

        # Freeze the entire pretrained model.
        for p in self.parameters():
            p.requires_grad_(False)

    def output_dim(self) -> int:
        """All feature modes are normalized to the backbone hidden size."""
        return self.hidden_size

    def _ensure_components(self, head_count: int) -> None:
        if self._head_count == head_count:
            return
        self._head_count = head_count
        # Place new trainable components on the same device as the backbone so
        # mixed CPU/GPU state cannot occur during lazy initialization.
        device = next(self.backbone.parameters()).device
        if self.use_gates:
            self.gates = nn.Parameter(torch.zeros(head_count, device=device))
        if self.use_projection:
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device)
        if self.use_residual_gate:
            self.attended_proj = nn.Linear(self.hidden_size, self.hidden_size).to(device)
            # Zero-init scalar gate: at initialization the output equals the
            # hidden-only (P1) feature, so training starts from the stable
            # regime and opens the aligned branch only if it reduces loss.
            self.residual_gate = nn.Parameter(torch.zeros(1, device=device))

    def reset_trainable_parameters(self) -> None:
        """Re-initialize gates and projection for a fresh optimizer search trial."""
        if self.gates is not None:
            nn.init.zeros_(self.gates)
        if self.projection is not None:
            nn.init.xavier_uniform_(self.projection.weight)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)
        if self.attended_proj is not None:
            nn.init.xavier_uniform_(self.attended_proj.weight)
            if self.attended_proj.bias is not None:
                nn.init.zeros_(self.attended_proj.bias)
        if self.residual_gate is not None:
            nn.init.zeros_(self.residual_gate)

    def _gated_attend(
        self,
        attn_maps: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``sum_h alpha_h (A_h @ H)``."""
        if self.gates is None:
            raise RuntimeError("gates not initialized")
        weights = F.softmax(self.gates, dim=0)  # [K]
        # [B, K, L, D]
        attended = torch.einsum("bkij,bjd->bkid", attn_maps, hidden)
        # weighted sum over heads
        return torch.einsum("k,bkid->bid", weights, attended)

    def _random_adapter_maps(
        self,
        head_outputs: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run each adapter on a different native head (deterministic control)."""
        if self.student is None:
            raise RuntimeError("student not available")
        K = head_outputs.shape[1]
        specs = self.student.adapter_specs[self.route_depth]
        preds: list[torch.Tensor] = []
        for _key, paired_head in specs:
            random_head = (paired_head + 1) % K
            pred = self.student.adapters[_key](
                head_outputs[:, random_head], key_padding_mask
            )["pred_map"]
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def _maybe_project(self, hidden: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        if self.use_projection:
            if self.projection is None:
                raise RuntimeError("projection not initialized")
            return self.projection(torch.cat([hidden, attended], dim=-1))
        return attended

    def _crop_or_pad_attention(
        self,
        attn: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Align a cached attention map ``[K, L, L]`` to the model sequence length."""
        K, L, _ = attn.shape
        if L == target_len:
            return attn
        out = torch.zeros(K, target_len, target_len, dtype=attn.dtype, device=attn.device)
        copy_len = min(L, target_len)
        out[:, :copy_len, :copy_len] = attn[:, :copy_len, :copy_len]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: list[str] | None = None,
    ) -> torch.Tensor:
        key_padding_mask = attention_mask == 0

        if self.feature_mode in ("vanilla_hidden", "distilled_hidden"):
            exits = self.backbone(input_ids, attention_mask)
            return exits[self.route_depth]["hidden"]

        if self.feature_mode == "distilled_hidden_native":
            exits = self.backbone(input_ids, attention_mask)
            hidden = exits[self.route_depth]["hidden"]
            head_outputs = exits[self.route_depth]["head_outputs"]
            K = head_outputs.shape[1]
            self._ensure_components(K)
            attn_maps = _native_head_maps(head_outputs, key_padding_mask)
            attended = self._gated_attend(attn_maps, hidden)
            return self._maybe_project(hidden, attended)

        if self.feature_mode in (
            "distilled_aligned",
            "distilled_hidden_aligned",
            "distilled_hidden_aligned_gated",
            "distilled_hidden_random_heads",
        ):
            if self.student is None:
                raise RuntimeError("student required")
            out = self.student(input_ids, attention_mask, return_backbone=True)
            hidden = out["backbone"][self.route_depth]["hidden"]
            head_outputs = out["backbone"][self.route_depth]["head_outputs"]

            if self.feature_mode == "distilled_hidden_random_heads":
                attn_maps = self._random_adapter_maps(head_outputs, key_padding_mask)
            else:
                attn_maps = out[f"pred_{self.route_key}"]

            K = attn_maps.shape[1]
            self._ensure_components(K)
            attended = self._gated_attend(attn_maps, hidden)

            if self.feature_mode == "distilled_aligned":
                return attended
            if self.feature_mode == "distilled_hidden_aligned_gated":
                if self.attended_proj is None or self.residual_gate is None:
                    raise RuntimeError("residual-gate components not initialized")
                return hidden + self.residual_gate * self.attended_proj(attended)
            return self._maybe_project(hidden, attended)

        if self.feature_mode == "distilled_hidden_teacher":
            if seq_ids is None:
                raise ValueError("seq_ids required for distilled_hidden_teacher")
            if self.teacher_cache is None:
                raise RuntimeError("teacher cache not initialized")
            exits = self.backbone(input_ids, attention_mask)
            hidden = exits[self.route_depth]["hidden"]
            teacher_attn = self.teacher_cache.lookup(seq_ids).to(
                device=hidden.device, dtype=hidden.dtype
            )
            B = hidden.shape[0]
            seq_len = hidden.shape[1]
            teacher_attn = torch.stack(
                [
                    self._crop_or_pad_attention(teacher_attn[i], seq_len)
                    for i in range(B)
                ],
                dim=0,
            )
            K = teacher_attn.shape[1]
            self._ensure_components(K)
            attended = self._gated_attend(teacher_attn, hidden)
            return self._maybe_project(hidden, attended)

        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def extra_repr(self) -> str:
        return (
            f"feature_mode={self.feature_mode}, task={self.task}, "
            f"route_depth={self.route_depth}, hidden_size={self.hidden_size}, "
            f"gates={self.use_gates}, projection={self.use_projection}"
        )
