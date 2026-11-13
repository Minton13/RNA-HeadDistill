from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rna_head_alignment.io.paths import load_yaml, resolve_path


@dataclass
class ModelSpec:
    name: str
    model_type: str
    local_dir: Path
    attention_teacher: bool = False
    hidden_control: bool = False
    student: bool = False
    optional: bool = False
    large: bool = False
    trust_remote_code: bool = True
    note: str = ""

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=name,
            model_type=str(cfg.get("type", "RNA")).upper(),
            local_dir=resolve_path(cfg["local_dir"]),
            attention_teacher=bool(cfg.get("attention_teacher", False)),
            hidden_control=bool(cfg.get("hidden_control", False)),
            student=bool(cfg.get("student", False)),
            optional=bool(cfg.get("optional", False)),
            large=bool(cfg.get("large", False)),
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            note=str(cfg.get("note", "")),
        )

    def normalize_sequence(self, sequence: str) -> str:
        seq = "".join(ch for ch in sequence.upper() if ch in "ACGTUN")
        if self.model_type == "RNA":
            return seq.replace("T", "U")
        if self.model_type == "DNA":
            return seq.replace("U", "T")
        return seq


def load_model_specs(path: str | Path) -> dict[str, ModelSpec]:
    cfg = load_yaml(path)
    models = cfg.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError(f"Expected models mapping in {path}")
    return {name: ModelSpec.from_config(name, item or {}) for name, item in models.items()}


def get_model_spec(path: str | Path, name: str) -> ModelSpec:
    specs = load_model_specs(path)
    if name not in specs:
        raise KeyError(f"Model {name!r} not found in {path}")
    return specs[name]
