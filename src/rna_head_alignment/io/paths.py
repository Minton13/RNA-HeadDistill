from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {p}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(base) if base is not None else project_root()) / p


def ensure_parent(path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_default_env_from_config(config: dict[str, Any]) -> None:
    for key, value in (config.get("hf") or {}).items():
        os.environ.setdefault(str(key), str(value))


def get_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def check_environment(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path) if config_path else {}
    set_default_env_from_config(cfg)
    out: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(project_root()),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
    }
    try:
        import torch

        out.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        )
    except Exception as e:
        out["torch_error"] = repr(e)
    try:
        import transformers

        out["transformers_version"] = transformers.__version__
    except Exception as e:
        out["transformers_error"] = repr(e)
    try:
        import multimolecule  # type: ignore

        out["multimolecule_available"] = True
        out["multimolecule_module"] = getattr(multimolecule, "__file__", "")
    except Exception as e:
        out["multimolecule_available"] = False
        out["multimolecule_error"] = repr(e)
    return out


@dataclass
class AssetMigrationResult:
    asset: str
    source: str
    target: str
    status: str
    method: str
    error: str | None = None


def _remove_empty_target(path: Path) -> None:
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _link_or_copy_dir(source: Path, target: Path) -> AssetMigrationResult:
    if not source.exists():
        return AssetMigrationResult(
            target.name, str(source), str(target), "missing_source", "none", None
        )
    if target.exists():
        return AssetMigrationResult(
            target.name,
            str(source),
            str(target),
            "exists",
            "existing",
            None,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    symlink_error = None
    try:
        os.symlink(source, target, target_is_directory=True)
        return AssetMigrationResult(target.name, str(source), str(target), "ok", "symlink")
    except Exception as e:
        symlink_error = repr(e)
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(target), str(source)],
                cwd=str(project_root()),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0 and target.exists():
                return AssetMigrationResult(
                    target.name, str(source), str(target), "ok", "junction"
                )
            symlink_error = f"{symlink_error}; junction failed: {completed.stderr.strip()}"
        except Exception as e:
            symlink_error = f"{symlink_error}; junction exception: {e!r}"
    _remove_empty_target(target)
    try:
        shutil.copytree(source, target)
        return AssetMigrationResult(
            target.name,
            str(source),
            str(target),
            "ok",
            "copy",
            symlink_error,
        )
    except Exception as e:
        return AssetMigrationResult(
            target.name,
            str(source),
            str(target),
            "failed",
            "copy",
            f"{symlink_error}; copy failed: {e!r}",
        )


def migrate_assets(old_root: str | Path, report_dir: str | Path = "outputs/model_checks") -> dict[str, Any]:
    old = resolve_path(old_root)
    new = project_root()
    results = []
    for asset in ("models", "data", "manifests"):
        results.append(_link_or_copy_dir(old / "assets" / asset, new / "assets" / asset))
    report = {
        "old_project_root": str(old),
        "new_project_root": str(new),
        "results": [asdict(r) for r in results],
        "missing_assets": [r.asset for r in results if r.status == "missing_source"],
        "skipped_assets": [r.asset for r in results if r.status == "exists"],
    }
    out_dir = resolve_path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "asset_migration_report.json"
    md_path = out_dir / "asset_migration_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Asset Migration Report",
        "",
        f"- old project root: `{old}`",
        f"- new project root: `{new}`",
        "",
        "| asset | status | method | source | target | error |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.asset} | {r.status} | {r.method} | `{r.source}` | `{r.target}` | {r.error or ''} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
