from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from rna_head_alignment.io.manifests import write_dataframe, write_json, write_markdown_table
from rna_head_alignment.io.paths import get_device, resolve_path

from .attention_extractors import extract_attention_bundle
from .registry import ModelSpec, load_model_specs


def _load_tokenizer(spec: ModelSpec) -> Any:
    errors: list[str] = []
    try:
        return AutoTokenizer.from_pretrained(
            spec.local_dir,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    except Exception as e:
        errors.append(f"transformers.AutoTokenizer: {e!r}")
    try:
        import multimolecule  # type: ignore

        return multimolecule.AutoTokenizer.from_pretrained(
            spec.local_dir,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
        )
    except Exception as e:
        errors.append(f"multimolecule.AutoTokenizer: {e!r}")
    raise RuntimeError("; ".join(errors))


def _load_raw_state_dict(local_dir: Path) -> dict[str, torch.Tensor] | None:
    """Load the raw checkpoint from a local model directory.

    Supports ``model.safetensors`` and ``pytorch_model.bin``, including simple
    sharded variants (``model-00001-of-NNNNN.safetensors``).
    """
    safetensor_files = sorted(local_dir.glob("*.safetensors"))
    bin_files = sorted(local_dir.glob("pytorch_model*.bin"))

    state: dict[str, torch.Tensor] = {}
    if safetensor_files:
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("safetensors package is required to load .safetensors files") from exc
        for path in safetensor_files:
            state.update(load_file(path, device="cpu"))
        return state

    if bin_files:
        for path in bin_files:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                state.update(checkpoint)
        if state:
            return state

    return None


def _strip_wrapper_prefix(state_dict: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Remap a state dict whose keys are wrapped in a top-level module prefix.

    Many staged checkpoints (e.g. RNA-FM, RiNALMo, UTR-LM, SpliceBERT) save the
    base encoder under a ``model.*`` prefix, but the base model class expects the
    encoder keys at the top level.  ESM-based checkpoints (PlantRNA-FM, NT-v2)
    use an ``esm.*`` prefix.  We detect the dominant wrapper prefix and strip it
    when none of the model's own parameters use that prefix.
    """
    param_names = {name for name, _ in model.named_parameters()}

    prefix_counts: dict[str, int] = {}
    for key in state_dict:
        first = key.split(".")[0]
        prefix_counts[first] = prefix_counts.get(first, 0) + 1

    if not prefix_counts:
        return state_dict

    total = len(state_dict)
    dominant = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[0][0]
    # Only treat it as a wrapper prefix if it dominates and is not already used
    # by the instantiated model class.
    if prefix_counts[dominant] / total <= 0.5:
        return state_dict
    if any(name.startswith(dominant + ".") for name in param_names):
        return state_dict

    new_state: dict[str, torch.Tensor] = {}
    prefix_with_dot = dominant + "."
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            new_state[key[len(prefix_with_dot):]] = value
        else:
            new_state[key] = value
    return new_state


def _load_model_from_local(spec: ModelSpec, device: str = "cpu") -> torch.nn.Module | None:
    """Load a base encoder from a local checkpoint, remapping common wrapper prefixes."""
    import multimolecule  # noqa: F401 - registers custom model types

    config = AutoConfig.from_pretrained(
        spec.local_dir,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    try:
        model = AutoModel.from_config(config, trust_remote_code=spec.trust_remote_code)
    except Exception:
        return None

    state_dict = _load_raw_state_dict(spec.local_dir)
    if state_dict is None:
        return None

    state_dict = _strip_wrapper_prefix(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # If the base encoder is still mostly uninitialized, fall back to the
    # standard ``from_pretrained`` path rather than silently returning a random
    # model.
    base_param_count = sum(1 for _ in model.parameters())
    if missing and len(missing) / max(base_param_count, 1) > 0.05:
        return None

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _load_model(spec: ModelSpec, device: str = "cpu") -> torch.nn.Module:
    # First try the local, prefix-remapping path.  This avoids the common
    # situation where ``AutoModel.from_pretrained`` instantiates a class whose
    # base encoder attribute name differs from the checkpoint prefix, leading to
    # a randomly-initialized model with only a warning.
    try:
        model = _load_model_from_local(spec, device)
        if model is not None:
            return model
    except Exception:
        pass

    errors: list[str] = []
    kwargs = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": True,
    }
    model_classes = (AutoModelForMaskedLM, AutoModel) if spec.name.startswith("nucleotide_transformer") else (
        AutoModel,
        AutoModelForMaskedLM,
    )
    for cls in model_classes:
        try:
            model = cls.from_pretrained(spec.local_dir, **kwargs, attn_implementation="eager")
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return model.to(device)
        except Exception as e:
            errors.append(f"{cls.__name__}+eager: {e!r}")
        try:
            model = cls.from_pretrained(spec.local_dir, **kwargs)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return model.to(device)
        except Exception as e:
            errors.append(f"{cls.__name__}: {e!r}")
    try:
        import multimolecule  # type: ignore

        for cls in (multimolecule.AutoModel, getattr(multimolecule, "AutoModelForMaskedLM", None)):
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(spec.local_dir, **kwargs)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                return model.to(device)
            except Exception as e:
                errors.append(f"multimolecule.{cls.__name__}: {e!r}")
    except Exception as e:
        errors.append(f"multimolecule import: {e!r}")
    raise RuntimeError("; ".join(errors))


def load_frozen_model(spec: ModelSpec, device: str = "auto") -> tuple[Any, torch.nn.Module]:
    resolved_device = get_device(device)
    tokenizer = _load_tokenizer(spec)
    model = _load_model(spec, resolved_device)
    # RiNALMo-mega is small enough that fp16 weights keep it well under the 40 GB
    # A100 budget even for 512-nt sequences, where fp32 activations OOM after a
    # few hundred forward passes (cublas/cudnn workspace accumulation).
    if spec.name == "rinalmo_mega":
        model = model.half()
    return tokenizer, model


def verify_one_model(
    spec: ModelSpec,
    device: str = "auto",
    max_len: int = 128,
    include_large: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": spec.name,
        "type": spec.model_type,
        "local_dir": str(spec.local_dir),
        "exists": spec.local_dir.exists(),
        "attention_teacher_expected": spec.attention_teacher,
        "optional": spec.optional,
        "large": spec.large,
        "load_ok": False,
        "forward_ok": False,
        "has_attentions": False,
        "supports_hidden_states": False,
        "n_layers": None,
        "n_heads": None,
        "hidden_size": None,
        "attn_shape": "",
        "error": "",
    }
    if spec.large and not include_large:
        row["error"] = "skipped_large_model; pass --include-large to verify"
        return row
    if not spec.local_dir.exists():
        row["error"] = "local_dir_missing"
        return row
    resolved_device = get_device(device)
    try:
        tokenizer, model = load_frozen_model(spec, resolved_device)
        row["load_ok"] = True
        row["tokenizer_class"] = tokenizer.__class__.__name__
        row["model_class"] = model.__class__.__name__
        sequence = "AUGCUUAGCUAGCUA" if spec.model_type == "RNA" else "ATGCTTAGCTAGCTA"
        bundle = extract_attention_bundle(
            spec, tokenizer, model, sequence, "verify", max_len=max_len, device=resolved_device
        )
        row["forward_ok"] = True
        if bundle.hidden_states is not None:
            row["supports_hidden_states"] = True
            row["hidden_size"] = int(bundle.hidden_states.shape[-1])
            row["hidden_layers_returned"] = int(bundle.hidden_states.shape[0])
        if bundle.attentions is not None:
            row["has_attentions"] = True
            row["n_layers"] = int(bundle.attentions.shape[0])
            row["n_heads"] = int(bundle.attentions.shape[1])
            row["attn_shape"] = "x".join(str(x) for x in bundle.attentions.shape)
        row["mapping_method"] = bundle.metadata.get("mapping_method")
        row["token_count"] = bundle.metadata.get("token_count")
    except Exception as e:
        row["error"] = repr(e)
    finally:
        try:
            del model  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def verify_models(
    models_yaml: str | Path,
    output_csv: str | Path,
    device: str = "auto",
    max_len: int = 128,
    include_large: bool = False,
) -> list[dict[str, Any]]:
    specs = load_model_specs(models_yaml)
    rows = [
        verify_one_model(spec, device=device, max_len=max_len, include_large=include_large)
        for spec in specs.values()
    ]
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = write_dataframe(df, output_csv)
    write_json(rows, csv_path.with_suffix(".json"))
    md_rows = [
        {
            "model": r["model"],
            "type": r["type"],
            "load_ok": r["load_ok"],
            "forward_ok": r["forward_ok"],
            "has_attentions": r["has_attentions"],
            "n_layers": r.get("n_layers"),
            "n_heads": r.get("n_heads"),
            "error": r.get("error", ""),
        }
        for r in rows
    ]
    write_markdown_table(md_rows, csv_path.with_suffix(".md"), "Model Attention Check")
    return rows
