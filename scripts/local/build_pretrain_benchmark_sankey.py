from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go


LEVEL_LABELS = {
    "L6": "L6: sequence-level bank",
    "L8": "L8: token/reactivity bank",
    "L12": "L12: pairwise structure bank",
}

TASK_LABELS = {
    "splice_site_seqcls": "splice_site_seqcls",
    "rbp_binding_seqcls": "rbp_binding_seqcls",
    "ncRNA_family_seqcls": "ncRNA_family_seqcls",
    "mrna_stability_cds_bart_regression": "mRNA_stability_cds_bart",
    "mrna_stability_gleghorn_regression": "mRNA_stability_gleghorn",
    "mrna_ribosome_loading_regression": "mRNA_ribosome_loading",
    "chemical_reactivity_token": "ribonanza_reactivity_all",
    "chemical_reactivity_ribonanza_2a3": "ribonanza_2A3",
    "chemical_reactivity_ribonanza_dms": "ribonanza_DMS",
    "chemical_reactivity_rnascope": "RNAscope_chemical_reactivity",
    "openvaccine_reactivity_token": "OpenVaccine_reactivity",
    "openvaccine_deg_mg_ph10_token": "OpenVaccine_deg_Mg_pH10",
    "openvaccine_deg_mg_50c_token": "OpenVaccine_deg_Mg_50C",
    "secondary_structure_pair": "secondary_structure_pair",
    "rna3d_rnaglib": "RNA3D_RNAGlib",
    "rnascope_contact_map": "RNAscope_contact_map",
}


@dataclass(frozen=True)
class PretrainRoute:
    source_group: str
    pretrain_corpus: str
    species: str
    tasks: tuple[str, ...]
    model_patterns: tuple[str, ...]
    record_count: float | None
    overlap_basis: str


PRETRAIN_ROUTES = [
    PretrainRoute(
        source_group="PT+BM: RNAcentral/Rfam ncRNA family corpora",
        pretrain_corpus="RNABERT pretraining corpus",
        species="human ncRNA",
        tasks=("ncRNA_family_seqcls",),
        model_patterns=("rnabert",),
        record_count=722370,
        overlap_basis="domain-level ncRNA/RNAcentral/Rfam overlap",
    ),
    PretrainRoute(
        source_group="PT+BM: RNAcentral/Rfam ncRNA family corpora",
        pretrain_corpus="RNA-FM pretraining corpus",
        species="multi-species/database ncRNA",
        tasks=("ncRNA_family_seqcls",),
        model_patterns=("rnafm", "rnafm_ss"),
        record_count=23700000,
        overlap_basis="domain-level ncRNA/RNAcentral/Rfam overlap",
    ),
    PretrainRoute(
        source_group="PT+BM: RNAcentral/Rfam ncRNA family corpora",
        pretrain_corpus="RiNALMo pretraining corpus",
        species="multi-species/database RNA",
        tasks=("ncRNA_family_seqcls",),
        model_patterns=("rinalmo_micro", "rinalmo_mega", "rinalmo_giga"),
        record_count=36000000,
        overlap_basis="source-family overlap: RiNALMo includes Rfam/RNAcentral",
    ),
    PretrainRoute(
        source_group="PT+BM: vertebrate pre-mRNA splicing corpora",
        pretrain_corpus="SpliceBERT pretraining corpus",
        species="vertebrate pre-mRNA",
        tasks=("splice_site_seqcls",),
        model_patterns=("splicebert",),
        record_count=2000000,
        overlap_basis="domain-level splice context overlap",
    ),
    PretrainRoute(
        source_group="PT+BM: UTR/mRNA function and stability corpora",
        pretrain_corpus="3UTRBERT pretraining corpus",
        species="human mRNA/3UTR",
        tasks=("mrna_stability_cds_bart_regression", "mrna_stability_gleghorn_regression", "mrna_ribosome_loading_regression"),
        model_patterns=("utrbert_3mer", "utrbert_4mer", "utrbert_5mer", "utrbert_6mer"),
        record_count=108573,
        overlap_basis="domain-level UTR/mRNA post-transcriptional overlap",
    ),
    PretrainRoute(
        source_group="PT+BM: UTR/mRNA function and stability corpora",
        pretrain_corpus="UTR-LM pretraining corpus",
        species="human/model-vertebrate 5UTR",
        tasks=("mrna_ribosome_loading_regression",),
        model_patterns=("utrlm_mrl", "utrlm_te_el"),
        record_count=None,
        overlap_basis="domain-level 5UTR translation/function overlap",
    ),
    PretrainRoute(
        source_group="PT+BM: mRNA transcript corpora",
        pretrain_corpus="mRNA-FM pretraining corpus",
        species="mRNA transcript corpus",
        tasks=("mrna_stability_cds_bart_regression", "mrna_stability_gleghorn_regression", "mrna_ribosome_loading_regression"),
        model_patterns=("mrnafm", "mrnabert_yyly66", "mrnabert_taykhoom"),
        record_count=None,
        overlap_basis="metadata-only mRNA domain overlap",
    ),
    PretrainRoute(
        source_group="PT+BM: broad RNA language corpora",
        pretrain_corpus="RNAErnie pretraining corpus",
        species="broad RNA corpus",
        tasks=("ncRNA_family_seqcls",),
        model_patterns=("rnaernie",),
        record_count=None,
        overlap_basis="metadata-only broad RNA domain overlap",
    ),
]


BENCHMARK_GROUPS = {
    "ncRNA_family": "PT+BM: RNAcentral/Rfam ncRNA family corpora",
    "pre_mRNA_splicing": "PT+BM: vertebrate pre-mRNA splicing corpora",
    "mRNA_stability": "PT+BM: UTR/mRNA function and stability corpora",
    "mRNA_function": "PT+BM: UTR/mRNA function and stability corpora",
    "RBP_interaction": "BM only: RBP binding assays",
    "chemical_reactivity": "BM only: chemical reactivity assays",
    "mRNA_vaccine_stability": "BM only: OpenVaccine designed RNA stability assays",
    "secondary_structure": "BM only: RNA secondary-structure resources",
    "RNA_3D_structure": "BM only: RNA 3D/contact-map resources",
}


SPECIES_CANONICAL = {
    "pre-mRNA splice contexts": "vertebrate/pre-mRNA context",
    "mixed eukaryotic RNA": "mixed eukaryotic RNA",
    "broad ncRNA families": "multi-species/database ncRNA",
    "mRNA/cell-line assays": "human/cell-line mRNA assay",
    "mixed RNA benchmark": "mixed benchmark RNA",
    "mixed synthetic/natural RNA": "synthetic/natural RNA assay",
    "viral/SARS-CoV-2 designed RNA": "synthetic viral/SARS-CoV-2 RNA",
    "mixed RNA structures": "mixed RNA structure database",
    "PDB RNA structures": "PDB-derived RNA structures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pretraining/benchmark to level Sankey diagrams.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--supervised-summary", type=Path, default=Path("outputs/asset_summaries/supervised_dataset_summary_latest.csv"))
    parser.add_argument("--model-manifest", type=Path, default=Path("assets/manifests/model_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/asset_summaries"))
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def clean_count(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        count = float(value)
    except Exception:
        return None
    return count if count > 0 else None


def model_counts(model_manifest: pd.DataFrame, patterns: Iterable[str]) -> tuple[list[str], list[str]]:
    names = set(patterns)
    rows = model_manifest[model_manifest["model"].isin(names)].copy()
    registered = rows["model"].astype(str).tolist()
    attention_ready = rows[
        (rows.get("load_ok", False) == True)
        & (rows.get("forward_ok", False) == True)
        & (rows.get("has_attentions", False) == True)
    ]["model"].astype(str).tolist()
    return registered, attention_ready


def canonical_species(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return SPECIES_CANONICAL.get(text, text or "unknown/metadata-only")


def task_level_lookup(supervised: pd.DataFrame) -> dict[str, str]:
    return dict(zip(supervised["task"].astype(str), supervised["alignment_layer"].astype(str)))


def build_rows(supervised: pd.DataFrame, model_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    task_to_level = task_level_lookup(supervised)

    for route in PRETRAIN_ROUTES:
        registered, attention_ready = model_counts(model_manifest, route.model_patterns)
        count = len(attention_ready)
        if count == 0:
            count = len(registered)
        if count == 0:
            continue
        for task in route.tasks:
            level = task_to_level.get(task)
            if level not in LEVEL_LABELS:
                continue
            rows.append(
                {
                    "row_type": "pretraining_model_coverage",
                    "source_group": route.source_group,
                    "data_source_node": f"PT: {route.pretrain_corpus}",
                    "data_source": route.pretrain_corpus,
                    "species": route.species,
                    "task": TASK_LABELS.get(task, task),
                    "level": LEVEL_LABELS[level],
                    "level_id": level,
                    "task_id": task,
                    "count_value": count,
                    "record_count": route.record_count,
                    "log_record_value": math.log10((route.record_count or 1) + 1),
                    "registered_models": ";".join(registered),
                    "attention_ready_models": ";".join(attention_ready),
                    "overlap_basis": route.overlap_basis,
                    "notes": "pretraining route width is usable attention model count",
                }
            )

    enabled = supervised[(supervised["enabled"] == True) & (supervised["read_ok"] == True)].copy()
    for _, row in enabled.iterrows():
        level_id = str(row["alignment_layer"])
        if level_id not in LEVEL_LABELS:
            continue
        bio_domain = str(row["bio_domain"])
        source_group = BENCHMARK_GROUPS.get(bio_domain, f"BM only: {bio_domain}")
        record_count = clean_count(row.get("record_count"))
        task = str(row["task"])
        rows.append(
            {
                "row_type": "benchmark_task",
                "source_group": source_group,
                "data_source_node": f"BM: {str(row.get('source_family') or source_group)}",
                "data_source": str(row.get("source_family") or source_group),
                "species": canonical_species(row.get("species_scope")),
                "task": TASK_LABELS.get(task, task),
                "level": LEVEL_LABELS[level_id],
                "level_id": level_id,
                "task_id": task,
                "count_value": 1,
                "record_count": record_count,
                "log_record_value": math.log10((record_count or 1) + 1),
                "registered_models": "",
                "attention_ready_models": "",
                "overlap_basis": "benchmark task registration",
                "notes": "benchmark route width is one enabled task in count view",
            }
        )
    return pd.DataFrame(rows)


def color_for(row_type: str, source_group: str) -> str:
    if row_type == "pretraining_model_coverage":
        return "rgba(92, 83, 165, 0.42)"
    if source_group.startswith("PT+BM"):
        return "rgba(31, 119, 180, 0.38)"
    return "rgba(255, 127, 14, 0.36)"


def node_color(label: str) -> str:
    if label.startswith("PT+BM"):
        return "rgba(31, 119, 180, 0.82)"
    if label.startswith("BM only"):
        return "rgba(255, 127, 14, 0.82)"
    if label.startswith("L6"):
        return "rgba(44, 160, 44, 0.88)"
    if label.startswith("L8"):
        return "rgba(214, 39, 40, 0.88)"
    if label.startswith("L12"):
        return "rgba(148, 103, 189, 0.88)"
    return "rgba(80, 80, 80, 0.72)"


def display_label(label: str) -> str:
    replacements = {
        "PT+BM: RNAcentral/Rfam ncRNA family corpora": "PT+BM<br>RNAcentral/Rfam<br>ncRNA family",
        "PT+BM: vertebrate pre-mRNA splicing corpora": "PT+BM<br>vertebrate<br>pre-mRNA splicing",
        "PT+BM: UTR/mRNA function and stability corpora": "PT+BM<br>UTR/mRNA<br>function + stability",
        "PT+BM: mRNA transcript corpora": "PT+BM<br>mRNA transcript<br>corpora",
        "PT+BM: broad RNA language corpora": "PT+BM<br>broad RNA<br>language corpora",
        "BM only: RBP binding assays": "BM only<br>RBP binding",
        "BM only: chemical reactivity assays": "BM only<br>chemical reactivity",
        "BM only: OpenVaccine designed RNA stability assays": "BM only<br>OpenVaccine<br>designed RNA stability",
        "BM only: RNA secondary-structure resources": "BM only<br>RNA secondary<br>structure",
        "BM only: RNA 3D/contact-map resources": "BM only<br>RNA 3D/contact<br>maps",
        "L6: sequence-level bank": "L6<br>sequence-level<br>bank",
        "L8: token/reactivity bank": "L8<br>token/reactivity<br>bank",
        "L12: pairwise structure bank": "L12<br>pairwise structure<br>bank",
        "PT: RNABERT pretraining corpus": "PT<br>RNABERT",
        "PT: RNA-FM pretraining corpus": "PT<br>RNA-FM",
        "PT: RiNALMo pretraining corpus": "PT<br>RiNALMo",
        "PT: SpliceBERT pretraining corpus": "PT<br>SpliceBERT",
        "PT: 3UTRBERT pretraining corpus": "PT<br>3UTRBERT",
        "PT: UTR-LM pretraining corpus": "PT<br>UTR-LM",
        "PT: mRNA-FM pretraining corpus": "PT<br>mRNA-FM",
        "PT: RNAErnie pretraining corpus": "PT<br>RNAErnie",
        "BM: Kaggle/OpenVaccine fallback": "BM<br>OpenVaccine",
        "BM: Kaggle/HF Ribonanza": "BM<br>Ribonanza",
        "BM: RNAscope Kaggle": "BM<br>RNAscope",
        "BM: RNAscope/splice benchmarks": "BM<br>RNAscope splice",
        "BM: RNAscope/RBP benchmarks": "BM<br>RNAscope RBP",
        "BM: HF/mRNA stability": "BM<br>HF mRNA stability",
        "BM: bpRNA/PDB/HF": "BM<br>bpRNA/PDB/HF",
        "BM: RNAGlib/PDB-derived": "BM<br>RNAGlib/PDB",
        "BM: Rfam/HF": "BM<br>Rfam/HF",
    }
    if label in replacements:
        return replacements[label]
    if len(label) <= 28:
        return label
    parts = label.replace("_", " ").replace("/", "/ ").split()
    lines: list[str] = []
    cur = ""
    for part in parts:
        if len(cur) + len(part) + 1 > 22 and cur:
            lines.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines[:4])


def level_rank(level_id: str) -> int:
    return {"L6": 0, "L8": 1, "L12": 2}.get(str(level_id), 9)


def build_sankey(
    rows: pd.DataFrame,
    value_column: str,
    title: str,
    output_html: Path,
    output_links: Path,
    output_nodes: Path,
    first_column: str = "source_group",
) -> None:
    labels: list[str] = []
    node_index: dict[str, int] = {}
    node_column: dict[str, int] = {}

    def add_node(label: str, column: int) -> int:
        if label not in node_index:
            node_index[label] = len(labels)
            labels.append(label)
            node_column[label] = column
        return node_index[label]

    source: list[int] = []
    target: list[int] = []
    values: list[float] = []
    colors: list[str] = []
    customdata: list[str] = []
    link_rows: list[dict[str, object]] = []
    node_level_values: dict[str, dict[str, float]] = {}
    node_type_hint: dict[str, str] = {}

    for _, row in rows.iterrows():
        path = [
            (first_column, str(row[first_column]), 0),
            ("species", str(row["species"]), 1),
            ("task", str(row["task"]), 2),
            ("level", str(row["level"]), 3),
        ]
        value = float(row[value_column])
        if value <= 0:
            continue
        for (from_kind, from_label, from_col), (to_kind, to_label, to_col) in zip(path, path[1:]):
            s = add_node(from_label, from_col)
            t = add_node(to_label, to_col)
            node_type_hint.setdefault(from_label, from_kind)
            node_type_hint.setdefault(to_label, to_kind)
            for label in (from_label, to_label):
                node_level_values.setdefault(label, {})
                node_level_values[label][str(row["level_id"])] = node_level_values[label].get(str(row["level_id"]), 0.0) + value
            source.append(s)
            target.append(t)
            values.append(value)
            colors.append(color_for(str(row["row_type"]), str(row["source_group"])))
            hover = (
                f"row_type={row['row_type']}<br>"
                f"data_source={row['data_source']}<br>"
                f"task={row['task_id']}<br>"
                f"level={row['level_id']}<br>"
                f"record_count={row['record_count']}<br>"
                f"registered_models={row['registered_models']}<br>"
                f"attention_ready_models={row['attention_ready_models']}<br>"
                f"overlap={row['overlap_basis']}"
            )
            customdata.append(hover)
            link_rows.append(
                {
                    "from_kind": from_kind,
                    "from": from_label,
                    "to_kind": to_kind,
                    "to": to_label,
                    "value": value,
                    "value_basis": value_column,
                    "row_type": row["row_type"],
                    "data_source": row["data_source"],
                    "source_group": row["source_group"],
                    "task_id": row["task_id"],
                    "level_id": row["level_id"],
                    "record_count": row["record_count"],
                    "registered_models": row["registered_models"],
                    "attention_ready_models": row["attention_ready_models"],
                    "overlap_basis": row["overlap_basis"],
                }
            )

    incoming = {label: 0.0 for label in labels}
    outgoing = {label: 0.0 for label in labels}
    for s, t, value in zip(source, target, values):
        outgoing[labels[s]] += value
        incoming[labels[t]] += value

    def dominant_level(label: str) -> str:
        lv = node_level_values.get(label, {})
        if not lv:
            return "L99"
        return max(lv.items(), key=lambda kv: (kv[1], -level_rank(kv[0])))[0]

    def sort_key(label: str) -> tuple[object, ...]:
        col = node_column[label]
        kind = node_type_hint.get(label, "")
        dom = dominant_level(label)
        weight = max(incoming.get(label, 0.0), outgoing.get(label, 0.0))
        if col == 0:
            group_rank = 0 if str(label).startswith("PT+BM") or str(label).startswith("PT:") else 1
            return (group_rank, level_rank(dom), -weight, label)
        if col == 3:
            return (level_rank(dom), label)
        if kind == "task":
            return (level_rank(dom), label)
        return (level_rank(dom), -weight, label)

    ordered_by_col: dict[int, list[str]] = {}
    for col in range(4):
        ordered_by_col[col] = sorted([label for label in labels if node_column[label] == col], key=sort_key)

    positions: dict[str, tuple[float, float]] = {}
    x_by_col = {0: 0.015, 1: 0.345, 2: 0.665, 3: 0.955}
    top_margin = 0.025
    bottom_margin = 0.025
    level_gap = 0.045
    node_gap = 0.008
    for col, ordered in ordered_by_col.items():
        if not ordered:
            continue
        by_level: dict[str, list[str]] = {"L6": [], "L8": [], "L12": [], "L99": []}
        for label in ordered:
            by_level.setdefault(dominant_level(label), []).append(label)
        active_levels = [lvl for lvl in ("L6", "L8", "L12", "L99") if by_level.get(lvl)]
        total_gap = level_gap * max(len(active_levels) - 1, 0)
        active_height = 1.0 - top_margin - bottom_margin - total_gap
        level_weights = {
            lvl: sum(max(incoming.get(label, 0.0), outgoing.get(label, 0.0), 0.15) for label in by_level[lvl])
            for lvl in active_levels
        }
        total_weight = sum(level_weights.values()) or 1.0
        y_cursor = top_margin
        for lvl in active_levels:
            labels_in_level = by_level[lvl]
            level_height = active_height * level_weights[lvl] / total_weight
            label_weight_total = sum(max(incoming.get(label, 0.0), outgoing.get(label, 0.0), 0.15) for label in labels_in_level) or 1.0
            usable = max(level_height - node_gap * max(len(labels_in_level) - 1, 0), 0.02)
            inner_cursor = y_cursor
            for label in labels_in_level:
                weight = max(incoming.get(label, 0.0), outgoing.get(label, 0.0), 0.15)
                slot = usable * weight / label_weight_total
                positions[label] = (x_by_col[col], min(max(inner_cursor, 0.001), 0.985))
                inner_cursor += max(slot, 0.018) + node_gap
            y_cursor += level_height + level_gap

    node_rows = []
    xs, ys = [], []
    for label in labels:
        x, y = positions.get(label, (x_by_col[node_column[label]], 0.5))
        xs.append(x)
        ys.append(y)
        node_rows.append(
            {
                "node": label,
                "display_label": display_label(label),
                "column": node_column[label],
                "dominant_level": dominant_level(label),
                "incoming": incoming[label],
                "outgoing": outgoing[label],
                "x": x,
                "y": y,
            }
        )

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node={
                    "pad": 9,
                    "thickness": 17,
                    "line": {"color": "rgba(40,40,40,0.35)", "width": 0.5},
                    "label": [display_label(label) for label in labels],
                    "color": [node_color(label) for label in labels],
                    "x": xs,
                    "y": ys,
                },
                link={
                    "source": source,
                    "target": target,
                    "value": values,
                    "color": colors,
                    "customdata": customdata,
                    "hovertemplate": "%{source.label} -> %{target.label}<br>value=%{value}<br>%{customdata}<extra></extra>",
                },
            )
        ]
    )
    fig.update_layout(
        title=title,
        font={"size": 12, "family": "Arial"},
        width=1650,
        height=920,
        margin={"l": 15, "r": 15, "t": 65, "b": 20},
        annotations=[
            {"x": 0.015, "y": 1.055, "xref": "paper", "yref": "paper", "text": "Data source", "showarrow": False, "font": {"size": 14}},
            {"x": 0.345, "y": 1.055, "xref": "paper", "yref": "paper", "text": "Species / source scope", "showarrow": False, "font": {"size": 14}},
            {"x": 0.665, "y": 1.055, "xref": "paper", "yref": "paper", "text": "Downstream task", "showarrow": False, "font": {"size": 14}},
            {"x": 0.955, "y": 1.055, "xref": "paper", "yref": "paper", "text": "Level", "showarrow": False, "font": {"size": 14}},
        ],
    )
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)
    pd.DataFrame(link_rows).to_csv(output_links, index=False, encoding="utf-8-sig")
    pd.DataFrame(node_rows).to_csv(output_nodes, index=False, encoding="utf-8-sig")


def format_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def svg_label(label: str) -> str:
    text = label.replace("<br>", " ").replace("_", " ")
    text = " ".join(text.split())
    if len(text) <= 34:
        return text
    return text[:31] + "..."


def build_svg_sankey(
    rows: pd.DataFrame,
    value_column: str,
    title: str,
    output_html: Path,
    output_svg: Path,
    output_links: Path,
    first_column: str = "data_source_node",
) -> None:
    width, height = 1700, 980
    top, bottom = 72, 68
    available = height - top - bottom
    col_x = [70.0, 500.0, 930.0, 1360.0]
    node_w = 20.0
    col_names = ["Data source", "Species / source scope", "Downstream task", "Alignment level"]
    col_colors = ["#2f5f73", "#6c7a20", "#9a5f1d", "#7d4d83"]
    link_colors = ["#2f5f73", "#6c7a20", "#9a5f1d"]

    edge_map: dict[tuple[int, str, str], dict[str, object]] = {}
    node_col: dict[str, int] = {}
    node_levels: dict[str, dict[str, float]] = {}

    for _, row in rows.iterrows():
        path = [
            (str(row[first_column]), 0),
            (str(row["species"]), 1),
            (str(row["task"]), 2),
            (str(row["level"]), 3),
        ]
        value = float(row[value_column])
        if value <= 0:
            continue
        for label, col in path:
            node_col[label] = col
            node_levels.setdefault(label, {})
            node_levels[label][str(row["level_id"])] = node_levels[label].get(str(row["level_id"]), 0.0) + value
        for (src, src_col), (dst, _) in zip(path, path[1:]):
            key = (src_col, src, dst)
            item = edge_map.setdefault(
                key,
                {
                    "stage": src_col,
                    "source": src,
                    "target": dst,
                    "value": 0.0,
                    "row_types": set(),
                    "tasks": set(),
                    "data_sources": set(),
                    "overlap_basis": set(),
                },
            )
            item["value"] = float(item["value"]) + value
            item["row_types"].add(str(row["row_type"]))
            item["tasks"].add(str(row["task_id"]))
            item["data_sources"].add(str(row["data_source"]))
            item["overlap_basis"].add(str(row["overlap_basis"]))

    edges = list(edge_map.values())
    max_edge = max(float(e["value"]) for e in edges) if edges else 1.0
    for edge in edges:
        edge["stroke_width"] = max(1.15, 72.0 * (float(edge["value"]) / max_edge) ** 0.62)

    def recalc_node_heights(scale: float) -> tuple[dict[str, float], float]:
        incoming = {label: 0.0 for label in node_col}
        outgoing = {label: 0.0 for label in node_col}
        for edge in edges:
            sw = float(edge["stroke_width"]) * scale
            outgoing[str(edge["source"])] += sw
            incoming[str(edge["target"])] += sw
        node_h = {
            label: max(13.0, incoming.get(label, 0.0), outgoing.get(label, 0.0))
            for label in node_col
        }
        max_block = 0.0
        for col in range(4):
            labels = [label for label, c in node_col.items() if c == col]
            gap = 10.0 if len(labels) > 12 else 14.0
            max_block = max(max_block, sum(node_h[label] for label in labels) + gap * max(len(labels) - 1, 0))
        return node_h, max_block

    scale = 1.0
    for _ in range(8):
        node_h, max_block = recalc_node_heights(scale)
        if max_block <= available:
            break
        scale *= max(0.35, available / max_block)
    node_h, _ = recalc_node_heights(scale)
    for edge in edges:
        edge["stroke_width"] = max(1.05, float(edge["stroke_width"]) * scale)

    def dominant_level(label: str) -> str:
        vals = node_levels.get(label, {})
        if not vals:
            return "L99"
        return max(vals.items(), key=lambda kv: (kv[1], -level_rank(kv[0])))[0]

    def sort_node(label: str) -> tuple[object, ...]:
        col = node_col[label]
        value = node_h[label]
        if col == 0:
            prefix = 0 if label.startswith("PT:") else 1 if label.startswith("BM:") else 2
            return (prefix, level_rank(dominant_level(label)), -value, label)
        if col == 3:
            return (level_rank(dominant_level(label)), label)
        return (level_rank(dominant_level(label)), -value, label)

    node_y: dict[str, float] = {}
    node_order: dict[int, list[str]] = {}
    for col in range(4):
        labels = sorted([label for label, c in node_col.items() if c == col], key=sort_node)
        node_order[col] = labels
        gap = 10.0 if len(labels) > 12 else 14.0
        block = sum(node_h[label] for label in labels) + gap * max(len(labels) - 1, 0)
        y = top + max(0.0, (available - block) * 0.42)
        for label in labels:
            node_y[label] = y
            y += node_h[label] + gap

    outgoing_offsets = {label: 0.0 for label in node_col}
    incoming_offsets = {label: 0.0 for label in node_col}

    def edge_sort(edge: dict[str, object]) -> tuple[object, ...]:
        stage = int(edge["stage"])
        src = str(edge["source"])
        dst = str(edge["target"])
        return (
            stage,
            node_y.get(src, 0.0),
            node_y.get(dst, 0.0),
            -float(edge["value"]),
            src,
            dst,
        )

    edges = sorted(edges, key=edge_sort)
    link_rows = []
    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:650}.col{font-size:13px;font-weight:650;fill:#555}.node{font-size:12px}.meta{font-size:10px;fill:#666}</style>",
        f'<text class="title" x="70" y="36">{html.escape(title)}</text>',
        '<text class="meta" x="70" y="56">Aggregated count-weighted links; transparent bands encode amount; hover shows grouped tasks/sources.</text>',
    ]
    for idx, name in enumerate(col_names):
        svg_parts.append(f'<text class="col" x="{col_x[idx] - 10:.1f}" y="{height - 24}">{html.escape(name)}</text>')

    for edge in sorted(edges, key=lambda e: float(e["stroke_width"]), reverse=True):
        stage = int(edge["stage"])
        src = str(edge["source"])
        dst = str(edge["target"])
        sw = float(edge["stroke_width"])
        sx = col_x[stage] + node_w
        tx = col_x[stage + 1]
        sy = node_y[src] + outgoing_offsets[src] + sw / 2
        ty = node_y[dst] + incoming_offsets[dst] + sw / 2
        outgoing_offsets[src] += sw
        incoming_offsets[dst] += sw
        c1 = sx + (tx - sx) * 0.44
        c2 = sx + (tx - sx) * 0.56
        opacity = 0.36 if sw >= 7 else 0.24
        title_txt = (
            f"{src} -> {dst}\n"
            f"value={format_count(float(edge['value']))}; "
            f"tasks={', '.join(sorted(edge['tasks'])[:8])}; "
            f"sources={', '.join(sorted(edge['data_sources'])[:6])}; "
            f"overlap={', '.join(sorted(edge['overlap_basis'])[:4])}"
        )
        svg_parts.append(
            f'<path d="M{sx:.1f},{sy:.1f} C{c1:.1f},{sy:.1f} {c2:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" '
            f'fill="none" stroke="{link_colors[stage]}" stroke-opacity="{opacity:.2f}" stroke-width="{sw:.2f}">'
            f"<title>{html.escape(title_txt)}</title></path>"
        )
        link_rows.append(
            {
                "stage": stage,
                "source": src,
                "target": dst,
                "value": edge["value"],
                "stroke_width": sw,
                "tasks": ";".join(sorted(edge["tasks"])),
                "data_sources": ";".join(sorted(edge["data_sources"])),
                "overlap_basis": ";".join(sorted(edge["overlap_basis"])),
            }
        )

    node_rows = []
    for col in range(4):
        for label in node_order[col]:
            x = col_x[col]
            y = node_y[label]
            h = node_h[label]
            node_value = max(
                sum(float(e["value"]) for e in edges if str(e["source"]) == label),
                sum(float(e["value"]) for e in edges if str(e["target"]) == label),
            )
            title_txt = f"{label}\nvisual_height={h:.1f}; value={format_count(node_value)}"
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w:.1f}" height="{h:.1f}" rx="4" '
                f'fill="{col_colors[col]}" opacity="0.92"><title>{html.escape(title_txt)}</title></rect>'
            )
            text_y = y + h / 2 - 2
            svg_parts.append(
                f'<text class="node" text-anchor="start" x="{x + 28:.1f}" y="{text_y:.1f}">{html.escape(svg_label(label))}</text>'
            )
            svg_parts.append(
                f'<text class="meta" text-anchor="start" x="{x + 28:.1f}" y="{text_y + 14:.1f}">sum={html.escape(format_count(node_value))}</text>'
            )
            node_rows.append(
                {
                    "node": label,
                    "column": col,
                    "x": x,
                    "y": y,
                    "height": h,
                    "value": node_value,
                    "dominant_level": dominant_level(label),
                }
            )

    svg_parts.append("</svg>")
    svg = "\n".join(svg_parts)
    output_svg.write_text(svg, encoding="utf-8")
    output_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Sankey</title>"
        "<style>body{margin:0;background:#fbfbf8}.wrap{padding:18px}</style></head>"
        f"<body><div class='wrap'>{svg}</div></body></html>",
        encoding="utf-8",
    )
    pd.DataFrame(link_rows).to_csv(output_links, index=False, encoding="utf-8-sig")
    output_nodes = output_links.with_name(output_links.stem.replace("links", "nodes") + output_links.suffix)
    pd.DataFrame(node_rows).to_csv(output_nodes, index=False, encoding="utf-8-sig")


def write_summary(
    rows: pd.DataFrame,
    output: Path,
    count_html: Path,
    volume_html: Path,
    detail_html: Path,
    count_stacked_html: Path,
    volume_stacked_html: Path,
    detail_stacked_html: Path,
    svg_detail_html: Path,
    svg_group_html: Path,
) -> None:
    pretrain = rows[rows["row_type"] == "pretraining_model_coverage"]
    bench = rows[rows["row_type"] == "benchmark_task"]
    level_counts = bench.groupby("level_id")["task_id"].nunique().sort_index()
    source_counts = rows.groupby("source_group")["count_value"].sum().sort_values(ascending=False)
    overlap = pretrain.groupby("task_id")["count_value"].sum().sort_values(ascending=False)

    lines = [
        "# Pretraining Benchmark Level Sankey",
        "",
        "Columns are fixed as: data source -> species/source scope -> downstream task -> alignment level.",
        "",
        "The count-weighted view uses mutually exclusive route events:",
        "- pretraining routes: number of locally registered usable attention checkpoints mapped to the task domain;",
        "- benchmark routes: one enabled/readable supervised task dataset.",
        "",
        "The log-volume view uses `log10(record_count + 1)` so small assays remain visible next to large corpora.",
        "",
        "## Outputs",
        f"- recommended SVG detail Sankey: `{svg_detail_html.as_posix()}`",
        f"- recommended SVG grouped Sankey: `{svg_group_html.as_posix()}`",
        f"- count Sankey: `{count_html.as_posix()}`",
        f"- log-volume Sankey: `{volume_html.as_posix()}`",
        f"- data-source detail Sankey: `{detail_html.as_posix()}`",
        f"- stacked count Sankey: `{count_stacked_html.as_posix()}`",
        f"- stacked log-volume Sankey: `{volume_stacked_html.as_posix()}`",
        f"- stacked data-source detail Sankey: `{detail_stacked_html.as_posix()}`",
        f"- route rows: `outputs/asset_summaries/pretrain_benchmark_level_sankey_rows.csv`",
        f"- count links: `outputs/asset_summaries/pretrain_benchmark_level_sankey_links_counts.csv`",
        f"- log-volume links: `outputs/asset_summaries/pretrain_benchmark_level_sankey_links_log_volume.csv`",
        "",
        "## Enabled Benchmark Tasks By Level",
    ]
    for level, n in level_counts.items():
        lines.append(f"- `{level}`: `{int(n)}` tasks")
    lines.extend(["", "## Pretraining Coverage Events By Task"])
    for task, value in overlap.items():
        lines.append(f"- `{task}`: `{int(value)}` usable attention-model coverage events")
    lines.extend(["", "## Top Source Groups By Count Events"])
    for source, value in source_counts.head(12).items():
        lines.append(f"- `{source}`: `{int(value)}` events")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- `PT+BM` nodes mean the current metadata indicates source-family or domain-level overlap between a model pretraining corpus and at least one supervised benchmark.",
            "- This is not a proof of exact sequence leakage unless the `overlap_basis` says exact asset/source match; most routes are conservative domain-level flags.",
            "- DNA/control pretraining sources without a mapped RNA downstream benchmark are intentionally excluded from the main three-level Sankey and should be reviewed separately.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supervised = pd.read_csv(resolve(root, args.supervised_summary))
    model_manifest = pd.read_csv(resolve(root, args.model_manifest))
    rows = build_rows(supervised, model_manifest)
    rows_path = output_dir / "pretrain_benchmark_level_sankey_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")

    count_html = output_dir / "pretrain_benchmark_level_sankey_counts.html"
    count_links = output_dir / "pretrain_benchmark_level_sankey_links_counts.csv"
    count_nodes = output_dir / "pretrain_benchmark_level_sankey_nodes_counts.csv"
    build_sankey(
        rows,
        "count_value",
        "RNA model pretraining / benchmark overlap to downstream levels (count events)",
        count_html,
        count_links,
        count_nodes,
        first_column="source_group",
    )
    count_stacked_html = output_dir / "pretrain_benchmark_level_sankey_counts_stacked.html"
    count_stacked_links = output_dir / "pretrain_benchmark_level_sankey_links_counts_stacked.csv"
    count_stacked_nodes = output_dir / "pretrain_benchmark_level_sankey_nodes_counts_stacked.csv"
    build_sankey(
        rows,
        "count_value",
        "RNA model pretraining / benchmark overlap to downstream levels (stacked count events)",
        count_stacked_html,
        count_stacked_links,
        count_stacked_nodes,
        first_column="source_group",
    )

    volume_html = output_dir / "pretrain_benchmark_level_sankey_log_volume.html"
    volume_links = output_dir / "pretrain_benchmark_level_sankey_links_log_volume.csv"
    volume_nodes = output_dir / "pretrain_benchmark_level_sankey_nodes_log_volume.csv"
    build_sankey(
        rows,
        "log_record_value",
        "RNA model pretraining / benchmark overlap to downstream levels (log10 record volume)",
        volume_html,
        volume_links,
        volume_nodes,
        first_column="source_group",
    )
    volume_stacked_html = output_dir / "pretrain_benchmark_level_sankey_log_volume_stacked.html"
    volume_stacked_links = output_dir / "pretrain_benchmark_level_sankey_links_log_volume_stacked.csv"
    volume_stacked_nodes = output_dir / "pretrain_benchmark_level_sankey_nodes_log_volume_stacked.csv"
    build_sankey(
        rows,
        "log_record_value",
        "RNA model pretraining / benchmark overlap to downstream levels (stacked log10 record volume)",
        volume_stacked_html,
        volume_stacked_links,
        volume_stacked_nodes,
        first_column="source_group",
    )

    detail_html = output_dir / "pretrain_benchmark_datasource_level_sankey_counts.html"
    detail_links = output_dir / "pretrain_benchmark_datasource_level_sankey_links_counts.csv"
    detail_nodes = output_dir / "pretrain_benchmark_datasource_level_sankey_nodes_counts.csv"
    build_sankey(
        rows,
        "count_value",
        "RNA model pretraining corpora and benchmark datasets to downstream levels (count events)",
        detail_html,
        detail_links,
        detail_nodes,
        first_column="data_source_node",
    )
    detail_stacked_html = output_dir / "pretrain_benchmark_datasource_level_sankey_counts_stacked.html"
    detail_stacked_links = output_dir / "pretrain_benchmark_datasource_level_sankey_links_counts_stacked.csv"
    detail_stacked_nodes = output_dir / "pretrain_benchmark_datasource_level_sankey_nodes_counts_stacked.csv"
    build_sankey(
        rows,
        "count_value",
        "RNA pretraining corpora and benchmark datasets to downstream levels (stacked count events)",
        detail_stacked_html,
        detail_stacked_links,
        detail_stacked_nodes,
        first_column="data_source_node",
    )

    svg_detail_html = output_dir / "pretrain_benchmark_datasource_level_sankey_counts_v2.html"
    svg_detail_svg = output_dir / "pretrain_benchmark_datasource_level_sankey_counts_v2.svg"
    svg_detail_links = output_dir / "pretrain_benchmark_datasource_level_sankey_links_counts_v2.csv"
    build_svg_sankey(
        rows,
        "count_value",
        "RNA pretraining corpora + benchmark datasets to downstream alignment levels",
        svg_detail_html,
        svg_detail_svg,
        svg_detail_links,
        first_column="data_source_node",
    )

    svg_group_html = output_dir / "pretrain_benchmark_grouped_level_sankey_counts_v2.html"
    svg_group_svg = output_dir / "pretrain_benchmark_grouped_level_sankey_counts_v2.svg"
    svg_group_links = output_dir / "pretrain_benchmark_grouped_level_sankey_links_counts_v2.csv"
    build_svg_sankey(
        rows,
        "count_value",
        "RNA pretraining / benchmark overlap groups to downstream alignment levels",
        svg_group_html,
        svg_group_svg,
        svg_group_links,
        first_column="source_group",
    )

    summary = output_dir / "pretrain_benchmark_level_sankey_summary.md"
    write_summary(
        rows,
        summary,
        count_html.relative_to(root),
        volume_html.relative_to(root),
        detail_html.relative_to(root),
        count_stacked_html.relative_to(root),
        volume_stacked_html.relative_to(root),
        detail_stacked_html.relative_to(root),
        svg_detail_html.relative_to(root),
        svg_group_html.relative_to(root),
    )
    print(f"wrote {count_html}")
    print(f"wrote {volume_html}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
