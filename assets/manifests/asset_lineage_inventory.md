# Asset Lineage Inventory

- generated_at: `2026-06-13T12:32:31`
- rows: `35`

| stream | rows |
|---|---:|
| `Benchmark dataset` | 20 |
| `Model training data` | 9 |
| `RNA3D benchmark` | 6 |

- CSV: `D:\code\RNA_LLM_v2\rna_circuit_distill\assets\manifests\asset_lineage_inventory.csv`
- JSON: `D:\code\RNA_LLM_v2\rna_circuit_distill\assets\manifests\asset_lineage_inventory.json`

## v2 Taxonomy

The v2 inventory replaces the earlier free-text categories with controlled, non-overlapping Sankey columns:

| column | meaning |
|---|---|
| `数据流` | 训练语料 / 本地归一化benchmark / RNAscope官方benchmark / RNA3D结构benchmark |
| `物种/来源范围` | RNA/DNA plus human, multi-species, assay, PDB/structure, or benchmark-mixed source |
| `生物学对象` | ncRNA、UTR/翻译、pre-mRNA剪接、RNA-蛋白互作、二级结构、三级结构等 |
| `标签/目标层级` | 自监督序列、序列整体分类/回归、位点标签、pairwise/contact、3D图/坐标、元数据 |
| `采样/切分方式` | 训练语料策略、官方split、本地full+mini、3D图快照、metadata-only |

The primary v2 Sankey uses real `count_value` for links. Rows with unknown counts are marked `count_missing=true` and contribute value 1 only so they remain visible.

- v2 inventory: `assets/manifests/asset_lineage_inventory_v2.csv`
- v2 exact link table: `assets/manifests/asset_lineage_links_counts_v2.csv`
- count-weighted Sankey: `outputs/lineage/asset_lineage_sankey_counts_v2.html`
- readable log-view Sankey: `outputs/lineage/asset_lineage_sankey_readable_v2.html`
