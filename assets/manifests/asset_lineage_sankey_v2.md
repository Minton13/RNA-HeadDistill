# Asset Lineage Sankey v2

- generated_at: `2026-06-15T19:49:27`
- rows: `35`
- Taxonomy columns are mutually controlled categories to reduce label noise and avoid same-column overlap.
- Count-weighted links use `count_value`; rows with missing counts are marked `count_missing=true` and use value 1 only to stay visible.

## Columns

| column | meaning |
|---|---|
| `数据流` | `data_stream_v2` |
| `物种/来源范围` | `origin_scope_v2` |
| `生物学对象` | `biology_domain_v2` |
| `标签/目标层级` | `target_level_v2` |
| `采样/切分方式` | `sampling_scope_v2` |

## Outputs

- `assets/manifests/asset_lineage_inventory_v2.csv`
- `assets/manifests/asset_lineage_links_counts_v2.csv`
- `outputs/lineage/asset_lineage_sankey_counts_v2.html`
- `outputs/lineage/asset_lineage_sankey_counts_v2.svg`
- `outputs/lineage/asset_lineage_sankey_readable_v2.html`
- `outputs/lineage/asset_lineage_sankey_readable_v2.svg`
