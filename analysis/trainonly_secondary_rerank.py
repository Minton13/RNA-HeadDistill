"""Rebuild the secondary_structure_pair head ranking from the train-only
re-screening and compare it with the stored (mixed-split) selection.

Background
----------
The stored atlas rows for secondary_structure_pair (and the 30 candidate rows
derived from them) are labeled split="mixed": each model was probed on the
first 100 records of the shuffled assets/data/secondary/secondary_full.jsonl
(~77 train / ~14 val / ~9 test out of 100). Job 65390 re-screened all 37
enabled models on a strictly train-only pool (split_filter=["train"],
post_filter_limit=100, same n=100 per model) into
outputs/head_screening_trainonly/secondary_structure_pair/<model>/head_atlas.csv.

This script
  1. collects the train-only per-model atlases, stamps split="train_only"
     (the scorer hardcodes "mixed"; the label is rewritten here because the
     sample pool is verified train-only by the config);
  2. applies the same cleaning as audit_atlas_compute.py (drop
     model_load_failed, NaN metric_primary, degenerate (task, model) combos
     with max metric <= 1e-9, head >= 0);
  3. re-derives the secondary candidates with the exact rules of
     propose_teacher_heads (per (task, model) top-5 -> per-task top-30 by
     metric_primary), on the train-only atlas;
  4. reports the overlap/rank shift vs the stored 30 secondary candidates and
     the fate of the two L12-route heads selected from this task
     (rinalmo_giga L31H13, rinalmo_giga L31H1).

Outputs (--outdir, default outputs/reproduction/trainonly/):
  trainonly_secondary_atlas.csv            full train-only atlas (26+ models)
  trainonly_secondary_top30.csv            re-derived task top-30
  trainonly_secondary_vs_mixed_report.md   comparison summary

Run:  python analysis/trainonly_secondary_rerank.py --help
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SELECTED_L12 = [('rinalmo_giga', 31, 13), ('rinalmo_giga', 31, 1)]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    is_fail = df['notes'].fillna('').str.contains('model_load_failed')
    d = df[~is_fail]
    d = d[d['metric_primary'].notna()]
    d = d[d['head'] >= 0]
    combo_max = d.groupby(['task', 'model'])['metric_primary'].max()
    deg = set(combo_max[combo_max <= 1e-9].index)
    keep = [(t, m) not in deg for t, m in zip(d['task'], d['model'])]
    return d[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--atlas', type=Path, default=ROOT / 'release-metadata/v1.0.1/trainonly_secondary_atlas.csv')
    parser.add_argument('--per-model-root', type=Path, help='optional root containing */head_atlas.csv from a new rerun')
    data_root = Path(os.environ.get('RNA_HEAD_DISTILL_DATA', ROOT / 'dataset-rna-head-distill'))
    parser.add_argument('--stored-candidates', type=Path, default=data_root / 'selection/teacher_head_candidates.csv')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/reproduction/trainonly')
    args = parser.parse_args()
    OUT = args.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    if args.per_model_root:
        frames = [pd.read_csv(p) for p in sorted(args.per_model_root.glob('*/head_atlas.csv')) if p.stat().st_size]
        if not frames:
            raise SystemExit(f'no train-only atlases under {args.per_model_root}')
        atlas = pd.concat(frames, ignore_index=True)
    else:
        atlas = pd.read_csv(args.atlas)
    atlas['split'] = 'train_only'
    valid = clean(atlas)
    valid.to_csv(OUT / 'trainonly_secondary_atlas.csv', index=False)
    print(f'train-only atlas: {len(valid)} rows, '
          f'{valid.model.nunique()} models')

    # per (task, model) top-5 -> per-task top-30 (propose_teacher_heads rules)
    a = valid.copy()
    a['metric_primary'] = pd.to_numeric(a['metric_primary'], errors='coerce')
    a = a[a['layer'] >= 0].sort_values('metric_primary', ascending=False)
    parts = [g.head(5) for _, g in a.groupby(['task', 'model'])]
    cand = pd.concat(parts, ignore_index=True)
    top30 = (cand.sort_values('metric_primary', ascending=False)
             .groupby('task').head(30).reset_index(drop=True))
    top30.to_csv(OUT / 'trainonly_secondary_top30.csv', index=False)

    stored = pd.read_csv(args.stored_candidates)
    stored30 = stored[stored['task'] == 'secondary_structure_pair'].copy()
    key = ['model', 'layer', 'head']
    new_set = set(map(tuple, top30[key].values))
    old_set = set(map(tuple, stored30[key].values))

    lines = [
        '# Train-only secondary re-ranking vs stored mixed-split selection',
        '',
        f'- train-only atlas rows (cleaned): {len(valid)} '
        f'({valid.model.nunique()} models)',
        f'- re-derived top-30 vs stored top-30 overlap: '
        f'{len(new_set & old_set)}/30',
        f'- heads only in train-only top-30: {len(new_set - old_set)}; '
        f'only in stored: {len(old_set - new_set)}',
        '',
        '## Fate of the two L12-route heads selected from this task',
    ]
    for m, l, h in SELECTED_L12:
        old_rank = (stored30.reset_index()
                    .query('model == @m and layer == @l and head == @h')
                    .index.tolist())
        new_rank = (top30.reset_index()
                    .query('model == @m and layer == @l and head == @h')
                    .index.tolist())
        old_s = f'rank {old_rank[0] + 1}' if old_rank else 'NOT in top-30'
        new_s = f'rank {new_rank[0] + 1}' if new_rank else 'NOT in top-30'
        old_metric = (stored30.query('model == @m and layer == @l and '
                                     'head == @h')['metric_primary'].tolist())
        new_metric = (top30.query('model == @m and layer == @l and '
                                  'head == @h')['metric_primary'].tolist())
        lines.append(f'- {m} L{l}H{h}: stored {old_s} '
                     f'(auprc {old_metric[0]:.4f}' if old_metric else
                     f'- {m} L{l}H{h}: stored {old_s}')
        lines[-1] += (f') | train-only {new_s}'
                      + (f' (auprc {new_metric[0]:.4f})' if new_metric
                         else ''))
    lines += [
        '',
        '## Top-10 of the re-derived train-only ranking',
        '',
        top30.head(10)[key + ['metric_primary']].to_string(index=False),
        '',
        '## Heads entering/leaving the top-30',
        '',
        f'entering: {sorted(new_set - old_set)}',
        '',
        f'leaving: {sorted(old_set - new_set)}',
    ]
    (OUT / 'trainonly_secondary_vs_mixed_report.md').write_text(
        '\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:14]))
    print('->', OUT / 'trainonly_secondary_vs_mixed_report.md')


if __name__ == '__main__':
    main()
