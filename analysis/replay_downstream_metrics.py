"""Offline replay: recompute per-seed downstream metrics from the packaged
per-sequence predictions and compare them with the published per-seed table.

Inputs
------
--pred-root   PAYLOAD/predictions/  (v1.0.1 package)
--points-csv  downstream_recomputed_points.csv  (from the v1.0.0 asset
              dataset-rna-head-distill_v1.0.0.tar.gz ->
              dataset-rna-head-distill/metrics/downstream/)

What it checks (per task/mode/seed cell present in the package)
  - binary seqcls parquets (splice_donor/acceptor, rbp_binding):
      AUROC, AUPRC from (label, logits)
  - ncRNA_family_seqcls parquets (logits = list of class scores):
      macro one-vs-rest AUROC, macro-F1, accuracy
  - chemical_reactivity_token .pt files (list of {seq_id, preds, labels}):
      Pearson/Spearman, pooled over all nucleotides (and per-sequence-mean
      as a secondary diagnostic)

Tolerance: absolute diff <= 1e-6 counts as MATCH (the same check in the
internal audit, `dumpcheck_*` vs `eval_*`, matched to 0.0 / < 1e-6).

Exit code 0 = every packaged cell matched; 2 = at least one mismatch or a
referenced prediction file missing; 1 = usage/IO error. A cell listed in the
points CSV but not packaged (secondary_structure_pair) is reported as
NOT_PACKAGED, never as a failure.

Usage:
  python replay_downstream_metrics.py --pred-root PAYLOAD/predictions \
      --points-csv downstream_recomputed_points.csv --out replay_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)

TOL = 1e-6


def check_binary(pq: Path, pts: pd.Series) -> dict:
    df = pd.read_parquet(pq)
    y = df['label'].to_numpy()
    s = df['logits'].to_numpy(dtype=float)
    out = {}
    if 'auroc' in pts and pd.notna(pts['auroc']):
        out['auroc'] = (roc_auc_score(y, s), float(pts['auroc']))
    if 'auprc' in pts and pd.notna(pts['auprc']):
        out['auprc'] = (average_precision_score(y, s), float(pts['auprc']))
    return out


def check_ncrna(pq: Path, pts: pd.Series) -> dict:
    """Replicate probes/metrics.sequence_classification_metrics multiclass branch.

    softmax over ALL stored logit columns (scipy.special.softmax on the raw
    float32 array, as the original eval did), then one-vs-rest over the labels
    present in y_true only; AUROC/AUPRC = unweighted mean over those classes.
    """
    import warnings

    from scipy.special import softmax
    from sklearn.metrics import matthews_corrcoef

    df = pd.read_parquet(pq)
    y = df['label'].to_numpy()
    S = np.vstack(df['logits'].to_numpy())  # [n, n_classes] raw logits
    S = np.nan_to_num(S, nan=0.0, posinf=1e6, neginf=-1e6)
    probs = softmax(S, axis=1)
    present_labels = sorted(np.unique(y).tolist())
    auroc_vals, auprc_vals = [], []
    for lab in present_labels:
        binary_true = (y == lab).astype(int)
        binary_score = probs[:, lab]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                auroc_vals.append(float(roc_auc_score(binary_true, binary_score)))
            except Exception:
                pass
            try:
                auprc_vals.append(float(average_precision_score(binary_true, binary_score)))
            except Exception:
                pass
    y_pred = S.argmax(axis=1)
    out = {}
    if 'auroc' in pts and pd.notna(pts['auroc']):
        out['auroc'] = (float(np.mean(auroc_vals)), float(pts['auroc']))
    if 'auprc' in pts and pd.notna(pts['auprc']):
        out['auprc'] = (float(np.mean(auprc_vals)), float(pts['auprc']))
    if 'f1' in pts and pd.notna(pts['f1']):
        out['f1_macro'] = (f1_score(y, y_pred, average='macro',
                                    zero_division=0), float(pts['f1']))
    if 'mcc' in pts and pd.notna(pts['mcc']):
        out['mcc'] = (matthews_corrcoef(y, y_pred), float(pts['mcc']))
    if 'accuracy' in pts and pd.notna(pts['accuracy']):
        out['accuracy'] = (float((y == y_pred).mean()), float(pts['accuracy']))
    return out


def check_chem(pt: Path, pts: pd.Series) -> dict:
    import torch
    recs = torch.load(pt, map_location='cpu', weights_only=False)
    preds = np.concatenate([np.asarray(r['preds'], dtype=float)
                            for r in recs])
    labels = np.concatenate([np.asarray(r['labels'], dtype=float)
                             for r in recs])
    out = {}
    if 'pearson' in pts and pd.notna(pts['pearson']):
        out['pearson'] = (pearsonr(preds, labels)[0], float(pts['pearson']))
    if 'spearman' in pts and pd.notna(pts['spearman']):
        out['spearman'] = (spearmanr(preds, labels)[0], float(pts['spearman']))
    per_seq = [pearsonr(np.asarray(r['preds'], float),
                        np.asarray(r['labels'], float))[0]
               for r in recs[:200]]
    out['_diagnostic_per_seq_pearson_first200'] = (float(np.nanmean(per_seq)),
                                                   None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-root', required=True, type=Path)
    ap.add_argument('--points-csv', required=True, type=Path)
    ap.add_argument('--gated-points-csv', type=Path, default=None,
                    help='optional per-seed point values incl. P4g/A8-P4g '
                         '(metadata/downstream_perseed_points_p4g_a8.csv); '
                         'A8_P4g maps to <task>/distilled_hidden_aligned_gated_a8/')
    ap.add_argument('--out', required=True, type=Path,
                    help='output prefix (.csv / .json)')
    args = ap.parse_args()

    pts = pd.read_csv(args.points_csv)
    rows = []
    for tup in pts.itertuples():
        r = tup._asdict()
        base = args.pred_root / r['task'] / r['mode'] / f"seed_{r['seed']}"
        pq, pt = base / 'pred_test.parquet', base / 'pred_test.pt'
        if not pq.exists() and not pt.exists():
            rows.append({'task': r['task'], 'mode': r['mode'], 'seed': r['seed'],
                         'metric': '', 'status': ('NOT_PACKAGED' if r['task'] == 'secondary_structure_pair' else 'ERROR: required prediction missing'), 'abs_diff': None})
            continue
        try:
            if pq.exists():
                df0 = pd.read_parquet(pq)
                is_list = df0['logits'].map(lambda v: isinstance(v, (list, np.ndarray))).any()
                checks = check_ncrna(pq, r) if is_list else check_binary(pq, r)
            else:
                checks = check_chem(pt, r)
        except Exception as e:  # noqa: BLE001 - report, don't fabricate
            rows.append({'task': r['task'], 'mode': r['mode'], 'seed': r['seed'],
                         'metric': '', 'status': f'ERROR: {e!r}',
                         'abs_diff': None})
            continue
        for metric, (got, want) in checks.items():
            if want is None:
                continue
            diff = abs(got - want)
            rows.append({'task': r['task'], 'mode': r['mode'], 'seed': r['seed'],
                         'metric': metric, 'recomputed': got,
                         'published': want, 'abs_diff': diff,
                         'status': 'MATCH' if diff <= TOL else 'MISMATCH'})
    rep = pd.DataFrame(rows)

    # Optional: P4g / A8-P4g per-seed point values (separate directory for A8).
    if args.gated_points_csv is not None:
        gp = pd.read_csv(args.gated_points_csv)
        grows = []
        for r in gp.itertuples():
            mode_dir = ('distilled_hidden_aligned_gated_a8'
                        if r.mode_or_line == 'A8_P4g' else r.mode_or_line)
            base = args.pred_root / r.task / mode_dir / f'seed_{r.seed}'
            pq, pt = base / 'pred_test.parquet', base / 'pred_test.pt'
            try:
                if pq.exists():
                    df0 = pd.read_parquet(pq)
                    is_list = df0['logits'].map(lambda v: isinstance(v, (list, np.ndarray))).any()
                    if is_list:
                        got = check_ncrna(pq, pd.Series({'auroc': r.value}))['auroc'][0]
                    else:
                        y = df0['label'].to_numpy()
                        s = df0['logits'].to_numpy(dtype=float)
                        got = roc_auc_score(y, s)
                elif pt.exists():
                    import torch
                    recs = torch.load(pt, map_location='cpu', weights_only=False)
                    preds = np.concatenate([np.asarray(x['preds'], dtype=float) for x in recs])
                    labels = np.concatenate([np.asarray(x['labels'], dtype=float) for x in recs])
                    got = pearsonr(preds, labels)[0]
                else:
                    grows.append({'task': r.task, 'mode': mode_dir, 'seed': r.seed,
                                  'metric': r.metric, 'status': ('NOT_PACKAGED' if r.task == 'secondary_structure_pair' else 'ERROR: required prediction missing'),
                                  'abs_diff': None})
                    continue
            except Exception as e:  # noqa: BLE001
                grows.append({'task': r.task, 'mode': mode_dir, 'seed': r.seed,
                              'metric': r.metric, 'status': f'ERROR: {e!r}',
                              'abs_diff': None})
                continue
            diff = abs(got - r.value)
            grows.append({'task': r.task, 'mode': mode_dir, 'seed': r.seed,
                          'metric': r.metric, 'recomputed': got, 'published': r.value,
                          'abs_diff': diff,
                          'status': 'MATCH' if diff <= TOL else 'MISMATCH'})
        grep = pd.DataFrame(grows)
        gated_csv = args.out.with_suffix('').as_posix() + '_gated.csv'
        grep.to_csv(gated_csv, index=False)
        rep = pd.concat([rep, grep], ignore_index=True)

    out_csv = args.out.with_suffix('.csv')
    rep.to_csv(out_csv, index=False)
    checked = rep[rep.status.isin(['MATCH', 'MISMATCH'])]
    summary = {
        'cells_checked': int(checked[['task', 'mode', 'seed']].drop_duplicates().shape[0]),
        'metric_checks': int(len(checked)),
        'match': int((rep.status == 'MATCH').sum()),
        'mismatch': int((rep.status == 'MISMATCH').sum()),
        'errors': int(rep.status.str.startswith('ERROR').sum()),
        'not_packaged_cells': int((rep.status == 'NOT_PACKAGED').sum()),
        'max_abs_diff': float(rep['abs_diff'].max()),
        'tolerance': TOL,
    }
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['mismatch'] == 0 and summary['errors'] == 0
                     else 2)


if __name__ == '__main__':
    main()
