"""Blue-edition redraw of Fig. 2 (a1-a3 atlas heatmaps + cross-task Jaccard).

Why this script exists
----------------------
The v3 manuscript ships a *blue* Fig. 2 (sequential white->steel-blue maps),
but its panels were digitized from the old raster and still carry the two
known digitization errors:

  * fig2d  openvaccine_deg_mg_50c x deg_mg_ph10  printed 0.49, raw = 0.48
  * fig2d  chem_rnascope x chem                  printed 0.18, raw = 0.19

All matrices here are recomputed from the raw per-head atlas
    rna_head_alignment/outputs/head_screening/all_head_atlas.csv
with the exact cleaning of paper/audit/heads/audit_atlas_compute.py, and the
Jaccard matrix is asserted against paper/audit/heads/fig2b_jaccard_matrix_audit.csv
(same check as plot_heatmaps_twotone_rawdata.py), so the printed annotations
are the raw values by construction.

Palette: the blue half of the project two-tone palette
    #FAF9F6 -> #D2E4EF -> #8EBBD7 -> #4F83AD   (off-white .. deep steel blue)
applied as a plain sequential map over the same data windows as the two-tone
rawdata edition (a1 [0,.92], a2 [.22,.78], a3 [.26,.87], Jaccard [0,1]).

Outputs (nothing overwritten; new directories):
  redesign_v2/panels_blue_rawdata/               PDF + white PNG per panel
  redesign_v2/panels_blue_rawdata_transparent/   transparent PNG per panel
  redesign_v2/figures/fig2_blue_rawdata.{pdf,png}  composed, tight-layout
                                                   full Fig. 2 (panels a-e;
                                                   e redrawn from
                                                   paper/tables/selected_teacher_heads.csv)

Run:  python redesign_v2/scripts/plot_heatmaps_blue_rawdata.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import panelstyle  # noqa: E402
from panelstyle import panel_figure, save_panel  # noqa: E402
from figstyle import ROUTE, INK, GRAY, header, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]
ATLAS_CSV = ROOT.parents[2] / 'rna_head_alignment' / 'outputs' / \
    'head_screening' / 'all_head_atlas.csv'
AUDIT_JAC = PAPER / 'audit' / 'heads' / 'fig2b_jaccard_matrix_audit.csv'
SEL_CSV = PAPER / 'tables' / 'selected_teacher_heads.csv'

BLUE = LinearSegmentedColormap.from_list('blue_rawdata', [
    '#FAF9F6', '#D2E4EF', '#8EBBD7', '#4F83AD',
])

OUT_W = ROOT / 'panels_blue_rawdata'
OUT_T = ROOT / 'panels_blue_rawdata_transparent'
OUT_W.mkdir(parents=True, exist_ok=True)
OUT_T.mkdir(parents=True, exist_ok=True)
panelstyle.PANELS, panelstyle.TRANSPARENT_PANELS = OUT_W, OUT_T


def load_valid_atlas():
    atlas = pd.read_csv(ATLAS_CSV, low_memory=False)
    is_fail = atlas['notes'].fillna('').str.contains('model_load_failed')
    a = atlas[~is_fail]
    a = a[a['metric_primary'].notna()]
    per_head = a[a['head'] >= 0]
    combo_max = per_head.groupby(['task', 'model'])['metric_primary'].max()
    deg = set(combo_max[combo_max <= 1e-9].index)
    keep = [(t, m) not in deg for t, m in
            zip(per_head['task'], per_head['model'])]
    return per_head[keep]


def atlas_matrix(valid, task, model):
    g = valid[(valid.task == task) & (valid.model == model)]
    mat = g.pivot(index='layer', columns='head',
                  values='metric_primary').sort_index()
    mat = mat[sorted(mat.columns)]
    arr = mat.values.astype(float)
    assert not np.isnan(arr).any(), f'{task}/{model}: missing cells'
    return arr


def plot_atlas_panels(valid):
    picks = [
        ('a1', 'Secondary structure · RiNALMo-giga',
         'secondary_structure_pair', 'rinalmo_giga', 0., .92,
         [0, .2, .4, .6, .8], 6, 'fig2a1_secondary_structure'),
        ('a2', 'RBP binding · RNA-FM',
         'rbp_binding_seqcls', 'rnafm', .22, .78,
         [.4, .5, .6], 8, 'fig2a2_rbp_binding'),
        ('a3', 'Splice site · SpliceBERT',
         'splice_site_seqcls', 'splicebert', .26, .87,
         [.4, .5, .6, .75], 12, 'fig2a3_splice_site'),
    ]
    for letter, title, task, model, vmin, vmax, ticks, route, stem in picks:
        arr = atlas_matrix(valid, task, model)
        nl, nh = arr.shape
        print(f'{stem}: raw range [{arr.min():.4f}, {arr.max():.4f}] '
              f'shape {arr.shape}')
        norm = Normalize(vmin=vmin, vmax=vmax)
        cell = .22
        tick_size, label_size = 9.0, 15.0
        plot_w, plot_h = nh * cell, nl * cell
        fig_w, fig_h = plot_w + 1.60, plot_h + 1.17
        fig = panel_figure(letter, title, figsize=(fig_w, fig_h), route=route)
        ax = fig.add_axes([.68 / fig_w, .62 / fig_h,
                           plot_w / fig_w, plot_h / fig_h])
        cax = fig.add_axes([(.68 + plot_w + .18) / fig_w, .62 / fig_h,
                            .14 / fig_w, plot_h / fig_h])
        sns.heatmap(arr, cmap=BLUE, norm=norm, ax=ax, cbar_ax=cax,
                    square=True, xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'probe score', 'ticks': ticks},
                    rasterized=True)
        xi = np.arange(0, nh, 5)
        yi = np.arange(0, nl, 5 if nl > 12 else 2)
        ax.set_xticks(xi + .5, xi)
        ax.set_yticks(yi + .5, yi)
        ax.set_xlabel('head', fontsize=label_size)
        ax.set_ylabel('layer', fontsize=label_size)
        ax.tick_params(length=0, pad=2, labelsize=tick_size)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = ax.collections[0].colorbar
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=2, width=.6, pad=2, labelsize=tick_size)
        colorbar.set_label('probe score', fontsize=label_size)
        save_panel(fig, stem)


def jaccard_matrix(valid):
    tasks = sorted(valid['task'].unique())
    K = 10
    jac_sum = pd.DataFrame(0.0, index=tasks, columns=tasks)
    jac_cnt = pd.DataFrame(0, index=tasks, columns=tasks)
    for model, gm in valid.groupby('model'):
        tops = {}
        for task, g in gm.groupby('task'):
            top = g.nlargest(K, 'metric_primary')
            tops[task] = set(zip(top['layer'], top['head']))
        for i, t1 in enumerate(tasks):
            if t1 not in tops:
                continue
            for t2 in tasks[i:]:
                if t2 not in tops:
                    continue
                a, b = tops[t1], tops[t2]
                j = len(a & b) / len(a | b)
                jac_sum.loc[t1, t2] += j
                jac_sum.loc[t2, t1] += j
                jac_cnt.loc[t1, t2] += 1
                jac_cnt.loc[t2, t1] += 1
    jac = jac_sum / jac_cnt.replace(0, np.nan)
    audit = pd.read_csv(AUDIT_JAC, index_col=0)
    audit = audit.loc[tasks, tasks]
    diff = (jac - audit).abs().values
    print(f'jaccard vs audit max|diff|: {np.nanmax(diff):.6f}')
    # audit CSV was saved with .round(4): tolerance = rounding half-step
    assert np.nanmax(diff) < 1e-4, 'recomputed Jaccard differs from audit CSV'
    return jac


DISPLAY = {
    'chemical_reactivity_ribonanza_2a3': 'chem_ribonanza_2a3',
    'chemical_reactivity_ribonanza_dms': 'chem_ribonanza_dms',
    'chemical_reactivity_rnascope': 'chem_rnascope',
    'chemical_reactivity_token': 'chem',
    'mrna_ribosome_loading_regression': 'mrna_ribosome_loading',
    'mrna_stability_cds_bart_regression': 'mrna_stability_cds_bart',
    'mrna_stability_gleghorn_regression': 'mrna_stability_gleghorn',
    'openvaccine_deg_mg_50c_token': 'openvaccine_deg_mg_50c',
    'openvaccine_deg_mg_ph10_token': 'openvaccine_deg_mg_ph10',
    'openvaccine_reactivity_token': 'openvaccine_reactivity',
    'rbp_binding_seqcls': 'rbp_binding',
    'rna3d_rnaglib': 'rna3d_rnaglib',
    'rnascope_contact_map': 'rnascope_contact_map',
    'secondary_structure_pair': 'secondary_structure_pair',
    'splice_site_seqcls': 'splice_site',
}


def annot_color(norm, value):
    r, g, b = BLUE(norm(value))[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.55 else 'black'


def jaccard_display(jac):
    order = [t for t in DISPLAY if t in jac.index]
    names = [DISPLAY[t] for t in order]
    mat = jac.loc[order, order].values.astype(float)
    mat = np.nan_to_num(mat, nan=1.0)  # diagonal may be NaN if never paired
    np.fill_diagonal(mat, 1.0)
    return order, names, mat


def plot_jaccard(jac):
    order, names, mat = jaccard_display(jac)
    norm = Normalize(vmin=0., vmax=1.)
    n = len(order)
    annotations = np.empty((n, n), dtype=object)
    annotations[:] = ''
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] >= .10:
                annotations[i, j] = f'{mat[i, j]:.2f}'
    fig = panel_figure('b', 'Cross-task head sharing', figsize=(9.0, 8.4),
                       route=6)
    plot_size = 4.8
    ax = fig.add_axes([.245, .280, plot_size / 9.0, plot_size / 8.4])
    cax = fig.add_axes([.245 + plot_size / 9.0 + .035, .280,
                        .018, plot_size / 8.4])
    sns.heatmap(mat, cmap=BLUE, norm=norm, ax=ax, cbar_ax=cax,
                square=True, xticklabels=names, yticklabels=names,
                annot=annotations, fmt='', annot_kws={'fontsize': 5.0},
                cbar_kws={'label': 'mean Jaccard'}, rasterized=True)
    for text, (i, j) in zip(ax.texts, [(i, j) for i in range(n)
                                       for j in range(n)]):
        if text.get_text():
            text.set_color(annot_color(norm, mat[i, j]))
    ax.set_xticklabels(names, rotation=90, fontsize=8.0)
    ax.set_yticklabels(names, rotation=0, fontsize=8.0)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.collections[0].colorbar.outline.set_visible(False)
    save_panel(fig, 'fig2b_cross_task_jaccard')
    ov = jac.loc[['openvaccine_deg_mg_50c_token',
                  'openvaccine_deg_mg_ph10_token',
                  'openvaccine_reactivity_token'],
                 ['openvaccine_deg_mg_50c_token',
                  'openvaccine_deg_mg_ph10_token',
                  'openvaccine_reactivity_token']]
    print('openvaccine block (raw):')
    print(ov.round(4))
    chem = jac.loc[['chemical_reactivity_rnascope'],
                   ['chemical_reactivity_token']]
    print('chem_rnascope x chem (raw):', chem.round(4).values[0, 0])


# ---------------------------------------------------------------------------
# Composed full Fig. 2 (tight two-column layout, letters a-e as in v3)
# ---------------------------------------------------------------------------

NAME_MAP = {'nucleotide_transformer_v2_500m': 'NT-v2-500M',
            'moderngena_base': 'ModernGENA-base', 'plantrna_fm': 'PlantRNA-FM',
            'rinalmo_mega': 'RiNALMo-mega', 'rinalmo_giga': 'RiNALMo-giga',
            'rnafm': 'RNA-FM', 'splicebert': 'SpliceBERT',
            'utrlm_mrl': 'UTR-LM (MRL)', 'mrnafm': 'mRNA-FM'}


def compose_fig2(valid, jac):
    plt.rcParams.update({'font.size': 7.2, 'axes.labelsize': 7.0,
                         'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3})
    fig = plt.figure(figsize=(10.4, 8.8), facecolor='white')

    def heat(ax, cax, arr, vmin, vmax, ticks, ylab='layer'):
        norm = Normalize(vmin=vmin, vmax=vmax)
        nl, nh = arr.shape
        sns.heatmap(arr, cmap=BLUE, norm=norm, ax=ax, cbar_ax=cax,
                    square=False, xticklabels=False, yticklabels=False,
                    cbar_kws={'ticks': ticks},
                    rasterized=True)
        xi = np.arange(0, nh, 5)
        yi = np.arange(0, nl, 5 if nl > 12 else 2)
        ax.set_xticks(xi + .5, xi)
        ax.set_yticks(yi + .5, yi)
        ax.set_xlabel('head')
        ax.set_ylabel(ylab)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = ax.collections[0].colorbar
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, width=.6, pad=2)
        cb.set_label('probe score', fontsize=7.0)
        cb.ax.yaxis.label.set_size(7.0)
        return norm

    # left column: a / b / c atlas heatmaps
    a1 = atlas_matrix(valid, 'secondary_structure_pair', 'rinalmo_giga')
    a2 = atlas_matrix(valid, 'rbp_binding_seqcls', 'rnafm')
    a3 = atlas_matrix(valid, 'splice_site_seqcls', 'splicebert')

    header(fig, 'a', 'Secondary structure · RiNALMo-giga', .968)
    axa = fig.add_axes([.075, .420, .205, .470])
    caxa = fig.add_axes([.292, .420, .010, .470])
    heat(axa, caxa, a1, 0., .92, [0, .2, .4, .6, .8])

    header(fig, 'b', 'RBP binding · RNA-FM', .390)
    axb = fig.add_axes([.075, .225, .205, .135])
    caxb = fig.add_axes([.292, .225, .010, .135])
    heat(axb, caxb, a2, .22, .78, [.4, .5, .6])

    header(fig, 'c', 'Splice site · SpliceBERT', .195)
    axc = fig.add_axes([.075, .070, .205, .095])
    caxc = fig.add_axes([.292, .070, .010, .095])
    heat(axc, caxc, a3, .26, .87, [.4, .5, .6, .75])

    # right column top: d cross-task Jaccard
    header(fig, 'd', 'Cross-task head sharing', .968, x=.360)
    order, names, mat = jaccard_display(jac)
    norm = Normalize(vmin=0., vmax=1.)
    n = len(order)
    annotations = np.empty((n, n), dtype=object)
    annotations[:] = ''
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] >= .10:
                annotations[i, j] = f'{mat[i, j]:.2f}'
    axd = fig.add_axes([.475, .430, .290, .460])
    caxd = fig.add_axes([.778, .430, .010, .460])
    sns.heatmap(mat, cmap=BLUE, norm=norm, ax=axd, cbar_ax=caxd,
                square=True, xticklabels=names, yticklabels=names,
                annot=annotations, fmt='', annot_kws={'fontsize': 4.6},
                cbar_kws={'label': 'mean Jaccard'}, rasterized=True)
    for text, (i, j) in zip(axd.texts, [(i, j) for i in range(n)
                                        for j in range(n)]):
        if text.get_text():
            text.set_color(annot_color(norm, mat[i, j]))
    axd.set_xticklabels(names, rotation=90, fontsize=6.0)
    axd.set_yticklabels(names, rotation=0, fontsize=6.0)
    axd.tick_params(length=0, pad=1.5)
    for spine in axd.spines.values():
        spine.set_visible(False)
    cbd = axd.collections[0].colorbar
    cbd.outline.set_visible(False)
    cbd.ax.tick_params(length=2, width=.6, pad=2)
    cbd.set_label('mean Jaccard', fontsize=7.0)
    cbd.ax.yaxis.label.set_size(7.0)

    # right column bottom: e supervised head selection
    header(fig, 'e', 'Supervised head selection', .360, x=.360)
    sel = pd.read_csv(SEL_CSV)
    labels = [f"{NAME_MAP.get(r.teacher_model, r.teacher_model)} "
              f"L{r.teacher_layer}H{r.teacher_head}"
              for r in sel.itertuples()]
    colors = [ROUTE[int(d)] for d in sel.route_depth]
    axe = fig.add_axes([.500, .085, .330, .235])
    axe.barh(range(len(sel)), sel.percentile_in_task_model, color=colors,
             height=.46, edgecolor='none', linewidth=0)
    axe.set_yticks(range(len(sel)), labels, fontsize=6.0)
    axe.invert_yaxis()
    axe.set_xlim(0, 108)
    axe.axvline(90, color='#777777', ls='--', lw=.65)
    axe.set_xticks([0, 20, 40, 60, 80, 100])
    axe.set_xlabel('percentile (100 = best)')
    axe.tick_params(axis='y', length=0, pad=2)
    clean(axe)
    fig.legend(handles=[Patch(fc=ROUTE[d], ec='none', label=f'L{d} route')
                        for d in (6, 8, 12)],
               loc='lower right', bbox_to_anchor=(.965, .018), ncol=3,
               frameon=False, fontsize=6.4, handlelength=1.1,
               columnspacing=.9)

    out = ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig2_blue_rawdata.pdf', facecolor='white')
    fig.savefig(out / 'fig2_blue_rawdata.png', facecolor='white', dpi=400)
    plt.close(fig)
    print(out / 'fig2_blue_rawdata.pdf')


if __name__ == '__main__':
    valid = load_valid_atlas()
    print(f'valid rows: {len(valid)}  models: {valid.model.nunique()}  '
          f'tasks: {valid.task.nunique()}')
    plot_atlas_panels(valid)
    jac = jaccard_matrix(valid)
    plot_jaccard(jac)
    compose_fig2(valid, jac)
    print('done ->', OUT_W, ',', OUT_T, 'and figures/fig2_blue_rawdata.*')
