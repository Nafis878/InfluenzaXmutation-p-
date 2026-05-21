#!/usr/bin/env python3
"""
Consolidate all pipeline PNGs into a single folder and
build section dashboards + a master overview gallery.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import shutil
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent
VIZ  = ROOT / 'all_visualizations'
VIZ.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
})

BLUE   = '#2471A3'; ORANGE = '#E67E22'; GREEN  = '#27AE60'
RED    = '#C0392B'; PURPLE = '#8E44AD'; GRAY   = '#7F8C8D'
DARK   = '#1A252F'

# ── Catalogue of all plots ─────────────────────────────────────────────────────
PLOTS = {
    'influenza': [
        ('outputs/phase1_h1n1_temporal_trend.png',
         'H1N1 Divergence Trend',          'Phase 1'),
        ('outputs/phase2_h3n2_silhouette_scores.png',
         'H3N2 Clustering Silhouette',     'Phase 2'),
        ('outputs/phase3_top_variations.png',
         'Top 25 AA Substitutions',        'Phase 3'),
        ('outputs/phase3_enrichment_summary.png',
         'Critical Region Enrichment',     'Phase 3'),
        ('outputs/phase4_mds_plot_temporal.png',
         'MDS Temporal Progression',       'Phase 4'),
        ('outputs/phase4_mds_plot_clusters.png',
         'H3N2 Historical Clusters (MDS)', 'Phase 4'),
        ('outputs/phase4_mds_plot_subtype.png',
         'H1N1 vs H3N2 Separation (MDS)', 'Phase 4'),
        ('outputs/phase5_variant_emergence_timeline.png',
         'Variant Emergence Timeline',     'Phase 5'),
        ('outputs/validation_dashboard.png',
         'Pipeline Validation Dashboard',  'Phases 1-5'),
    ],
    'agentic': [
        ('agentic_drift_models/06_visualizations/drift_trajectory.png',
         'Agentic Drift Trajectory',        'Drift'),
        ('agentic_drift_models/06_visualizations/feature_importance.png',
         'Feature Importance',              'Drift'),
        ('agentic_drift_models/06_visualizations/attention_heatmap.png',
         'Transformer Attention Heatmap',   'Drift'),
        ('agentic_drift_models/06_visualizations/ensemble_confidence.png',
         'Ensemble Model Confidence',       'Drift'),
        ('agentic_drift_models/06_visualizations/anomaly_detection.png',
         'Anomaly Detection',               'Drift'),
    ],
    'mda': [
        ('phase8_outputs/phase8_model_comparison.png',
         'Model Comparison (MDA vs RF vs XGB)', 'Phase 8'),
        ('phase8_outputs/phase8_training_curves.png',
         'MDA Training Curves',             'Phase 8'),
        ('phase8_outputs/phase8_drift_prob_distribution.png',
         'Drift Probability Distribution',  'Phase 8'),
        ('phase8_outputs/phase8_mutation_scatter.png',
         'Mutation Position vs Drift Prob', 'Phase 8'),
        ('phase8_outputs/phase8_cluster_forecast.png',
         'Next-Cluster Forecast',           'Phase 8'),
        ('phase8_outputs/phase8_attention_analysis.png',
         'Attention by Position Band',      'Phase 8'),
    ],
}

# ── Step 1: Copy every PNG to all_visualizations/ ─────────────────────────────
print('Copying PNGs to all_visualizations/ ...')
copied = []
for section, items in PLOTS.items():
    for src_rel, title, phase in items:
        src = ROOT / src_rel
        if src.exists():
            dest = VIZ / src.name
            shutil.copy2(src, dest)
            copied.append((src, dest, title, phase, section))
            print(f'  copied  {src.name}')
        else:
            print(f'  MISSING {src_rel}')
print(f'Copied {len(copied)} files.\n')


# ── Helper: read image safely ──────────────────────────────────────────────────
def load_img(path):
    try:
        return mpimg.imread(str(path))
    except Exception:
        return None


def add_section_label(ax, text, color):
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
            fontsize=7, color='white', fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.18', facecolor=color,
                      edgecolor='none', alpha=0.88))


def thumb_grid(fig, items, ncols, title, title_color, section_color, y_start, height):
    """Draw a row-group of thumbnails inside fig using absolute positions."""
    nrows = int(np.ceil(len(items) / ncols))
    w = 1.0 / ncols
    h = height / nrows

    # Section header bar
    header_ax = fig.add_axes([0.0, y_start + height - 0.015, 1.0, 0.028])
    header_ax.set_facecolor(title_color)
    header_ax.set_xticks([]); header_ax.set_yticks([])
    for sp in header_ax.spines.values():
        sp.set_visible(False)
    header_ax.text(0.012, 0.45, title, transform=header_ax.transAxes,
                   fontsize=11, fontweight='bold', color='white', va='center')

    for idx, (src, dest, label, phase, _section) in enumerate(items):
        row = idx // ncols
        col = idx  % ncols
        left   = col * w + 0.002
        bottom = y_start + (nrows - 1 - row) * h + 0.002
        ax = fig.add_axes([left, bottom, w - 0.004, h - 0.030])
        img = load_img(dest)
        if img is not None:
            ax.imshow(img, aspect='auto')
        else:
            ax.set_facecolor('#eeeeee')
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_edgecolor('#cccccc')

        # Caption
        cap_ax = fig.add_axes([left, bottom - 0.026, w - 0.004, 0.024])
        cap_ax.set_facecolor('#f8f9fa')
        cap_ax.set_xticks([]); cap_ax.set_yticks([])
        for sp in cap_ax.spines.values():
            sp.set_linewidth(0.3); sp.set_edgecolor('#cccccc')
        tag_color = {'influenza': BLUE, 'agentic': ORANGE, 'mda': PURPLE}.get(_section, GRAY)
        cap_ax.text(0.04, 0.55, f'[{phase}]', transform=cap_ax.transAxes,
                    fontsize=6.5, color=tag_color, fontweight='bold', va='center')
        cap_ax.text(0.22, 0.55, label, transform=cap_ax.transAxes,
                    fontsize=6.5, color=DARK, va='center',
                    clip_on=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD 1: Influenza Mutation Pipeline (Phases 1-5)
# ══════════════════════════════════════════════════════════════════════════════
print('Building dashboard_influenza_pipeline.png ...')
items_inf = [(ROOT/s, VIZ/Path(s).name, t, p, 'influenza') for s, t, p in PLOTS['influenza']]
ncols = 3
nrows = int(np.ceil(len(items_inf) / ncols))
fig = plt.figure(figsize=(22, nrows * 5.5 + 1.2))
fig.patch.set_facecolor('#FAFAFA')

# Title
fig.text(0.5, 0.985, 'Influenza Mutation Analysis Pipeline  |  Phases 1–5',
         ha='center', va='top', fontsize=17, fontweight='bold', color=DARK)
fig.text(0.5, 0.970, 'H1N1 Divergence  ·  H3N2 Clustering  ·  Variation Detection  ·  MDS  ·  Variant Timeline',
         ha='center', va='top', fontsize=10, color=GRAY)

h_per_row = 0.88 / nrows
for idx, (src, dest, label, phase, section) in enumerate(items_inf):
    row = idx // ncols
    col = idx  % ncols
    left   = col * (1/ncols) + 0.008
    bottom = 0.06 + (nrows - 1 - row) * h_per_row + 0.02
    w_ax   = 1/ncols - 0.016
    h_ax   = h_per_row - 0.045

    ax = fig.add_axes([left, bottom, w_ax, h_ax])
    img = load_img(dest)
    if img is not None:
        ax.imshow(img, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor(BLUE)

    # Title bar under image
    tb = fig.add_axes([left, bottom - 0.032, w_ax, 0.028])
    tb.set_facecolor(BLUE); tb.set_xticks([]); tb.set_yticks([])
    for sp in tb.spines.values(): sp.set_visible(False)
    tb.text(0.5, 0.5, f'[{phase}]  {label}',
            transform=tb.transAxes, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

# Legend row
legend_ax = fig.add_axes([0.02, 0.01, 0.96, 0.04])
legend_ax.set_facecolor('#EBF5FB'); legend_ax.set_xticks([]); legend_ax.set_yticks([])
for sp in legend_ax.spines.values(): sp.set_edgecolor(BLUE); sp.set_linewidth(0.5)
legend_ax.text(0.5, 0.5,
    'Phase 1: H1N1 Divergence Rate  ·  Phase 2: H3N2 K-Means Clustering  ·  '
    'Phase 3: AA Substitution Enrichment  ·  Phase 4: MDS Sequence Space  ·  Phase 5: Variant Timeline',
    ha='center', va='center', fontsize=8, color=DARK, transform=legend_ax.transAxes)

fig.savefig(VIZ / 'dashboard_influenza_pipeline.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  saved dashboard_influenza_pipeline.png')


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD 2: Agentic Drift Monitoring
# ══════════════════════════════════════════════════════════════════════════════
print('Building dashboard_agentic_drift.png ...')
items_ag = [(ROOT/s, VIZ/Path(s).name, t, p, 'agentic') for s, t, p in PLOTS['agentic']]
fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor('#FAFAFA')
fig.text(0.5, 0.985, 'Agentic Drift Alignment Monitoring  |  10-Model Ensemble System',
         ha='center', va='top', fontsize=17, fontweight='bold', color=DARK)
fig.text(0.5, 0.969,
         'Chi-Square Enrichment  ·  Polynomial Regression  ·  ARIMA Forecast  ·  Transformer Attention  ·  Isolation Forest',
         ha='center', va='top', fontsize=10, color=GRAY)

ncols_ag = 3
positions = [
    (0.008,  0.09, 0.310, 0.84),
    (0.338,  0.09, 0.310, 0.84),
    (0.668,  0.09, 0.310, 0.84),
    (0.008,  0.09 - 0.00, 0.310, 0.84),  # will be offset below
    (0.338,  0.09 - 0.00, 0.310, 0.84),
]
# 5 items: 3 top row, 2 bottom row (centred)
pos5 = [
    (0.008, 0.51, 0.310, 0.43),
    (0.338, 0.51, 0.310, 0.43),
    (0.668, 0.51, 0.310, 0.43),
    (0.173, 0.06, 0.310, 0.43),
    (0.503, 0.06, 0.310, 0.43),
]
for idx, (src, dest, label, phase, section) in enumerate(items_ag):
    l, b, w_ax, h_ax = pos5[idx]
    h_ax -= 0.05
    ax = fig.add_axes([l, b, w_ax, h_ax])
    img = load_img(dest)
    if img is not None:
        ax.imshow(img, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor(ORANGE)
    tb = fig.add_axes([l, b - 0.038, w_ax, 0.034])
    tb.set_facecolor(ORANGE); tb.set_xticks([]); tb.set_yticks([])
    for sp in tb.spines.values(): sp.set_visible(False)
    tb.text(0.5, 0.5, label, transform=tb.transAxes,
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

fig.savefig(VIZ / 'dashboard_agentic_drift.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  saved dashboard_agentic_drift.png')


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD 3: MDA Transformer (Phase 8)
# ══════════════════════════════════════════════════════════════════════════════
print('Building dashboard_mda_transformer.png ...')
items_mda = [(ROOT/s, VIZ/Path(s).name, t, p, 'mda') for s, t, p in PLOTS['mda']]
fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor('#FAFAFA')
fig.text(0.5, 0.985, 'MDA Transformer — Phase 8  |  MutationDriftAttention',
         ha='center', va='top', fontsize=17, fontweight='bold', color=DARK)
fig.text(0.5, 0.969,
         'Model Comparison  ·  Training Curves  ·  Drift Distribution  ·  Mutation Scatter  ·  Cluster Forecast  ·  Attention Analysis',
         ha='center', va='top', fontsize=10, color=GRAY)
# 6 items: 3 top row + 3 bottom row
pos6m = [
    (0.008, 0.51, 0.310, 0.43),
    (0.338, 0.51, 0.310, 0.43),
    (0.668, 0.51, 0.310, 0.43),
    (0.008, 0.06, 0.310, 0.43),
    (0.338, 0.06, 0.310, 0.43),
    (0.668, 0.06, 0.310, 0.43),
]
for idx, (src, dest, label, phase, section) in enumerate(items_mda):
    if idx >= len(pos6m):
        break
    l, b, w_ax, h_ax = pos6m[idx]
    h_ax -= 0.05
    ax = fig.add_axes([l, b, w_ax, h_ax])
    img = load_img(dest)
    if img is not None:
        ax.imshow(img, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor(PURPLE)
    tb = fig.add_axes([l, b - 0.038, w_ax, 0.034])
    tb.set_facecolor(PURPLE); tb.set_xticks([]); tb.set_yticks([])
    for sp in tb.spines.values(): sp.set_visible(False)
    tb.text(0.5, 0.5, label, transform=tb.transAxes,
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

fig.savefig(VIZ / 'dashboard_mda_transformer.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  saved dashboard_mda_transformer.png')


# ══════════════════════════════════════════════════════════════════════════════
# MASTER OVERVIEW GALLERY (all 19 unique plots, 4-column grid)
# ══════════════════════════════════════════════════════════════════════════════
print('Building master_overview_gallery.png ...')

ALL_ITEMS = (
    [(ROOT/s, VIZ/Path(s).name, t, p, 'influenza') for s, t, p in PLOTS['influenza']] +
    [(ROOT/s, VIZ/Path(s).name, t, p, 'agentic')   for s, t, p in PLOTS['agentic']]   +
    [(ROOT/s, VIZ/Path(s).name, t, p, 'mda')        for s, t, p in PLOTS['mda']]
)

SECTION_COLORS = {'influenza': BLUE, 'agentic': ORANGE, 'mda': PURPLE}
SECTION_LABELS = {
    'influenza': 'INFLUENZA PIPELINE  (Phases 1–5)',
    'agentic':   'AGENTIC DRIFT MONITORING',
    'mda':       'MDA TRANSFORMER  (Phase 8)',
}

NCOLS = 4
NROWS = int(np.ceil(len(ALL_ITEMS) / NCOLS))

fig = plt.figure(figsize=(26, NROWS * 5.8 + 2.4))
fig.patch.set_facecolor('#F0F3F4')

# ── Master title ────────────────────────────────────────────────────────────
title_ax = fig.add_axes([0.0, 0.965, 1.0, 0.038])
title_ax.set_facecolor(DARK)
title_ax.set_xticks([]); title_ax.set_yticks([])
for sp in title_ax.spines.values(): sp.set_visible(False)
title_ax.text(0.5, 0.52,
    'Influenza Mutation Analysis  ·  Agentic Drift Monitoring  ·  MDA Transformer',
    ha='center', va='center', fontsize=15, fontweight='bold', color='white',
    transform=title_ax.transAxes)
title_ax.text(0.98, 0.52,
    f'{len(ALL_ITEMS)} visualisations across 3 analysis modules  ·  incl. model comparison dashboard',
    ha='right', va='center', fontsize=8.5, color='#aaaaaa',
    transform=title_ax.transAxes)

# ── Section divider tracker ─────────────────────────────────────────────────
cell_h  = 0.88 / NROWS
cell_w  = 1.0  / NCOLS
img_h   = cell_h - 0.055

prev_section = None

for idx, (src, dest, label, phase, section) in enumerate(ALL_ITEMS):
    row = idx // NCOLS
    col = idx  % NCOLS
    color = SECTION_COLORS[section]

    # Section banner when section changes and col==0
    if section != prev_section and col == 0:
        by = 0.06 + (NROWS - row) * cell_h - 0.010
        ban = fig.add_axes([0.0, by, 1.0, 0.022])
        ban.set_facecolor(color)
        ban.set_xticks([]); ban.set_yticks([])
        for sp in ban.spines.values(): sp.set_visible(False)
        ban.text(0.012, 0.5, SECTION_LABELS[section],
                 transform=ban.transAxes, fontsize=10.5, fontweight='bold',
                 color='white', va='center')
    prev_section = section

    left   = col * cell_w + 0.006
    bottom = 0.06 + (NROWS - 1 - row) * cell_h + 0.030

    # Image
    ax = fig.add_axes([left, bottom, cell_w - 0.012, img_h])
    img = load_img(dest)
    if img is not None:
        ax.imshow(img, aspect='auto')
    else:
        ax.set_facecolor('#e0e0e0')
        ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.0); sp.set_edgecolor(color)

    # Caption
    cap = fig.add_axes([left, bottom - 0.028, cell_w - 0.012, 0.026])
    cap.set_facecolor(color)
    cap.set_xticks([]); cap.set_yticks([])
    for sp in cap.spines.values(): sp.set_visible(False)
    cap.text(0.04, 0.52,
             f'[{phase}]',
             transform=cap.transAxes, fontsize=7.5, color='#ffffffcc',
             fontweight='bold', va='center')
    cap.text(0.20, 0.52,
             label,
             transform=cap.transAxes, fontsize=8, color='white',
             va='center', clip_on=True)

# ── Bottom legend bar ────────────────────────────────────────────────────────
leg = fig.add_axes([0.0, 0.005, 1.0, 0.030])
leg.set_facecolor('#D5D8DC')
leg.set_xticks([]); leg.set_yticks([])
for sp in leg.spines.values(): sp.set_visible(False)
patches = [
    mpatches.Patch(facecolor=BLUE,   label='Influenza Pipeline (9 plots)'),
    mpatches.Patch(facecolor=ORANGE, label='Agentic Drift Monitoring (5 plots)'),
    mpatches.Patch(facecolor=PURPLE, label='MDA Transformer Phase 8 (6 plots)'),
]
leg.legend(handles=patches, loc='center', ncol=3, fontsize=9,
           frameon=False, labelcolor=DARK)

fig.savefig(VIZ / 'master_overview_gallery.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  saved master_overview_gallery.png\n')

# ── Final listing ────────────────────────────────────────────────────────────
all_out = sorted(VIZ.glob('*.png'))
print('='*62)
print(f' all_visualizations/  ({len(all_out)} files)')
print('='*62)
for f in all_out:
    kb = f.stat().st_size / 1024
    print(f'  {f.name:<52}  {kb:6.0f} KB')
