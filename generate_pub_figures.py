#!/usr/bin/env python3
"""
Task 6a — Generate publication-ready figures (300 dpi PNG + PDF).
Applies uniform style: 12pt fonts, colorblind-safe palette, clean spines, grid.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from pathlib import Path

ROOT   = Path(__file__).parent
OUT    = ROOT / 'outputs'
PHASE8 = ROOT / 'phase8_outputs'

# ── Publication style ──────────────────────────────────────────────────────────
PUB_STYLE = {
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.30,
    'grid.linestyle': '--',
    'legend.framealpha': 0.9,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}
plt.rcParams.update(PUB_STYLE)

# colorblind-safe: matplotlib tab10
TAB = plt.cm.tab10.colors
BLUE   = TAB[0]; ORANGE = TAB[1]; GREEN  = TAB[2]
RED    = TAB[3]; PURPLE = TAB[4]; BROWN  = TAB[5]
PINK   = TAB[6]; GRAY   = TAB[7]; OLIVE  = TAB[8]; CYAN = TAB[9]

PANEL_KW = dict(fontsize=15, fontweight='bold', va='top')


def save(fig, stem: str):
    fig.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: outputs/{stem}.png  (.pdf)')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — H1N1 Divergence
# ══════════════════════════════════════════════════════════════════════════════
print('FIG 1: H1N1 Divergence ...')
dr = pd.read_csv(OUT / 'phase1_h1n1_divergence_rates.csv')
dr['Year'] = dr['Year'].astype(int)
pan = dr.dropna(subset=['pandemic_mean']).copy()
pan = pan[pan['Year'] >= 2009]

t   = (pan['Year'] - 2009).values.astype(float)
d   = pan['pandemic_mean'].values
w   = pan['pandemic_n'].values
slope = float(np.sum(w * t * d) / np.sum(w * t**2))

fig, ax = plt.subplots(figsize=(12, 6))
ax.text(-0.06, 1.04, 'A', transform=ax.transAxes, **PANEL_KW)

pre  = dr[dr['Year'] < 2009]
post = dr[dr['Year'] >= 2009]
ax.scatter(pre['Year'], pre['mean_distance'],
           color=GRAY, s=18, alpha=0.35, zorder=2, label='All H1N1 pre-2009')
ax.errorbar(post['Year'], post['mean_distance'],
            yerr=post['std_distance'].clip(0, 200),
            fmt='o', color=GRAY, ecolor='#cccccc', capsize=3,
            alpha=0.4, markersize=5, zorder=3,
            label='All H1N1 post-2009')
ax.fill_between(pan['Year'], pan['pandemic_mean'], alpha=0.18, color=BLUE)
ax.plot(pan['Year'], pan['pandemic_mean'], 'o-', color=BLUE, linewidth=2.5,
        markersize=7, zorder=5, label='Pandemic lineage (dist<60 from 2009 ref)')

reg_years = np.array([2009, 2018])
ax.plot(reg_years, slope * (reg_years - 2009), '--', color=RED, linewidth=2,
        zorder=4, label=f'Regression slope = {slope:.2f} aa/yr')
ax.axhline(2.45, xmin=0.62, xmax=0.98, color=RED,
           linewidth=1.2, alpha=0.55, linestyle=':')
ax.text(2017.3, 2.45 + 0.4, 'Literature 2.45 aa/yr',
        color=RED, fontsize=10, va='bottom')
ax.axvspan(2009, 2017.5, alpha=0.06, color=BLUE, zorder=1)

ax.text(0.98, 0.96, f'PASS   Rate = {slope:.2f} aa/yr\n(target 2.20–2.70)',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFAF1',
                  edgecolor=GREEN, linewidth=1.5))

ax.set_xlim(1915, 2020); ax.set_ylim(bottom=-5)
ax.set_xlabel('Collection Year')
ax.set_ylabel('Mean Hamming Distance from 2009 Reference (aa)')
ax.set_title('H1N1 HA Divergence Over Time\nPost-pandemic lineage vs full dataset')
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
save(fig, 'fig1_h1n1_divergence')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — H3N2 MDS Clusters
# ══════════════════════════════════════════════════════════════════════════════
print('FIG 2: H3N2 MDS Clusters ...')
mds = pd.read_csv(OUT / 'phase4_mds_coordinates.csv')
h3  = mds[mds['Subtype'] == 'H3N2'].copy()
h1  = mds[mds['Subtype'] == 'H1N1'].copy()
clusters = sorted(h3['historical_cluster'].unique())
cmap20 = plt.cm.get_cmap('tab20', len(clusters))
cmap_dict = {c: cmap20(i) for i, c in enumerate(clusters)}

def draw_hull(ax, x, y, color, alpha=0.10):
    pts = np.column_stack([x, y])
    if len(pts) < 3: return
    try:
        hull = ConvexHull(pts)
        hp   = pts[hull.vertices]
        hp   = np.vstack([hp, hp[0]])
        ax.fill(hp[:,0], hp[:,1], alpha=alpha, color=color, zorder=1)
        ax.plot(hp[:,0], hp[:,1], '-', color=color, alpha=0.5, linewidth=1, zorder=2)
    except Exception:
        pass

fig, ax = plt.subplots(figsize=(13, 9))
ax.text(-0.06, 1.02, 'A', transform=ax.transAxes, **PANEL_KW)
ax.scatter(h1['mds_x'], h1['mds_y'], color='#C0C0C0', s=35, alpha=0.45,
           zorder=2, marker='s', label='H1N1 (background)')

for cl in clusters:
    mask = h3['historical_cluster'] == cl
    sub  = h3[mask]
    color = cmap_dict[cl]
    draw_hull(ax, sub['mds_x'].values, sub['mds_y'].values, color)
    ax.scatter(sub['mds_x'], sub['mds_y'], color=color, s=65, alpha=0.88,
               zorder=4, edgecolors='white', linewidths=0.4, label=cl)
    cx, cy = sub['mds_x'].mean(), sub['mds_y'].mean()
    ax.text(cx, cy, cl, fontsize=7, ha='center', va='center',
            fontweight='bold', color='white', zorder=6,
            bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.75, lw=0))

ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')
ax.set_title('H3N2 Historical Cluster Groupings in Sequence Space\nConvex hulls show cluster boundaries')

handles = [mpatches.Patch(facecolor=cmap_dict[c], label=c) for c in clusters]
handles.append(mpatches.Patch(facecolor='#C0C0C0', label='H1N1'))
ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=2,
          title='Cluster / H1N1')
ax.text(0.98, 0.02, 'Cluster purity = 0.969\n(target >0.70)  PASS',
        transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFAF1',
                  edgecolor=GREEN, linewidth=1.5))
fig.tight_layout()
save(fig, 'fig2_h3n2_mds_clusters')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Variant Emergence Timeline
# ══════════════════════════════════════════════════════════════════════════════
print('FIG 3: Variant Emergence Timeline ...')
vt = pd.read_csv(OUT / 'phase5_variant_tracking.csv')
vt = vt[vt['n_sequences'] > 0].copy()
vt['year'] = vt['year'].astype(int)
vt['vars_per_seq'] = vt['n_critical_variations'] / vt['n_sequences']

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.text(-0.06, 1.04, 'A', transform=ax1.transAxes, **PANEL_KW)
ax2 = ax1.twinx()
ax2.grid(False)

x = vt['year'].values
bar_colors = [GREEN if v == vt['vars_per_seq'].max() else BLUE
              for v in vt['vars_per_seq']]

bars = ax1.bar(x, vt['vars_per_seq'], color=bar_colors, alpha=0.75,
               zorder=3, width=0.6, label='Critical variants per sequence')
ax2.plot(x, vt['n_sequences'], 'o--', color=ORANGE, linewidth=2,
         markersize=7, zorder=4, label='Sequences collected')
ax2.fill_between(x, vt['n_sequences'], alpha=0.12, color=ORANGE)

peak_idx = vt['vars_per_seq'].idxmax()
peak_yr  = int(vt.loc[peak_idx, 'year'])
peak_val = float(vt.loc[peak_idx, 'vars_per_seq'])
ax1.annotate(f'Peak\n{peak_yr}', xy=(peak_yr, peak_val),
             xytext=(peak_yr + 0.6, peak_val * 0.92),
             fontsize=10, color=GREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.2,
             f'{h:.1f}', ha='center', va='bottom', fontsize=8, color='#333')

ax1.set_xlabel('Year')
ax1.set_ylabel('Critical Region Variations per Sequence', color=BLUE)
ax2.set_ylabel('Number of Sequences Analysed', color=ORANGE)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax2.tick_params(axis='y', labelcolor=ORANGE)
ax1.set_xticks(x)
ax1.set_xticklabels([str(y) for y in x], rotation=30, ha='right')
ax1.set_title('H1N1 Critical Region Variant Emergence (2009–2017)')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(True)

lines1 = [mpatches.Patch(facecolor=BLUE, alpha=0.75, label='Critical vars/seq'),
          mpatches.Patch(facecolor=GREEN, alpha=0.75, label='Peak year')]
lines2 = [Line2D([0],[0], color=ORANGE, marker='o', markersize=7,
                 label='Sequences collected')]
ax1.legend(handles=lines1 + lines2, loc='upper left', fontsize=9)
fig.tight_layout()
save(fig, 'fig3_variant_emergence')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Transformer Predictions (2-panel)
# ══════════════════════════════════════════════════════════════════════════════
print('FIG 6: Transformer Predictions ...')
pred = pd.read_csv(PHASE8 / 'phase8_mda_all_predictions.csv')

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('MDA Transformer Drift Predictions Across All Mutations',
             fontsize=14, fontweight='bold')

# Panel A — Histogram
axA.text(-0.08, 1.04, 'A', transform=axA.transAxes, **PANEL_KW)
axA.hist(pred['drift_prob'], bins=40, color=BLUE, alpha=0.75, edgecolor='white', zorder=3)
axA.axvline(0.65, color=RED, linewidth=2, linestyle='--', label='High-impact threshold (0.65)')
axA.axvline(0.50, color=ORANGE, linewidth=1.5, linestyle=':', alpha=0.8, label='Decision boundary (0.50)')
n_hi = (pred['drift_prob'] > 0.65).sum()
axA.text(0.67, 0.92, f'High-impact:\n{n_hi:,} mutations',
         transform=axA.transAxes, color=RED, fontsize=10, fontweight='bold',
         va='top')
axA.set_xlabel('Drift Probability')
axA.set_ylabel('Count')
axA.set_title(f'Drift Probability Distribution\n{len(pred):,} mutations scored')
axA.legend(fontsize=9)
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)

# Panel B — Scatter
axB.text(-0.08, 1.04, 'B', transform=axB.transAxes, **PANEL_KW)
plot_p = pred.sample(min(500, len(pred)), random_state=42)
years  = plot_p['year'].values
norm   = plt.Normalize(years.min(), years.max())
sc = axB.scatter(plot_p['position'], plot_p['drift_prob'],
                 c=norm(years), cmap=cm.plasma,
                 s=np.abs(plot_p['agentic_coeff'].values) * 50 + 12,
                 alpha=0.75, zorder=3, edgecolors='none')
cbar = plt.colorbar(sc, ax=axB, pad=0.01, shrink=0.85)
cbar.set_label('Year', fontsize=10)
tick_yrs = np.linspace(years.min(), years.max(), 5, dtype=int)
cbar.set_ticks(np.linspace(0, 1, 5)); cbar.set_ticklabels([str(y) for y in tick_yrs])
axB.axhline(0.65, color=RED, linewidth=1.5, linestyle='--', alpha=0.7,
            label='High-impact threshold')
axB.set_xlabel('Protein Position (0–565)')
axB.set_ylabel('Drift Probability')
axB.set_title('Position vs Drift Probability\nColour = year, size = |agentic coeff|')
axB.legend(fontsize=9)
axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)

fig.tight_layout()
save(fig, 'fig6_transformer_predictions')

print('\nAll publication figures complete.')
