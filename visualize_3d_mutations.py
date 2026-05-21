"""
3D Visualizations — Top 10 Influenza HA Mutations
Outputs four PNG files to the outputs directory.
"""
import sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from pathlib import Path

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR  = Path("C:/Users/UseR/outputs")
VIS_DIR     = Path("C:/Users/UseR/Downloads/InfluenzaXmutation/all_visualizations")
VIS_DIR.mkdir(exist_ok=True)

# ─── Global dark theme ───────────────────────────────────────────────────────
BG      = '#0D1117'
GRID_C  = '#1E2530'
TICK_C  = '#AAAAAA'
TEXT_C  = 'white'
BLUE    = '#2196F3'
RED     = '#EF5350'
ORANGE  = '#FF8F00'
GREEN   = '#66BB6A'
PURPLE  = '#AB47BC'

def dark_axes(ax3d, title, xlabel, ylabel, zlabel):
    ax3d.set_facecolor(BG)
    ax3d.set_title(title, fontsize=12, fontweight='bold', color=TEXT_C, pad=18)
    ax3d.set_xlabel(xlabel, fontsize=9, color=TEXT_C, labelpad=10)
    ax3d.set_ylabel(ylabel, fontsize=9, color=TEXT_C, labelpad=10)
    ax3d.set_zlabel(zlabel, fontsize=9, color=TEXT_C, labelpad=10)
    ax3d.tick_params(colors=TICK_C, labelsize=7)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_C)
    ax3d.xaxis.line.set_color(GRID_C)
    ax3d.yaxis.line.set_color(GRID_C)
    ax3d.zaxis.line.set_color(GRID_C)

def dark_colorbar(fig, ax3d, sm, label):
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.45, pad=0.10, aspect=20)
    cbar.set_label(label, color=TEXT_C, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TICK_C, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TICK_C)
    return cbar

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading mutation data...")
# mutation_frequency.csv has all columns including Frequency, BLOSUM62, entropy
mut_df   = pd.read_csv(OUTPUT_DIR / "mutation_frequency.csv")
trend_df = pd.read_csv(OUTPUT_DIR / "mutation_trends_by_year.csv")

for col in ['In_Antigenic_Site', 'In_RBS', 'Known_Virulence_Marker', 'Conservative']:
    mut_df[col] = mut_df[col].astype(bool)

# Top 5 per subtype → balanced top 10
top5_h1 = mut_df[mut_df['Subtype'] == 'H1N1'].nlargest(5, 'Count').copy()
top5_h3 = mut_df[mut_df['Subtype'] == 'H3N2'].nlargest(5, 'Count').copy()
top10 = pd.concat([top5_h1, top5_h3]).reset_index(drop=True)
top10['Label']  = top10['WT_AA'] + top10['Position'].astype(str) + top10['Mutant_AA']
top10['FullLbl']= top10['Label'] + '\n(' + top10['Subtype'] + ')'
top10['MutKey'] = (top10['Position'].astype(str) + top10['WT_AA'] + '>' + top10['Mutant_AA'])

print(f"\nTop 10 mutations (top 5 per subtype):")
print(f"  {'#':>2}  {'Subtype':>6}  {'Mutation':>8}  {'Count':>7}  "
      f"{'Freq':>6}  {'Antigenic':>9}  {'RBS':>4}  {'BLOSUM62':>8}  {'Entropy':>7}")
print("  " + "-"*72)
for i, r in top10.iterrows():
    print(f"  {i+1:>2}  {r.Subtype:>6}  {r.Label:>8}  {r.Count:>7,}  "
          f"{r.Frequency:>6.4f}  {'Y' if r.In_Antigenic_Site else 'N':>9}  "
          f"{'Y' if r.In_RBS else 'N':>4}  {r.BLOSUM62_Score:>8}  {r.Position_Entropy:>7.3f}")

def bar_color(row, idx):
    """Color bars: H1N1=blue family, H3N2=red family; antigenic site=orange."""
    if row['In_Antigenic_Site']:
        return ORANGE
    h1_palette = ['#1565C0','#1976D2','#1E88E5','#42A5F5','#82B1FF']
    h3_palette = ['#B71C1C','#C62828','#D32F2F','#E53935','#FF5252']
    rank = idx % 5
    return h1_palette[rank] if row['Subtype'] == 'H1N1' else h3_palette[rank]

top10['BarColor'] = [bar_color(r, i) for i, r in top10.iterrows()]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: 3D Bar Chart — Mutation × Count, coloured by BLOSUM62
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering Plot 1 — 3D Bar Chart...")
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection='3d')

blosum_vals = top10['BLOSUM62_Score'].values.astype(float)
norm_b  = mcolors.Normalize(vmin=blosum_vals.min() - 0.5, vmax=blosum_vals.max() + 0.5)
cmap_b  = cm.RdYlGn

dx = dy = 0.5
for i, row in top10.iterrows():
    x = i
    y = 0
    z = float(row['Count'])
    col = cmap_b(norm_b(float(row['BLOSUM62_Score'])))
    ax.bar3d(x - dx/2, y - dy/2, 0, dx, dy, z,
             color=col, alpha=0.90, shade=True, zsort='average')
    # Value label floating above bar
    ax.text(x, 0, z + z * 0.04,
            f"{row['Label']}\n{int(z):,}",
            ha='center', va='bottom', fontsize=7, color=TEXT_C, fontweight='bold')

dark_axes(ax,
          'Top 10 Mutations — 3D Bar Chart\nColour = BLOSUM62 score  (green = conservative, red = disruptive)',
          'Mutation Index', '', 'Observation Count')

ax.set_xticks(range(10))
ax.set_xticklabels(top10['Label'].tolist(), rotation=35, ha='right',
                   fontsize=7.5, color=TICK_C)
ax.set_yticks([])
ax.set_ylim(-1, 1)

# Subtype region labels
ax.text2D(0.12, 0.04, 'H1N1 top 5 →', transform=ax.transAxes,
          fontsize=9, color=BLUE, fontstyle='italic')
ax.text2D(0.58, 0.04, 'H3N2 top 5 →', transform=ax.transAxes,
          fontsize=9, color=RED, fontstyle='italic')

sm = cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
sm.set_array([])
dark_colorbar(fig, ax, sm, 'BLOSUM62 Score')

# Antigenic site marker legend
antigen_patch = mpatches.Patch(color=ORANGE, label='In antigenic site')
ax.legend(handles=[antigen_patch], loc='upper right', fontsize=9,
          facecolor='#1A1A2E', edgecolor='#444', labelcolor=TEXT_C)

ax.view_init(elev=28, azim=-50)
fig.tight_layout()
fig.savefig(VIS_DIR / 'top10_mutations_3d_bar.png', dpi=300,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ top10_mutations_3d_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: 3D Scatter Landscape — Position × Entropy × Frequency
# All mutations background; top 10 highlighted
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Plot 2 — 3D Scatter Landscape...")
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection='3d')

# Background scatter (all mutations, colour by subtype)
for st, col in [('H1N1', '#1565C0'), ('H3N2', '#B71C1C')]:
    sub = mut_df[mut_df['Subtype'] == st]
    ax.scatter(sub['Position'], sub['Position_Entropy'], sub['Frequency'],
               c=col, s=4, alpha=0.14, depthshade=True, zorder=2)

# Top 10 foreground
for i, row in top10.iterrows():
    fc = ORANGE if row['In_Antigenic_Site'] else (BLUE if row['Subtype'] == 'H1N1' else RED)
    ec = 'white'
    ax.scatter([row['Position']], [row['Position_Entropy']], [row['Frequency']],
               c=fc, s=220, alpha=0.97, edgecolors=ec, linewidths=0.9, zorder=8)
    offset_z = 0.04
    ax.text(row['Position'], row['Position_Entropy'], row['Frequency'] + offset_z,
            row['Label'], ha='center', va='bottom', fontsize=7.5,
            color=TEXT_C, fontweight='bold')

dark_axes(ax,
          'Mutation Landscape — 3D Scatter\nAll mutations (faded) + Top 10 highlighted',
          'HA Position', 'Position Entropy (bits)', 'Mutation Frequency')

legend_elems = [
    mpatches.Patch(color=BLUE,   alpha=0.8, label='H1N1 (all)'),
    mpatches.Patch(color=RED,    alpha=0.8, label='H3N2 (all)'),
    mpatches.Patch(color=BLUE,   label='H1N1 top 5'),
    mpatches.Patch(color=RED,    label='H3N2 top 5'),
    mpatches.Patch(color=ORANGE, label='Antigenic site'),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=8.5,
          facecolor='#1A1A2E', edgecolor='#444', labelcolor=TEXT_C)

ax.view_init(elev=22, azim=42)
fig.tight_layout()
fig.savefig(VIS_DIR / 'top10_mutations_3d_landscape.png', dpi=300,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ top10_mutations_3d_landscape.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: 3D Temporal Surface — Year × Mutation × Frequency
# Two sub-surfaces: H1N1 (2009–2017) and H3N2 (1968–2010)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Plot 3 — 3D Temporal Surface...")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)

for panel_idx, (st, year_lo, year_hi, cmap_name, title_suffix) in enumerate([
    ('H1N1', 2009, 2017, 'Blues',   'H1N1 (2009–2017)'),
    ('H3N2', 1968, 2010, 'Reds',    'H3N2 (1968–2010)'),
]):
    ax = fig.add_subplot(1, 2, panel_idx + 1, projection='3d')

    st_muts  = top10[top10['Subtype'] == st].copy().reset_index(drop=True)
    mut_keys = st_muts['MutKey'].tolist()
    mut_lbls = st_muts['Label'].tolist()

    years = sorted(trend_df[
        (trend_df['Subtype'] == st) & trend_df['Year'].between(year_lo, year_hi)
    ]['Year'].unique().astype(int))

    if not years or not mut_keys:
        continue

    freq_mat = np.zeros((len(years), len(mut_keys)))
    for j, mkey in enumerate(mut_keys):
        for i, yr in enumerate(years):
            row = trend_df[
                (trend_df['Subtype'] == st) &
                (trend_df['Year']    == yr) &
                (trend_df['Mutation']== mkey)
            ]
            if not row.empty:
                freq_mat[i, j] = float(row['Frequency'].iloc[0])

    X_idx = np.arange(len(mut_keys))
    Y_idx = np.arange(len(years))
    XX, YY = np.meshgrid(X_idx, Y_idx)
    ZZ = freq_mat

    norm_z = mcolors.Normalize(vmin=0, vmax=max(ZZ.max(), 0.01))
    cmap   = cm.get_cmap(cmap_name)
    surf   = ax.plot_surface(XX, YY, ZZ,
                              facecolors=cmap(norm_z(ZZ)),
                              alpha=0.88, shade=True,
                              linewidth=0, antialiased=True)
    ax.plot_wireframe(XX, YY, ZZ, color='white', alpha=0.06, linewidth=0.35)

    ax.set_xticks(X_idx)
    ax.set_xticklabels(mut_lbls, rotation=30, ha='right', fontsize=7, color=TICK_C)
    step = max(1, len(years) // 6)
    ax.set_yticks(Y_idx[::step])
    ax.set_yticklabels([str(years[i]) for i in range(0, len(years), step)],
                       fontsize=7, color=TICK_C)
    dark_axes(ax, f'Temporal Frequency Surface\n{title_suffix}',
              'Mutation', 'Year', 'Frequency')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm_z)
    sm.set_array([])
    dark_colorbar(fig, ax, sm, 'Frequency')
    ax.view_init(elev=30, azim=-55)

fig.suptitle('Top 5 Mutations per Subtype — Temporal Frequency Surfaces',
             fontsize=14, fontweight='bold', color=TEXT_C, y=1.01)
fig.tight_layout()
fig.savefig(VIS_DIR / 'top10_mutations_3d_temporal.png', dpi=300,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ top10_mutations_3d_temporal.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: 3D Stem Plot — H1N1 vs H3N2, position × count
# Each mutation drawn as a vertical spike with a sphere cap
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Plot 4 — 3D Stem / Spike Plot...")
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection='3d')

# Y-plane: H1N1 = 0, H3N2 = 2  (spread apart for clarity)
subtype_y = {'H1N1': 0.0, 'H3N2': 2.0}

for i, row in top10.iterrows():
    x  = float(row['Position'])
    y  = subtype_y[row['Subtype']]
    z  = float(row['Count'])
    fc = row['BarColor']

    # Drop shadow on floor
    ax.plot([x, x], [y, y], [0, 0], color=fc, lw=2.5, alpha=0.18)
    # Vertical spike
    ax.plot([x, x], [y, y], [0, z], color=fc, lw=2.2, alpha=0.80, zorder=3)
    # Sphere cap
    ax.scatter([x], [y], [z], c=fc, s=300, alpha=0.97,
               edgecolors='white', linewidths=0.9, zorder=7, depthshade=False)
    # Label above sphere
    ax.text(x, y, z + float(top10['Count'].max()) * 0.045,
            f"{row['Label']}\n{int(z):,}",
            ha='center', va='bottom', fontsize=7.5, color=TEXT_C, fontweight='bold')

# Subtype plane labels
z_mid = top10['Count'].mean()
ax.text(top10[top10.Subtype=='H1N1']['Position'].mean(), 0, z_mid * 1.55,
        'H1N1', ha='center', fontsize=13, color=BLUE, fontweight='bold', alpha=0.75)
ax.text(top10[top10.Subtype=='H3N2']['Position'].mean(), 2, z_mid * 1.55,
        'H3N2', ha='center', fontsize=13, color=RED, fontweight='bold', alpha=0.75)

dark_axes(ax,
          'Top 10 Mutations — 3D Spike Plot\nH1N1 vs H3N2 separated by subtype plane',
          'HA Position', 'Subtype Plane', 'Observation Count')
ax.set_yticks([0, 2])
ax.set_yticklabels(['H1N1', 'H3N2'], fontsize=10, color=TICK_C)
ax.set_zlim(0, top10['Count'].max() * 1.20)

legend_elems = [
    mpatches.Patch(color=BLUE,   label='H1N1 (non-antigenic)'),
    mpatches.Patch(color=RED,    label='H3N2 (non-antigenic)'),
    mpatches.Patch(color=ORANGE, label='In antigenic site'),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=9,
          facecolor='#1A1A2E', edgecolor='#444', labelcolor=TEXT_C)

ax.view_init(elev=22, azim=25)
fig.tight_layout()
fig.savefig(VIS_DIR / 'top10_mutations_3d_stem.png', dpi=300,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ top10_mutations_3d_stem.png")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"""
{'='*65}
4 × 3D PNG files saved to {VIS_DIR}

  top10_mutations_3d_bar.png       — bar chart: count, BLOSUM62 colour
  top10_mutations_3d_landscape.png — scatter: position × entropy × freq
  top10_mutations_3d_temporal.png  — surface: year × mutation × freq
  top10_mutations_3d_stem.png      — spike: H1N1 vs H3N2 by position
{'='*65}
""")
