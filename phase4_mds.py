"""
Phase 4: Antigenic Map Visualization via MDS
Tasks 4.1–4.5
All sequences are amino-acid HA proteins (540-570 aa, modal 566).
'Complete' = within 5% of modal length (538–570 aa); no 800-aa threshold applies.
"""
import sys, io, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from collections import Counter
from datetime import datetime
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = "C:/Users/UseR/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def out(f): return os.path.join(OUTPUT_DIR, f)

np.random.seed(42)

# ─── Colour palettes ──────────────────────────────────────────────────────────
SMITH_COLORS = {
    "HK68":"#e41a1c","EN72":"#ff7f00","VI75":"#e6b800","TX77":"#4daf4a",
    "BK79":"#984ea3","SI87":"#a65628","BE89":"#f781bf","BE92":"#377eb8",
    "WU95":"#17becf","SY97":"#2ca02c","FU02":"#8c564b","OTHER":"#aaaaaa",
}
H1N1_LIN_COLORS = {"Pandemic1918":"#e41a1c","Seasonal1977":"#ff7f00","H1N1pdm09":"#377eb8"}
HOST_COLORS     = {"Human":"#1976D2","Avian":"#D32F2F","Other":"#757575"}

def h1n1_lineage(year):
    try: yr = int(year)
    except: return "Unknown"
    if yr <= 1957:  return "Pandemic1918"
    if yr <= 2008:  return "Seasonal1977"
    return "H1N1pdm09"

# ─── Helper: pairwise Hamming distance matrix (vectorised, shared-prefix) ────
def pairwise_hamming(seqs):
    """Returns (N,N) matrix of hamming distances as float32."""
    N = len(seqs)
    # Convert to array; pad shorter sequences with '_' (won't match any AA)
    max_len = max(len(s) for s in seqs)
    arr = np.full((N, max_len), '_', dtype='U1')
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = list(s)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        mismatches = (arr[i] != arr).sum(axis=1).astype(np.float32)
        # Normalise by comparison length = min of two actual lengths
        lens = np.array([min(len(seqs[i]), len(seqs[j])) for j in range(N)], dtype=np.float32)
        D[i] = mismatches / np.maximum(lens, 1)
    return D

# ─── Greedy farthest-point diversity selection ───────────────────────────────
def greedy_select(seqs_df, n, rng=None):
    """From seqs_df (DataFrame with 'Sequence'), pick n maximally diverse rows."""
    if len(seqs_df) <= n:
        return seqs_df.copy()
    if rng is None:
        rng = np.random.default_rng(42)
    # Candidate pool: all if ≤ 200, else random 200
    pool = seqs_df if len(seqs_df) <= 200 else seqs_df.sample(200, random_state=42)
    pool = pool.reset_index(drop=True)
    seqs = pool['Sequence'].tolist()
    N    = len(seqs)
    D    = pairwise_hamming(seqs)        # full pool pairwise distances

    # Start with most-common sequence (lowest mean distance to others)
    start = int(np.argmin(D.mean(axis=1)))
    selected_idx = [start]
    remaining    = list(range(N))
    remaining.remove(start)

    while len(selected_idx) < min(n, N):
        min_dists = D[remaining][:, selected_idx].min(axis=1)
        pick      = remaining[int(np.argmax(min_dists))]
        selected_idx.append(pick)
        remaining.remove(pick)

    return pool.iloc[selected_idx].copy()

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.1 — Select Representative Strains
# ══════════════════════════════════════════════════════════════════════════════
print("Loading sequences...")
h1  = pd.read_csv('C:/Users/UseR/outputs/h1n1_filtered_sequences.csv', low_memory=False)
h3  = pd.read_csv('C:/Users/UseR/outputs/h3n2_filtered_sequences.csv', low_memory=False)
sc  = pd.read_csv('C:/Users/UseR/outputs/h3n2_smith_clusters.csv')

for df in [h1, h3]:
    df['Sequence'] = df['Sequence'].astype(str).str.strip().str.upper()
    df['Year']     = pd.to_numeric(df['Year'], errors='coerce')

# Merge Smith clusters onto H3N2
h3 = h3.merge(sc[['Accession','Smith_Cluster']], on='Accession', how='left')
h3['Smith_Cluster'] = h3['Smith_Cluster'].fillna('OTHER')

# Quality filter: within 5% of modal length (538-570 aa)
MODAL = 566
MIN_LEN, MAX_LEN = int(MODAL * 0.95), int(MODAL * 1.05)
h1 = h1[h1['Sequence'].str.len().between(MIN_LEN, MAX_LEN)].copy()
h3 = h3[h3['Sequence'].str.len().between(MIN_LEN, MAX_LEN)].copy()
print(f"  H1N1 after quality filter: {len(h1):,}")
print(f"  H3N2 after quality filter: {len(h3):,}")

print("\n✓ Task 4.1 — Selecting representative strains...")

# ── H1N1: ~40 strains across three lineages ──
h1['Lineage'] = h1['Year'].apply(h1n1_lineage)
h1_rep_parts  = []

lineage_quotas = {
    'Pandemic1918': 10,   # ~50 seqs, pick 10 spread across 1918-1957
    'Seasonal1977': 12,   # ~206 seqs, pick 12 spread across 1976-2008
    'H1N1pdm09'  : 20,   # ~6000 seqs, pick 20 across 2009-2017
}
for lin, quota in lineage_quotas.items():
    pool = h1[h1['Lineage'] == lin].dropna(subset=['Year'])
    if len(pool) == 0:
        continue
    # Within-lineage: sample by year strata for even temporal coverage
    years = sorted(pool['Year'].unique().astype(int))
    # Distribute quota across years
    per_yr = max(1, quota // len(years))
    yr_samples = []
    for yr in years:
        yp = pool[pool['Year'] == yr]
        n  = min(per_yr, len(yp))
        yr_samples.append(yp.sample(min(n*3, len(yp)), random_state=42))  # 3× candidates
    candidates = pd.concat(yr_samples).drop_duplicates('Accession')
    selected   = greedy_select(candidates, quota)
    selected['Lineage'] = lin
    h1_rep_parts.append(selected)

h1_rep = pd.concat(h1_rep_parts).drop_duplicates('Accession').reset_index(drop=True)
print(f"  H1N1 representatives: {len(h1_rep)}")
for lin, g in h1_rep.groupby('Lineage'):
    print(f"    {lin}: {len(g)}  years {int(g.Year.min())}-{int(g.Year.max())}")

# ── H3N2: ~88 strains across Smith clusters + post-2004 ──
h3_rep_parts = []
smith_quotas = {
    'HK68':8,'EN72':6,'VI75':6,'TX77':5,'BK79':8,'SI87':5,
    'BE89':7,'BE92':8,'WU95':8,'SY97':8,'FU02':8,
}
for cluster, quota in smith_quotas.items():
    pool = h3[h3['Smith_Cluster'] == cluster].dropna(subset=['Year'])
    if len(pool) == 0:
        continue
    selected = greedy_select(pool, quota)
    selected['Smith_Cluster'] = cluster
    h3_rep_parts.append(selected)

# Post-2004 OTHER: pick 1 per year 2005-2020
other = h3[(h3['Smith_Cluster'] == 'OTHER') & h3['Year'].between(2005, 2020)]
for yr in sorted(other['Year'].dropna().unique().astype(int)):
    yp   = other[other['Year'] == yr]
    pick = greedy_select(yp, 1)
    pick['Smith_Cluster'] = 'OTHER'
    h3_rep_parts.append(pick)

h3_rep = pd.concat(h3_rep_parts).drop_duplicates('Accession').reset_index(drop=True)
h3_rep['Lineage'] = h3_rep['Smith_Cluster']
print(f"  H3N2 representatives: {len(h3_rep)}")
for cl, g in h3_rep.groupby('Smith_Cluster'):
    print(f"    {cl}: {len(g)}  years {int(g.Year.min())}-{int(g.Year.max())}")

# Save
rep_all = pd.concat([
    h1_rep[['Accession','Year','Subtype','VirusName','Lineage']].assign(Subtype='H1N1'),
    h3_rep[['Accession','Year','Subtype','VirusName','Lineage','Smith_Cluster']].assign(Subtype='H3N2'),
], ignore_index=True)
rep_all.to_csv(out("representative_strains.csv"), index=False)
print(f"\n✓ Saved representative_strains.csv ({len(rep_all)} total)")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.2 — Pairwise Distance Matrices
# ══════════════════════════════════════════════════════════════════════════════
print("\n✓ Task 4.2 — Computing pairwise distance matrices...")

def compute_and_save_dist(rep_df, subtype, label):
    seqs = rep_df['Sequence'].tolist()
    accs = rep_df['Accession'].tolist()
    print(f"  {subtype}: {len(seqs)} sequences...", end=' ', flush=True)
    D = pairwise_hamming(seqs)
    # Enforce symmetry and zero diagonal
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    df_dist = pd.DataFrame(D, index=accs, columns=accs)
    df_dist.to_csv(out(f"distance_matrix_{label}.csv"))
    print(f"shape {D.shape}  max={D.max():.3f}  mean={D.mean():.3f}")
    return D, accs

D_h1, acc_h1 = compute_and_save_dist(h1_rep, 'H1N1', 'h1n1')
D_h3, acc_h3 = compute_and_save_dist(h3_rep, 'H3N2', 'h3n2')
print("✓ Saved distance_matrix_h1n1.csv and distance_matrix_h3n2.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.3 — MDS Coordinates
# ══════════════════════════════════════════════════════════════════════════════
print("\n✓ Task 4.3 — Applying MDS (2 components)...")

def run_mds(D, rep_df, accs, subtype):
    mds = MDS(n_components=2, dissimilarity='precomputed',
              normalized_stress='auto', random_state=42,
              n_init=10, max_iter=1000, eps=1e-6)
    coords = mds.fit_transform(D)
    raw_stress = mds.stress_
    # Normalised Kruskal stress-1
    norm_stress = np.sqrt(raw_stress / (0.5 * (D**2).sum()))
    print(f"  {subtype}: raw_stress={raw_stress:.4f}  "
          f"normalised_stress={norm_stress:.4f}  "
          f"({'excellent' if norm_stress<0.05 else 'good' if norm_stress<0.10 else 'fair' if norm_stress<0.20 else 'poor'})")
    result = rep_df.copy().reset_index(drop=True)
    result['MDS1'] = coords[:, 0]
    result['MDS2'] = coords[:, 1]
    result['norm_stress'] = norm_stress
    return result, norm_stress

mds_h1, stress_h1 = run_mds(D_h1, h1_rep, acc_h1, 'H1N1')
mds_h3, stress_h3 = run_mds(D_h3, h3_rep, acc_h3, 'H3N2')

mds_h1[['Accession','MDS1','MDS2','Year','Subtype']].assign(Subtype='H1N1')\
      .to_csv(out("mds_coordinates_h1n1.csv"), index=False)
mds_h3[['Accession','MDS1','MDS2','Year','Subtype']].assign(Subtype='H3N2')\
      .to_csv(out("mds_coordinates_h3n2.csv"), index=False)
print("✓ Saved mds_coordinates_h1n1.csv and mds_coordinates_h3n2.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.4 — Antigenic Map Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\n✓ Task 4.4 — Generating antigenic maps...")

def add_style(ax, title, stress):
    ax.set_xlabel('MDS Dimension 1', fontsize=11)
    ax.set_ylabel('MDS Dimension 2', fontsize=11)
    ax.set_title(f'{title}\nStress = {stress:.4f}', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#FAFAFA')

def draw_temporal_arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2, alpha=0.5,
                                connectionstyle='arc3,rad=0.05'))

def plot_by_year(mds_df, subtype, stress, fname, color_col='Year',
                 show_arrows=True, label_every_n=1):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FAFAFA')

    years = mds_df['Year'].dropna().astype(int)
    yr_min, yr_max = int(years.min()), int(years.max())
    norm = mcolors.Normalize(vmin=yr_min, vmax=yr_max)
    cmap = cm.plasma

    sc = ax.scatter(mds_df['MDS1'], mds_df['MDS2'],
                    c=mds_df['Year'], cmap=cmap, s=60, alpha=0.85,
                    edgecolors='white', linewidths=0.5, zorder=4, norm=norm)
    plt.colorbar(sc, ax=ax, label='Year', shrink=0.85)

    # Temporal arrows: connect year centroids chronologically
    if show_arrows:
        yr_cents = (mds_df.groupby('Year')[['MDS1','MDS2']]
                    .mean().sort_index().reset_index())
        for i in range(len(yr_cents)-1):
            r0, r1 = yr_cents.iloc[i], yr_cents.iloc[i+1]
            draw_temporal_arrow(ax, r0['MDS1'], r0['MDS2'], r1['MDS1'], r1['MDS2'])

    # Year labels at centroids
    yr_cents2 = (mds_df.groupby('Year')[['MDS1','MDS2']].mean().reset_index())
    for i, row in yr_cents2.iterrows():
        if i % label_every_n == 0:
            ax.annotate(str(int(row['Year'])),
                        (row['MDS1'], row['MDS2']),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=7, color='#333', zorder=5)

    add_style(ax, f'{subtype} Antigenic Map — Coloured by Year', stress)
    plt.tight_layout()
    fig.savefig(out(fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")

def plot_by_cluster(mds_df, subtype, stress, fname, cluster_col, color_map, title_suffix):
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#FAFAFA')

    clusters = [c for c in sorted(mds_df[cluster_col].unique())
                if c in color_map or c == 'OTHER']
    for cl in clusters:
        sub = mds_df[mds_df[cluster_col] == cl]
        col = color_map.get(cl, '#aaaaaa')
        ax.scatter(sub['MDS1'], sub['MDS2'], c=col, s=60, alpha=0.85,
                   edgecolors='white', linewidths=0.5, label=cl, zorder=4)
        # Cluster centroid label
        cx, cy = sub['MDS1'].mean(), sub['MDS2'].mean()
        ax.annotate(cl, (cx, cy), fontsize=8.5, fontweight='bold',
                    color=col, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                    zorder=6)

    # Add year labels to each point (small)
    for _, row in mds_df.iterrows():
        ax.annotate(str(int(row['Year'])) if pd.notna(row['Year']) else '',
                    (row['MDS1'], row['MDS2']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=6, color='#555', zorder=5)

    ax.legend(fontsize=8.5, loc='best', framealpha=0.85, ncol=2)
    add_style(ax, f'{subtype} Antigenic Map — {title_suffix}', stress)
    plt.tight_layout()
    fig.savefig(out(fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")

def plot_by_host(mds_df, subtype, stress, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#FAFAFA')

    hosts = mds_df['Host'].fillna('Other').unique()
    for host in hosts:
        sub = mds_df[mds_df['Host'].fillna('Other') == host]
        col = HOST_COLORS.get(host, '#999999')
        ax.scatter(sub['MDS1'], sub['MDS2'], c=col, s=60, alpha=0.85,
                   edgecolors='white', linewidths=0.5,
                   label=f'{host} (n={len(sub)})', zorder=4)

    # Year labels
    for _, row in mds_df.iterrows():
        ax.annotate(str(int(row['Year'])) if pd.notna(row['Year']) else '',
                    (row['MDS1'], row['MDS2']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=6.5, color='#444', zorder=5)

    ax.legend(fontsize=10, loc='best', framealpha=0.85)
    add_style(ax, f'{subtype} Antigenic Map — Coloured by Host', stress)
    plt.tight_layout()
    fig.savefig(out(fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")

# ── H1N1 plots ──
plot_by_year(mds_h1, 'H1N1', stress_h1,
             'antigenic_map_h1n1_by_year.png', show_arrows=True, label_every_n=2)
plot_by_cluster(mds_h1, 'H1N1', stress_h1,
                'antigenic_map_h1n1_by_cluster.png',
                'Lineage', H1N1_LIN_COLORS, 'Coloured by Lineage')
plot_by_host(mds_h1, 'H1N1', stress_h1, 'antigenic_map_h1n1_by_subtype.png')

# ── H3N2 plots ──
plot_by_year(mds_h3, 'H3N2', stress_h3,
             'antigenic_map_h3n2_by_year.png', show_arrows=True, label_every_n=3)
plot_by_cluster(mds_h3, 'H3N2', stress_h3,
                'antigenic_map_h3n2_by_cluster.png',
                'Smith_Cluster', SMITH_COLORS, 'Coloured by Smith Cluster')
plot_by_host(mds_h3, 'H3N2', stress_h3, 'antigenic_map_h3n2_by_subtype.png')

print("✓ Task 4.4 complete — all 6 maps saved")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4.5 — K-means Clustering in MDS Space
# ══════════════════════════════════════════════════════════════════════════════
print("\n✓ Task 4.5 — K-means clustering in MDS space...")

results_lines = [
    "MDS CLUSTERING ANALYSIS",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 65,
]

for subtype, mds_df, ref_cluster_col, stress_val in [
        ('H1N1', mds_h1, 'Lineage',        stress_h1),
        ('H3N2', mds_h3, 'Smith_Cluster',  stress_h3),
]:
    X     = mds_df[['MDS1','MDS2']].values
    N     = len(X)
    K_MIN, K_MAX = 3, min(15, N-1)

    sil_records, km_models = [], {}
    for k in range(K_MIN, K_MAX+1):
        km     = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(X, labels)
        sil_records.append({'K': k, 'Silhouette': sil})
        km_models[k] = (km, labels)

    sil_df  = pd.DataFrame(sil_records)
    best_k  = int(sil_df.loc[sil_df['Silhouette'].idxmax(), 'K'])
    best_sil= float(sil_df.loc[sil_df['Silhouette'].idxmax(), 'Silhouette'])
    km_best, labels_best = km_models[best_k]

    mds_df[f'MDS_Cluster'] = labels_best
    print(f"  [{subtype}] Optimal K={best_k}, Silhouette={best_sil:.4f}")
    print(f"            Silhouette curve: " +
          "  ".join(f"K{r.K}={r.Silhouette:.3f}" for _, r in sil_df.iterrows()))

    # Cluster composition
    mds_df['YearGroup'] = pd.cut(
        mds_df['Year'], bins=[1915,1960,1980,2000,2010,2025],
        labels=['1918-1960','1961-1980','1981-2000','2001-2010','2011-2025']
    )
    comp = pd.crosstab(mds_df['MDS_Cluster'], mds_df['YearGroup'])

    # Temporal pattern assessment
    dom_ref = (pd.crosstab(mds_df['MDS_Cluster'], mds_df[ref_cluster_col])
               .idxmax(axis=1))

    # Compare K-means to reference clusters for purity
    ref_labels = mds_df[ref_cluster_col].values
    confusion  = pd.crosstab(mds_df['MDS_Cluster'], mds_df[ref_cluster_col])
    row_max    = confusion.max(axis=1)
    purity     = row_max.sum() / N

    # Temporal coherence: do clusters map to time periods?
    yr_kend, yr_p = 0.0, 1.0
    if 'Year' in mds_df.columns:
        from scipy.stats import kendalltau
        yr_kend, yr_p = kendalltau(mds_df['Year'].fillna(mds_df['Year'].median()),
                                   mds_df['MDS1'])

    # Pass/fail
    temporal_ok = abs(yr_kend) > 0.30 and yr_p < 0.05
    purity_ok   = purity > 0.70
    if temporal_ok and purity_ok:   verdict = "PASS ✓"
    elif temporal_ok or purity_ok:  verdict = "MARGINAL ⚠"
    else:                           verdict = "FAIL ✗"

    print(f"  [{subtype}] MDS cluster purity={purity:.3f}  "
          f"Kendall-tau(Year,MDS1)={yr_kend:.3f} p={yr_p:.4f}  Verdict={verdict}")

    results_lines += [
        "",
        f"SUBTYPE: {subtype}",
        "-" * 65,
        f"  Sequences in MDS: {N}",
        f"  MDS stress (normalised): {stress_val:.4f}  "
        f"({'excellent' if stress_val<0.05 else 'good' if stress_val<0.10 else 'fair' if stress_val<0.20 else 'poor'})",
        "",
        f"  K-MEANS SWEEP (K={K_MIN}..{K_MAX})",
    ]
    for _, row in sil_df.iterrows():
        marker = " <-- OPTIMAL" if row.K == best_k else ""
        results_lines.append(f"    K={int(row.K):>2}  Silhouette={row.Silhouette:.4f}{marker}")

    results_lines += [
        "",
        f"  OPTIMAL K = {best_k}  (Silhouette = {best_sil:.4f})",
        "",
        "  CLUSTER COMPOSITION (per year group)",
    ]
    for row_idx in comp.index:
        dom = dom_ref.get(row_idx, '?')
        row_str = "  ".join(f"{col}:{int(v)}" for col, v in comp.loc[row_idx].items() if v > 0)
        results_lines.append(f"    Cluster {row_idx} (dominant {ref_cluster_col}: {dom}): {row_str}")

    results_lines += [
        "",
        f"  PURITY vs {ref_cluster_col}: {purity:.4f}  "
        f"({'PASS' if purity>0.70 else 'FAIL'} threshold 0.70)",
        f"  Temporal coherence (Kendall-tau MDS1 vs Year): "
        f"tau={yr_kend:.4f} p={yr_p:.4f} {'significant' if yr_p<0.05 else 'not significant'}",
        f"  Verdict: {verdict}",
        "",
        "  TEMPORAL OBSERVATIONS",
    ]
    # Describe what older vs newer clusters look like
    cluster_yr_means = mds_df.groupby('MDS_Cluster')['Year'].mean().sort_values()
    oldest_cl  = int(cluster_yr_means.index[0])
    newest_cl  = int(cluster_yr_means.index[-1])
    results_lines += [
        f"    Cluster {oldest_cl}: oldest sequences (mean year {cluster_yr_means.iloc[0]:.0f})",
        f"    Cluster {newest_cl}: newest sequences (mean year {cluster_yr_means.iloc[-1]:.0f})",
        f"    Year range per cluster:",
    ]
    for cl_idx, yr_mean in cluster_yr_means.items():
        sub = mds_df[mds_df['MDS_Cluster'] == cl_idx]
        results_lines.append(
            f"      Cluster {cl_idx}: yr {int(sub.Year.min())}-{int(sub.Year.max())}  "
            f"n={len(sub)}  mean_yr={yr_mean:.0f}"
        )

    # Supplementary K-means overlay plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle(f'{subtype} MDS Space — K-means Clustering (K={best_k})',
                 fontsize=13, fontweight='bold')

    cmap_k   = cm.get_cmap('tab20', best_k)
    years_c  = mds_df['Year'].values.astype(float)
    norm_yr  = mcolors.Normalize(vmin=np.nanmin(years_c), vmax=np.nanmax(years_c))

    for k_idx in range(best_k):
        mask = labels_best == k_idx
        ax1.scatter(X[mask,0], X[mask,1], c=[cmap_k(k_idx)], s=55,
                    alpha=0.80, edgecolors='white', lw=0.5, label=f'K{k_idx}', zorder=4)
    ctr = km_best.cluster_centers_
    ax1.scatter(ctr[:,0], ctr[:,1], marker='*', s=200, c='black', zorder=6)
    for k_idx, (cx,cy) in enumerate(ctr):
        ax1.annotate(f'K{k_idx}', (cx,cy), fontsize=8, fontweight='bold',
                     ha='center', va='bottom', color='black', zorder=7)
    ax1.legend(fontsize=7.5, ncol=3, loc='best', framealpha=0.8)
    ax1.set_xlabel('MDS1'); ax1.set_ylabel('MDS2')
    ax1.set_title(f'K-means Clusters (K={best_k}, sil={best_sil:.3f})', fontweight='bold')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2, linestyle='--'); ax1.set_facecolor('#FAFAFA')

    sc2 = ax2.scatter(X[:,0], X[:,1], c=years_c, cmap='plasma', s=55,
                      alpha=0.85, edgecolors='white', lw=0.5,
                      norm=norm_yr, zorder=4)
    plt.colorbar(sc2, ax=ax2, label='Year', shrink=0.85)
    ax2.set_xlabel('MDS1'); ax2.set_ylabel('MDS2')
    ax2.set_title('Same points coloured by Year', fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2, linestyle='--'); ax2.set_facecolor('#FAFAFA')

    plt.tight_layout()
    fname_km = f'mds_kmeans_{subtype.lower()}.png'
    fig.savefig(out(fname_km), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fname_km}")

with open(out("mds_clustering_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
print("✓ Saved mds_clustering_results.txt")

# ══════════════════════════════════════════════════════════════════════════════
# OVERALL SUCCESS VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
All outputs saved to: {OUTPUT_DIR}
  representative_strains.csv       H1N1={len(h1_rep)}, H3N2={len(h3_rep)}
  distance_matrix_h1n1.csv         {len(h1_rep)}x{len(h1_rep)}
  distance_matrix_h3n2.csv         {len(h3_rep)}x{len(h3_rep)}
  mds_coordinates_h1n1.csv
  mds_coordinates_h3n2.csv
  antigenic_map_h1n1_by_year.png   (300 dpi)
  antigenic_map_h1n1_by_cluster.png
  antigenic_map_h1n1_by_subtype.png
  antigenic_map_h3n2_by_year.png   (300 dpi)
  antigenic_map_h3n2_by_cluster.png
  antigenic_map_h3n2_by_subtype.png
  mds_clustering_results.txt
  mds_kmeans_h1n1.png
  mds_kmeans_h3n2.png
{'='*65}
  H1N1 MDS stress : {stress_h1:.4f}
  H3N2 MDS stress : {stress_h3:.4f}
{'='*65}
""")
