"""
Phase 2: H3N2 Smith 2004 Clustering Validation
Tasks 2.1 – 2.4
Feature method: Dipeptide (2-mer) amino-acid frequency profiles + PCA + K-means
(ESM embeddings unavailable; dipeptide composition is a standard bioinformatics
proxy for sequence-based clustering of viral proteins)
"""
import sys, io, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

INPUT_CSV  = "final_fixed_influenza_ha_v2ok.csv"
OUTPUT_DIR = "C:/Users/UseR/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(fname):
    return os.path.join(OUTPUT_DIR, fname)

# ─── Smith 2004 cluster definitions ───────────────────────────────────────────
# Boundaries: year >= start and year < next_start (end year inclusive in last cluster)
SMITH_CLUSTERS = [
    ("HK68", 1968, 1971),
    ("EN72", 1972, 1974),
    ("VI75", 1975, 1976),
    ("TX77", 1977, 1978),
    ("BK79", 1979, 1986),
    ("SI87", 1987, 1988),
    ("BE89", 1989, 1991),
    ("BE92", 1992, 1994),
    ("WU95", 1995, 1996),
    ("SY97", 1997, 2001),
    ("FU02", 2002, 2004),
]
CLUSTER_COLORS = {
    "HK68":"#e41a1c","EN72":"#ff7f00","VI75":"#e6b800",
    "TX77":"#4daf4a","BK79":"#984ea3","SI87":"#a65628",
    "BE89":"#f781bf","BE92":"#377eb8","WU95":"#17becf",
    "SY97":"#2ca02c","FU02":"#8c564b","OTHER":"#aaaaaa",
}

def assign_smith_cluster(year):
    try:
        yr = int(year)
    except (ValueError, TypeError):
        return "UNKNOWN"
    for name, start, end in SMITH_CLUSTERS:
        if start <= yr <= end:
            return name
    return "OTHER"

# ─── Amino-acid alphabet & dipeptide feature names ────────────────────────────
AA = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = [''.join(p) for p in product(AA, repeat=2)]   # 400 features

def dipeptide_features(seq: str) -> np.ndarray:
    """Normalized dipeptide (2-mer) frequency profile — length-independent."""
    seq = seq.upper().strip()
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    total = max(sum(counts.values()), 1)
    return np.array([counts.get(dp, 0) / total for dp in DIPEPTIDES], dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.1 — Extract H3N2 Sequences
# ══════════════════════════════════════════════════════════════════════════════
print("Loading original dataset...")
df_all = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  Total records: {len(df_all):,}")

df_h3 = df_all[df_all['Subtype'] == 'H3N2'].copy()
df_h3['Sequence'] = df_h3['Sequence'].astype(str).str.strip().str.upper()
df_h3['Year'] = pd.to_numeric(df_h3['Year'], errors='coerce')
df_h3 = df_h3[df_h3['Sequence'].str.len() > 50].copy()

total_h3 = len(df_h3)
yr_min, yr_max = int(df_h3['Year'].min()), int(df_h3['Year'].max())
in_smith = df_h3[df_h3['Year'].between(1968, 2004)]

print(f"\n✓ Task 2.1 — H3N2 extraction")
print(f"  Total H3N2        : {total_h3:,}")
print(f"  Year range        : {yr_min}–{yr_max}")
print(f"  In Smith 1968-2004: {len(in_smith):,}")
print(f"  Host distribution : {df_h3['Host'].value_counts().to_dict()}")

# Year coverage check
print(f"\n  Smith cluster coverage:")
for name, start, end in SMITH_CLUSTERS:
    yrs_present = sorted(df_h3[df_h3['Year'].between(start,end)]['Year'].dropna().unique().astype(int))
    n = len(df_h3[df_h3['Year'].between(start,end)])
    gap_yrs = [y for y in range(start, end+1) if y not in yrs_present]
    gap_str = f"  GAPS:{gap_yrs}" if gap_yrs else ""
    print(f"    {name} ({start}-{end}): {n:>4} seqs  years={yrs_present}{gap_str}")

# Data quality
dup_acc = df_h3['Accession'].duplicated().sum() if 'Accession' in df_h3.columns else 0
dup_seq = df_h3['Sequence'].duplicated().sum()
print(f"\n  Duplicate accessions : {dup_acc}")
print(f"  Duplicate sequences  : {dup_seq}")
print(f"  Sequence length range: {df_h3['Sequence'].str.len().min()}–{df_h3['Sequence'].str.len().max()} aa")

df_h3.to_csv(out("h3n2_filtered_sequences.csv"), index=False)
print(f"\n✓ Saved h3n2_filtered_sequences.csv ({total_h3:,} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.2 — Map Smith's 11 Clusters
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("Task 2.2 — Assigning Smith cluster labels...")

df_h3['Smith_Cluster'] = df_h3['Year'].apply(assign_smith_cluster)

# Representative sequence per cluster = most common sequence in that cluster
rep_seqs = {}
for name, _, _ in SMITH_CLUSTERS:
    sub = df_h3[df_h3['Smith_Cluster'] == name]
    if len(sub) > 0:
        rep_seqs[name] = sub['Sequence'].value_counts().index[0]

df_h3['Representative_Seq'] = df_h3['Smith_Cluster'].map(
    lambda c: rep_seqs.get(c, '')
)

smith_counts = df_h3['Smith_Cluster'].value_counts()
print(f"\n  Cluster assignments:")
for name, _, _ in SMITH_CLUSTERS:
    n = smith_counts.get(name, 0)
    print(f"    {name}: {n:>4} sequences")
print(f"    OTHER (post-2004): {smith_counts.get('OTHER', 0):>4}")

save_cols = ['Accession','Year','Smith_Cluster','Representative_Seq'] + \
            ([c for c in ['Host','Country','VirusName'] if c in df_h3.columns])
df_h3[save_cols].to_csv(out("h3n2_smith_clusters.csv"), index=False)
print(f"\n✓ Task 2.2 complete — Saved h3n2_smith_clusters.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.3 — K-means Clustering on Dipeptide Features
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("Task 2.3 — Generating sequence features & clustering...")
print("  Feature method: Dipeptide (2-mer) amino-acid frequency profiles (400-dim)")
print("  No ESM embeddings available; using dipeptide composition as proxy")

# Focus on 1968-2004 for clustering (Smith framework period)
df_smith = df_h3[df_h3['Year'].between(1968, 2004)].copy().reset_index(drop=True)
N = len(df_smith)
print(f"\n  Sequences for clustering (1968-2004): {N:,}")

# Build feature matrix
print("  Building dipeptide feature matrix...")
X_raw = np.vstack([dipeptide_features(s) for s in df_smith['Sequence']])
print(f"  Feature matrix shape: {X_raw.shape}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA — retain 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
n_components = X_pca.shape[1]
var_explained = pca.explained_variance_ratio_.sum() * 100
print(f"  PCA components: {n_components}  ({var_explained:.1f}% variance explained)")

# K-means for K = 3..15
K_RANGE = range(3, 16)
silhouette_records = []
kmeans_models = {}

print(f"\n  K-means sweep (K=3..15):")
print(f"  {'K':>3}  {'Silhouette':>12}  {'Inertia':>12}")
print(f"  {'─'*3}  {'─'*12}  {'─'*12}")

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=min(N, 2000), random_state=42)
    silhouette_records.append({'K': k, 'Silhouette_Score': round(sil, 6)})
    kmeans_models[k] = (km, labels)
    print(f"  K={k:>2}  sil={sil:.4f}  inertia={km.inertia_:.1f}")

sil_df = pd.DataFrame(silhouette_records)
best_row = sil_df.loc[sil_df['Silhouette_Score'].idxmax()]
K_BEST = int(best_row['K'])
SIL_BEST = float(best_row['Silhouette_Score'])

print(f"\n✓ Task 2.3 — Optimal K={K_BEST}, Silhouette Score={SIL_BEST:.4f}")

sil_df.to_csv(out("h3n2_kmeans_silhouette_scores.csv"), index=False)
print("✓ Saved h3n2_kmeans_silhouette_scores.csv")

# Per-sample silhouette for optimal K
km_best, labels_best = kmeans_models[K_BEST]
sil_samples = silhouette_samples(X_pca, labels_best)

df_smith['KMeans_Cluster'] = labels_best
df_smith['Silhouette_Coef'] = sil_samples.round(6)

cluster_out_cols = ['Accession','Year','KMeans_Cluster','Silhouette_Coef','Smith_Cluster']
cluster_out_cols = [c for c in cluster_out_cols if c in df_smith.columns]
df_smith[cluster_out_cols].to_csv(out("h3n2_cluster_assignments.csv"), index=False)
print("✓ Saved h3n2_cluster_assignments.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2.4 — Cluster Purity Analysis
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("Task 2.4 — Calculating cluster purity...")

smith_labels   = df_smith['Smith_Cluster'].values
kmeans_labels  = df_smith['KMeans_Cluster'].values
smith_names    = [c[0] for c in SMITH_CLUSTERS]

# Confusion matrix (K-means rows × Smith columns)
confusion = pd.crosstab(
    df_smith['KMeans_Cluster'],
    df_smith['Smith_Cluster'],
    rownames=['KMeans_Cluster'],
    colnames=['Smith_Cluster']
)
# Add only Smith cluster columns in order
for nm in smith_names:
    if nm not in confusion.columns:
        confusion[nm] = 0
confusion = confusion[[nm for nm in smith_names if nm in confusion.columns]]

# Overall purity = sum of row-maxima / total
row_maxima = confusion.max(axis=1)
purity = row_maxima.sum() / N
dominant_smith_per_k = confusion.idxmax(axis=1)

# Per-cluster purity
cluster_purities = (row_maxima / confusion.sum(axis=1)).round(4)

print(f"\n  Overall purity : {purity:.4f} ({purity*100:.2f}%)")
print(f"\n  Per K-means cluster:")
print(f"  {'K':>3}  {'Size':>6}  {'Dominant Smith':>15}  {'Purity':>7}")
print(f"  {'─'*3}  {'─'*6}  {'─'*15}  {'─'*7}")
for k_idx in sorted(confusion.index):
    sz = confusion.loc[k_idx].sum()
    dom = dominant_smith_per_k[k_idx]
    pur = cluster_purities[k_idx]
    print(f"  K={k_idx:<2}  {sz:>6}  {dom:>15}  {pur:>7.3f}")

# Pass/Fail
if purity > 0.70:
    verdict = "PASS ✓"
    interp  = "K-means dipeptide clustering aligns well with Smith antigenic clusters."
elif purity >= 0.60:
    verdict = "MARGINAL ⚠"
    interp  = "Partial alignment; investigate cluster boundaries and feature quality."
else:
    verdict = "FAIL ✗"
    interp  = "Poor alignment; dipeptide features may not fully capture antigenicity."

print(f"\n  Verdict: {verdict}  (threshold PASS>0.70, MARGINAL 0.60-0.70, FAIL<0.60)")

# ── Purity report text ──
purity_lines = [
    "H3N2 CLUSTER PURITY ANALYSIS — Smith (2004) Validation",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 65,
    "",
    "METHODOLOGY",
    "  Feature type   : Dipeptide (2-mer) amino-acid frequency profiles (400-dim)",
    "  Note           : ESM protein embeddings unavailable; dipeptide composition",
    "                   is a standard sequence-based proxy for viral protein clustering",
    f"  Dimensionality : PCA to {n_components} components ({var_explained:.1f}% variance retained)",
    f"  Clustering     : K-means, optimal K={K_BEST} (silhouette={SIL_BEST:.4f})",
    f"  Sequences      : {N:,} H3N2 HA (1968-2004, Smith framework period)",
    "",
    "SMITH CLUSTER ASSIGNMENTS (year-based)",
    "-" * 65,
]
for name, start, end in SMITH_CLUSTERS:
    n = smith_counts.get(name, 0)
    purity_lines.append(f"  {name} ({start}-{end}): {n:>4} sequences")

purity_lines += [
    "",
    "OVERALL PURITY",
    "-" * 65,
    f"  Overall purity : {purity:.4f}  ({purity*100:.2f}%)",
    f"  Verdict        : {verdict}",
    f"  Interpretation : {interp}",
    "",
    "PER K-MEANS CLUSTER PURITY",
    "-" * 65,
    f"  {'K-Cluster':>9}  {'Size':>6}  {'Dominant Smith':>15}  {'Purity':>8}  {'Match n':>7}",
]
for k_idx in sorted(confusion.index):
    sz       = int(confusion.loc[k_idx].sum())
    dom      = dominant_smith_per_k[k_idx]
    pur      = cluster_purities[k_idx]
    match_n  = int(confusion.loc[k_idx, dom])
    purity_lines.append(
        f"  K={k_idx:<6}  {sz:>6}  {dom:>15}  {pur:>8.3f}  {match_n:>7}"
    )

purity_lines += [
    "",
    "CLUSTER RESOLUTION ANALYSIS",
    "-" * 65,
    f"  Smith clusters correctly distinguished (dominant): "
    f"{len(dominant_smith_per_k.unique())} of {len(smith_names)} Smith clusters represented",
    f"  K-means clusters dominated by single Smith cluster: "
    f"{(cluster_purities >= 0.80).sum()} of {K_BEST} (≥80% purity)",
    "",
    "NOTES",
    "-" * 65,
    "  1. Year-based Smith cluster assignment is the standard approach.",
    "     Boundary years (e.g., 1972, 1975) assigned to the newer cluster.",
    "  2. Dipeptide features capture amino-acid composition and local context",
    "     but not spatial structure of antigenic sites (HA1 epitopes).",
    "  3. Pre-pandemic sequences (1968-1986) are sparse (<30 seq/year);",
    "     small clusters may show artificially high or low purity.",
    "  4. For antigenicity-aware clustering, HI titres or HA1 structure-based",
    "     embeddings (ESM, ProtT5) would be more appropriate.",
]
with open(out("h3n2_cluster_purity_analysis.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(purity_lines))
print("✓ Saved h3n2_cluster_purity_analysis.txt")

confusion.to_csv(out("h3n2_cluster_comparison_matrix.csv"))
print("✓ Saved h3n2_cluster_comparison_matrix.csv")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('H3N2 Clustering Validation — Smith (2004) Framework\n'
             f'Dipeptide Features + PCA + K-means (K={K_BEST})',
             fontsize=14, fontweight='bold', y=1.01)

# ── Plot 1: PCA scatter coloured by Smith cluster ──
ax = axes[0, 0]
ax.set_facecolor('#FAFAFA')
X_pca2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
for name in smith_names:
    mask = df_smith['Smith_Cluster'] == name
    if mask.sum() > 0:
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                   c=CLUSTER_COLORS[name], s=12, alpha=0.65, label=name)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('PCA Coloured by Smith Cluster', fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='best', framealpha=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Plot 2: PCA scatter coloured by K-means cluster ──
ax = axes[0, 1]
ax.set_facecolor('#FAFAFA')
cmap = plt.cm.get_cmap('tab20', K_BEST)
sc = ax.scatter(X_pca2[:, 0], X_pca2[:, 1],
                c=labels_best, cmap=cmap, s=12, alpha=0.70)
plt.colorbar(sc, ax=ax, label='K-means cluster')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'PCA Coloured by K-means (K={K_BEST})', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Plot 3: Silhouette scores vs K ──
ax = axes[1, 0]
ax.set_facecolor('#FAFAFA')
ks  = sil_df['K'].values
ss  = sil_df['Silhouette_Score'].values
ax.plot(ks, ss, '-o', color='#1565C0', lw=2, ms=7,
        markerfacecolor='white', markeredgewidth=2)
ax.axvline(x=K_BEST, color='#D32F2F', ls='--', lw=1.5,
           label=f'Optimal K={K_BEST} (sil={SIL_BEST:.3f})')
ax.axvline(x=11, color='#388E3C', ls=':', lw=1.5, label='Smith K=11')
for k, s in zip(ks, ss):
    ax.annotate(f'{s:.3f}', (k, s), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=7.5)
ax.set_xlabel('Number of Clusters (K)'); ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score vs K', fontweight='bold')
ax.legend(fontsize=9); ax.set_xticks(ks)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Plot 4: Purity bar chart ──
ax = axes[1, 1]
ax.set_facecolor('#FAFAFA')
k_labels = [f'K{i}' for i in sorted(confusion.index)]
purity_vals = [cluster_purities[i] for i in sorted(confusion.index)]
dom_clusters = [dominant_smith_per_k[i] for i in sorted(confusion.index)]
bar_colors   = [CLUSTER_COLORS.get(d, '#aaaaaa') for d in dom_clusters]
bars = ax.bar(k_labels, purity_vals, color=bar_colors, edgecolor='white', width=0.7)
ax.axhline(y=0.70, color='#2E7D32', ls='--', lw=1.5, label='PASS threshold (0.70)')
ax.axhline(y=0.60, color='#F57F17', ls=':', lw=1.5, label='MARGINAL threshold (0.60)')
ax.axhline(y=purity, color='#D32F2F', ls='-', lw=1.8, alpha=0.8,
           label=f'Overall purity={purity:.3f}  {verdict}')
for bar, val, dom in zip(bars, purity_vals, dom_clusters):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            dom, ha='center', va='bottom', fontsize=6.5, rotation=60, color='#333')
ax.set_xlabel('K-means Cluster'); ax.set_ylabel('Purity (fraction matching dominant Smith cluster)')
ax.set_title('Per-Cluster Purity vs Smith Clusters', fontweight='bold')
ax.set_ylim(0, 1.18); ax.legend(fontsize=8.5, loc='upper right', framealpha=0.88)
ax.tick_params(axis='x', rotation=45)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(out("h3n2_clustering_validation.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved h3n2_clustering_validation.png (300 dpi)")

# ── Temporal K-means vs Smith cluster plot ──
fig2, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig2.patch.set_facecolor('#F8F9FA')

years_arr = df_smith['Year'].values.astype(int)
for name in smith_names:
    mask = df_smith['Smith_Cluster'] == name
    jitter = np.random.default_rng(42).uniform(-0.25, 0.25, mask.sum())
    ax_t.scatter(years_arr[mask], np.ones(mask.sum()) * smith_names.index(name) + jitter,
                 c=CLUSTER_COLORS[name], s=12, alpha=0.55)
ax_t.set_yticks(range(len(smith_names))); ax_t.set_yticklabels(smith_names, fontsize=9)
ax_t.set_ylabel('Smith Cluster'); ax_t.set_title('Smith Cluster Assignment (year-based)', fontweight='bold')
for name, start, end in SMITH_CLUSTERS:
    ax_t.axvspan(start-0.4, end+0.4, alpha=0.08, color=CLUSTER_COLORS[name])
ax_t.grid(True, alpha=0.2, axis='x')
ax_t.spines['top'].set_visible(False); ax_t.spines['right'].set_visible(False)

cmap2 = plt.cm.get_cmap('tab20', K_BEST)
jitter2 = np.random.default_rng(42).uniform(-0.25, 0.25, len(years_arr))
ax_b.scatter(years_arr, labels_best + jitter2, c=labels_best, cmap=cmap2,
             s=12, alpha=0.60)
ax_b.set_xlabel('Year'); ax_b.set_ylabel(f'K-means Cluster (K={K_BEST})')
ax_b.set_title(f'K-means Cluster Assignment (K={K_BEST}, dipeptide features)', fontweight='bold')
ax_b.grid(True, alpha=0.2, axis='x')
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
ax_b.set_xlim(1967, 2005)

plt.tight_layout()
fig2.savefig(out("h3n2_temporal_cluster_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved h3n2_temporal_cluster_comparison.png (300 dpi)")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
All outputs saved to: {OUTPUT_DIR}
  h3n2_filtered_sequences.csv          ({total_h3:,} rows)
  h3n2_smith_clusters.csv              ({len(df_h3):,} rows)
  h3n2_kmeans_silhouette_scores.csv    ({len(sil_df)} K values)
  h3n2_cluster_assignments.csv         ({N:,} rows)
  h3n2_cluster_purity_analysis.txt
  h3n2_cluster_comparison_matrix.csv
  h3n2_clustering_validation.png       (300 dpi, 4-panel)
  h3n2_temporal_cluster_comparison.png (300 dpi)
{'='*65}

PHASE 2 SUMMARY
  H3N2 sequences (total)       : {total_h3:,}
  Smith framework (1968-2004)  : {N:,} sequences
  PCA components               : {n_components} ({var_explained:.1f}% variance)
  Optimal K                    : {K_BEST}
  Best silhouette score        : {SIL_BEST:.4f}
  Smith clusters at K=11 sil   : {sil_df[sil_df['K']==11]['Silhouette_Score'].iloc[0]:.4f}
  Overall cluster purity       : {purity:.4f} ({purity*100:.2f}%)
  Verdict                      : {verdict}
""")
