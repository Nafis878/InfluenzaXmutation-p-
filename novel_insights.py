"""
Novel Biological Insights — Beyond Benchmark Validation
Four analyses extracting genuinely new signal from H1N1/H3N2 datasets:
  1. Epistatic co-mutation pairs  (per-sequence co-occurrence, Fisher's exact)
  2. Convergent evolution H1N1 <-> H3N2
  3. Positive selection pressure hotspots (dN/dS proxy via BLOSUM62)
  4. Mutation-specific temporal acceleration
"""
import sys, io, warnings
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import fisher_exact, linregress
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = Path("C:/Users/UseR/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def out(f): return OUTPUT_DIR / f

AA_VALID = set("ARNDCQEGHILKMFPSTWYV")

print("=" * 65)
print("  NOVEL BIOLOGICAL INSIGHTS — H1N1 & H3N2 HA")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

# ─── Load aggregated mutation data ────────────────────────────────────────────
print("\nLoading mutation data...")
mut_freq = pd.read_csv(out("mutation_frequency.csv"))
trend_df = pd.read_csv(out("mutation_trends_by_year.csv"))
for col in ['In_Antigenic_Site', 'In_RBS', 'Known_Virulence_Marker', 'Conservative']:
    mut_freq[col] = mut_freq[col].astype(bool)
print(f"  mutation_frequency : {len(mut_freq):,} unique variants")
print(f"  mutation_trends    : {len(trend_df):,} rows")

# ─── Load raw sequences for per-sequence epistatic analysis ───────────────────
print("  Loading raw sequences for epistatic analysis...")
h1_seq = pd.read_csv(out("h1n1_filtered_sequences.csv"), low_memory=False)
h3_seq = pd.read_csv(out("h3n2_filtered_sequences.csv"), low_memory=False)
for df_ in [h1_seq, h3_seq]:
    df_['Sequence'] = df_['Sequence'].astype(str).str.strip().str.upper()
print(f"  H1N1 sequences     : {len(h1_seq):,}")
print(f"  H3N2 sequences     : {len(h3_seq):,}")

# Reference sequences: most common 2009 H1N1, most common H3N2 (1968 or mode year)
ref_h1 = (h1_seq[h1_seq['Year'] == 2009]['Sequence']
          .value_counts().index[0])
h3_mode_yr = int(h3_seq['Year'].mode()[0])
ref_h3 = (h3_seq[h3_seq['Year'] == h3_mode_yr]['Sequence']
          .value_counts().index[0])
REFS   = {'H1N1': ref_h1, 'H3N2': ref_h3}
SEQ_DF = {'H1N1': h1_seq, 'H3N2': h3_seq}
print(f"  H1N1 ref : 2009 ({len(ref_h1)} aa)")
print(f"  H3N2 ref : {h3_mode_yr} ({len(ref_h3)} aa)")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Epistatic Co-Mutation Pairs
# For each pair of HA positions, test whether simultaneous mutation is observed
# more often than expected by independence (Fisher's exact test).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("Analysis 1 — Epistatic Co-Mutation Pairs")
print("─" * 65)

epi_results = []

for st in ['H1N1', 'H3N2']:
    raw    = SEQ_DF[st].dropna(subset=['Sequence']).copy()
    ref    = REFS[st]
    ref_arr = np.array(list(ref))
    ref_len = len(ref_arr)

    # Sample up to 2000 sequences for speed (preserves statistical power)
    MAX_SEQS = 2000
    if len(raw) > MAX_SEQS:
        raw = raw.sample(MAX_SEQS, random_state=42)
    n_seqs = len(raw)

    # Build per-sequence mutated-position sets (vectorised row-by-row)
    print(f"  {st}: computing mutation sets for {n_seqs:,} sequences...", end=' ', flush=True)
    pos_sets = []
    for seq_str in raw['Sequence']:
        arr = np.array(list(seq_str[:ref_len]))
        L   = min(len(arr), ref_len)
        muts = set(np.where(arr[:L] != ref_arr[:L])[0].tolist())
        # Keep only standard-AA positions
        muts = {p for p in muts if (arr[p] in AA_VALID and ref_arr[p] in AA_VALID)}
        pos_sets.append(muts)

    # Marginal counts per position
    pos_counts: dict[int, int] = {}
    for pset in pos_sets:
        for p in pset:
            pos_counts[p] = pos_counts.get(p, 0) + 1

    # Limit to positions mutated in ≥ 5 % of sampled sequences
    min_cnt  = max(5, int(n_seqs * 0.05))
    common   = [p for p, c in pos_counts.items() if c >= min_cnt]
    pos_idx  = {p: i for i, p in enumerate(common)}
    n_p      = len(common)
    print(f"{n_p} common positions")

    if n_p < 2:
        continue

    # Co-occurrence count matrix (upper triangle only needed)
    cooccur = np.zeros((n_p, n_p), dtype=np.int32)
    for pset in pos_sets:
        in_common = [pos_idx[p] for p in pset if p in pos_idx]
        for a, b in combinations(in_common, 2):
            cooccur[a, b] += 1
            cooccur[b, a] += 1

    # Fisher's exact test for significant pairs
    pairs = []
    for i in range(n_p):
        for j in range(i + 1, n_p):
            n_both   = int(cooccur[i, j])
            ci       = pos_counts[common[i]]
            cj       = pos_counts[common[j]]
            n_i_only = ci - n_both
            n_j_only = cj - n_both
            n_neither= n_seqs - n_both - n_i_only - n_j_only
            if n_neither < 0 or n_i_only < 0 or n_j_only < 0:
                continue
            table = [[n_both, n_j_only], [n_i_only, n_neither]]
            odds_ratio, p_val = fisher_exact(table, alternative='greater')
            if p_val < 0.001 and float(odds_ratio) > 2.0 and n_both >= 5:
                pairs.append({
                    'Subtype'     : st,
                    'Position_A'  : common[i] + 1,  # 1-indexed
                    'Position_B'  : common[j] + 1,
                    'CoOccurrence': n_both,
                    'Count_A'     : ci,
                    'Count_B'     : cj,
                    'OddsRatio'   : round(float(odds_ratio), 3),
                    'P_value'     : round(float(p_val), 8),
                })

    pairs.sort(key=lambda x: x['OddsRatio'], reverse=True)
    epi_results.extend(pairs)
    print(f"  {st}: {len(pairs)} significant epistatic pairs (OR>2, p<0.001)")
    for r in pairs[:5]:
        print(f"    Pos {r['Position_A']:>3}–{r['Position_B']:>3}  "
              f"co-occur={r['CoOccurrence']:>4}  OR={r['OddsRatio']:.2f}  "
              f"p={r['P_value']:.2e}")

epi_df = pd.DataFrame(epi_results) if epi_results else pd.DataFrame(
    columns=['Subtype','Position_A','Position_B','CoOccurrence',
             'Count_A','Count_B','OddsRatio','P_value'])
epi_df.to_csv(out("epistatic_pairs.csv"), index=False)
print(f"✓ Saved epistatic_pairs.csv  ({len(epi_df)} pairs)")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Convergent Evolution Across Subtypes
# Same amino acid change at same position in both H1N1 AND H3N2.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("Analysis 2 — Convergent Evolution (H1N1 ↔ H3N2)")
print("─" * 65)

h1_vars = mut_freq[mut_freq['Subtype'] == 'H1N1'].copy()
h3_vars = mut_freq[mut_freq['Subtype'] == 'H3N2'].copy()

conv = h1_vars.merge(h3_vars, on=['Position', 'WT_AA', 'Mutant_AA'],
                     suffixes=('_H1N1', '_H3N2'))
conv['Combined_Frequency'] = conv['Frequency_H1N1'] + conv['Frequency_H3N2']
conv['In_Antigenic_H1N1'] = conv['In_Antigenic_Site_H1N1'].astype(bool)
conv['In_Antigenic_H3N2'] = conv['In_Antigenic_Site_H3N2'].astype(bool)
conv['In_RBS_Either']     = conv['In_RBS_H1N1'].astype(bool) | conv['In_RBS_H3N2'].astype(bool)
conv['BLOSUM62_Score']    = conv['BLOSUM62_Score_H1N1']
conv['Mutation']          = conv['WT_AA'] + conv['Position'].astype(str) + conv['Mutant_AA']

conv_out = conv.sort_values('Combined_Frequency', ascending=False)[[
    'Mutation', 'Position', 'WT_AA', 'Mutant_AA',
    'Count_H1N1', 'Frequency_H1N1', 'Count_H3N2', 'Frequency_H3N2',
    'Combined_Frequency', 'In_Antigenic_H1N1', 'In_Antigenic_H3N2',
    'In_RBS_Either', 'BLOSUM62_Score'
]].reset_index(drop=True)

conv_out.to_csv(out("convergent_evolution.csv"), index=False)
print(f"  Convergently evolved mutations: {len(conv_out)}")
ag_either = (conv_out['In_Antigenic_H1N1'] | conv_out['In_Antigenic_H3N2']).sum()
print(f"  In antigenic site (≥1 subtype): {ag_either}")
print(f"  In RBS (either subtype)        : {conv_out['In_RBS_Either'].sum()}")
if len(conv_out) > 0:
    print(f"\n  Top 10 by combined frequency:")
    print(f"  {'Mutation':>10}  {'Freq_H1N1':>9}  {'Freq_H3N2':>9}  {'AntSite':>8}  {'RBS':>4}")
    for _, r in conv_out.head(10).iterrows():
        ant = ('H1+' if r.In_Antigenic_H1N1 else '') + ('H3' if r.In_Antigenic_H3N2 else '')
        if not ant: ant = '-'
        print(f"  {r.Mutation:>10}  {r.Frequency_H1N1:>9.4f}  {r.Frequency_H3N2:>9.4f}  "
              f"{ant:>8}  {'Y' if r.In_RBS_Either else 'N':>4}")
print(f"✓ Saved convergent_evolution.csv")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Positive Selection Pressure Hotspots (dN/dS proxy)
# Positions where > 70% of observed mutations are physicochemically disruptive
# (BLOSUM62 < 0) are under positive selection pressure.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("Analysis 3 — Positive Selection Pressure Hotspots")
print("─" * 65)

sel_records = []
for st in ['H1N1', 'H3N2']:
    sub = mut_freq[mut_freq['Subtype'] == st].copy()
    for pos, grp in sub.groupby('Position'):
        total_mut  = grp['Count'].sum()
        disrupt_ct = grp.loc[grp['BLOSUM62_Score'] < 0, 'Count'].sum()
        if total_mut < 10:
            continue
        sel_records.append({
            'Subtype'             : st,
            'Position'            : int(pos),
            'Total_Mutations'     : int(total_mut),
            'Disruptive_Count'    : int(disrupt_ct),
            'Disruptive_Fraction' : round(disrupt_ct / total_mut, 4),
            'Unique_Variants'     : int(grp['Mutant_AA'].nunique()),
            'In_Antigenic_Site'   : bool(grp['In_Antigenic_Site'].any()),
            'In_RBS'              : bool(grp['In_RBS'].any()),
            'Position_Entropy'    : round(float(grp['Position_Entropy'].iloc[0]), 4),
            'Mean_Frequency'      : round(float(grp['Frequency'].mean()), 5),
        })

sel_df = pd.DataFrame(sel_records)
sel_df['Under_Positive_Selection'] = sel_df['Disruptive_Fraction'] >= 0.70
sel_df = sel_df.sort_values(['Subtype', 'Disruptive_Fraction'], ascending=[True, False])
sel_df.to_csv(out("positive_selection_hotspots.csv"), index=False)

for st in ['H1N1', 'H3N2']:
    hot = sel_df[(sel_df.Subtype == st) & sel_df['Under_Positive_Selection']]
    print(f"\n  {st}: {len(hot)} positive-selection hotspots "
          f"(disruptive≥70%, n≥10)")
    print(f"    Antigenic site: {hot['In_Antigenic_Site'].sum()}  "
          f"RBS: {hot['In_RBS'].sum()}")
    for _, r in hot.head(5).iterrows():
        print(f"    Pos {int(r.Position):>4}  disruptive={r.Disruptive_Fraction:.2f}  "
              f"n={int(r.Total_Mutations):>5}  "
              f"ag={'Y' if r.In_Antigenic_Site else 'N'}  "
              f"rbs={'Y' if r.In_RBS else 'N'}")
print(f"\n✓ Saved positive_selection_hotspots.csv")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Mutation Temporal Acceleration
# Per-mutation linear slope on frequency vs year, identify sweeping variants.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("Analysis 4 — Mutation Temporal Acceleration")
print("─" * 65)

accel_records = []
for st in ['H1N1', 'H3N2']:
    sub = trend_df[trend_df['Subtype'] == st].copy()
    for mkey, grp in sub.groupby('Mutation'):
        grp_s = grp.sort_values('Year').dropna(subset=['Frequency'])
        if len(grp_s) < 4:
            continue
        years = grp_s['Year'].values.astype(float)
        freqs = grp_s['Frequency'].values
        slope, _, r_val, p_val, _ = linregress(years, freqs)
        # Parse mutation key e.g. "49I>L"
        try:
            parts  = mkey.split('>')
            pos    = int(''.join(filter(str.isdigit, parts[0])))
            wt_aa  = ''.join(filter(str.isalpha, parts[0]))
            mut_aa = parts[1]
        except (IndexError, ValueError):
            continue
        ann = mut_freq[(mut_freq['Subtype'] == st) &
                       (mut_freq['Position'] == pos) &
                       (mut_freq['WT_AA'] == wt_aa) &
                       (mut_freq['Mutant_AA'] == mut_aa)]
        accel_records.append({
            'Subtype'          : st,
            'Mutation'         : mkey,
            'Position'         : pos,
            'Slope_per_year'   : round(float(slope), 6),
            'R_squared'        : round(float(r_val ** 2), 4),
            'P_value'          : round(float(p_val), 6),
            'N_years'          : len(grp_s),
            'Freq_start'       : round(float(freqs[0]), 5),
            'Freq_end'         : round(float(freqs[-1]), 5),
            'In_Antigenic_Site': bool(ann['In_Antigenic_Site'].any()) if len(ann) else False,
            'In_RBS'           : bool(ann['In_RBS'].any()) if len(ann) else False,
            'BLOSUM62_Score'   : int(ann['BLOSUM62_Score'].iloc[0]) if len(ann) else 0,
        })

accel_df = pd.DataFrame(accel_records)
accel_df['Accelerating'] = (accel_df['Slope_per_year'] > 0) & (accel_df['P_value'] < 0.05)
accel_df['Decelerating'] = (accel_df['Slope_per_year'] < 0) & (accel_df['P_value'] < 0.05)
accel_df = accel_df.sort_values('Slope_per_year', ascending=False)
accel_df.to_csv(out("mutation_acceleration.csv"), index=False)

for st in ['H1N1', 'H3N2']:
    sub_a = accel_df[accel_df.Subtype == st]
    print(f"\n  {st}: {len(sub_a)} mutations tracked")
    print(f"    Accelerating (slope>0, p<0.05): {sub_a['Accelerating'].sum()}")
    print(f"    Decelerating (slope<0, p<0.05): {sub_a['Decelerating'].sum()}")
    top3 = sub_a[sub_a['Accelerating']].head(3)
    for _, r in top3.iterrows():
        print(f"      {r.Mutation:<15} slope={r.Slope_per_year:+.5f}/yr  "
              f"R²={r.R_squared:.3f}  p={r.P_value:.4f}  "
              f"ag={'Y' if r.In_Antigenic_Site else 'N'}")
print(f"\n✓ Saved mutation_acceleration.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — 4-panel Novel Insights Dashboard
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating novel_insights_figure.png...")

BLUE   = '#1565C0'; RED    = '#C62828'
ORANGE = '#E65100'; GREEN  = '#2E7D32'; GRAY = '#757575'

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#FAFAFA')

# ── Panel A: Epistatic pairs scatter ──────────────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor('white')
if len(epi_df) > 0:
    for st, c, mk in [('H1N1', BLUE, 'o'), ('H3N2', RED, 's')]:
        sub_e = epi_df[epi_df.Subtype == st]
        if len(sub_e):
            ax.scatter(sub_e['Position_A'], sub_e['Position_B'],
                       s=sub_e['OddsRatio'].clip(2, 20) * 18,
                       c=c, alpha=0.65, marker=mk, edgecolors='white',
                       linewidths=0.4, label=f'{st} ({len(sub_e)} pairs)')
    ax.set_xlabel('HA Position A', fontsize=10)
    ax.set_ylabel('HA Position B', fontsize=10)
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, 'No significant epistatic pairs\nfound at current thresholds\n(p<0.001, OR>2)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color=GRAY,
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor=GRAY))
ax.set_title('A.  Epistatic Co-Mutation Pairs\n'
             '(size = odds ratio; p<0.001, OR>2)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Panel B: Convergent evolution ─────────────────────────────────────────────
ax = axes[0, 1]
ax.set_facecolor('white')
if len(conv_out) >= 1:
    top_c = conv_out.head(min(15, len(conv_out)))
    y_pos = np.arange(len(top_c))
    h1_col = [ORANGE if r.In_Antigenic_H1N1 else BLUE for _, r in top_c.iterrows()]
    h3_col = [ORANGE if r.In_Antigenic_H3N2 else RED  for _, r in top_c.iterrows()]
    ax.barh(y_pos,        top_c['Frequency_H1N1'], height=0.36,
            color=h1_col, alpha=0.82, label='H1N1 frequency')
    ax.barh(y_pos + 0.38, top_c['Frequency_H3N2'], height=0.36,
            color=h3_col, alpha=0.82, label='H3N2 frequency')
    ax.set_yticks(y_pos + 0.19)
    ax.set_yticklabels(top_c['Mutation'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mutation Frequency', fontsize=10)
    handles = [mpatches.Patch(facecolor=BLUE,   label='H1N1'),
               mpatches.Patch(facecolor=RED,    label='H3N2'),
               mpatches.Patch(facecolor=ORANGE, label='Antigenic site')]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
else:
    ax.text(0.5, 0.5, 'No convergent mutations found', ha='center',
            va='center', transform=ax.transAxes, fontsize=11)
ax.set_title(f'B.  Convergent Evolution H1N1 ↔ H3N2\n'
             f'({len(conv_out)} shared mutations; orange = antigenic site)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--', axis='x')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Panel C: Positive selection hotspots ──────────────────────────────────────
ax = axes[1, 0]
ax.set_facecolor('white')
for st, color in [('H1N1', BLUE), ('H3N2', RED)]:
    all_pos = sel_df[(sel_df.Subtype == st)]
    hot     = sel_df[(sel_df.Subtype == st) & sel_df['Under_Positive_Selection']]
    ax.scatter(all_pos['Position'], all_pos['Disruptive_Fraction'],
               c=color, s=10, alpha=0.18, zorder=2)
    if len(hot):
        ax.scatter(hot['Position'], hot['Disruptive_Fraction'],
                   c=color, s=60, alpha=0.90, edgecolors='white',
                   linewidths=0.5, zorder=5,
                   label=f'{st} hotspot ({len(hot)})')
ax.axhline(0.70, color=GRAY, lw=1.3, linestyle='--', alpha=0.7,
           label='Positive selection (0.70)')
ax.set_xlabel('HA Position', fontsize=10)
ax.set_ylabel('Disruptive Mutation Fraction\n(BLOSUM62 < 0)', fontsize=10)
ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=9)
ax.set_title('C.  Positive Selection Hotspots\n'
             '(n≥10 mutations; large = above threshold)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Panel D: Mutation temporal acceleration ───────────────────────────────────
ax = axes[1, 1]
ax.set_facecolor('white')
if len(accel_df):
    for st, color in [('H1N1', BLUE), ('H3N2', RED)]:
        slopes = accel_df[accel_df.Subtype == st]['Slope_per_year']
        if len(slopes):
            ax.hist(slopes, bins=12, color=color, alpha=0.52,
                    label=f'{st} (n={len(slopes)})', edgecolor='white')
    ax.axvline(0, color='black', lw=1.2, linestyle='--', alpha=0.6)
    n_acc = int(accel_df['Accelerating'].sum())
    n_dec = int(accel_df['Decelerating'].sum())
    ax.text(0.97, 0.95,
            f'Accelerating: {n_acc}\nDecelerating: {n_dec}\n(p < 0.05)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#EBF5FB', edgecolor='#2471A3'))
    ax.set_xlabel('Frequency Slope (per year)', fontsize=10)
    ax.set_ylabel('Number of Mutations', fontsize=10)
    ax.legend(fontsize=9)
ax.set_title('D.  Mutation Temporal Acceleration\n'
             '(positive = rising variant)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.suptitle(
    'Novel Biological Insights — H1N1 & H3N2 HA Mutations\n'
    'Epistasis · Convergent evolution · Positive selection · Temporal acceleration',
    fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3.5, w_pad=3)
fig.savefig(out("novel_insights_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved novel_insights_figure.png (300 dpi)")

# ─── Summary report ───────────────────────────────────────────────────────────
summary = "\n".join([
    "=" * 65,
    "  NOVEL INSIGHTS SUMMARY",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 65,
    "",
    f"1. EPISTATIC CO-MUTATION PAIRS : {len(epi_df)} significant pairs (OR>2, p<0.001)",
    f"   H1N1: {len(epi_df[epi_df.Subtype=='H1N1'])}  "
    f"H3N2: {len(epi_df[epi_df.Subtype=='H3N2'])}",
    "   Implication: position pairs mutated together > chance — functional coupling",
    "",
    f"2. CONVERGENT EVOLUTION        : {len(conv_out)} mutations shared by both subtypes",
    f"   In antigenic sites          : {(conv_out['In_Antigenic_H1N1']|conv_out['In_Antigenic_H3N2']).sum() if len(conv_out) else 0}",
    "   Implication: convergent immune escape under shared human-host selection",
    "",
    "3. POSITIVE SELECTION HOTSPOTS :",
] + [
    f"   {st}: "
    f"{len(sel_df[(sel_df.Subtype==st)&sel_df['Under_Positive_Selection']])} positions "
    f"(disruptive ≥ 70%, n≥10)"
    for st in ['H1N1', 'H3N2']
] + [
    "   Implication: radical changes preferred — ongoing positive selection",
    "",
    f"4. TEMPORAL ACCELERATION       : {int(accel_df['Accelerating'].sum())} rising variants (p<0.05)",
    f"   Decelerating                : {int(accel_df['Decelerating'].sum())}",
    "   Implication: identifies currently sweeping variants for surveillance",
    "",
    "OUTPUTS",
    "  epistatic_pairs.csv             novel_insights_figure.png",
    "  convergent_evolution.csv        positive_selection_hotspots.csv",
    "  mutation_acceleration.csv       novel_insights_report.txt",
    "=" * 65,
])
(out("novel_insights_report.txt")).write_text(summary, encoding='utf-8')
print("✓ Saved novel_insights_report.txt")
print(f"\n{'='*65}")
print("  Novel insights complete — 4 analyses, 5 CSV files, 1 figure")
print(f"{'='*65}")
