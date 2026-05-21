#!/usr/bin/env python3
"""
dataset_transparency.py — Comprehensive audit of final_fixed_influenza_ha_v2ok.csv.

Outputs:
  outputs/dataset_transparency/supplementary_table_S1.csv  (year × subtype counts)
  outputs/dataset_transparency/supplementary_table_S2.csv  (length stats per subtype/year)
  outputs/dataset_transparency/accession_sample.csv        (first 50 accessions per subtype)
  outputs/dataset_transparency/dataset_quality_report.txt
  outputs/dataset_transparency/dataset_visualization.png   (2-panel stacked bar + violin)
"""

import sys, re, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

ROOT   = Path(__file__).parent
OUT    = ROOT / 'outputs'
DTOUT  = OUT / 'dataset_transparency'
DTOUT.mkdir(exist_ok=True)

print('\n' + '='*62)
print(' Dataset Transparency Report')
print('='*62)

# ── 1. Load data ───────────────────────────────────────────────────────────────

print('\nLoading final_fixed_influenza_ha_v2ok.csv ...')
df = pd.read_csv(ROOT / 'final_fixed_influenza_ha_v2ok.csv')
print(f'  Columns: {list(df.columns)}')
print(f'  Shape:   {df.shape}')

ACCESSION_COL = 'Accession'
SUBTYPE_COL   = 'Subtype'
YEAR_COL      = 'Year'
SEQ_COL       = 'Sequence'
LEN_COL       = 'Length'


# ── 2. Basic counts ────────────────────────────────────────────────────────────

total_seqs = len(df)
h1n1_mask  = df[SUBTYPE_COL].str.contains('H1N1', case=False, na=False)
h3n2_mask  = df[SUBTYPE_COL].str.contains('H3N2', case=False, na=False)
other_mask = ~h1n1_mask & ~h3n2_mask

h1n1_count  = h1n1_mask.sum()
h3n2_count  = h3n2_mask.sum()
other_count = other_mask.sum()

print(f'\n  Total sequences: {total_seqs:,}')
print(f'  H1N1: {h1n1_count:,}  H3N2: {h3n2_count:,}  Other: {other_count:,}')
print(f'  Subtypes present: {df[SUBTYPE_COL].value_counts().head(10).to_dict()}')


# ── 3. Year × subtype count matrix (Supplementary Table S1) ───────────────────

# Create simplified subtype grouping
def classify_subtype(st):
    if isinstance(st, str):
        if 'H1N1' in st.upper(): return 'H1N1'
        if 'H3N2' in st.upper(): return 'H3N2'
    return 'Other'

df['subtype_group'] = df[SUBTYPE_COL].apply(classify_subtype)
pivot = df.pivot_table(index=YEAR_COL, columns='subtype_group', aggfunc='size', fill_value=0)
# Ensure columns order
for col in ['H1N1', 'H3N2', 'Other']:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[['H1N1', 'H3N2', 'Other']].sort_index()
pivot['Total'] = pivot.sum(axis=1)
pivot.index.name = 'Year'
pivot.to_csv(DTOUT / 'supplementary_table_S1.csv')
print(f'\n  Year range: {int(df[YEAR_COL].min())}–{int(df[YEAR_COL].max())}')
print(f'  Year × subtype matrix shape: {pivot.shape}')


# ── 4. Sequence length stats per subtype per year (Table S2) ──────────────────

len_stats = (df.groupby([YEAR_COL, 'subtype_group'])[LEN_COL]
             .agg(['count','min','max','mean','std'])
             .round(2).reset_index())

# Mode per group
modes = (df.groupby([YEAR_COL, 'subtype_group'])[LEN_COL]
         .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else float('nan'))
         .reset_index().rename(columns={LEN_COL: 'mode'}))
len_stats = len_stats.merge(modes, on=[YEAR_COL, 'subtype_group'], how='left')
len_stats.to_csv(DTOUT / 'supplementary_table_S2.csv', index=False)


# ── 5. Accession sample (first 50 per subtype) ────────────────────────────────

if ACCESSION_COL in df.columns:
    acc_samples = []
    for grp in ['H1N1', 'H3N2', 'Other']:
        sub = df[df['subtype_group'] == grp]
        acc_samples.append(sub[[ACCESSION_COL, SUBTYPE_COL, YEAR_COL]].head(50))
    pd.concat(acc_samples, ignore_index=True).to_csv(DTOUT / 'accession_sample.csv', index=False)
    print(f'  Accession sample saved (50 per subtype)')
else:
    print('  No Accession column found')


# ── 6. Missing values ─────────────────────────────────────────────────────────

missing = df.isnull().sum()
missing_rate = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_rate})
missing_summary = missing_summary[missing_summary['missing_count'] > 0]


# ── 7. Accession validation ───────────────────────────────────────────────────

if ACCESSION_COL in df.columns:
    valid_pattern = re.compile(r'^([A-Z]{2}\d{6}|[A-Z]{3}\d{5}|[A-Z]{2}_\d+)$')
    malformed = df[ACCESSION_COL].apply(
        lambda x: not bool(valid_pattern.match(str(x))) if pd.notna(x) else True)
    n_malformed = malformed.sum()
    malformed_rate = 100 * n_malformed / len(df)
else:
    n_malformed = 0; malformed_rate = 0.0


# ── 8. Duplicate detection ────────────────────────────────────────────────────

dup_mask  = df.duplicated(subset=[SEQ_COL, YEAR_COL], keep=False) if SEQ_COL in df.columns else pd.Series([False]*len(df))
n_dups    = dup_mask.sum()
dup_rate  = 100 * n_dups / len(df)

# Exact Accession duplicates
if ACCESSION_COL in df.columns:
    acc_dups = df[ACCESSION_COL].duplicated().sum()
else:
    acc_dups = 0

print(f'\n  Duplicates (seq+year): {n_dups:,} ({dup_rate:.2f}%)')
print(f'  Malformed accessions:  {n_malformed:,} ({malformed_rate:.2f}%)')


# ── 9. Length outlier detection ───────────────────────────────────────────────

q1, q3 = df[LEN_COL].quantile(0.25), df[LEN_COL].quantile(0.75)
iqr     = q3 - q1
outliers= ((df[LEN_COL] < q1 - 3*iqr) | (df[LEN_COL] > q3 + 3*iqr)).sum()


# ── 10. Quality report ─────────────────────────────────────────────────────────

report_lines = [
    'Dataset Transparency & Quality Report',
    '=' * 60,
    f'File: final_fixed_influenza_ha_v2ok.csv',
    f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
    '',
    '── Basic Statistics ──────────────────────────────────────────',
    f'Total sequences:          {total_seqs:,}',
    f'H1N1 sequences:           {h1n1_count:,} ({100*h1n1_count/total_seqs:.1f}%)',
    f'H3N2 sequences:           {h3n2_count:,} ({100*h3n2_count/total_seqs:.1f}%)',
    f'Other subtypes:           {other_count:,} ({100*other_count/total_seqs:.1f}%)',
    f'Year range:               {int(df[YEAR_COL].min())}–{int(df[YEAR_COL].max())}',
    f'Columns:                  {len(df.columns)}  ({", ".join(df.columns)})',
    '',
    '── Sequence Length ───────────────────────────────────────────',
    f'Min length:               {int(df[LEN_COL].min())} aa',
    f'Max length:               {int(df[LEN_COL].max())} aa',
    f'Mean length:              {df[LEN_COL].mean():.1f} aa',
    f'Std length:               {df[LEN_COL].std():.1f} aa',
    f'Mode length:              {int(df[LEN_COL].mode()[0])} aa',
    f'Length outliers (3×IQR):  {outliers:,} ({100*outliers/total_seqs:.2f}%)',
    '',
    '── Data Quality ──────────────────────────────────────────────',
    f'Exact duplicates (seq+yr):{n_dups:,} ({dup_rate:.2f}%)',
    f'Accession duplicates:     {acc_dups:,}',
    f'Malformed accessions:     {n_malformed:,} ({malformed_rate:.2f}%)',
]
if len(missing_summary) > 0:
    report_lines.append('')
    report_lines.append('── Missing Values ────────────────────────────────────────────')
    for col, row in missing_summary.iterrows():
        report_lines.append(f'  {col:<25} {int(row.missing_count):>8,} missing  ({row.missing_pct:.2f}%)')
else:
    report_lines.append(f'Missing values:           None (all columns complete)')

report_lines += [
    '',
    '── Subtype Distribution ──────────────────────────────────────',
]
for subtype, count in df[SUBTYPE_COL].value_counts().head(15).items():
    report_lines.append(f'  {str(subtype):<15} {count:>8,}  ({100*count/total_seqs:.2f}%)')

report_lines += [
    '',
    '── Data Integrity Summary ────────────────────────────────────',
    f'Overall duplicate rate:   {dup_rate:.2f}%  {"[CLEAN]" if dup_rate < 1 else "[FLAG: >1% duplicates]"}',
    f'Missing value rate:       {100*missing.sum()/(len(df)*len(df.columns)):.2f}% overall',
    f'Accession validity:       {100-malformed_rate:.1f}% valid format',
    f'Length consistency:       {100-100*outliers/total_seqs:.1f}% within 3×IQR of median',
]

report_text = '\n'.join(report_lines)
(DTOUT / 'dataset_quality_report.txt').write_text(report_text, encoding='utf-8')
print('\n' + report_text)


# ── 11. 2-panel visualization ──────────────────────────────────────────────────

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Influenza HA Sequence Dataset — Composition Overview',
             fontsize=14, fontweight='bold')

# Panel A: stacked bar chart H1N1 vs H3N2 by year
recent = pivot[(pivot.index >= 2000) & (pivot.index <= 2021)]
if len(recent) > 0:
    ax1.bar(recent.index, recent['H1N1'], label='H1N1', color='#2471A3', alpha=0.85, width=0.7)
    ax1.bar(recent.index, recent['H3N2'], bottom=recent['H1N1'], label='H3N2',
            color='#E67E22', alpha=0.85, width=0.7)
    if recent['Other'].sum() > 0:
        ax1.bar(recent.index, recent['Other'],
                bottom=recent['H1N1']+recent['H3N2'], label='Other',
                color='#7F8C8D', alpha=0.85, width=0.7)
ax1.set_xlabel('Year'); ax1.set_ylabel('Sequence Count')
ax1.set_title('Panel A: Sequence Counts by Year and Subtype\n(2000–2021)', fontweight='bold')
ax1.legend(); ax1.tick_params(axis='x', rotation=45)

# Panel B: violin plot of sequence lengths by subtype
groups_to_plot = ['H1N1', 'H3N2']
if other_count > 0:
    groups_to_plot.append('Other')
violin_data = [df[df['subtype_group']==g][LEN_COL].dropna().values for g in groups_to_plot]
violin_data = [d for d in violin_data if len(d) > 1]
valid_labels = [g for g, d in zip(groups_to_plot, [df[df['subtype_group']==g][LEN_COL].dropna().values for g in groups_to_plot]) if len(d) > 1]

if violin_data:
    parts = ax2.violinplot(violin_data, positions=range(len(violin_data)), showmedians=True)
    colors_v = ['#2471A3','#E67E22','#7F8C8D']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_v[i % len(colors_v)])
        pc.set_alpha(0.75)
ax2.set_xticks(range(len(valid_labels)))
ax2.set_xticklabels(valid_labels)
ax2.set_xlabel('Subtype'); ax2.set_ylabel('Sequence Length (aa)')
ax2.set_title('Panel B: Sequence Length Distribution by Subtype', fontweight='bold')
ax2.grid(True, alpha=0.25, linestyle='--')

fig.tight_layout()
fig.savefig(DTOUT / 'dataset_visualization.png')
plt.close(fig)
print(f'\n  Saved all outputs to {DTOUT}')
