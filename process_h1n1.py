import pandas as pd
import numpy as np
import json
import os
import sys
from collections import Counter
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Paths
INPUT_FILE = "final_fixed_influenza_ha_v2ok.csv"
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(name):
    return os.path.join(OUTPUT_DIR, name)

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
total_original = len(df)
print(f"  Total records loaded: {total_original:,}")

# ─────────────────────────────────────────────
# TASK 1: Filter to H1N1 only
# ─────────────────────────────────────────────
df_h1n1 = df[df['Subtype'] == 'H1N1'].copy()
removed = total_original - len(df_h1n1)
print(f"\n✓ Task 1 – Filter to H1N1")
print(f"  H1N1 sequences kept : {len(df_h1n1):,}")
print(f"  Non-H1N1 removed    : {removed:,}")

# ─────────────────────────────────────────────
# TASK 2: Count H1N1 sequences by year
# ─────────────────────────────────────────────
df_h1n1['Year'] = pd.to_numeric(df_h1n1['Year'], errors='coerce')
year_counts = df_h1n1['Year'].value_counts().sort_index()
print(f"\n✓ Task 2 – H1N1 sequence counts")
print(f"  Total H1N1 sequences: {len(df_h1n1):,}")
print(f"  Breakdown by year:")
for yr, cnt in year_counts.items():
    print(f"    {int(yr) if not np.isnan(yr) else 'NaN'}: {cnt:,}")

# ─────────────────────────────────────────────
# TASK 3: Verify year range 2009-2020
# ─────────────────────────────────────────────
min_year = df_h1n1['Year'].min()
max_year = df_h1n1['Year'].max()
years_present = sorted([int(y) for y in df_h1n1['Year'].dropna().unique()])
expected_range = list(range(2009, 2021))
gaps = [y for y in expected_range if y not in years_present]
print(f"\n✓ Task 3 – Year range verification")
print(f"  Min year : {min_year}")
print(f"  Max year : {max_year}")
print(f"  Years in 2009-2020 present: {sorted([y for y in years_present if 2009 <= y <= 2020])}")
print(f"  Gaps in 2009-2020 range   : {gaps if gaps else 'None'}")

# ─────────────────────────────────────────────
# TASK 4: Completeness classification
# ─────────────────────────────────────────────
df_h1n1['Length'] = pd.to_numeric(df_h1n1['Length'], errors='coerce')
df_h1n1['Completeness'] = df_h1n1['Length'].apply(
    lambda x: 'Complete' if pd.notna(x) and x >= 800 else 'Truncated'
)
complete_n = (df_h1n1['Completeness'] == 'Complete').sum()
truncated_n = (df_h1n1['Completeness'] == 'Truncated').sum()
total_n = len(df_h1n1)
print(f"\n✓ Task 4 – Completeness check (≥800bp = Complete)")
print(f"  Complete  : {complete_n:,} ({complete_n/total_n*100:.1f}%)")
print(f"  Truncated : {truncated_n:,} ({truncated_n/total_n*100:.1f}%)")
print(f"  Length stats:")
print(f"    Mean   : {df_h1n1['Length'].mean():.1f} bp")
print(f"    Median : {df_h1n1['Length'].median():.1f} bp")
print(f"    Min    : {df_h1n1['Length'].min():.0f} bp")
print(f"    Max    : {df_h1n1['Length'].max():.0f} bp")
print(f"    Std    : {df_h1n1['Length'].std():.1f} bp")

# ─────────────────────────────────────────────
# TASK 5: Full statistics
# ─────────────────────────────────────────────

# Top 20 countries
country_counts = df_h1n1['Country'].value_counts().head(20)

# Host distribution
host_counts = df_h1n1['Host'].value_counts()

# Metadata completeness
meta_cols = ['Accession', 'Host', 'Subtype', 'Year', 'Sequence', 'Length', 'Country', 'VirusName']
meta_completeness = {}
for col in meta_cols:
    if col in df_h1n1.columns:
        filled = df_h1n1[col].notna().sum()
        pct = filled / total_n * 100
        meta_completeness[col] = {'filled': int(filled), 'missing': int(total_n - filled), 'pct': round(pct, 2)}

# Duplicates
dup_accession = df_h1n1['Accession'].duplicated().sum() if 'Accession' in df_h1n1.columns else 0
dup_sequence = df_h1n1['Sequence'].duplicated().sum() if 'Sequence' in df_h1n1.columns else 0

print(f"\n✓ Task 5 – Comprehensive statistics compiled")

# ─────────────────────────────────────────────
# Year distribution with completeness rate
# ─────────────────────────────────────────────
year_completeness = df_h1n1.groupby('Year').apply(
    lambda g: pd.Series({
        'Count': len(g),
        'Completeness_Rate': round((g['Completeness'] == 'Complete').sum() / len(g) * 100, 2)
    })
).reset_index()
year_completeness.columns = ['Year', 'Count', 'Completeness_Rate']
year_completeness['Year'] = year_completeness['Year'].astype(int)

# Length distribution bins
bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, float('inf')]
labels = ['0-199', '200-399', '400-599', '600-799', '800-999',
          '1000-1199', '1200-1399', '1400-1599', '1600-1799', '1800+']
df_h1n1['LengthBin'] = pd.cut(df_h1n1['Length'], bins=bins, labels=labels, right=False)
len_dist = df_h1n1['LengthBin'].value_counts().sort_index().reset_index()
len_dist.columns = ['Length_Range', 'Count']

# ─────────────────────────────────────────────
# OUTPUT 1: Filtered CSV with Completeness
# ─────────────────────────────────────────────
save_cols = [c for c in ['Accession','Host','Subtype','Year','Length','Country','VirusName','Protein','Completeness','Sequence'] if c in df_h1n1.columns]
df_h1n1[save_cols].to_csv(out("h1n1_filtered_sequences.csv"), index=False)
print(f"\n✓ Saved h1n1_filtered_sequences.csv ({len(df_h1n1):,} rows)")

# ─────────────────────────────────────────────
# OUTPUT 4: Year distribution CSV
# ─────────────────────────────────────────────
year_completeness.to_csv(out("h1n1_year_distribution.csv"), index=False)
print("✓ Saved h1n1_year_distribution.csv")

# ─────────────────────────────────────────────
# OUTPUT 5: Length distribution CSV
# ─────────────────────────────────────────────
len_dist.to_csv(out("h1n1_length_distribution.csv"), index=False)
print("✓ Saved h1n1_length_distribution.csv")

# ─────────────────────────────────────────────
# OUTPUT 6: Top countries CSV
# ─────────────────────────────────────────────
country_df = country_counts.reset_index()
country_df.columns = ['Country', 'Count']
country_df.to_csv(out("h1n1_top_countries.csv"), index=False)
print("✓ Saved h1n1_top_countries.csv")

# ─────────────────────────────────────────────
# OUTPUT 2: Full statistics report
# ─────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "  H1N1 INFLUENZA HA SEQUENCE — BASELINE STATISTICS REPORT",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 60,
    "",
    "1. DATASET OVERVIEW",
    "-" * 40,
    f"  Original records (all subtypes) : {total_original:,}",
    f"  Non-H1N1 removed               : {removed:,}",
    f"  H1N1 sequences retained        : {total_n:,}",
    f"  Year range                     : {int(min_year)} – {int(max_year)}",
    f"  Gaps in 2009-2020              : {gaps if gaps else 'None'}",
    "",
    "2. SEQUENCE LENGTH STATISTICS",
    "-" * 40,
    f"  Mean   : {df_h1n1['Length'].mean():.2f} bp",
    f"  Median : {df_h1n1['Length'].median():.2f} bp",
    f"  Min    : {df_h1n1['Length'].min():.0f} bp",
    f"  Max    : {df_h1n1['Length'].max():.0f} bp",
    f"  Std    : {df_h1n1['Length'].std():.2f} bp",
    "",
    "3. COMPLETENESS (≥800 bp = Complete)",
    "-" * 40,
    f"  Complete  : {complete_n:,} ({complete_n/total_n*100:.2f}%)",
    f"  Truncated : {truncated_n:,} ({truncated_n/total_n*100:.2f}%)",
    "",
    "4. YEAR DISTRIBUTION",
    "-" * 40,
]
for _, row in year_completeness.iterrows():
    report_lines.append(f"  {int(row['Year'])}: {int(row['Count']):>6,} sequences  (Complete: {row['Completeness_Rate']}%)")

report_lines += [
    "",
    "5. TOP 20 COUNTRIES",
    "-" * 40,
]
for i, (country, cnt) in enumerate(country_counts.items(), 1):
    pct = cnt / total_n * 100
    report_lines.append(f"  {i:>2}. {str(country):<30} {cnt:>6,} ({pct:.2f}%)")

report_lines += [
    "",
    "6. HOST DISTRIBUTION",
    "-" * 40,
]
for host, cnt in host_counts.items():
    pct = cnt / total_n * 100
    report_lines.append(f"  {str(host):<20} {cnt:>6,} ({pct:.2f}%)")

report_lines += [
    "",
    "7. METADATA COMPLETENESS",
    "-" * 40,
]
for col, stats in meta_completeness.items():
    report_lines.append(f"  {col:<15} Filled: {stats['filled']:>6,} | Missing: {stats['missing']:>4,} | {stats['pct']:.2f}%")

report_lines += [
    "",
    "8. DUPLICATE CHECK",
    "-" * 40,
    f"  Duplicate Accessions : {dup_accession:,}",
    f"  Duplicate Sequences  : {dup_sequence:,}",
    "",
    "=" * 60,
    "  END OF REPORT",
    "=" * 60,
]

report_text = "\n".join(report_lines)
with open(out("h1n1_baseline_statistics.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)
print("✓ Saved h1n1_baseline_statistics.txt")

# ─────────────────────────────────────────────
# OUTPUT 3: JSON summary
# ─────────────────────────────────────────────
json_summary = {
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "original_total": total_original,
        "non_h1n1_removed": removed,
        "h1n1_total": total_n,
        "year_min": int(min_year),
        "year_max": int(max_year),
        "year_gaps_2009_2020": gaps
    },
    "length_stats": {
        "mean": round(float(df_h1n1['Length'].mean()), 2),
        "median": round(float(df_h1n1['Length'].median()), 2),
        "min": int(df_h1n1['Length'].min()),
        "max": int(df_h1n1['Length'].max()),
        "std": round(float(df_h1n1['Length'].std()), 2)
    },
    "completeness": {
        "complete_count": int(complete_n),
        "complete_pct": round(complete_n / total_n * 100, 2),
        "truncated_count": int(truncated_n),
        "truncated_pct": round(truncated_n / total_n * 100, 2)
    },
    "year_distribution": {str(int(row['Year'])): {"count": int(row['Count']), "completeness_rate": float(row['Completeness_Rate'])}
                          for _, row in year_completeness.iterrows()},
    "top_20_countries": {str(k): int(v) for k, v in country_counts.items()},
    "host_distribution": {str(k): int(v) for k, v in host_counts.items()},
    "metadata_completeness": meta_completeness,
    "duplicates": {
        "duplicate_accessions": int(dup_accession),
        "duplicate_sequences": int(dup_sequence)
    }
}

with open(out("h1n1_statistics_summary.json"), "w", encoding="utf-8") as f:
    json.dump(json_summary, f, indent=2)
print("✓ Saved h1n1_statistics_summary.json")

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
print(report_text)
