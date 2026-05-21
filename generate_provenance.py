#!/usr/bin/env python3
"""
Task 5 — Data Provenance Documentation.
Generates: sequence_accession_list.txt, DATA_PROVENANCE.md,
           dataset_summary_stats.csv
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import hashlib, time
import numpy as np
import pandas as pd
import torch, sklearn, scipy
from pathlib import Path

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

print('Loading dataset ...')
df = pd.read_csv(ROOT / 'final_fixed_influenza_ha_v2ok.csv')
print(f'  {len(df):,} sequences loaded')

# ── OUTPUT A: sequence_accession_list.txt ─────────────────────────────────────
print('\nWriting sequence_accession_list.txt ...')
cols = ['Accession', 'Subtype', 'Host', 'Year', 'Country', 'Length', 'VirusName']
with open(OUT / 'sequence_accession_list.txt', 'w', encoding='utf-8') as f:
    f.write('#' + '\t'.join(cols) + '\n')
    for _, row in df[cols].iterrows():
        f.write('\t'.join(str(row[c]) for c in cols) + '\n')
print(f'  Written: {len(df):,} accessions')

# ── Compute SHA-256 ────────────────────────────────────────────────────────────
print('Computing SHA-256 of source CSV ...')
sha = hashlib.sha256()
raw_path = ROOT / 'final_fixed_influenza_ha_v2ok.csv'
with open(raw_path, 'rb') as f:
    for chunk in iter(lambda: f.read(65536), b''):
        sha.update(chunk)
sha256 = sha.hexdigest()
print(f'  SHA-256: {sha256}')

# ── Counts ─────────────────────────────────────────────────────────────────────
subtypes_all = sorted(df['Subtype'].unique())
n_human  = (df['Host'] == 'Human').sum()
n_avian  = (df['Host'] == 'Avian').sum()
n_countries = df['Country'].nunique()

h1  = df[df['Subtype'] == 'H1N1']
h3  = df[df['Subtype'] == 'H3N2']

h1_human = (h1['Host'] == 'Human').sum()
h1_avian = (h1['Host'] == 'Avian').sum()
h3_human = (h3['Host'] == 'Human').sum()
h3_avian = (h3['Host'] == 'Avian').sum()

# ── Package versions ───────────────────────────────────────────────────────────
import sys as _sys
py_ver = _sys.version.split(' ')[0]
import pandas as _pd; import numpy as _np
pkg_versions = (f'pandas {_pd.__version__}, numpy {_np.__version__}, '
                f'scipy {scipy.__version__}, sklearn {sklearn.__version__}, '
                f'torch {torch.__version__}')

# ── OUTPUT B: DATA_PROVENANCE.md ──────────────────────────────────────────────
print('Writing DATA_PROVENANCE.md ...')
md_lines = [
    '# Data Provenance',
    f'Generated: {time.strftime("%Y-%m-%d %H:%M")}',
    '',
    '## Dataset Summary',
    f'- Total sequences: {len(df):,}',
    f'- Subtypes represented: {", ".join(subtypes_all[:20])} ... ({len(subtypes_all)} total)',
    f'- Hosts: Human ({n_human:,}), Avian ({n_avian:,}), Other ({len(df)-n_human-n_avian:,})',
    f'- Year range: {int(df["Year"].min())}–{int(df["Year"].max())}',
    f'- Countries: {n_countries} unique',
    f'- Median sequence length: {int(df["Length"].median())} aa',
    f'- Mean sequence length: {df["Length"].mean():.1f} aa',
    '',
    '## Analysis Subset (H1N1 + H3N2 only)',
    f'- H1N1: {len(h1):,} sequences (Human: {h1_human:,}, Avian: {h1_avian:,}) '
    f'| Years: {int(h1["Year"].min())}–{int(h1["Year"].max())}',
    f'- H3N2: {len(h3):,} sequences (Human: {h3_human:,}, Avian: {h3_avian:,}) '
    f'| Years: {int(h3["Year"].min())}–{int(h3["Year"].max())}',
    '',
    '## Data Availability Statement',
    '',
    '"All sequences analysed in this study are publicly available from the NCBI '
    'Influenza Virus Resource (https://www.ncbi.nlm.nih.gov/genomes/FLU/). '
    'Protein accession numbers for all 31,619 sequences are provided in '
    'Supplementary Table S1 (sequence_accession_list.txt). The curated analysis '
    'dataset is available at [GITHUB REPO URL]."',
    '',
    '## Reproducibility',
    '',
    f'SHA-256 hash of final_fixed_influenza_ha_v2ok.csv: `{sha256}`',
    f'',
    f'Python version: {py_ver}',
    f'Key package versions: {pkg_versions}',
    '',
    '## Acknowledgements',
    '',
    '"Sequence data were obtained from the NCBI Influenza Virus Database. '
    'The authors thank the originating laboratories for submitting sequences."',
]
(OUT / 'DATA_PROVENANCE.md').write_text('\n'.join(md_lines), encoding='utf-8')
print('  Written: DATA_PROVENANCE.md')

# ── OUTPUT C: dataset_summary_stats.csv ───────────────────────────────────────
print('Writing dataset_summary_stats.csv ...')
target_subtypes = ['H1N1', 'H3N2', 'H5N1', 'H9N2']
rows = []

for st in target_subtypes:
    sub = df[df['Subtype'] == st]
    if len(sub) == 0:
        rows.append(dict(Subtype=st, N_total=0, N_human=0, N_avian=0,
                         Year_min='N/A', Year_max='N/A', N_countries=0,
                         Mean_length=0, Std_length=0, Median_length=0))
        continue
    rows.append(dict(
        Subtype=st,
        N_total=len(sub),
        N_human=(sub['Host']=='Human').sum(),
        N_avian=(sub['Host']=='Avian').sum(),
        Year_min=int(sub['Year'].min()),
        Year_max=int(sub['Year'].max()),
        N_countries=sub['Country'].nunique(),
        Mean_length=round(sub['Length'].mean(), 1),
        Std_length=round(sub['Length'].std(), 1),
        Median_length=int(sub['Length'].median()),
    ))

# All subtypes summary row
rows.append(dict(
    Subtype='All subtypes',
    N_total=len(df),
    N_human=n_human,
    N_avian=n_avian,
    Year_min=int(df['Year'].min()),
    Year_max=int(df['Year'].max()),
    N_countries=n_countries,
    Mean_length=round(df['Length'].mean(), 1),
    Std_length=round(df['Length'].std(), 1),
    Median_length=int(df['Length'].median()),
))

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT / 'dataset_summary_stats.csv', index=False)
print('  Written: dataset_summary_stats.csv')
print(stats_df.to_string(index=False))

print(f'\nSHA-256: {sha256}')
print('Provenance generation complete.')
