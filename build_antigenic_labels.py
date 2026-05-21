#!/usr/bin/env python3
"""
Task 2a — Build biologically grounded antigenic drift labels.
Uses published WHO/CDC cluster transition data for H3N2 and
published divergence eras for H1N1 post-pandemic lineage.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

# ── WHO/CDC H3N2 antigenic cluster map ────────────────────────────────────────
CLUSTER_MAP = {
    'HK68': {'years': range(1968, 1973), 'distance': 0},
    'EN72': {'years': range(1973, 1976), 'distance': 1},
    'VI75': {'years': range(1976, 1978), 'distance': 2},
    'TX77': {'years': range(1978, 1980), 'distance': 3},
    'BK79': {'years': range(1980, 1988), 'distance': 4},
    'SI87': {'years': range(1988, 1990), 'distance': 5},
    'BE89': {'years': range(1990, 1993), 'distance': 6},
    'BE92': {'years': range(1993, 1996), 'distance': 7},
    'WU95': {'years': range(1996, 1998), 'distance': 8},
    'SY97': {'years': range(1998, 2003), 'distance': 9},
    'FU02': {'years': range(2003, 2009), 'distance': 10},
    'PE09': {'years': range(2009, 2012), 'distance': 11},
    'VI11': {'years': range(2012, 2015), 'distance': 12},
    'SW13': {'years': range(2015, 2018), 'distance': 13},
    'HK14': {'years': range(2018, 2021), 'distance': 14},
}
# Build year → (cluster_name, distance) lookup
YEAR_TO_H3N2 = {}
for name, info in CLUSTER_MAP.items():
    for yr in info['years']:
        YEAR_TO_H3N2[yr] = (name, info['distance'])

# ── H1N1 era labels ───────────────────────────────────────────────────────────
def h1n1_label(year: int) -> tuple[str, int, int]:
    """Return (era_name, antigenic_distance, drift_binary) for a year."""
    if year <= 2008:
        return 'pre-pandemic', 0, 0
    elif year <= 2011:
        return 'pandemic-founding', 1, 0
    elif year <= 2014:
        return 'early-drift', 2, 1
    else:
        return 'immune-escape', 3, 1

# ── Load the full dataset ─────────────────────────────────────────────────────
print('Loading dataset ...')
df = pd.read_csv(ROOT / 'final_fixed_influenza_ha_v2ok.csv')

# ── H3N2 labels ───────────────────────────────────────────────────────────────
print('Building H3N2 antigenic labels ...')
h3n2 = df[df['Subtype'] == 'H3N2'][['Accession', 'Year', 'Subtype', 'Host']].copy()

def get_h3n2_row(year):
    yr = int(year)
    if yr in YEAR_TO_H3N2:
        name, dist = YEAR_TO_H3N2[yr]
    else:
        # Extrapolate: before 1968 → HK68, after 2020 → HK14
        if yr < 1968:
            name, dist = 'HK68', 0
        else:
            name, dist = 'HK14', 14
    return name, dist, int(dist >= 9)

h3n2[['cluster_name', 'antigenic_distance', 'drift_binary_label']] = \
    h3n2['Year'].apply(lambda y: pd.Series(get_h3n2_row(y)))

h3n2.to_csv(OUT / 'antigenic_labels_h3n2.csv', index=False)
print(f'  H3N2: {len(h3n2)} sequences')
print(f'  Positive (drift=1): {h3n2["drift_binary_label"].sum()}  |  '
      f'Negative (drift=0): {(h3n2["drift_binary_label"]==0).sum()}')
print(f'  Saved: outputs/antigenic_labels_h3n2.csv')

# ── H1N1 labels ───────────────────────────────────────────────────────────────
print('\nBuilding H1N1 antigenic labels ...')
h1n1 = df[df['Subtype'] == 'H1N1'][['Accession', 'Year', 'Subtype', 'Host']].copy()

rows = h1n1['Year'].apply(lambda y: pd.Series(h1n1_label(int(y))))
rows.columns = ['cluster_name', 'antigenic_distance', 'drift_binary_label']
h1n1 = pd.concat([h1n1, rows], axis=1)

h1n1.to_csv(OUT / 'antigenic_labels_h1n1.csv', index=False)
print(f'  H1N1: {len(h1n1)} sequences')
print(f'  Positive (drift=1): {h1n1["drift_binary_label"].sum()}  |  '
      f'Negative (drift=0): {(h1n1["drift_binary_label"]==0).sum()}')
print(f'  Saved: outputs/antigenic_labels_h1n1.csv')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\nH3N2 cluster distribution:')
print(h3n2.groupby(['cluster_name', 'antigenic_distance', 'drift_binary_label'])
         .size().reset_index(name='n').to_string(index=False))
print('\nH1N1 era distribution:')
print(h1n1.groupby(['cluster_name', 'antigenic_distance', 'drift_binary_label'])
         .size().reset_index(name='n').to_string(index=False))
print('\nDone.')
