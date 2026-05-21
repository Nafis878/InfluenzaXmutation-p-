#!/usr/bin/env python3
"""
Task 1a — Export H1N1 and H3N2 Human sequences to FASTA.
Stratified sample of up to 800 sequences per subtype, random_state=42.
Header format: >Accession|Year|Country
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).parent
OUT   = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

T0 = time.perf_counter()

print('Loading dataset ...')
df = pd.read_csv(ROOT / 'final_fixed_influenza_ha_v2ok.csv')
print(f'  {len(df):,} total sequences loaded')


def stratified_sample(sub_df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Stratified random sample of n rows across Year bins."""
    rng = np.random.default_rng(seed)
    sub_df = sub_df.copy()
    sub_df['_year_bin'] = pd.cut(sub_df['Year'], bins=20, labels=False)
    counts = sub_df['_year_bin'].value_counts()
    # Proportional allocation with at least 1 per bin
    total = len(sub_df)
    alloc = (counts / total * n).astype(int)
    alloc = alloc.clip(lower=1)
    # Adjust to hit exactly n
    diff = n - alloc.sum()
    if diff > 0:
        for idx in counts.sort_values(ascending=False).index:
            if diff <= 0:
                break
            alloc[idx] += 1
            diff -= 1
    elif diff < 0:
        for idx in counts.sort_values(ascending=True).index:
            if diff >= 0:
                break
            if alloc[idx] > 1:
                alloc[idx] -= 1
                diff += 1

    pieces = []
    for bin_id, k in alloc.items():
        chunk = sub_df[sub_df['_year_bin'] == bin_id]
        k = min(k, len(chunk))
        if k > 0:
            pieces.append(chunk.sample(k, random_state=seed, replace=False))
    result = pd.concat(pieces).drop(columns=['_year_bin'])
    # If still short, top up randomly
    if len(result) < n:
        remaining = sub_df[~sub_df.index.isin(result.index)]
        extra = min(n - len(result), len(remaining))
        if extra > 0:
            result = pd.concat([result,
                                 remaining.sample(extra, random_state=seed)])
    return result.head(n)


def write_fasta(df_sub: pd.DataFrame, path: Path) -> int:
    rows_written = 0
    with path.open('w', encoding='utf-8') as f:
        for _, row in df_sub.iterrows():
            country = str(row.get('Country', 'Unknown')).replace(' ', '_')
            header  = f">{row['Accession']}|{int(row['Year'])}|{country}"
            seq     = str(row['Sequence']).strip().upper()
            # Write in 70-char line blocks
            f.write(header + '\n')
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + '\n')
            rows_written += 1
    return rows_written


CAP = 800

for subtype, fname in [('H1N1', 'h1n1_human.fasta'), ('H3N2', 'h3n2_human.fasta')]:
    human = df[(df['Subtype'] == subtype) & (df['Host'] == 'Human')].copy()
    print(f'\n{subtype}: {len(human):,} Human sequences available')

    if len(human) > CAP:
        sampled = stratified_sample(human, CAP, seed=42)
        print(f'  Stratified sample: {len(sampled)} sequences')
    else:
        sampled = human
        print(f'  Using all {len(sampled)} (< {CAP} cap)')

    fasta_path = ROOT / fname
    n = write_fasta(sampled, fasta_path)
    size_kb = fasta_path.stat().st_size / 1024
    print(f'  Written: {fasta_path.name}  ({n} seqs, {size_kb:.0f} KB)')

    # Also write a companion metadata CSV for later join
    sampled[['Accession', 'Host', 'Subtype', 'Year', 'Length', 'Country',
             'VirusName', 'Protein']].to_csv(
        OUT / f'{subtype.lower()}_human_meta.csv', index=False)
    print(f'  Metadata: outputs/{subtype.lower()}_human_meta.csv')

print(f'\nDone in {time.perf_counter()-T0:.1f}s')
