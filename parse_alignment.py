#!/usr/bin/env python3
"""
Task 1d — Parse aligned FASTAs back to CSV.
Rebuilds DataFrames with original metadata columns, replaces Sequence
with the gapped aligned sequence and recalculates Length.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from pathlib import Path
from Bio import SeqIO

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'


def parse_aligned(fasta_path: Path, meta_csv: Path, out_csv: Path) -> pd.DataFrame:
    # Load metadata
    meta = pd.read_csv(meta_csv)
    meta['Accession'] = meta['Accession'].astype(str)

    # Parse FASTA — header is Accession|Year|Country
    rows = []
    for rec in SeqIO.parse(str(fasta_path), 'fasta'):
        acc = rec.id.split('|')[0]
        rows.append({'Accession': acc, 'AlignedSequence': str(rec.seq)})
    aln_df = pd.DataFrame(rows)

    # Merge on Accession
    merged = meta.merge(aln_df, on='Accession', how='inner')
    merged['Sequence'] = merged['AlignedSequence']
    merged['Length']   = merged['Sequence'].str.len()
    merged = merged.drop(columns=['AlignedSequence'])

    merged.to_csv(out_csv, index=False)
    print(f'  {out_csv.name}: {len(merged)} rows, aligned_length={merged["Length"].iloc[0]}')
    return merged


print('Parsing aligned FASTAs ...')
h1 = parse_aligned(ROOT / 'h1n1_aligned.fasta',
                   OUT  / 'h1n1_human_meta.csv',
                   OUT  / 'h1n1_aligned.csv')
h3 = parse_aligned(ROOT / 'h3n2_aligned.fasta',
                   OUT  / 'h3n2_human_meta.csv',
                   OUT  / 'h3n2_aligned.csv')
print('Done.')
