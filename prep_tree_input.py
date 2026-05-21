#!/usr/bin/env python3
"""
Task 3b — Prepare stratified FASTA inputs for phylogenetic analysis.
H1N1: 150 seqs stratified by decade (2009s get 50% of slots)
H3N2: 200 seqs stratified by decade
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import Seq

ROOT = Path(__file__).parent


def decade(year):
    return (year // 10) * 10


def stratified_fasta_sample(fasta_path, n_total, seed=42,
                             boost_decade=2009, boost_frac=0.5):
    """
    Read aligned FASTA, stratify by decade, return n_total records.
    boost_decade gets boost_frac of slots; rest distributed evenly.
    """
    records = list(SeqIO.parse(str(fasta_path), 'fasta'))
    rng = np.random.default_rng(seed)

    # Parse year from header Accession|Year|Country
    for rec in records:
        parts = rec.id.split('|')
        rec.annotations['year'] = int(parts[1]) if len(parts) > 1 else 2000

    by_decade = {}
    for rec in records:
        d = decade(rec.annotations['year'])
        by_decade.setdefault(d, []).append(rec)

    decades = sorted(by_decade.keys())
    n_boost     = int(n_total * boost_frac)
    n_rest      = n_total - n_boost
    other_decs  = [d for d in decades if d != boost_decade]
    per_other   = n_rest // max(1, len(other_decs))

    selected = []
    # Boost decade
    pool = by_decade.get(boost_decade, [])
    k = min(n_boost, len(pool))
    idxs = rng.choice(len(pool), k, replace=False)
    selected.extend([pool[i] for i in idxs])

    # Other decades
    for d in other_decs:
        pool = by_decade[d]
        k = min(per_other, len(pool))
        if k > 0:
            idxs = rng.choice(len(pool), k, replace=False)
            selected.extend([pool[i] for i in idxs])

    # Top-up if short
    if len(selected) < n_total:
        used_ids = {r.id for r in selected}
        rest = [r for r in records if r.id not in used_ids]
        extra = min(n_total - len(selected), len(rest))
        if extra > 0:
            idxs = rng.choice(len(rest), extra, replace=False)
            selected.extend([rest[i] for i in idxs])

    return selected[:n_total]


def write_fasta(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(f'>{rec.id}\n')
            seq_str = str(rec.seq)
            for i in range(0, len(seq_str), 70):
                f.write(seq_str[i:i+70] + '\n')
    return len(records)


# H1N1 tree input
print('Preparing H1N1 tree input (150 seqs) ...')
h1_sel = stratified_fasta_sample(ROOT / 'h1n1_aligned.fasta', 150,
                                  boost_decade=2009, boost_frac=0.5)
n = write_fasta(h1_sel, ROOT / 'h1n1_tree_input.fasta')
print(f'  Written: h1n1_tree_input.fasta  ({n} seqs)')

# H3N2 tree input
print('Preparing H3N2 tree input (200 seqs) ...')
h3_sel = stratified_fasta_sample(ROOT / 'h3n2_aligned.fasta', 200,
                                  boost_decade=2009, boost_frac=0.3)
n = write_fasta(h3_sel, ROOT / 'h3n2_tree_input.fasta')
print(f'  Written: h3n2_tree_input.fasta  ({n} seqs)')
print('Done.')
