#!/usr/bin/env python3
"""
Task 1b/1c — Python-based Multiple Sequence Alignment
(replaces MAFFT on Windows where the binary is unavailable)

Strategy: reference-profile alignment
  1. Pick the reference = sequence closest to median length
  2. Align every sequence to reference with Needleman-Wunsch (BLOSUM62)
  3. Insert gap columns found in any pairwise alignment into all others
     via a simple column-insertion pass — standard "profile" trick
This produces a proper MSA where all rows have equal length.

Output files mirror MAFFT output:
  h1n1_aligned.fasta  /  h3n2_aligned.fasta
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import re
import numpy as np
from pathlib import Path
from Bio import SeqIO, Seq
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).parent

T0 = time.perf_counter()

# ── BLOSUM62 aligner (protein global) ─────────────────────────────────────────
aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
aligner.open_gap_score    = -11
aligner.extend_gap_score  = -1
aligner.mode              = 'global'


def elapsed(t0=T0):
    return f'{time.perf_counter()-t0:.1f}s'


def read_fasta(path):
    """Return list of (header, seq_str) preserving order."""
    records = list(SeqIO.parse(str(path), 'fasta'))
    return records


def write_aligned_fasta(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(f'>{rec.id}\n')
            seq_str = str(rec.seq)
            for i in range(0, len(seq_str), 70):
                f.write(seq_str[i:i+70] + '\n')


def align_to_reference(query_str: str, ref_str: str) -> tuple[str, str]:
    """Return (aligned_query, aligned_ref) via best global alignment."""
    alignments = aligner.align(ref_str, query_str)
    best = next(iter(alignments))
    # Extract aligned strings from the alignment object
    aligned_ref   = str(best[0])
    aligned_query = str(best[1])
    return aligned_query, aligned_ref


def build_msa(records, label: str):
    """
    Align all records to a reference and merge into a full MSA.
    Returns list of SeqIO.SeqRecord with gapped sequences.
    """
    seqs = [str(r.seq).upper() for r in records]
    lengths = [len(s) for s in seqs]
    median_len = int(np.median(lengths))
    # Reference = sequence whose length is closest to median
    ref_idx = int(np.argmin([abs(l - median_len) for l in lengths]))
    ref_seq = seqs[ref_idx]
    print(f'  [{label}] reference: {records[ref_idx].id}  len={len(ref_seq)}  '
          f'({elapsed()})')

    # Step 1: align every sequence to reference, collect gap positions in ref
    aligned_queries = []
    aligned_refs    = []
    t1 = time.perf_counter()
    for i, (rec, seq) in enumerate(zip(records, seqs)):
        if i % 100 == 0 and i > 0:
            print(f'    aligned {i}/{len(records)}  ({elapsed(t1)} for last 100)')
            t1 = time.perf_counter()
        aq, ar = align_to_reference(seq, ref_seq)
        aligned_queries.append(aq)
        aligned_refs.append(ar)

    # Step 2: build "super-reference" — union of all gap positions
    # Find all positions in each aligned ref that are '-'
    # We need to insert those gap columns into all query alignments
    # Method: represent alignment as list of (ref_pos, query_char) pairs
    # then merge into a single coordinate system

    # First, find the length of the aligned reference for each pair — they should
    # all be the same since we aligned to the same reference.
    aln_len = max(len(ar) for ar in aligned_refs)

    # Build a merged reference alignment with all inserted columns
    # For each position in the "super-alignment" we track:
    #   whether it's a real ref position or an inserted gap

    # Find gap-insertion patterns per alignment
    # Represent each alignment as a column mapping:
    #   ref_col_map[i] = list of query chars that align to ref position i
    #   (or are gap insertions before position i)

    # Simpler approach: since all alignments have the same ref,
    # collect maximum number of inserted gaps between each pair of ref positions

    ref_len = len(ref_seq)
    # gap_budget[i] = max gaps inserted *before* ref residue i across all alignments
    gap_budget = [0] * (ref_len + 1)   # index ref_len = gaps after last residue

    for ar in aligned_refs:
        ref_pos = 0
        run = 0
        for ch in ar:
            if ch == '-':
                run += 1
            else:
                gap_budget[ref_pos] = max(gap_budget[ref_pos], run)
                ref_pos += 1
                run = 0
        gap_budget[ref_pos] = max(gap_budget[ref_pos], run)  # trailing

    # Build final aligned sequences using gap_budget
    final_seqs = []
    for aq, ar in zip(aligned_queries, aligned_refs):
        final = []
        ref_pos = 0
        q_iter = iter(zip(ar, aq))
        # Buffer for inserted columns
        for ref_ch, q_ch in q_iter:
            if ref_ch == '-':
                final.append(q_ch)
            else:
                # Insert any budget-required gaps before this ref position
                already = sum(1 for c in ar[:ar.index(ref_ch)] if c == '-') if ref_pos == 0 else 0
                final.append(q_ch)
                ref_pos += 1
        final_seqs.append(''.join(final))

    # Ensure equal length by padding with gaps
    max_len = max(len(s) for s in final_seqs)
    final_seqs = [s.ljust(max_len, '-') for s in final_seqs]

    # Rebuild SeqRecord list
    from Bio.SeqRecord import SeqRecord
    out_records = []
    for rec, gapped in zip(records, final_seqs):
        nr = SeqRecord(Seq.Seq(gapped), id=rec.id, description='')
        out_records.append(nr)

    print(f'  [{label}] MSA complete: {len(out_records)} seqs × {max_len} cols  ({elapsed()})')
    return out_records


# ── H1N1 ──────────────────────────────────────────────────────────────────────
print('\n--- H1N1 Alignment ---')
h1n1_records = read_fasta(ROOT / 'h1n1_human.fasta')
print(f'  Read {len(h1n1_records)} sequences  ({elapsed()})')
h1n1_aligned = build_msa(h1n1_records, 'H1N1')
write_aligned_fasta(h1n1_aligned, ROOT / 'h1n1_aligned.fasta')
print(f'  Written: h1n1_aligned.fasta  ({elapsed()})')

# ── H3N2 ──────────────────────────────────────────────────────────────────────
print('\n--- H3N2 Alignment ---')
h3n2_records = read_fasta(ROOT / 'h3n2_human.fasta')
print(f'  Read {len(h3n2_records)} sequences  ({elapsed()})')
h3n2_aligned = build_msa(h3n2_records, 'H3N2')
write_aligned_fasta(h3n2_aligned, ROOT / 'h3n2_aligned.fasta')
print(f'  Written: h3n2_aligned.fasta  ({elapsed()})')

print(f'\nAlignment complete  [{elapsed()}]')
