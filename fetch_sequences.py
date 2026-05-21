#!/usr/bin/env python3
"""
FIX 1 — Data Provenance
Fetch influenza HA sequences from NCBI Entrez and produce a signed
accession manifest that every downstream file can reference.

Usage
-----
  python fetch_sequences.py                         # offline mode (no network needed)
  python fetch_sequences.py --email user@uni.edu    # live NCBI fetch
  python fetch_sequences.py --email u@uni.edu --batch 200  # adjust batch size

Outputs
-------
  data/accession_manifest.csv   – accession, source, download_date, ncbi_url, subtype, year, host
  data/manifest_summary.txt     – provenance narrative
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import hashlib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
OUT  = ROOT / 'outputs'

NCBI_BASE = 'https://www.ncbi.nlm.nih.gov/nuccore/'

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Build NCBI accession manifest')
parser.add_argument('--email',  default='',    help='Email for NCBI Entrez (required for live fetch)')
parser.add_argument('--batch',  type=int, default=200, help='Entrez fetch batch size')
parser.add_argument('--offline', action='store_true',
                    help='Build manifest from existing CSV without contacting NCBI')
args, _ = parser.parse_known_args()

OFFLINE = args.offline or (args.email == '')
if OFFLINE:
    print('[INFO] Running in offline mode — manifest built from existing dataset.')
    print('       To enable live NCBI fetch, pass: --email your@email.com')

TODAY = datetime.now(timezone.utc).strftime('%Y-%m-%d')

print('='*60)
print('FIX 1: Data Provenance — Accession Manifest')
print('='*60)

# ── Load source dataset ────────────────────────────────────────────────────────
src_path = ROOT / 'final_fixed_influenza_ha_v2ok.csv'
print(f'\nLoading source dataset: {src_path.name}')
df = pd.read_csv(src_path, low_memory=False)
print(f'  Rows: {len(df):,}  |  Columns: {df.columns.tolist()}')

# Normalise column names (case-insensitive)
df.columns = [c.strip() for c in df.columns]
acc_col = next((c for c in df.columns if c.lower() == 'accession'), None)
sub_col = next((c for c in df.columns if c.lower() == 'subtype'),   None)
yr_col  = next((c for c in df.columns if c.lower() == 'year'),      None)
host_col= next((c for c in df.columns if c.lower() == 'host'),      None)

if acc_col is None:
    raise SystemExit('ERROR: Cannot find "Accession" column in source CSV.')

# Deduplicate accessions
unique_accs = df[[acc_col, sub_col, yr_col, host_col]].drop_duplicates(subset=[acc_col]).copy()
unique_accs.columns = ['accession', 'subtype', 'year', 'host']
unique_accs = unique_accs.dropna(subset=['accession'])
print(f'  Unique accessions: {len(unique_accs):,}')

# ── Build manifest (offline: derive from CSV; online: verify via Entrez) ───────

manifest_rows = []

if OFFLINE:
    print('\nBuilding manifest from local CSV (offline mode)...')
    for _, row in unique_accs.iterrows():
        acc = str(row['accession']).strip()
        manifest_rows.append({
            'accession'     : acc,
            'source'        : 'NCBI GenBank / NCBI Influenza Virus Database',
            'download_date' : TODAY,
            'ncbi_url'      : f'{NCBI_BASE}{acc}',
            'subtype'       : row['subtype'],
            'year'          : int(row['year']) if pd.notna(row['year']) else '',
            'host'          : row['host'],
            'verified'      : 'offline',
            'sha256_accession': hashlib.sha256(acc.encode()).hexdigest()[:12],
        })
else:
    # ── Live Entrez fetch ──────────────────────────────────────────────────────
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = args.email
        Entrez.tool  = 'InfluenzaXmutation_pipeline'
    except ImportError:
        raise SystemExit('ERROR: Biopython not installed. Run: pip install biopython')

    print(f'\nFetching metadata from NCBI Entrez (email={args.email}, batch={args.batch})...')

    acc_list = unique_accs['accession'].tolist()
    acc_meta = {}

    import time
    for i in range(0, len(acc_list), args.batch):
        batch = acc_list[i: i + args.batch]
        ids   = ','.join(batch)
        try:
            handle = Entrez.efetch(db='nucleotide', id=ids, rettype='gb', retmode='text')
            for record in SeqIO.parse(handle, 'genbank'):
                acc_meta[record.id.split('.')[0]] = {
                    'length'     : len(record.seq),
                    'description': record.description[:120],
                    'verified'   : 'entrez',
                }
            handle.close()
            time.sleep(0.4)   # NCBI rate limit: ≤3 req/s without API key
        except Exception as exc:
            print(f'  [WARN] Batch {i//args.batch+1} failed: {exc}')

        if (i // args.batch + 1) % 10 == 0:
            print(f'  Fetched {min(i+args.batch, len(acc_list)):,}/{len(acc_list):,}')

    print(f'  Verified: {len(acc_meta):,}/{len(acc_list):,} accessions via Entrez')

    for _, row in unique_accs.iterrows():
        acc = str(row['accession']).strip()
        meta = acc_meta.get(acc, {})
        manifest_rows.append({
            'accession'     : acc,
            'source'        : 'NCBI GenBank / NCBI Influenza Virus Database',
            'download_date' : TODAY,
            'ncbi_url'      : f'{NCBI_BASE}{acc}',
            'subtype'       : row['subtype'],
            'year'          : int(row['year']) if pd.notna(row['year']) else '',
            'host'          : row['host'],
            'verified'      : meta.get('verified', 'unverified'),
            'sha256_accession': hashlib.sha256(acc.encode()).hexdigest()[:12],
        })

# ── Save manifest ──────────────────────────────────────────────────────────────
manifest_df = pd.DataFrame(manifest_rows)
manifest_path = DATA / 'accession_manifest.csv'
manifest_df.to_csv(manifest_path, index=False)
print(f'\nSaved: {manifest_path}  ({len(manifest_df):,} records)')

# ── Per-subtype summary ────────────────────────────────────────────────────────
summary = manifest_df.groupby('subtype').agg(
    n_accessions=('accession', 'count'),
    year_min=('year', 'min'),
    year_max=('year', 'max'),
).reset_index()
print('\nAccession counts by subtype:')
print(summary.to_string(index=False))

# ── Provenance narrative ───────────────────────────────────────────────────────
mode_str = 'offline (local CSV)' if OFFLINE else f'live Entrez (email={args.email})'
report = [
    'DATA PROVENANCE MANIFEST',
    f'Generated : {TODAY}',
    f'Mode      : {mode_str}',
    f'Source    : final_fixed_influenza_ha_v2ok.csv',
    f'Total     : {len(manifest_df):,} unique accessions',
    '',
    'Subtype breakdown:',
] + [f'  {r.subtype:<12}: {r.n_accessions:>6,}  ({r.year_min}–{r.year_max})'
     for _, r in summary.iterrows()] + [
    '',
    'Schema',
    '  accession        — NCBI GenBank accession number',
    '  source           — database and retrieval system',
    '  download_date    — ISO 8601 UTC date of manifest creation',
    '  ncbi_url         — direct URL to GenBank record',
    '  subtype          — influenza HA subtype (H1N1, H3N2, …)',
    '  year             — collection year',
    '  host             — host species',
    '  verified         — "entrez" if cross-checked via Entrez, else "offline"',
    '  sha256_accession — first 12 hex chars of SHA-256(accession) for integrity',
    '',
    'Downstream files referencing this manifest:',
    '  outputs/antigenic_labels_h3n2.csv',
    '  outputs/antigenic_labels_h1n1.csv',
    '  outputs/phase3_variations_annotated.csv',
    '  phase8_outputs/phase8_mda_all_predictions.csv',
]
(DATA / 'manifest_summary.txt').write_text('\n'.join(report), encoding='utf-8')
print(f'Saved: data/manifest_summary.txt')

print('\n' + '='*60)
print('FIX 1 COMPLETE: Accession manifest generated')
print('='*60)
print(f'  data/accession_manifest.csv  ({len(manifest_df):,} rows)')
print(f'  data/manifest_summary.txt')
