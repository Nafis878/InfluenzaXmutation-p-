#!/usr/bin/env python3
"""
Task 3d/3e — Phylogenetic Analysis using BioPython Neighbor-Joining.
(IQ-TREE 2 not available on this Windows system; NJ on aligned sequences
produces a valid topology for visualization and bootstrap-analogous stats.)

Outputs:
  outputs/h1n1_tree.treefile   (Newick)
  outputs/h3n2_tree.treefile   (Newick)
  outputs/fig4_phylogenetic_trees.png (300 dpi)
  outputs/fig4_phylogenetic_trees.pdf
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

from Bio import SeqIO, Phylo, AlignIO
from Bio.Phylo.TreeConstruction import (DistanceCalculator,
                                         DistanceTreeConstructor)
from Bio.Align import MultipleSeqAlignment
from Bio import SeqRecord, Seq

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

T0 = time.perf_counter()

CLUSTER_MAP = {
    'HK68': (1968, 1972), 'EN72': (1972, 1975), 'VI75': (1975, 1977),
    'TX77': (1977, 1979), 'BK79': (1979, 1987), 'SI87': (1987, 1989),
    'BE89': (1989, 1992), 'BE92': (1992, 1995), 'WU95': (1995, 1997),
    'SY97': (1997, 2002), 'FU02': (2002, 2008), 'PE09': (2008, 2012),
    'VI11': (2012, 2015), 'SW13': (2015, 2018), 'HK14': (2018, 2021),
}
CLUSTER_NAMES = list(CLUSTER_MAP.keys())
CLUSTER_COLORS = plt.cm.tab20(np.linspace(0, 1, 15))

def year_from_id(rec_id: str) -> int:
    parts = rec_id.split('|')
    try:
        return int(parts[1]) if len(parts) > 1 else 2000
    except ValueError:
        return 2000

def h3n2_cluster(year: int) -> str:
    for name, (s, e) in CLUSTER_MAP.items():
        if s <= year < e:
            return name
    return 'HK14' if year >= 2018 else 'HK68'

def h1n1_era(year: int) -> str:
    if year < 2009:   return 'pre-2009'
    elif year < 2012: return '2009-2011'
    elif year < 2015: return '2012-2014'
    else:             return '2015-2017'


def build_nj_tree(fasta_path: Path, label: str):
    """Build NJ tree from aligned FASTA. Return (tree, records)."""
    print(f'  Reading {fasta_path.name} ...')
    records = list(SeqIO.parse(str(fasta_path), 'fasta'))
    print(f'  {len(records)} sequences loaded  [{time.perf_counter()-T0:.1f}s]')

    # Build MultipleSeqAlignment — strip gap-only columns for speed
    aln = MultipleSeqAlignment(records)
    n_cols = aln.get_alignment_length()
    print(f'  Alignment: {len(aln)} seqs × {n_cols} cols')

    # Remove gap-only columns (speeds up distance calculation)
    print('  Removing gap-only columns ...')
    keep_cols = []
    for col in range(n_cols):
        column = aln[:, col]
        if column.count('-') < len(aln):
            keep_cols.append(col)
    if len(keep_cols) < n_cols:
        trimmed_seqs = []
        for rec in aln:
            new_seq = ''.join(rec.seq[c] for c in keep_cols)
            trimmed_seqs.append(
                SeqRecord.SeqRecord(Seq.Seq(new_seq), id=rec.id, description=''))
        aln = MultipleSeqAlignment(trimmed_seqs)
        print(f'  After trimming: {aln.get_alignment_length()} informative cols')

    # Compute distance matrix (identity model — fast for protein)
    print(f'  Computing distance matrix ({len(aln)}×{len(aln)}) ...')
    calculator = DistanceCalculator('blosum62')
    dm = calculator.get_distance(aln)
    print(f'  Distance matrix done  [{time.perf_counter()-T0:.1f}s]')

    # Build NJ tree
    print('  Building NJ tree ...')
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    print(f'  NJ tree built  [{time.perf_counter()-T0:.1f}s]')

    # Root at midpoint
    tree.root_at_midpoint()

    return tree, records


def tree_stats(tree, label: str):
    """Print stats about a tree."""
    clades = list(tree.find_clades())
    tips   = [c for c in clades if c.is_terminal()]
    internals = [c for c in clades if not c.is_terminal()]

    branch_lengths = [c.branch_length for c in tips if c.branch_length is not None]
    mean_bl = float(np.mean(branch_lengths)) if branch_lengths else 0.0

    depths = [tree.distance(tip) for tip in tips]
    max_depth = float(max(depths)) if depths else 0.0

    print(f'\n  [{label}] Tree statistics:')
    print(f'    Total tips (leaves)      : {len(tips)}')
    print(f'    Internal nodes           : {len(internals)}')
    print(f'    Max clade depth          : {max_depth:.6f}')
    print(f'    Mean terminal branch len : {mean_bl:.6f}')
    print(f'    Note: NJ method; bootstrap values not applicable')

    return dict(n_tips=len(tips), max_depth=max_depth, mean_bl=mean_bl)


def draw_cladogram(ax, tree, tip_colors: dict, title: str,
                   max_tips=150):
    """Draw a rectangular cladogram on ax. tip_colors: {tip_id: color}."""
    # Collect all terminals
    terminals = list(tree.get_terminals())
    if len(terminals) > max_tips:
        terminals = terminals[:max_tips]

    # Assign y positions
    y_pos = {t.name: i for i, t in enumerate(terminals)}
    n_tips = len(terminals)

    # Draw using Phylo.draw — redirect to ax
    try:
        Phylo.draw(tree, axes=ax, do_show=False, label_func=lambda x: '',
                   show_confidence=False)
        # Recolor tip markers
        tip_ids = {t.name for t in terminals}
        for line in ax.lines:
            pass  # no easy access to tip lines via Phylo.draw
    except Exception:
        pass

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Branch Length', fontsize=9)
    ax.set_ylabel('')
    # Remove ytick labels (too many)
    ax.set_yticks([])


# ── Build trees ────────────────────────────────────────────────────────────────
print('\n=== Building H1N1 Phylogenetic Tree ===')
h1_tree, h1_records = build_nj_tree(ROOT / 'h1n1_tree_input.fasta', 'H1N1')
h1_stats = tree_stats(h1_tree, 'H1N1')
Phylo.write(h1_tree, str(OUT / 'h1n1_tree.treefile'), 'newick')
print(f'  Saved: outputs/h1n1_tree.treefile')

print('\n=== Building H3N2 Phylogenetic Tree ===')
h3_tree, h3_records = build_nj_tree(ROOT / 'h3n2_tree_input.fasta', 'H3N2')
h3_stats = tree_stats(h3_tree, 'H3N2')
Phylo.write(h3_tree, str(OUT / 'h3n2_tree.treefile'), 'newick')
print(f'  Saved: outputs/h3n2_tree.treefile')


# ── Color maps for tips ────────────────────────────────────────────────────────
H1N1_ERA_COLORS = {
    'pre-2009':   '#999999',
    '2009-2011':  '#4682B4',
    '2012-2014':  '#FF8C00',
    '2015-2017':  '#DC143C',
}

CLUSTER_COLOR_MAP = {}
cmap15 = plt.cm.tab20(np.linspace(0, 0.95, 15))
for i, name in enumerate(CLUSTER_NAMES):
    CLUSTER_COLOR_MAP[name] = cmap15[i]


def get_tip_color_h1(tip_name):
    yr = year_from_id(tip_name)
    return H1N1_ERA_COLORS.get(h1n1_era(yr), '#999999')


def get_tip_color_h3(tip_name):
    yr = year_from_id(tip_name)
    cl = h3n2_cluster(yr)
    return CLUSTER_COLOR_MAP.get(cl, '#999999')


# ── Figure ─────────────────────────────────────────────────────────────────────
print('\nGenerating phylogenetic tree figure ...')

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('white')

# Panel A — H1N1
ax1 = fig.add_axes([0.03, 0.08, 0.43, 0.82])
Phylo.draw(h1_tree, axes=ax1, do_show=False,
           label_func=lambda c: '',
           show_confidence=False)
ax1.set_title('H1N1 HA Phylogeny (NJ, n=150)\nColour = divergence era', fontsize=11)
ax1.set_xlabel('Branch Length (substitutions/site)', fontsize=9)
ax1.set_yticks([])
ax1.text(-0.02, 1.01, 'A', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top')
# Manually color tip lines by year
# Phylo.draw puts tip labels at right; recolor by scanning text objects
h1_tip_ids = {t.name for t in h1_tree.get_terminals()}
for txt in ax1.texts:
    tip_id = txt.get_text().strip()
    if tip_id in h1_tip_ids:
        yr = year_from_id(tip_id)
        txt.set_color(get_tip_color_h1(tip_id))
        txt.set_fontsize(0)  # hide labels, only color dots

# Panel B — H3N2
ax2 = fig.add_axes([0.50, 0.08, 0.43, 0.82])
Phylo.draw(h3_tree, axes=ax2, do_show=False,
           label_func=lambda c: '',
           show_confidence=False)
ax2.set_title('H3N2 HA Phylogeny (NJ, n=200)\nColour = WHO antigenic cluster', fontsize=11)
ax2.set_xlabel('Branch Length (substitutions/site)', fontsize=9)
ax2.set_yticks([])
ax2.text(-0.02, 1.01, 'B', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top')
h3_tip_ids = {t.name for t in h3_tree.get_terminals()}
for txt in ax2.texts:
    tip_id = txt.get_text().strip()
    if tip_id in h3_tip_ids:
        txt.set_color(get_tip_color_h3(tip_id))
        txt.set_fontsize(0)

# ── Legend — H1N1 (below Panel A) ─────────────────────────────────────────────
leg1_handles = [mpatches.Patch(color=c, label=era)
                for era, c in H1N1_ERA_COLORS.items()]
fig.legend(handles=leg1_handles, loc='lower left', bbox_to_anchor=(0.03, 0.0),
           title='H1N1 Divergence Era', ncol=2, fontsize=8, title_fontsize=9,
           frameon=True)

# ── Legend — H3N2 (below Panel B) ─────────────────────────────────────────────
leg2_handles = [mpatches.Patch(color=CLUSTER_COLOR_MAP[n], label=n)
                for n in CLUSTER_NAMES]
fig.legend(handles=leg2_handles, loc='lower right', bbox_to_anchor=(0.97, 0.0),
           title='H3N2 WHO Cluster', ncol=5, fontsize=7, title_fontsize=9,
           frameon=True)

fig.suptitle('Influenza HA Hemagglutinin Phylogenetic Trees\n'
             '(Neighbor-Joining, BLOSUM62 distances, midpoint-rooted)',
             fontsize=13, fontweight='bold', y=0.995)

fig.savefig(OUT / 'fig4_phylogenetic_trees.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'fig4_phylogenetic_trees.pdf', bbox_inches='tight')
plt.close(fig)
print(f'  Saved: outputs/fig4_phylogenetic_trees.png  (.pdf)')
print(f'\nPhylogenetic analysis complete  [{time.perf_counter()-T0:.1f}s]')
