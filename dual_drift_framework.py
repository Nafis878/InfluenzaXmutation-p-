"""
Dual-Drift Framework — Formal Mathematical Foundation
Establishes the rigorous conditions under which statistical methods developed for
biological sequence drift are applicable to LLM behavioral drift monitoring.

This is NOT an analogy. It is a formal mapping between two instances of the same
abstract problem: detecting non-stationary drift from a reference state in a
high-dimensional discrete-or-continuous space under selective pressure.
"""
import sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = Path("C:/Users/UseR/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Abstract Drift Model
#
# Both biological sequence drift and LLM behavioral drift are instances of:
#
#   Definition 1 (Reference Drift Process):
#   Let S = {s_0, s_1, ..., s_T} be a sequence of states in a measurable space
#   (Ω, F). Let r* ∈ Ω be a fixed reference state. Define the drift distance as:
#
#       D_t = d(s_t, r*)
#
#   where d: Ω × Ω → ℝ≥0 is a distance metric on Ω. The system exhibits
#   REFERENCE DRIFT if E[D_t] is a strictly increasing function of t.
#
# Biological realization:
#   Ω = A^L  (sequences of length L over amino acid alphabet A, |A|=20)
#   r* = pandemic reference sequence (e.g., 2009 H1N1 founder)
#   d  = Hamming distance (count of positional mismatches)
#   s_t = set of circulating sequences in year t
#
# LLM behavioral realization:
#   Ω = ℝ^D  (D-dimensional behavioral metric space)
#   r* = initial deployment behavioral vector (baseline)
#   d  = L2 or Mahalanobis distance
#   s_t = behavioral metric vector at version t
#
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Isomorphism Conditions
#
# The detection methods (chi-square enrichment, regression analysis, ARIMA,
# attention-based neural detection) transfer between the two realizations
# IF AND ONLY IF the following four conditions hold:
#
# Condition C1 (Reference Stability):
#   r* is well-defined, fixed, and non-ambiguous in both domains.
#   - Biological: the 2009 founder sequence is a single, identified genome.
#   - LLM: the "aligned" baseline behavior must be explicitly defined and
#     captured before deployment (e.g., via behavioral evaluation suite).
#
# Condition C2 (Compositional Decomposability):
#   The total drift D_t can be decomposed into contributions from individual
#   positions/components:   D_t = Σ_i δ_{t,i}
#   where δ_{t,i} ≥ 0 measures the deviation at component i at time t.
#   - Biological: Hamming distance is the sum of per-position indicators.
#   - LLM: L1 behavioral distance satisfies this; L2 requires coordinate choice.
#
# Condition C3 (Heterogeneous Sensitivity — "Critical Regions"):
#   There exists a partition of components into CRITICAL (C) and NON-CRITICAL (N̄)
#   subsets such that changes in C have disproportionate functional consequences.
#   This partition must be established INDEPENDENTLY of the drift observations.
#   - Biological: antigenic sites defined by structural crystallography (pre-existing).
#   - LLM: alignment constraints defined in RLHF reward model or safety spec.
#   Without an independently defined critical region, chi-square enrichment
#   is circular (finding what you put in).
#
# Condition C4 (Non-Stationarity):
#   The drift process is non-stationary: E[D_t] is not constant over time.
#   Both ARIMA and polynomial regression are meaningful only if this holds.
#   - Biological: validated by linear regression R² > 0.70 on pandemic lineage.
#   - LLM: requires longitudinal monitoring data across versions.
#
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Where the Analogy Breaks
#
# Condition C3 is the critical constraint for the LLM application:
#
# In virology, critical regions (antigenic sites) are identified from
# structural biology and experimental neutralization assays — completely
# independent of the mutation data being analyzed.
#
# For LLM behavioral monitoring, critical behavioral regions (e.g., "do not
# assist with X") MUST be defined from alignment specifications BEFORE
# observing behavioral trajectories. If critical behaviors are identified
# post-hoc from the same data used to detect drift, the enrichment test
# is circular — identical to the failure mode of generate_agent_data().
#
# Prediction: the framework WILL produce valid results for an LLM system that:
#   (a) has a formal behavioral specification (defines C)
#   (b) has longitudinal monitoring data across real versions (provides s_t)
#   (c) has a stable deployment baseline (provides r*)
#
# Prediction: the framework WILL NOT produce valid results when:
#   (a) critical regions are defined from the same behavioral data used for testing
#   (b) monitoring is applied to simulated rather than real behavioral trajectories
#
# ══════════════════════════════════════════════════════════════════════════════

def generate_framework_document():
    doc = f"""
DUAL-DRIFT FRAMEWORK: FORMAL MATHEMATICAL SPECIFICATION
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*72}

I. ABSTRACT DRIFT MODEL
{'─'*72}
Both biological sequence drift and LLM behavioral drift are instances of a
Reference Drift Process (Definition 1):

  Let S = {{s_0, s_1, ..., s_T}} be a sequence of states in (Ω, F).
  Let r* ∈ Ω be a fixed reference state.
  Define drift distance: D_t = d(s_t, r*)

  The system exhibits REFERENCE DRIFT iff E[D_t] is strictly increasing in t.

Biological Realization:
  Ω   = A^L  (amino acid sequences, |A|=20, L≈566 for H1N1 HA)
  r*  = pandemic founder (2009 H1N1 reference sequence)
  d   = Hamming distance (Σ_i I[s_i ≠ r*_i] over shared prefix)
  s_t = circulating sequences in year t

LLM Behavioral Realization:
  Ω   = ℝ^D  (D-dimensional behavioral metric space)
  r*  = initial deployment behavior vector (baseline version)
  d   = L2 distance ‖s_t - r*‖₂ or Mahalanobis distance
  s_t = behavioral metric vector for agent version t


II. ISOMORPHISM CONDITIONS (C1–C4)
{'─'*72}
The detection methods transfer between realizations IF AND ONLY IF:

  C1 Reference Stability:
     r* is fixed and unambiguously defined BEFORE analysis begins.
     Biological PASS: 2009 founder is a single identified sequence.
     LLM condition : behavioral baseline must be captured at deployment.

  C2 Compositional Decomposability:
     D_t = Σ_i δ_{{t,i}} where δ_{{t,i}} ≥ 0 is per-component deviation.
     Biological PASS: Hamming = sum of per-position binary indicators.
     LLM condition : use L1 distance or per-metric absolute deviations.

  C3 Independent Critical Regions:
     A CRITICAL set C ⊂ {{1,...,|Ω|}} is defined INDEPENDENTLY of drift data.
     Chi-square enrichment tests whether Pr[δ_{{t,i}}>0 | i∈C] > Pr[δ_{{t,i}}>0 | i∉C].
     Biological PASS: antigenic sites from crystallography (Smith 2004, Koel 2013).
     LLM condition : alignment constraints from RLHF spec or safety evaluation.
     FAILURE MODE  : defining C from the same behavioral data being tested
                     makes the chi-square test circular (self-fulfilling).

  C4 Non-Stationarity:
     E[D_t] is not constant; drift rate > 0 detectable by regression.
     Biological PASS: R² = 0.78 on pandemic lineage (2009–2017).
     LLM condition : requires longitudinal monitoring across real versions.


III. DETECTION METHOD APPLICABILITY
{'─'*72}
Under C1–C4, each detection method is formally valid:

  Method                 | Requires | Transfers iff
  ───────────────────────|──────────|───────────────────────────────
  Chi-square enrichment  | C1, C3   | C3 independent (not circular)
  Regression through     | C1, C4   | Non-stationary process
    origin (slope)       |          |
  ARIMA forecasting      | C4       | Non-stationary, sufficient T
  Polynomial velocity    | C1, C4   | Non-linear drift dynamics
  Transformer attention  | C2, C4   | Per-component decomposability
  MDS visualization      | C1, C2   | Distance metric structure


IV. CRITICAL LIMITATION — SIMULATION VALIDITY
{'─'*72}
The current agentic_drift.py module violates C3 by design:
generate_agent_data() injects drift via sigmoid with in_critical_prob ∝ drift.
Therefore critical function co-occurrence IS the drift signal — chi-square
detects what was programmed, not an emergent property of the data.

REMEDIATION OPTIONS:
  Option A (Preferred): Apply the framework to REAL LLM behavioral logs from
    a deployed system with a pre-specified behavioral audit checklist.
    Requirement: T ≥ 8 model versions, D ≥ 3 behavioral metrics, documented
    alignment spec defining C before deployment.

  Option B (Acceptable for paper): Use the influenza pipeline as a PROOF OF
    CONCEPT that the detection framework identifies real drift in a known
    positive control (H1N1 post-2009) while correctly NOT detecting drift in
    a negative control (H3N2 pre-1990 stable period).
    Then argue by analogy that a system meeting C1–C4 would be similarly
    detectable. This is honest about the scope of the claim.

  Option C (Minimum bar): Reframe all agentic_drift results explicitly as
    "under-simulation-assumption performance bounds" — not as detection of
    real drift. Report the metrics as theoretical detection limits, not
    empirical results.


V. MATHEMATICAL EQUIVALENCE PROOF SKETCH
{'─'*72}
Claim: The weighted regression-through-origin estimator used in Phase 1
(slope = Σ w_t·t·D_t / Σ w_t·t²) is the SAME estimator as would be applied
to an LLM behavioral time series under C1–C4, with only the distance metric
and time variable changing.

Proof sketch:
  Both minimize the weighted sum of squared residuals E[D_t - β·t]² with
  intercept forced to 0 (origin = reference state at t=0). The estimator
  β̂ = Σ w_t·t·D_t / Σ w_t·t² is metric-agnostic: substitute Hamming → L2
  and amino acid years → agent version numbers, and the formula is identical.
  QED: the same numerical procedure applies to both domains.


VI. THEORETICAL PREDICTIONS
{'─'*72}
If C1–C4 hold for a real LLM behavioral monitoring system, the framework
predicts:
  P1: Chi-square enrichment ratio > 1.5 for alignment-critical behaviors
      in drifted vs stable versions (same threshold as virology).
  P2: Regression through origin slope > 0 for behavioral distance over
      versions, with R² > 0.70 if T ≥ 8.
  P3: ARIMA one-step forecast MAPE < 15% for behavioral distance.
  P4: Transformer attention focuses on version windows immediately preceding
      drift onset (lag 1–3), mirroring the influenza pandemic transition.

These are FALSIFIABLE predictions. Failure to find them in real LLM data
would invalidate the framework for that specific system.

{'='*72}
"""
    return doc


doc = generate_framework_document()
framework_path = OUTPUT_DIR / "dual_drift_framework.txt"
framework_path.write_text(doc, encoding='utf-8')
print(f"✓ Saved dual_drift_framework.txt")
print(doc)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Dual-Drift Isomorphism Diagram
# ══════════════════════════════════════════════════════════════════════════════
print("Generating dual_drift_framework_figure.png...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor('white')

BBLUE  = '#1565C0'
RRED   = '#C62828'
GGRAY  = '#546E7A'
GGREEN = '#2E7D32'
LIGHT_B = '#E3F2FD'
LIGHT_R = '#FFEBEE'
LIGHT_G = '#E8F5E9'

# ── Panel 1: Protein sequence space ──
ax1 = axes[0]
ax1.set_facecolor(LIGHT_B)
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.set_aspect('equal')

# Reference point
ax1.scatter([5], [8], s=400, c=BBLUE, zorder=5, marker='*')
ax1.text(5, 8.5, 'r* = 2009 founder\nsequence', ha='center', fontsize=8.5,
         color=BBLUE, fontweight='bold')

# Drifted points
rng = np.random.default_rng(7)
for i, (yr, col, sz) in enumerate(
    [(2010, BBLUE, 40), (2012, '#1976D2', 60), (2014, '#42A5F5', 80), (2016, '#90CAF9', 100)]):
    pts = rng.normal([5, 8 - i * 1.5], [0.3 + i * 0.2, 0.3 + i * 0.2], (6, 2))
    ax1.scatter(pts[:, 0], pts[:, 1], s=sz, c=col, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    ax1.text(pts[:, 0].mean() + 0.5, pts[:, 1].mean(),
             str(yr), fontsize=7.5, color=col)

ax1.annotate('', xy=(6.5, 3.0), xytext=(5.1, 7.5),
             arrowprops=dict(arrowstyle='->', color=BBLUE, lw=2.0, alpha=0.8))
ax1.text(7.2, 5.2, 'D_t\nHamming', fontsize=8, color=BBLUE, fontstyle='italic')
ax1.set_title('Biological Domain\nΩ = A^L (amino acid sequences)',
              fontsize=10, fontweight='bold', color=BBLUE)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_xlabel('Sequence space dimension 1', fontsize=8.5)

# ── Panel 2: Conditions C1–C4 ──
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.set_xticks([]); ax2.set_yticks([])

conditions = [
    ('C1', 'Reference Stability\nr* fixed before analysis', GGREEN, 8.0),
    ('C2', 'Decomposability\nD_t = Σ δ_{t,i}', BBLUE, 6.0),
    ('C3', 'Independent Critical Regions\nC defined from structural/spec data', RRED, 4.0),
    ('C4', 'Non-Stationarity\nE[D_t] strictly increasing', GGRAY, 2.0),
]
for cid, desc, col, y in conditions:
    ax2.add_patch(mpatches.FancyBboxPatch((0.5, y - 0.6), 9, 1.2,
                                     boxstyle='round,pad=0.15',
                                     facecolor=col + '22', edgecolor=col,
                                     linewidth=1.5))
    ax2.text(1.2, y, f'{cid}', fontsize=14, color=col, fontweight='bold', va='center')
    ax2.text(2.5, y, desc, fontsize=8.5, color='#333', va='center')
ax2.set_title('Isomorphism Conditions (C1–C4)\nRequired for method transfer',
              fontsize=10, fontweight='bold')
ax2.text(5, 0.6, 'IF C1–C4 hold, THEN the same\ndetection statistics apply to both domains',
         ha='center', va='bottom', fontsize=9, style='italic', color=GGRAY,
         bbox=dict(boxstyle='round', facecolor='#F0F4C3', edgecolor='#AFB42B', linewidth=1.5))

# ── Panel 3: LLM behavioral space ──
ax3 = axes[2]
ax3.set_facecolor(LIGHT_R)
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
ax3.set_aspect('equal')

ax3.scatter([5], [8], s=400, c=RRED, zorder=5, marker='*')
ax3.text(5, 8.5, 'r* = baseline\nbehavior (v1.0)', ha='center', fontsize=8.5,
         color=RRED, fontweight='bold')

for i, (ver, col, sz) in enumerate(
    [('v1.4', RRED, 40), ('v2.2', '#D32F2F', 60),
     ('v3.0', '#EF5350', 80), ('v4.0', '#EF9A9A', 100)]):
    pts = rng.normal([5, 8 - i * 1.5], [0.3 + i * 0.25, 0.3 + i * 0.25], (6, 2))
    ax3.scatter(pts[:, 0], pts[:, 1], s=sz, c=col, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    ax3.text(pts[:, 0].mean() + 0.5, pts[:, 1].mean(),
             ver, fontsize=7.5, color=col)

ax3.annotate('', xy=(6.5, 3.0), xytext=(5.1, 7.5),
             arrowprops=dict(arrowstyle='->', color=RRED, lw=2.0, alpha=0.8))
ax3.text(7.2, 5.2, 'D_t\nL2 dist', fontsize=8, color=RRED, fontstyle='italic')
ax3.set_title('LLM Behavioral Domain\nΩ ⊆ ℝ^D (behavioral metrics)',
              fontsize=10, fontweight='bold', color=RRED)
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_xlabel('Behavioral metric space dimension 1', fontsize=8.5)

# Arrow between panels 1 and 3 (through middle)
fig.text(0.495, 0.50, '⟺', fontsize=26, ha='center', va='center',
         color=GGREEN, fontweight='bold')
fig.text(0.495, 0.40, 'IF C1–C4', fontsize=8, ha='center', color=GGREEN)

plt.suptitle('Dual-Drift Framework: Formal Isomorphism\n'
             'Same statistical detection methods apply to both domains under conditions C1–C4',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=2)
fig.savefig(OUTPUT_DIR / "dual_drift_framework_figure.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved dual_drift_framework_figure.png (300 dpi)")

print(f"\n{'='*65}")
print("  Dual-Drift Framework complete")
print("  - dual_drift_framework.txt    (full formal specification)")
print("  - dual_drift_framework_figure.png  (isomorphism diagram)")
print(f"{'='*65}")
