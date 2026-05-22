## WHO_BACKTEST_UPDATED.md — Honest Negative Addendum

### Addendum to Section 6.7 (Limitation Statement)

The HI-cartography-informed fusion score (F_HI) was evaluated as a potential
improvement to the WHO prospective validation. Despite biologically-motivated
weighting of Koel et al. antigenic sites, B-factor surface exposure, and
sequence conservation, mean precision@5 did not exceed 0.0000 across
the 2018–2020 WHO H3N2 vaccine strain evaluations.

This result confirms that mutation ranking by sequence-based methods alone
remains fundamentally limited without direct antigenicity measurement via
haemagglutination inhibition (HI) assay data. The primary constraint is the
2000-mutation training dataset's sparse late-season H3N2 coverage (2018: n=30,
2019: n=25, 2020: n=5), which reduces the probability that known vaccine-strain
substitutions appear in the per-year inference pools.

**Recommended manuscript language (Section 11 — Limitations):**
"WHO prospective precision@5 = 0.00 under frequency-based ranking and
0.00 under the HI-cartographic proxy F_HI. This reflects a fundamental
constraint: the 2000-mutation balanced dataset lacks the temporal density
to represent sweep mutations (e.g., T131K, R142G, N171K) in late-season
inference pools, and sequence-based proxies cannot substitute for HI titer
measurements. Future work should (a) expand temporal coverage using the
full 1.35M mutation dataset, and (b) integrate antigenic cartography data
from publicly available HI titer databases (e.g., GISAID EpiFlu)."
