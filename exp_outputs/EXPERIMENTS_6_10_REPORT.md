# Experiments 6–10: Validation & Interpretability Suite
**Generated:** 2026-05-21 22:51:52
**Runtime:** 30.2s (0.5 min)

---

## Experiment 6: South Asia Geographic Hold-Out Validation

**Methodology:** Mutations associated with South Asian sequences (Afghanistan, Bangladesh, Bhutan, India, Maldives, Nepal, Pakistan, Sri Lanka) were held out as an external cohort. A Random Forest (300 trees) was trained exclusively on non-South-Asian mutations and evaluated on the held-out cohort.

**South Asian cohort:** 70 sequences (743 unique mutations in SA context)

| Cohort | N mutations | AUC-ROC | 95% CI | F1-Score | ΔAUC |
|--------|-------------|---------|--------|----------|------|
| Global balanced test  | 800 | 0.9725 | [0.963–0.980] | 0.8982 | — |
| South Asia hold-out   | 743 | 0.9877 | [0.980–0.995] | 0.7848 | +0.0152 |

**Interpretation:** Model generalizes well across geographic cohorts, with performance degradation within acceptable bounds.

---

## Experiment 7: Phase-Output Ablation Study

**Methodology:** Each pipeline phase's feature group was systematically removed from the 16-feature RF model (identical train/test split as Phase 8). ΔAUC = AUC_ablated − AUC_full.

**Baseline RF (all 16 features):**  AUC=0.9663  [0.950–0.978]  F1=0.8732

| Phase Group | Dropped Features | AUC Ablated | ΔAUC | Impact |
|-------------|-----------------|-------------|------|--------|
| Phase 2: Temporal clustering (year, days, era signal) | year_norm, days_norm | 0.9079 [0.878–0.940] | -0.0584 | CRITICAL |
| Phase 5: Evolutionary persistence (frequency, persistence years) | freq_norm, n_years_norm | 0.9385 [0.916–0.958] | -0.0278 | significant |
| Phase 4: Drift era intensity (MDS-derived drift signal) | drift_inten | 0.9581 [0.941–0.973] | -0.0082 | moderate |
| Phase 1: Structural (position, critical/binding flags) | position_norm, crit_flag, bind_flag | 0.9659 [0.949–0.979] | -0.0004 | moderate |
| Phase 3: Mutation biochemistry (hydrophobicity, volume, charge) | ref_hydro, var_hydro, hydro_delta, ref_vol, var_vol, vol_delta, charge_chg, polar_chg | 0.9698 [0.956–0.982] | +0.0035 | negligible |

**Top individual feature (by ablation impact):** year_norm (ΔAUC=-0.0571)

---

## Experiment 8: Prospective 2022–2024 WHO Validation

**Methodology:** Cluster forecast (N→N+1) applied to WHO H3N2 vaccine strain recommendations 2009–2024. Concordance = fraction of seasons where model predicted exact or adjacent (±1 step) cluster correctly.

| Period | Seasons | Exact Match | Adjacent (±1) |
|--------|---------|-------------|--------------|
| Historical (2009–2020) | 12 | 17% | 67% |
| Prospective (2021–2024) | 4 | 75% | 100% |
| **Overall** | 16 | **31%** | **75%** |

**Mean cluster distance:** 2.06 steps (0=exact, 1=adjacent)

---

## Experiment 9: FluSurver Rule-Based Comparison

**Methodology:** FluSurver-style rule-based classifier implemented using WHO antigenic sites (A–E + RBS) and Koel 2013 critical positions. Scoring = evidence-count heuristic; no training data used.

| Model | Type | AUC-ROC | F1-Score | ΔAUC vs FluSurver |
|-------|------|---------|----------|------------------|
| DualBranchMDA Transformer (Exp 8) | Deep Learning | 0.9457 | 0.8518 | +0.4887 |
| Random Forest (300 trees) | Machine Learning | 0.9663 | 0.8732 | +0.5093 |
| FluSurver-style rule-based | Rule-based | 0.4570 | 0.0505 | +0.0000 |
| EVEscape-inspired (zero-shot) | Zero-shot | 0.2659 | 0.0094 | -0.1911 |

**MDA Transformer ΔAUC vs FluSurver:** +0.4887

---

## Experiment 10: SHAP + Attention Visualization

**SHAP analysis (RF, TreeExplainer):**
  Top feature by mean |SHAP|: **year_norm** (0.1445)
  Second: **days_norm** (0.0959)

**Top 5 features by SHAP importance:**
  1. year_norm: 0.1445
  2. days_norm: 0.0959
  3. freq_norm: 0.0684
  4. drift_inten: 0.0534
  5. n_years_norm: 0.0181

**Transformer attention analysis:**
  Highest-attended token: **era** (0.3787)
  (biochemical branch cross-attending to token sequence branch)

---

## Output Files

```
exp_outputs/
├── exp6_geographic_results.csv         — SA hold-out AUC/F1 vs global
├── exp6_geographic_validation.png/pdf  — ROC, AUC bar, country distribution
├── exp7_group_ablation.csv             — Phase-group ΔAUC
├── exp7_individual_ablation.csv        — Per-feature ΔAUC
├── exp7_feature_importances.csv        — Gini importances
├── exp7_ablation_study.png/pdf         — ΔAUC bars + importance chart
├── exp8_who_concordance.csv            — Season-by-season forecast vs WHO
├── exp8_who_prospective_validation.png/pdf
├── exp9_flusurver_comparison.csv       — AUC/F1 across 4 classifier types
├── exp9_flusurver_comparison.png/pdf
├── exp10_shap_summary.csv              — Mean |SHAP| per feature
├── exp10_token_attention.csv           — Transformer token attention weights
└── exp10_shap_attention_viz.png/pdf    — 6-panel supplementary figure
```

**Total runtime:** 30.2s