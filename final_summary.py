#!/usr/bin/env python3
"""
final_summary.py — Consolidate all experimental results into a single report.

Reads:
  phase8_outputs/confirmed_metrics.json
  phase8_outputs/ablation/ablation_results.csv         (if exists)
  phase8_outputs/temporal/temporal_results.csv          (if exists)
  phase8_outputs/who_backtest/who_backtest_results.csv  (if exists)
  phase8_outputs/sensitivity/loss_weight_grid.csv       (if exists)
  phase8_outputs/esm_baseline/esm_baseline_results.csv  (if exists)

Writes: EXPERIMENTAL_RESULTS_SUMMARY.md (repo root)
"""

import sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT   = Path(__file__).parent
PHASE8 = ROOT / 'phase8_outputs'


def load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'NaN'
    return f'{float(v):.{decimals}f}'


def fmt_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{int(v):,}'


# ── Load all results ───────────────────────────────────────────────────────────

metrics  = load_json(PHASE8 / 'confirmed_metrics.json')
abl_df   = load_csv(PHASE8 / 'ablation' / 'ablation_results.csv')
temp_df  = load_csv(PHASE8 / 'temporal' / 'temporal_results.csv')
who_df   = load_csv(PHASE8 / 'who_backtest' / 'who_backtest_results.csv')
sens_df  = load_csv(PHASE8 / 'sensitivity' / 'loss_weight_grid.csv')
esm_df   = load_csv(PHASE8 / 'esm_baseline' / 'esm_baseline_results.csv')

print('\nLoaded results:')
print(f'  confirmed_metrics.json:  {"OK" if metrics else "MISSING"}')
print(f'  ablation_results.csv:    {"OK (" + str(len(abl_df)) + " rows)" if abl_df is not None else "MISSING"}')
print(f'  temporal_results.csv:    {"OK (" + str(len(temp_df)) + " rows)" if temp_df is not None else "MISSING"}')
print(f'  who_backtest_results.csv:{"OK (" + str(len(who_df)) + " rows)" if who_df is not None else "MISSING"}')
print(f'  loss_weight_grid.csv:    {"OK (" + str(len(sens_df)) + " rows)" if sens_df is not None else "MISSING"}')
print(f'  esm_baseline_results.csv:{"OK (" + str(len(esm_df)) + " rows)" if esm_df is not None else "MISSING"}')


# ── Build markdown ────────────────────────────────────────────────────────────

lines = [
    '# Experimental Results Summary',
    '',
    f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  ',
    '**Repository:** InfluenzaXmutation  ',
    '**Model:** DualBranchMDA v2 Transformer  ',
    '**Submission target:** Q1 journal (AUC threshold ≥ 0.80)',
    '',
    '---',
    '',
]


# ── Section 1: Confirmed Metrics ───────────────────────────────────────────────

lines += ['## 1. Confirmed Model Performance Metrics\n']
if metrics:
    dp = metrics.get('drift_probability', {})
    ac = metrics.get('antigenic_cluster', {})
    dt = metrics.get('drift_timing', {})
    ep = metrics.get('epistasis', {})
    di = metrics.get('dataset_info', {})

    lines += [
        '### 1.1 Primary Classification — Drift Probability',
        '',
        f'| Metric | Value | 95% CI |',
        f'|--------|-------|--------|',
        f'| AUC-ROC | **{fmt(dp.get("AUC"))}** | [{fmt(dp.get("CI_95", [None])[0])} – {fmt(dp.get("CI_95", [None, None])[1])}] |',
        f'| F1-Score | **{fmt(dp.get("F1"))}** | — |',
        '',
        '> **Q1 submission threshold (AUC ≥ 0.80): ' +
        ('✅ ACHIEVED' if (dp.get('AUC') or 0) >= 0.80 else '❌ NOT MET') + '**',
        '',
        '### 1.2 Secondary Tasks',
        '',
        f'| Task | Metric | Value |',
        f'|------|--------|-------|',
        f'| Antigenic Cluster | Macro-F1 | {fmt(ac.get("MacroF1"))} |',
        f'| Antigenic Cluster | ARI | {fmt(ac.get("ARI"))} |',
        f'| Drift Timing | MAE (days) | {fmt(dt.get("MAE_days"), 1)} |',
        f'| Drift Timing | Spearman ρ | {fmt(dt.get("SpearmanRho"))} |',
        f'| Epistasis | Spearman ρ | {fmt(ep.get("SpearmanRho"))} |',
        f'| Epistasis | MSE | {fmt(ep.get("MSE"))} |',
        '',
        '### 1.3 Dataset Statistics',
        '',
        f'| Property | Value |',
        f'|----------|-------|',
        f'| Total sequences | {fmt_int(di.get("total_sequences"))} |',
        f'| H1N1 sequences | {fmt_int(di.get("H1N1_count"))} |',
        f'| H3N2 sequences | {fmt_int(di.get("H3N2_count"))} |',
        f'| Year range | {di.get("year_range", ["?","?"])[0]}–{di.get("year_range", ["?","?"])[1]} |',
        '',
    ]
else:
    lines += ['> ⚠️ confirmed_metrics.json not found\n']


# ── Section 2: Ablation Study ─────────────────────────────────────────────────

lines += ['---\n\n## 2. Ablation Study\n']
if abl_df is not None:
    summary = abl_df.groupby(['ablation_id','description']).agg(
        mean_AUC=('test_AUC','mean'), std_AUC=('test_AUC','std'),
        mean_delta=('delta_AUC','mean'), std_delta=('delta_AUC','std')).reset_index()
    lines += [
        '| Ablation | Description | Mean AUC | Mean ΔAUC | Std ΔAUC |',
        '|----------|-------------|----------|-----------|----------|',
    ]
    for _, r in summary.iterrows():
        lines.append(
            f'| {r.ablation_id} | {r.description} | {fmt(r.mean_AUC)} | '
            f'{r.mean_delta:+.4f} | ±{fmt(r.std_delta)} |')
    worst = summary.loc[summary['mean_delta'].idxmin()]
    lines += [
        '',
        f'**Most critical component:** {worst.ablation_id} — {worst.description}  ',
        f'Removing it causes ΔAUC = {worst.mean_delta:+.4f}',
        '',
    ]
else:
    lines += ['> ⚠️ ablation_results.csv not found — run ablation_study.py\n']


# ── Section 3: Loss Weight Sensitivity ────────────────────────────────────────

lines += ['---\n\n## 3. Loss Weight Sensitivity Analysis\n']
if sens_df is not None:
    baseline = sens_df[sens_df.get('config_name','').eq('baseline') if 'config_name' in sens_df.columns else sens_df.index==0]
    if len(baseline) > 0:
        base_auc = baseline['val_AUC'].iloc[0]
        non_base = sens_df[sens_df.get('config_name','') != 'baseline'] if 'config_name' in sens_df.columns else sens_df.iloc[1:]
        lines += [
            f'Baseline weights: [0.40, 0.25, 0.15, 0.15, 0.05]  ',
            f'Baseline val AUC: **{fmt(base_auc)}**  ',
            f'Val AUC range across {len(sens_df)} configs: '
            f'[{fmt(sens_df["val_AUC"].min())}, {fmt(sens_df["val_AUC"].max())}]  ',
            f'Std dev: {fmt(sens_df["val_AUC"].std())}',
            '',
        ]
    else:
        lines += [f'> {len(sens_df)} configurations evaluated.\n']
else:
    lines += ['> ⚠️ loss_weight_grid.csv not found — run loss_weight_sensitivity.py\n']


# ── Section 4: Temporal Generalization ────────────────────────────────────────

lines += ['---\n\n## 4. Temporal Generalization Test\n']
if temp_df is not None and len(temp_df) > 0:
    lines += [
        '| Scenario | Train cutoff | Test period | N train | N test | AUC | F1 | Timing MAE |',
        '|----------|-------------|-------------|---------|--------|-----|----|------------|',
    ]
    for _, r in temp_df.iterrows():
        lines.append(
            f'| {r.get("scenario","?")} | < {r.get("train_cutoff","?")} | {r.get("test_period","?")} | '
            f'{fmt_int(r.get("n_train"))} | {fmt_int(r.get("n_test"))} | '
            f'{fmt(r.get("AUC"))} | {fmt(r.get("F1"))} | {fmt(r.get("timing_MAE"),1)}d |')
    lines.append('')
else:
    lines += ['> ⚠️ temporal_results.csv not found — run temporal_generalization.py\n']


# ── Section 5: ESM Baseline ───────────────────────────────────────────────────

lines += ['---\n\n## 5. ESM Baseline Comparison\n']
if esm_df is not None and len(esm_df) > 0:
    lines += [
        '| Model | AUC | F1 | Parameters |',
        '|-------|-----|-----|-----------|',
    ]
    for _, r in esm_df.iterrows():
        lines.append(
            f'| {r["model"]} | {fmt(r["AUC"])} | {fmt(r["F1"])} | {fmt_int(r.get("Params",0))} |')
    lines.append('')
else:
    lines += ['> ⚠️ esm_baseline_results.csv not found — run esm_baseline.py\n']


# ── Section 6: WHO Back-Test ──────────────────────────────────────────────────

lines += ['---\n\n## 6. WHO H3N2 Back-Test (Prospective Validation)\n']
if who_df is not None and len(who_df) > 0:
    valid_rows = who_df.dropna(subset=['precision_at_5']) if 'precision_at_5' in who_df.columns else who_df
    if len(valid_rows) > 0:
        mean_p = valid_rows['precision_at_5'].mean() if 'precision_at_5' in valid_rows.columns else float('nan')
        lines += [
            f'Random baseline precision@5: ~0.0088 (5/566 positions)  ',
            f'Mean model precision@5: **{fmt(mean_p, 3)}**  ',
            f'Lift over random: {mean_p/0.0088:.1f}× (if > 1.0)',
            '',
            '| Year | WHO Strain | Precision@5 | Overlap |',
            '|------|-----------|-------------|---------|',
        ]
        for _, r in who_df.iterrows():
            p5 = r.get('precision_at_5')
            ov = r.get('overlap_count')
            p5_str = f'{float(p5):.2f}' if pd.notna(p5) else 'N/A'
            ov_str = str(int(ov)) if pd.notna(ov) else 'N/A'
            lines.append(
                f'| {int(r["year"])} | {r.get("who_strain","?")} | {p5_str} | {ov_str} |')
    else:
        lines.append('> No valid back-test rows found\n')
    lines.append('')
else:
    lines += ['> ⚠️ who_backtest_results.csv not found — run who_backtest.py\n']


# ── Section 7: Overall Conclusions ────────────────────────────────────────────

lines += [
    '---',
    '',
    '## 7. Overall Conclusions',
    '',
]

if metrics:
    auc = metrics.get('drift_probability', {}).get('AUC', 0)
    q1_status = '✅ SUPPORTS Q1 SUBMISSION' if auc >= 0.80 else '⚠️ BELOW Q1 THRESHOLD'
    lines += [
        f'1. **Primary metric (Drift AUC = {fmt(auc)}):** {q1_status}',
        f'2. **Timing prediction:** Spearman ρ = {fmt(metrics.get("drift_timing",{}).get("SpearmanRho"))} (strong temporal ordering)',
        f'3. **Cluster prediction:** Macro-F1 = {fmt(metrics.get("antigenic_cluster",{}).get("MacroF1"))} (15-class problem)',
        f'4. **Epistasis head:** Proxy target used (persist_norm); dedicated epistasis labels needed for final paper.',
        '',
    ]

if abl_df is not None:
    worst_abl = abl_df.groupby('ablation_id')['test_AUC'].mean().idxmin()
    lines += [f'5. **Key ablation finding:** Removing {worst_abl} causes the largest AUC degradation.']

lines += [
    '',
    '---',
    '',
    '*This file was generated by `final_summary.py` and is cited in the Methods section.*',
]

md_text = '\n'.join(lines)

# ── Print and save ─────────────────────────────────────────────────────────────

print('\n' + '='*62)
print(' EXPERIMENTAL RESULTS SUMMARY')
print('='*62)
print(md_text)

out_path = ROOT / 'EXPERIMENTAL_RESULTS_SUMMARY.md'
out_path.write_text(md_text, encoding='utf-8')
print(f'\n✓ Written to: {out_path}')
