#!/usr/bin/env python3
"""
Fix 7: Bootstrap CIs (n=1000) and Benjamini-Hochberg Correction.

Adds 95% bootstrap confidence intervals to EVERY reported metric.
Applies Benjamini-Hochberg (BH) FDR correction to all collected p-values
across the pipeline (enrichment tests, rate comparisons, CV tests).

Outputs:
  outputs/bootstrap_ci_summary.csv    — all metrics with 95% CI
  outputs/bh_corrected_pvalues.csv    — BH-corrected p-values
  outputs/bootstrap_stats_report.txt  — narrative summary
  outputs/fig_bootstrap_summary.png/pdf
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'
P8   = ROOT / 'phase8_outputs'
OUT.mkdir(exist_ok=True)

np.random.seed(42)

print('='*60)
print('Fix 7: Bootstrap CIs + Benjamini-Hochberg Correction')
print('='*60)

# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap utility functions
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_metric(y_true, y_prob_or_pred, metric_fn,
                     n_boot=1000, ci=0.95, stratify=True):
    """
    Returns (point_estimate, lower_ci, upper_ci).
    metric_fn(y_true, y_pred) -> float
    """
    rng = np.random.RandomState(42)
    n   = len(y_true)
    point = metric_fn(y_true, y_prob_or_pred)
    boots = []

    classes = np.unique(y_true)
    for _ in range(n_boot):
        if stratify and len(classes) > 1:
            idx = []
            for c in classes:
                c_idx = np.where(y_true == c)[0]
                idx.extend(rng.choice(c_idx, len(c_idx), replace=True))
            idx = np.array(idx)
        else:
            idx = rng.choice(n, n, replace=True)
        try:
            boots.append(metric_fn(y_true[idx], y_prob_or_pred[idx]))
        except Exception:
            pass

    if not boots:
        return point, float('nan'), float('nan')
    boots = np.sort(boots)
    alpha = (1 - ci) / 2
    lo    = float(np.percentile(boots, alpha * 100))
    hi    = float(np.percentile(boots, (1 - alpha) * 100))
    return float(point), lo, hi

def bootstrap_mean(values, n_boot=1000, ci=0.95):
    """Bootstrap CI around the sample mean."""
    rng = np.random.RandomState(42)
    n   = len(values)
    point = float(np.mean(values))
    boots = [np.mean(rng.choice(values, n, replace=True)) for _ in range(n_boot)]
    boots = np.sort(boots)
    alpha = (1 - ci) / 2
    return point, float(np.percentile(boots, alpha*100)), float(np.percentile(boots, (1-alpha)*100))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MDA Transformer test set bootstrap CIs
# ══════════════════════════════════════════════════════════════════════════════
print('\n[Section 1] MDA Transformer bootstrap CIs...')

from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, precision_score, recall_score)

bootstrap_results = []

test_path = P8 / 'phase8_test_data.csv'
pred_path = P8 / 'phase8_mda_all_predictions.csv'

if test_path.exists() and pred_path.exists():
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(pred_path)

    # Match test set predictions by (position, ref_aa, var_aa) — avoids accession
    # duplicates where one accession is the representative for multiple mutations.
    if {'position','ref_char','var_char'}.issubset(test_df.columns) and \
       {'position','ref_aa','mut_aa'}.issubset(pred_df.columns):
        pred_key = pred_df[['position','ref_aa','mut_aa','drift_prob']].copy()
        pred_key = pred_key.drop_duplicates(subset=['position','ref_aa','mut_aa'])
        pred_key.rename(columns={'ref_aa':'ref_char','mut_aa':'var_char'}, inplace=True)
        matched = test_df[['position','ref_char','var_char','label_drift_prob']].merge(
            pred_key, on=['position','ref_char','var_char'], how='left')
        y_true = matched['label_drift_prob'].values.astype(int)
        y_prob = matched['drift_prob'].fillna(0.5).values
    elif 'accession' in test_df.columns and 'accession' in pred_df.columns:
        pred_dedup = pred_df.drop_duplicates(subset='accession')[['accession','drift_prob']]
        matched = test_df[['accession','label_drift_prob']].merge(
            pred_dedup, on='accession', how='left')
        y_true = matched['label_drift_prob'].values.astype(int)
        y_prob = matched['drift_prob'].fillna(0.5).values
    else:
        n = min(len(test_df), len(pred_df))
        y_true = test_df['label_drift_prob'].values[:n].astype(int)
        y_prob = pred_df['drift_prob'].values[:n]

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = [
        ('AUC-ROC',   lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt))>1 else 0.5,
                      y_prob),
        ('Accuracy',  lambda yt, yp: accuracy_score(yt, (yp>=0.5).astype(int)), y_prob),
        ('F1',        lambda yt, yp: f1_score(yt, (yp>=0.5).astype(int), zero_division=0), y_prob),
        ('Precision', lambda yt, yp: precision_score(yt,(yp>=0.5).astype(int),zero_division=0), y_prob),
        ('Recall',    lambda yt, yp: recall_score(yt,(yp>=0.5).astype(int),zero_division=0), y_prob),
    ]

    for name, fn, yp in metrics:
        pt, lo, hi = bootstrap_metric(y_true, yp, fn, n_boot=1000)
        bootstrap_results.append({
            'model': 'MDA Transformer', 'metric': name,
            'point': round(pt,4), 'ci_lo95': round(lo,4), 'ci_hi95': round(hi,4),
            'ci_width': round(hi-lo, 4),
        })
        print(f'  MDA {name:<12}: {pt:.4f}  95%CI=[{lo:.4f},{hi:.4f}]')
else:
    print('  phase8 test data not found — skipping transformer bootstrap')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Baseline model bootstrap CIs
# ══════════════════════════════════════════════════════════════════════════════
print('\n[Section 2] Baseline model bootstrap CIs...')

AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}
POSITIVE_CHARGE = {'R', 'K', 'H'}
NEGATIVE_CHARGE = {'D', 'E'}

def charge_group(aa):
    if aa in POSITIVE_CHARGE: return 'pos'
    if aa in NEGATIVE_CHARGE: return 'neg'
    return 'neutral'

try:
    var_df = pd.read_csv(OUT / 'phase3_variations_annotated.csv')
    var_df = var_df[var_df['ref_char'].isin(AA_VOCAB) & var_df['var_char'].isin(AA_VOCAB)].copy()

    agg = (var_df.groupby(['position','ref_char','var_char','subtype',
                           'in_critical_region','in_binding_region'])
           .agg(frequency=('accession','count'), first_year=('year','min'))
           .reset_index())
    first_acc = (var_df.sort_values('year')
                 .groupby(['position','ref_char','var_char','subtype'])['accession']
                 .first().reset_index())
    agg = agg.merge(first_acc, on=['position','ref_char','var_char','subtype'], how='left')

    lbl_h3 = pd.read_csv(OUT/'antigenic_labels_h3n2.csv')[
        ['Accession','drift_binary_label']].rename(columns={'Accession':'accession'})
    lbl_h1 = pd.read_csv(OUT/'antigenic_labels_h1n1.csv')[
        ['Accession','drift_binary_label']].rename(columns={'Accession':'accession'})
    lbl_all = pd.concat([lbl_h3, lbl_h1])
    agg = agg.merge(lbl_all, on='accession', how='left')
    agg['label_drift_prob'] = agg['drift_binary_label'].fillna(
        (agg['first_year'] >= 2012).astype(int)).astype(int)

    agg['ref_idx']   = agg['ref_char'].apply(lambda c: AA2IDX.get(c,0))
    agg['var_idx']   = agg['var_char'].apply(lambda c: AA2IDX.get(c,0))
    agg['year_norm'] = (agg['first_year'] - 2009) / 11.0
    agg['freq_norm'] = np.log1p(agg['frequency']) / np.log1p(agg['frequency'].max())
    agg['crit_flag'] = agg['in_critical_region'].astype(float)
    agg['bind_flag'] = agg['in_binding_region'].astype(float)

    hi  = agg[agg['label_drift_prob']==1].sample(min(1000,int((agg['label_drift_prob']==1).sum())),random_state=42)
    bg  = agg[agg['label_drift_prob']==0].sample(min(1000,int((agg['label_drift_prob']==0).sum())),random_state=42)
    sampled = pd.concat([hi,bg]).reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    FEAT_COLS = ['position','ref_idx','var_idx','freq_norm','year_norm','crit_flag','bind_flag']
    X = sampled[FEAT_COLS].values
    y = sampled['label_drift_prob'].values
    tr_idx, tmp = train_test_split(range(len(sampled)), test_size=0.40, random_state=42, stratify=y)
    va_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=42)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    # LR
    clf_lr = LogisticRegression(max_iter=1000, random_state=42)
    clf_lr.fit(X_tr, y_tr)
    p_lr = clf_lr.predict_proba(X_te)[:,1]

    # RF
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf_rf.fit(X_tr, y_tr)
    p_rf = clf_rf.predict_proba(X_te)[:,1]

    # Rule-based
    test_samp = sampled.iloc[te_idx].reset_index(drop=True)
    charge_chg = test_samp.apply(
        lambda r: charge_group(r['ref_char']) != charge_group(r['var_char']), axis=1).values
    p_rule = (test_samp['crit_flag'].values.astype(bool) & charge_chg).astype(float)

    for mdl_name, yp in [('Logistic Regression', p_lr),
                          ('Random Forest',       p_rf),
                          ('Rule-based (FluSurver)', p_rule)]:
        for metric_name, fn in [
            ('AUC-ROC',   lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt))>1 else 0.5),
            ('Accuracy',  lambda yt, yp: accuracy_score(yt, (yp>=0.5).astype(int))),
            ('F1',        lambda yt, yp: f1_score(yt, (yp>=0.5).astype(int), zero_division=0)),
        ]:
            pt, lo, hi = bootstrap_metric(y_te, yp, fn, n_boot=1000)
            bootstrap_results.append({
                'model': mdl_name, 'metric': metric_name,
                'point': round(pt,4), 'ci_lo95': round(lo,4), 'ci_hi95': round(hi,4),
                'ci_width': round(hi-lo,4),
            })
        print(f'  {mdl_name:<26}: AUC={bootstrap_results[-3]["point"]:.4f} '
              f'[{bootstrap_results[-3]["ci_lo95"]:.4f},{bootstrap_results[-3]["ci_hi95"]:.4f}]')

except Exception as e:
    print(f'  Baseline bootstrap failed: {e}')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Collect all pipeline p-values for BH correction
# ══════════════════════════════════════════════════════════════════════════════
print('\n[Section 3] Collecting p-values for BH correction...')

pvalue_entries = []

# Phase 3: chi-square enrichment p-values
p3_path = OUT / 'phase3_variation_statistics.txt'
if p3_path.exists():
    txt = p3_path.read_text(encoding='utf-8')
    import re
    for line in txt.splitlines():
        m = re.search(r'chi2=([\d.]+)\s+p=([\d.e+-]+)', line)
        if m:
            chi2 = float(m.group(1))
            p    = float(m.group(2).replace('e+','e').replace('e-0','e-'))
            subtype = 'H1N1' if 'H1N1' in line else ('H3N2' if 'H3N2' in line else 'combined')
            pvalue_entries.append({
                'test'    : f'Critical-region enrichment ({subtype})',
                'statistic': chi2,
                'p_raw'   : max(p, 1e-300),
                'source'  : 'phase3_variation_statistics.txt',
                'method'  : 'chi-square',
            })

# Phase 1: divergence rate — construct a z-test p-value against literature
p1_path = OUT / 'phase1_h1n1_literature_comparison.txt'
if p1_path.exists():
    txt = p1_path.read_text(encoding='utf-8')
    rate_match = re.search(r'Calculated rate.*?:\s*([\d.]+)', txt)
    if rate_match:
        calc_rate = float(rate_match.group(1))
        # Simple approximation: z = (calc - lit) / (lit * 0.05) as SE proxy
        z = abs(calc_rate - 2.45) / (2.45 * 0.05)
        p_rate = float(2 * stats.norm.sf(z))
        pvalue_entries.append({
            'test'     : 'H1N1 divergence rate vs literature (2.45 aa/yr)',
            'statistic': round(z, 4),
            'p_raw'    : max(p_rate, 1e-300),
            'source'   : 'phase1_h1n1_literature_comparison.txt',
            'method'   : 'z-test (approximate)',
        })

# External validation Koel position enrichment (if available)
ext_path = OUT / 'external_validation_report.txt'
if ext_path.exists():
    txt = ext_path.read_text(encoding='utf-8')
    m = re.search(r'Mann-Whitney p-value\s*:\s*([\d.e+-]+)', txt)
    if m:
        pvalue_entries.append({
            'test'     : 'Koel position enrichment (Mann-Whitney U)',
            'statistic': float('nan'),
            'p_raw'    : max(float(m.group(1)), 1e-300),
            'source'   : 'external_validation_report.txt',
            'method'   : 'Mann-Whitney U',
        })

print(f'  Collected {len(pvalue_entries)} p-values for BH correction')

# ── Apply Benjamini-Hochberg FDR correction ───────────────────────────────────
if pvalue_entries:
    pval_df = pd.DataFrame(pvalue_entries)
    p_raw   = pval_df['p_raw'].values

    # BH procedure (Benjamini & Hochberg 1995)
    n   = len(p_raw)
    idx = np.argsort(p_raw)
    p_sorted = p_raw[idx]
    bh_thresh = (np.arange(1, n+1) / n) * 0.05  # FDR = 5%
    reject_sorted = p_sorted <= bh_thresh
    # All rejected: if rank k is rejected, all ranks ≤ k are rejected
    if reject_sorted.any():
        max_reject = np.where(reject_sorted)[0].max()
        reject_sorted[:max_reject+1] = True
    reject = np.zeros(n, dtype=bool)
    reject[idx] = reject_sorted

    # BH-adjusted p-values (Yekutieli-Benjamini step-up)
    p_adj = np.minimum(1.0, p_sorted * n / np.arange(1, n+1))
    p_adj = np.minimum.accumulate(p_adj[::-1])[::-1]
    p_adj_full = np.zeros(n)
    p_adj_full[idx] = p_adj

    pval_df['p_bh_adjusted'] = p_adj_full
    pval_df['reject_H0_fdr05'] = reject
    pval_df.to_csv(OUT / 'bh_corrected_pvalues.csv', index=False)

    print('\n  BH-corrected p-values:')
    for _, row in pval_df.iterrows():
        sig = '**SIGNIFICANT**' if row['reject_H0_fdr05'] else 'ns'
        print(f'    {row["test"][:50]:<52}: '
              f'p_raw={row["p_raw"]:.3e}  p_bh={row["p_bh_adjusted"]:.3e}  {sig}')

# ── Save bootstrap CI table ───────────────────────────────────────────────────
ci_df = pd.DataFrame(bootstrap_results)
ci_df.to_csv(OUT / 'bootstrap_ci_summary.csv', index=False)

# ── Figure: CI forest plot ────────────────────────────────────────────────────
print('\nGenerating bootstrap summary figure...')
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'savefig.dpi': 300, 'figure.facecolor': 'white',
})

auc_rows = ci_df[ci_df['metric'] == 'AUC-ROC'].copy()

if len(auc_rows) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bootstrap 95% Confidence Intervals (n=1,000 resamples)',
                 fontsize=12, fontweight='bold')

    # Panel A: AUC forest plot
    ax = axes[0]
    models_f = auc_rows['model'].tolist()
    ypos     = np.arange(len(models_f))
    ax.errorbar(auc_rows['point'], ypos,
                xerr=[auc_rows['point']-auc_rows['ci_lo95'],
                      auc_rows['ci_hi95']-auc_rows['point']],
                fmt='o', color='#2471A3', capsize=5, lw=1.5, ms=7, zorder=3)
    ax.axvline(0.80, color='red', lw=1.2, linestyle='--', alpha=0.6, label='AUC=0.80')
    ax.axvline(0.70, color='orange', lw=1.0, linestyle=':', alpha=0.6, label='AUC=0.70')
    ax.set_yticks(ypos)
    ax.set_yticklabels(models_f, fontsize=9)
    ax.set_xlabel('AUC-ROC  (point estimate ± 95% CI)')
    ax.set_title('A  AUC Forest Plot', fontweight='bold', loc='left')
    ax.set_xlim(0.3, 1.05)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    # Panel B: CI width (precision) comparison
    ax = axes[1]
    ax.barh(ypos, auc_rows['ci_width'], 0.5,
            color='#27AE60', alpha=0.8, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(models_f, fontsize=9)
    ax.set_xlabel('95% CI Width (smaller = more precise)')
    ax.set_title('B  CI Precision by Model', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    fig.tight_layout()
    fig.savefig(OUT / 'fig_bootstrap_summary.png', dpi=300)
    fig.savefig(OUT / 'fig_bootstrap_summary.pdf')
    plt.close(fig)
    print('  Saved: outputs/fig_bootstrap_summary.png (.pdf)')

# ── Write narrative report ────────────────────────────────────────────────────
report_lines = [
    '='*60,
    'BOOTSTRAP CI AND BENJAMINI-HOCHBERG CORRECTION REPORT',
    f'Generated: {datetime.now().isoformat()}',
    '='*60,
    '',
    '## Bootstrap Methodology',
    '  Resamples: n=1,000',
    '  CI level: 95% (α=0.05)',
    '  Strategy: stratified bootstrap (preserves class balance)',
    '  Implementation: numpy RandomState(42) for reproducibility',
    '',
    '## All Metrics with 95% CI',
    f'{"Model":<28} {"Metric":<12} {"Point":>7} {"Lo95%":>7} {"Hi95%":>7} {"Width":>7}',
    '-'*68,
]
for _, row in ci_df.iterrows():
    report_lines.append(
        f'  {row["model"]:<26} {row["metric"]:<12} '
        f'{row["point"]:>7.4f} {row["ci_lo95"]:>7.4f} '
        f'{row["ci_hi95"]:>7.4f} {row["ci_width"]:>7.4f}'
    )

report_lines += [
    '',
    '## Benjamini-Hochberg FDR Correction (α=0.05)',
    '  Reference: Benjamini Y, Hochberg Y (1995). J R Stat Soc B 57:289-300.',
    '             doi:10.1111/j.2517-6161.1995.tb02031.x',
    '',
    f'  Tests corrected: {len(pvalue_entries)}',
    f'{"Test":<54} {"p_raw":>10} {"p_BH":>10} {"Reject":>8}',
    '-'*88,
]
if pvalue_entries:
    for _, row in pval_df.iterrows():
        report_lines.append(
            f'  {row["test"][:52]:<52} {row["p_raw"]:>10.3e} '
            f'{row["p_bh_adjusted"]:>10.3e} {"YES" if row["reject_H0_fdr05"] else "NO":>8}'
        )
report_lines.append('='*60)
(OUT / 'bootstrap_stats_report.txt').write_text('\n'.join(report_lines), encoding='utf-8')

print('\n' + '='*60)
print('Fix 7 COMPLETE: Bootstrap CIs + BH Correction')
print('='*60)
print(f'  Metrics with 95% CI : {len(ci_df)}')
print(f'  P-values BH-corrected: {len(pvalue_entries)}')
print('  Outputs: bootstrap_ci_summary.csv, bh_corrected_pvalues.csv,')
print('           bootstrap_stats_report.txt, fig_bootstrap_summary.png/pdf')
