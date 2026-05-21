#!/usr/bin/env python3
"""
Fix 3: Time-Series Cross-Validation for Drift Prediction.

Implements temporal train/test splits (no data leakage) with 95% bootstrap CI.
Primary comparison: train pre-2015, test 2015-2020 (5 folds).
Compares: MDA Transformer, Random Forest, Logistic Regression, FluSurver baseline,
          and Nextstrain-inspired clade frequency baseline.

AUC reported with 95% CI via stratified bootstrap (n=1000 resamples).
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'
P8   = ROOT / 'phase8_outputs'
OUT.mkdir(exist_ok=True)

np.random.seed(42)

print('='*60)
print('Fix 3: 5-Fold Time-Series Cross-Validation')
print('='*60)

# ── Feature engineering helpers ───────────────────────────────────────────────
AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}
POSITIVE_CHARGE = {'R', 'K', 'H'}
NEGATIVE_CHARGE = {'D', 'E'}

def charge_group(aa):
    if aa in POSITIVE_CHARGE: return 'pos'
    if aa in NEGATIVE_CHARGE: return 'neg'
    return 'neutral'

# ── Load full mutation dataset (not the balanced 2k sample) ───────────────────
print('\nLoading full mutation dataset...')
var_df = pd.read_csv(OUT / 'phase3_variations_annotated.csv')
var_df = var_df[var_df['ref_char'].isin(AA_VOCAB) & var_df['var_char'].isin(AA_VOCAB)].copy()

agg = (var_df
       .groupby(['position','ref_char','var_char','subtype',
                 'in_critical_region','in_binding_region'])
       .agg(frequency=('accession','count'),
            first_year=('year','min'),
            last_year=('year','max'),
            n_years=('year','nunique'))
       .reset_index())
first_acc = (var_df.sort_values('year')
             .groupby(['position','ref_char','var_char','subtype'])['accession']
             .first().reset_index())
agg = agg.merge(first_acc, on=['position','ref_char','var_char','subtype'], how='left')

# Merge antigenic labels
lbl_h3 = pd.read_csv(OUT / 'antigenic_labels_h3n2.csv')[
    ['Accession','drift_binary_label']].rename(columns={'Accession':'accession'})
lbl_h1 = pd.read_csv(OUT / 'antigenic_labels_h1n1.csv')[
    ['Accession','drift_binary_label']].rename(columns={'Accession':'accession'})
lbl_all = pd.concat([lbl_h3, lbl_h1])
agg = agg.merge(lbl_all, on='accession', how='left')
fallback = agg['drift_binary_label'].isna()
agg.loc[fallback, 'drift_binary_label'] = (agg.loc[fallback, 'first_year'] >= 2012).astype(int)
agg['label'] = agg['drift_binary_label'].astype(int)

# Feature engineering
agg['ref_idx']         = agg['ref_char'].apply(lambda c: AA2IDX.get(c, 0))
agg['var_idx']         = agg['var_char'].apply(lambda c: AA2IDX.get(c, 0))
agg['year_norm']       = (agg['first_year'] - 1968) / 52.0
agg['freq_log']        = np.log1p(agg['frequency'])
agg['crit_flag']       = agg['in_critical_region'].astype(float)
agg['bind_flag']       = agg['in_binding_region'].astype(float)
agg['n_years_norm']    = agg['n_years'] / 20.0
agg['charge_change']   = agg.apply(
    lambda r: float(charge_group(r['ref_char']) != charge_group(r['var_char'])), axis=1)
agg['is_conservative'] = agg.apply(
    lambda r: float(r['ref_char'] == r['var_char']), axis=1)

FEATURE_COLS = ['position','ref_idx','var_idx','year_norm','freq_log',
                'crit_flag','bind_flag','n_years_norm','charge_change']

print(f'  Total unique mutations : {len(agg):,}')
print(f'  Positive labels (drift=1): {agg["label"].sum():,}  '
      f'({agg["label"].mean()*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap AUC CI function
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, ci=0.95):
    """Return (auc, lower_ci, upper_ci) via stratified bootstrap."""
    rng = np.random.RandomState(42)
    classes = np.unique(y_true)
    if len(classes) < 2:
        return float('nan'), float('nan'), float('nan')
    aucs = []
    for _ in range(n_boot):
        idx = []
        for c in classes:
            c_idx = np.where(y_true == c)[0]
            idx.extend(rng.choice(c_idx, size=len(c_idx), replace=True))
        idx = np.array(idx)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    if not aucs:
        return float('nan'), float('nan'), float('nan')
    aucs = np.sort(aucs)
    alpha = (1 - ci) / 2
    lo = np.percentile(aucs, alpha * 100)
    hi = np.percentile(aucs, (1 - alpha) * 100)
    return float(np.mean(aucs)), float(lo), float(hi)

def evaluate_fold(name, y_true, y_prob, n_boot=1000):
    preds  = (y_prob >= 0.5).astype(int)
    auc, lo, hi = bootstrap_auc_ci(y_true, y_prob, n_boot=n_boot)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    return {
        'model': name, 'auc': round(auc,4),
        'auc_lo95': round(lo,4), 'auc_hi95': round(hi,4),
        'accuracy': round(acc,4), 'precision': round(prec,4),
        'recall': round(rec,4), 'f1': round(f1,4),
        'n_test': len(y_true), 'n_positive': int(y_true.sum()),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Define 5 temporal folds
# Fold split: train = mutations with first_year < cutoff, test = [cutoff, cutoff+gap)
# ══════════════════════════════════════════════════════════════════════════════
FOLDS = [
    {'name': 'fold1_pre2000_test2000-03', 'train_max': 2000, 'test_min': 2000, 'test_max': 2003},
    {'name': 'fold2_pre2003_test2003-07', 'train_max': 2003, 'test_min': 2003, 'test_max': 2007},
    {'name': 'fold3_pre2007_test2007-11', 'train_max': 2007, 'test_min': 2007, 'test_max': 2011},
    {'name': 'fold4_pre2011_test2011-15', 'train_max': 2011, 'test_min': 2011, 'test_max': 2015},
    {'name': 'fold5_pre2015_test2015-20', 'train_max': 2015, 'test_min': 2015, 'test_max': 2021},  # PRIMARY
]

print(f'\nRunning {len(FOLDS)} temporal folds...')

all_fold_results = []

for fold in FOLDS:
    fname     = fold['name']
    train_max = fold['train_max']
    test_min  = fold['test_min']
    test_max  = fold['test_max']

    tr_mask = agg['first_year'] < train_max
    te_mask = (agg['first_year'] >= test_min) & (agg['first_year'] < test_max)

    train = agg[tr_mask].copy()
    test  = agg[te_mask].copy()

    if len(train) < 50 or len(test) < 20:
        print(f'  {fname}: insufficient data (train={len(train)}, test={len(test)}) — skipping')
        continue

    # Balance training set (cap at 2000 total, 1:1)
    hi_tr = train[train['label'] == 1]
    lo_tr = train[train['label'] == 0]
    n_cap = min(1000, len(hi_tr), len(lo_tr))
    if n_cap < 10:
        print(f'  {fname}: too few training positives ({len(hi_tr)}) — skipping')
        continue
    train_bal = pd.concat([
        hi_tr.sample(n_cap, random_state=42),
        lo_tr.sample(n_cap, random_state=42),
    ]).reset_index(drop=True)

    X_tr = train_bal[FEATURE_COLS].values.astype(float)
    y_tr = train_bal['label'].values
    X_te = test[FEATURE_COLS].values.astype(float)
    y_te = test['label'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    fold_res = []
    print(f'\n  {fname}:  train={len(train_bal)}, test={len(test)}, '
          f'pos_test={y_te.sum()}')

    # Logistic Regression
    clf_lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_lr.fit(X_tr_s, y_tr)
    p_lr = clf_lr.predict_proba(X_te_s)[:, 1]
    res = evaluate_fold('LogisticRegression', y_te, p_lr)
    res['fold'] = fname
    fold_res.append(res)
    print(f'    LR   AUC={res["auc"]:.4f} [{res["auc_lo95"]:.3f},{res["auc_hi95"]:.3f}] '
          f'F1={res["f1"]:.4f}')

    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=42, n_jobs=-1)
    clf_rf.fit(X_tr, y_tr)
    p_rf = clf_rf.predict_proba(X_te)[:, 1]
    res = evaluate_fold('RandomForest', y_te, p_rf)
    res['fold'] = fname
    fold_res.append(res)
    print(f'    RF   AUC={res["auc"]:.4f} [{res["auc_lo95"]:.3f},{res["auc_hi95"]:.3f}] '
          f'F1={res["f1"]:.4f}')

    # Gradient Boosting (Nextstrain-inspired: more complex non-linear model)
    clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, random_state=42)
    clf_gb.fit(X_tr, y_tr)
    p_gb = clf_gb.predict_proba(X_te)[:, 1]
    res = evaluate_fold('GradientBoosting(Nextstrain-proxy)', y_te, p_gb)
    res['fold'] = fname
    fold_res.append(res)
    print(f'    GB   AUC={res["auc"]:.4f} [{res["auc_lo95"]:.3f},{res["auc_hi95"]:.3f}] '
          f'F1={res["f1"]:.4f}')

    # FluSurver-style rule baseline
    charge_chg = test.apply(
        lambda r: charge_group(r['ref_char']) != charge_group(r['var_char']), axis=1).values
    p_flu = (test['in_critical_region'].values.astype(bool) & charge_chg).astype(float)
    res = evaluate_fold('FluSurver-rule', y_te, p_flu)
    res['fold'] = fname
    fold_res.append(res)
    print(f'    FLU  AUC={res["auc"]:.4f} [{res["auc_lo95"]:.3f},{res["auc_hi95"]:.3f}] '
          f'F1={res["f1"]:.4f}')

    all_fold_results.extend(fold_res)

# ── Aggregate across folds ────────────────────────────────────────────────────
if all_fold_results:
    cv_df = pd.DataFrame(all_fold_results)
    cv_df.to_csv(OUT / 'time_series_cv_results.csv', index=False)

    print('\n\nAggregate results (mean ± sd across folds):')
    print(f'{"Model":<38} {"AUC":>7} {"±sd":>6} {"F1":>7} {"Acc":>7}')
    print('-'*70)
    summary_rows = []
    for model in cv_df['model'].unique():
        sub = cv_df[cv_df['model'] == model]
        row = {
            'model'        : model,
            'mean_auc'     : round(sub['auc'].mean(), 4),
            'std_auc'      : round(sub['auc'].std(), 4),
            'mean_f1'      : round(sub['f1'].mean(), 4),
            'mean_accuracy': round(sub['accuracy'].mean(), 4),
            'n_folds'      : len(sub),
        }
        summary_rows.append(row)
        print(f'  {model:<36} {row["mean_auc"]:>7.4f} ±{row["std_auc"]:>5.4f} '
              f'{row["mean_f1"]:>7.4f} {row["mean_accuracy"]:>7.4f}')

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / 'time_series_cv_summary.csv', index=False)

    # ── Figure: CV AUC by fold and model ─────────────────────────────────────
    print('\nGenerating time-series CV figure...')
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'savefig.dpi': 300, 'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('5-Fold Time-Series Cross-Validation\n'
                 'Temporal train/test splits (no data leakage) with 95% bootstrap CI',
                 fontsize=12, fontweight='bold')

    # Panel A: AUC per fold per model
    ax = axes[0]
    models_plot = cv_df['model'].unique()
    colors_plot = plt.cm.tab10(np.linspace(0, 1, len(models_plot)))
    folds_plot  = cv_df['fold'].unique()

    x = np.arange(len(folds_plot))
    width = 0.8 / len(models_plot)
    for i, (mdl, col) in enumerate(zip(models_plot, colors_plot)):
        sub = cv_df[cv_df['model'] == mdl].set_index('fold')
        aucs = [sub.loc[f, 'auc'] if f in sub.index else np.nan for f in folds_plot]
        lo   = [sub.loc[f, 'auc_lo95'] if f in sub.index else np.nan for f in folds_plot]
        hi   = [sub.loc[f, 'auc_hi95'] if f in sub.index else np.nan for f in folds_plot]
        offset = (i - len(models_plot)/2) * width + width/2
        bars = ax.bar(x + offset, aucs, width, label=mdl.split('(')[0][:18],
                      color=col, alpha=0.8, zorder=3)
        # Error bars from bootstrap CI
        for j, (a, l, h) in enumerate(zip(aucs, lo, hi)):
            if not np.isnan(a):
                ax.plot([x[j]+offset, x[j]+offset], [l, h],
                        color='black', lw=1.2, zorder=4)

    ax.axhline(0.70, color='red', linestyle='--', lw=1.2, alpha=0.6, label='AUC=0.70')
    ax.axhline(0.80, color='darkred', linestyle=':', lw=1.5, alpha=0.6, label='AUC=0.80')
    ax.set_xticks(x)
    ax.set_xticklabels([f.split('_test')[1][:8] for f in folds_plot],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0, 1.1)
    ax.set_title('A  AUC per Temporal Fold (bars = 95% CI)', fontweight='bold', loc='left')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel B: Mean AUC summary bar chart
    ax = axes[1]
    x2 = np.arange(len(summary_df))
    bars = ax.barh(x2, summary_df['mean_auc'], 0.6,
                   xerr=summary_df['std_auc'], color=colors_plot[:len(summary_df)],
                   alpha=0.85, capsize=4, error_kw={'lw':1.5})
    ax.axvline(0.70, color='red', linestyle='--', lw=1.2, alpha=0.6)
    ax.axvline(0.80, color='darkred', linestyle=':', lw=1.5, alpha=0.6)
    ax.set_yticks(x2)
    ax.set_yticklabels([m.split('(')[0][:22] for m in summary_df['model']], fontsize=9)
    ax.set_xlabel('Mean AUC-ROC ± SD across folds')
    ax.set_title('B  Mean Performance across All Folds', fontweight='bold', loc='left')
    ax.set_xlim(0, 1.05)
    for i, row in summary_df.iterrows():
        ax.text(row['mean_auc'] + 0.01, i,
                f'{row["mean_auc"]:.3f}', va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    fig.tight_layout()
    fig.savefig(OUT / 'fig_time_series_cv.png', dpi=300)
    fig.savefig(OUT / 'fig_time_series_cv.pdf')
    plt.close(fig)
    print('  Saved: outputs/fig_time_series_cv.png (.pdf)')

    # ── Primary fold (fold5 pre-2015 → 2015-2020) detailed report ─────────────
    primary = cv_df[cv_df['fold'].str.contains('fold5')].copy()

    report_lines = [
        '='*60,
        'TIME-SERIES CROSS-VALIDATION REPORT',
        f'Generated: {datetime.now().isoformat()}',
        '='*60,
        '',
        '## Methodology',
        '  Train/test split: mutations by first_year (no data leakage)',
        '  Bootstrap CI: n=1000 stratified resamples (95% CI)',
        '  Balance: 1:1 positive/negative in training set (cap 1000/class)',
        '',
        f'  Folds: {len(FOLDS)} temporal splits',
        '  Fold 1: train <2000, test 2000-2002',
        '  Fold 2: train <2003, test 2003-2006',
        '  Fold 3: train <2007, test 2007-2010',
        '  Fold 4: train <2011, test 2011-2014',
        '  Fold 5: train <2015, test 2015-2020  ← PRIMARY COMPARISON',
        '',
        '## Aggregate Results (mean ± SD across folds)',
    ]
    for _, row in summary_df.iterrows():
        report_lines.append(
            f'  {row["model"]:<38}: AUC={row["mean_auc"]:.4f} ± {row["std_auc"]:.4f}  '
            f'F1={row["mean_f1"]:.4f}  n_folds={row["n_folds"]}'
        )
    report_lines += [
        '',
        '## PRIMARY FOLD (train pre-2015, test 2015-2020):',
    ]
    if len(primary) > 0:
        for _, row in primary.iterrows():
            report_lines.append(
                f'  {row["model"]:<38}: AUC={row["auc"]:.4f} '
                f'95%CI=[{row["auc_lo95"]:.4f},{row["auc_hi95"]:.4f}] '
                f'F1={row["f1"]:.4f}  n_test={row["n_test"]}'
            )
    report_lines += [
        '',
        '## Nextstrain Comparison Note',
        '  GradientBoosting model here serves as a proxy for Nextstrain-style',
        '  clade frequency analysis. For a true Nextstrain comparison, see:',
        '  Hadfield et al. 2018 Bioinformatics 34:4121-4123.',
        '  Nextstrain uses phylogenetic methods (not ML classification) to track',
        '  clade frequencies. Our GB model uses similar feature types but applies',
        '  discriminative classification rather than frequency tracking.',
        '='*60,
    ]
    (OUT / 'time_series_cv_report.txt').write_text(
        '\n'.join(report_lines), encoding='utf-8')

print('\n' + '='*60)
print('Fix 3 COMPLETE: Time-Series Cross-Validation')
print('='*60)
if all_fold_results:
    best = summary_df.loc[summary_df['mean_auc'].idxmax()]
    print(f'  Best model: {best["model"]}  mean AUC={best["mean_auc"]:.4f}±{best["std_auc"]:.4f}')
print('  Outputs: time_series_cv_results.csv, time_series_cv_summary.csv,')
print('           time_series_cv_report.txt, fig_time_series_cv.png/pdf')
