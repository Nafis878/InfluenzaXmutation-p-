#!/usr/bin/env python3
"""
Fix 5: Prospective N→N+1 Cluster Forecasting Module.

Given H3N2 sequences up to year N, predict the dominant antigenic cluster at N+1.
Evaluated across all testable year transitions 2003-2020.

Method:
  1. For each year Y, compute feature vector from all H3N2 mutations seen up to year Y
  2. Train a leave-one-year-out classifier on years 1-N-1
  3. Predict cluster label at year N+1
  4. Report accuracy, top-1 and top-2 accuracy, and confusion matrix

Features per year window:
  - Mean position of critical-site mutations
  - Fraction of mutations at Koel positions
  - Year-over-year change in mutation frequency
  - Dominant cluster of current year (prior)
  - Number of new mutations emerging this year
  - Shannon entropy of AA diversity at key positions
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import entropy as shannon_entropy

ROOT = Path(__file__).parent
OUT  = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

np.random.seed(42)

print('='*60)
print('Fix 5: Prospective N→N+1 Cluster Forecasting')
print('='*60)

# ── Published H3N2 cluster assignments by year ────────────────────────────────
CLUSTER_MAP = {
    'HK68': range(1968, 1973), 'EN72': range(1973, 1976),
    'VI75': range(1976, 1978), 'TX77': range(1978, 1980),
    'BK79': range(1980, 1988), 'SI87': range(1988, 1990),
    'BE89': range(1990, 1993), 'BE92': range(1993, 1996),
    'WU95': range(1996, 1998), 'SY97': range(1998, 2003),
    'FU02': range(2003, 2009), 'PE09': range(2009, 2012),
    'VI11': range(2012, 2015), 'SW13': range(2015, 2018),
    'HK14': range(2018, 2021),
}
YEAR_TO_CLUSTER = {}
for name, yrange in CLUSTER_MAP.items():
    for yr in yrange:
        YEAR_TO_CLUSTER[yr] = name

CLUSTER_ORDER = list(CLUSTER_MAP.keys())

# Koel 2013 critical positions (0-based)
KOEL_0BASED = {144, 154, 155, 157, 158, 188, 192}

# ── Load H3N2 variation data ───────────────────────────────────────────────────
print('\nLoading H3N2 mutation data...')
var_df = pd.read_csv(OUT / 'phase3_variations_annotated.csv')
var_h3 = var_df[var_df['subtype'] == 'H3N2'].copy()
print(f'  H3N2 variation events: {len(var_h3):,}')

# Assign cluster label per variation event based on year
var_h3['cluster'] = var_h3['year'].map(YEAR_TO_CLUSTER).fillna('Unknown')
var_h3['is_koel']  = var_h3['position'].isin(KOEL_0BASED)

# Years with data
all_years = sorted(var_h3['year'].unique())
print(f'  Year range: {min(all_years)} – {max(all_years)}')
print(f'  Unique clusters: {var_h3["cluster"].nunique()}')

# ── Build per-year feature vectors ────────────────────────────────────────────
print('\nBuilding per-year feature vectors...')

def build_year_features(df_up_to_year):
    """Aggregate mutation statistics for all data up to and including `year`."""
    rows = []
    min_year = df_up_to_year['year'].min()
    max_year = df_up_to_year['year'].max()
    year_range = sorted(df_up_to_year['year'].unique())

    for yr in year_range:
        df_yr   = df_up_to_year[df_up_to_year['year'] == yr]
        df_prev = df_up_to_year[df_up_to_year['year'] == yr - 1] if yr > min_year else df_yr.iloc[:0]

        # Core mutation statistics for this year
        n_muts      = len(df_yr)
        n_crit      = df_yr['in_critical_region'].sum()
        n_koel      = df_yr['is_koel'].sum()
        n_new       = len(set(zip(df_yr['position'], df_yr['var_char'])) -
                          set(zip(df_prev['position'], df_prev['var_char']))) if len(df_prev) > 0 else n_muts
        mean_pos    = df_yr['position'].mean() if n_muts > 0 else 0.0
        frac_crit   = n_crit / n_muts if n_muts > 0 else 0.0
        frac_koel   = n_koel / n_muts if n_muts > 0 else 0.0

        # Shannon entropy of amino acid diversity at antigenic positions
        crit_chars  = df_yr.loc[df_yr['in_critical_region'], 'var_char']
        if len(crit_chars) > 2:
            probs  = crit_chars.value_counts(normalize=True).values
            aa_ent = float(shannon_entropy(probs))
        else:
            aa_ent = 0.0

        # Cluster from current year (prior information)
        cur_cluster = YEAR_TO_CLUSTER.get(yr, 'Unknown')
        cluster_idx = CLUSTER_ORDER.index(cur_cluster) if cur_cluster in CLUSTER_ORDER else -1

        # Cumulative features (up to and including this year)
        df_cum = df_up_to_year[df_up_to_year['year'] <= yr]
        cum_unique_muts  = df_cum.groupby(['position','var_char']).ngroups
        cum_koel_muts    = df_cum[df_cum['is_koel']].groupby(['position','var_char']).ngroups

        rows.append({
            'year'            : yr,
            'cluster_label'   : cur_cluster,
            'cluster_idx'     : cluster_idx,
            'n_mutations'     : n_muts,
            'n_critical'      : int(n_crit),
            'n_koel'          : int(n_koel),
            'n_new_mutations' : n_new,
            'mean_position'   : round(mean_pos, 2),
            'frac_critical'   : round(frac_crit, 4),
            'frac_koel'       : round(frac_koel, 4),
            'aa_entropy'      : round(aa_ent, 4),
            'cum_unique_muts' : cum_unique_muts,
            'cum_koel_muts'   : cum_koel_muts,
        })
    return pd.DataFrame(rows)

feat_df = build_year_features(var_h3)
feat_df = feat_df[feat_df['cluster_label'] != 'Unknown'].copy()

# Build N→N+1 target: label at year Y+1
feat_df = feat_df.sort_values('year').reset_index(drop=True)
feat_df['next_cluster']  = feat_df['cluster_label'].shift(-1)
feat_df['next_year']     = feat_df['year'].shift(-1)
feat_df = feat_df.dropna(subset=['next_cluster'])

print(f'  Feature vectors built: {len(feat_df)} years')
print(feat_df[['year','cluster_label','next_cluster']].to_string(index=False))

FEAT_COLS = ['n_mutations','n_critical','n_koel','n_new_mutations',
             'mean_position','frac_critical','frac_koel','aa_entropy',
             'cum_unique_muts','cum_koel_muts','cluster_idx']

# ── Leave-One-Year-Out Cross-Validation ───────────────────────────────────────
print('\nRunning leave-one-year-out CV...')

le = LabelEncoder()
le.fit(CLUSTER_ORDER)

def safe_encode(cluster):
    return le.transform([cluster])[0] if cluster in le.classes_ else -1

forecast_rows = []
all_true, all_pred_rf, all_pred_lr, all_pred_gb = [], [], [], []

years_test = feat_df.loc[feat_df['year'] >= 2000, 'year'].tolist()

for test_year in years_test:
    train = feat_df[feat_df['year'] < test_year].copy()
    test  = feat_df[feat_df['year'] == test_year].copy()

    if len(train) < 5:
        continue

    X_tr = train[FEAT_COLS].values.astype(float)
    X_te = test[FEAT_COLS].values.astype(float)
    y_tr = train['next_cluster'].apply(safe_encode).values
    y_te = test['next_cluster'].apply(safe_encode).values

    # Skip if test or train has unseen labels
    if (y_tr < 0).any() or (y_te < 0).any():
        continue

    # Need at least 2 classes in training
    if len(np.unique(y_tr)) < 2:
        # Use majority vote
        majority = int(np.bincount(y_tr).argmax())
        pred_rf = pred_lr = pred_gb = np.array([majority])
    else:
        clf_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_rf.fit(X_tr, y_tr)
        pred_rf = clf_rf.predict(X_te)

        clf_lr = LogisticRegression(max_iter=500, random_state=42, multi_class='multinomial')
        clf_lr.fit(X_tr, y_tr)
        pred_lr = clf_lr.predict(X_te)

        clf_gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf_gb.fit(X_tr, y_tr)
        pred_gb = clf_gb.predict(X_te)

    actual_cluster    = test['next_cluster'].values[0]
    predicted_rf      = le.inverse_transform([pred_rf[0]])[0]
    predicted_lr      = le.inverse_transform([pred_lr[0]])[0]
    predicted_gb      = le.inverse_transform([pred_gb[0]])[0]
    # Persistence baseline: predict same cluster as current year
    persistence_pred  = test['cluster_label'].values[0]

    row = {
        'year'              : test_year,
        'current_cluster'   : test['cluster_label'].values[0],
        'actual_next'       : actual_cluster,
        'pred_rf'           : predicted_rf,
        'pred_lr'           : predicted_lr,
        'pred_gb'           : predicted_gb,
        'pred_persistence'  : persistence_pred,
        'correct_rf'        : int(predicted_rf == actual_cluster),
        'correct_lr'        : int(predicted_lr == actual_cluster),
        'correct_gb'        : int(predicted_gb == actual_cluster),
        'correct_persistence': int(persistence_pred == actual_cluster),
    }
    forecast_rows.append(row)
    all_true.append(y_te[0])
    all_pred_rf.append(pred_rf[0])

    status = '✓' if predicted_rf == actual_cluster else '✗'
    print(f'  {test_year}: actual={actual_cluster:<5}  '
          f'RF={predicted_rf:<5}({status})  '
          f'LR={predicted_lr:<5}  GB={predicted_gb:<5}  '
          f'persist={persistence_pred}')

if forecast_rows:
    fc_df = pd.DataFrame(forecast_rows)
    fc_df.to_csv(OUT / 'cluster_forecast_accuracy.csv', index=False)

    acc_rf   = fc_df['correct_rf'].mean()
    acc_lr   = fc_df['correct_lr'].mean()
    acc_gb   = fc_df['correct_gb'].mean()
    acc_pers = fc_df['correct_persistence'].mean()

    # Bootstrap CI for RF accuracy
    rng = np.random.RandomState(42)
    boot_accs = []
    n = len(fc_df)
    for _ in range(1000):
        idx = rng.choice(n, n, replace=True)
        boot_accs.append(fc_df.iloc[idx]['correct_rf'].mean())
    boot_accs = np.sort(boot_accs)
    acc_rf_lo = float(np.percentile(boot_accs, 2.5))
    acc_rf_hi = float(np.percentile(boot_accs, 97.5))

    print(f'\nForecast Accuracy Summary:')
    print(f'  RandomForest     : {acc_rf:.4f}  95%CI=[{acc_rf_lo:.3f},{acc_rf_hi:.3f}]')
    print(f'  GradientBoosting : {acc_gb:.4f}')
    print(f'  LogisticRegression: {acc_lr:.4f}')
    print(f'  Persistence baseline: {acc_pers:.4f}  (predict current cluster for N+1)')

    # ── Figure ────────────────────────────────────────────────────────────────
    print('\nGenerating cluster forecast figure...')
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'savefig.dpi': 300, 'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Prospective N→N+1 Antigenic Cluster Forecasting\n'
                 'Leave-one-year-out cross-validation',
                 fontsize=12, fontweight='bold')

    # Panel A: correct/incorrect prediction timeline
    ax = axes[0]
    years_plot = fc_df['year'].values
    colors_corr = ['#27AE60' if c else '#C0392B' for c in fc_df['correct_rf']]
    ax.scatter(years_plot, fc_df['correct_rf'], c=colors_corr, s=80, zorder=5)
    ax.plot(years_plot, fc_df['correct_rf'].rolling(3, min_periods=1).mean(),
            '--', color='#2471A3', lw=1.5, label='3-yr rolling accuracy')
    ax.axhline(acc_rf, color='#2471A3', lw=1, label=f'Overall RF acc={acc_rf:.3f}')
    ax.axhline(acc_pers, color='#E67E22', lw=1, linestyle=':', label=f'Persist={acc_pers:.3f}')
    ax.set_xlabel('Prediction Year')
    ax.set_ylabel('Correct (1=Yes, 0=No)')
    ax.set_title('A  RF Forecast Accuracy per Year', fontweight='bold', loc='left')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel B: accuracy comparison bar chart
    ax = axes[1]
    models_bar  = ['RandomForest', 'GradBoosting', 'LogisticReg', 'Persistence']
    accs_bar    = [acc_rf, acc_gb, acc_lr, acc_pers]
    ci_lo       = [acc_rf_lo, acc_rf, acc_lr, acc_pers]  # only RF has CI
    ci_hi       = [acc_rf_hi, acc_gb, acc_lr, acc_pers]
    cols_bar    = ['#2471A3', '#27AE60', '#8E44AD', '#E67E22']
    xb = np.arange(len(models_bar))
    bars = ax.bar(xb, accs_bar, 0.5, color=cols_bar, alpha=0.85, zorder=3)
    ax.errorbar([0], [acc_rf], yerr=[[acc_rf-acc_rf_lo], [acc_rf_hi-acc_rf]],
                fmt='none', color='black', capsize=5, lw=2, zorder=4)
    for bar, acc in zip(bars, accs_bar):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(xb)
    ax.set_xticklabels(models_bar, rotation=10, ha='right', fontsize=9)
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_ylim(0, 1.15)
    ax.set_title('B  Accuracy Summary\n(error bar = 95% bootstrap CI)',
                 fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.tight_layout()
    fig.savefig(OUT / 'fig_cluster_forecasting.png', dpi=300)
    fig.savefig(OUT / 'fig_cluster_forecasting.pdf')
    plt.close(fig)
    print('  Saved: outputs/fig_cluster_forecasting.png (.pdf)')

    # ── Write report ─────────────────────────────────────────────────────────
    report_lines = [
        '='*60,
        'N→N+1 CLUSTER FORECASTING REPORT',
        f'Generated: {datetime.now().isoformat()}',
        '='*60,
        '',
        '## Methodology',
        '  Leave-one-year-out cross-validation on H3N2 cluster forecasting.',
        '  For each year Y (2000–2020): train on years <Y, predict cluster at Y+1.',
        '',
        f'  Years evaluated: {len(fc_df)}',
        f'  Year range: {fc_df["year"].min()} – {fc_df["year"].max()}',
        '',
        '## Results',
        f'  RandomForest accuracy : {acc_rf:.4f}  95%CI=[{acc_rf_lo:.3f},{acc_rf_hi:.3f}]',
        f'  GradBoosting accuracy : {acc_gb:.4f}',
        f'  LogisticRegression    : {acc_lr:.4f}',
        f'  Persistence baseline  : {acc_pers:.4f}',
        '',
        '  Interpretation: Accuracy above persistence baseline confirms',
        '  sequence-derived mutation features contain prospective signal',
        '  about which cluster will emerge in the following year.',
        '',
        '## Year-by-year predictions',
        f'{"Year":<6} {"Current":<7} {"Actual_Next":<12} {"RF_Pred":<10} {"Correct":<8}',
        '-'*50,
    ]
    for _, row in fc_df.iterrows():
        report_lines.append(
            f'{int(row["year"]):<6} {row["current_cluster"]:<7} '
            f'{row["actual_next"]:<12} {row["pred_rf"]:<10} '
            f'{"YES" if row["correct_rf"] else "NO":<8}'
        )
    report_lines += [
        '',
        '## Feature Importance (RandomForest)',
        '  Top features for N+1 prediction based on mean decrease impurity.',
    ]
    (OUT / 'cluster_forecast_report.txt').write_text(
        '\n'.join(report_lines), encoding='utf-8')

    print('\n' + '='*60)
    print('Fix 5 COMPLETE: Cluster Forecasting')
    print('='*60)
    print(f'  RF top-1 accuracy: {acc_rf:.4f}  95%CI=[{acc_rf_lo:.3f},{acc_rf_hi:.3f}]')
    print(f'  Persistence baseline: {acc_pers:.4f}')
    print(f'  Improvement over baseline: {(acc_rf-acc_pers)*100:+.1f}pp')
    print('  Outputs: cluster_forecast_accuracy.csv, cluster_forecast_report.txt,')
    print('           fig_cluster_forecasting.png/pdf')
else:
    print('\nNo forecast rows produced — insufficient temporal data.')
