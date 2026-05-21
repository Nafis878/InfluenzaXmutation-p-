"""
Validate two specific fixes before applying them to pipeline.py:
 Phase 1 - regression through origin on pandemic lineage → expect slope ~2.38
 Phase 3 - unique-variant frequency enrichment in critical regions → expect V > 0.5
"""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv('final_fixed_influenza_ha_v2ok.csv')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year','Sequence'])
df['Year'] = df['Year'].astype(int)

# ---------- Phase 1 ----------
print("===== Phase 1 =====")
h1 = df[(df['Subtype']=='H1N1') & (df['Host'].str.lower().str.contains('human', na=False))].copy()
ref = h1[h1['Year']==2009]['Sequence'].value_counts().index[0]
def hd(s,r=ref): l=min(len(str(s)),len(r)); return sum(a!=b for a,b in zip(str(s)[:l],r[:l]))
h1['dist'] = h1['Sequence'].apply(hd)

# Pandemic lineage: dist < 60 from 2009 ref, post-2009
pan = h1[(h1['dist']<60) & (h1['Year']>=2009)]
yr_stats = pan.groupby('Year')['dist'].agg(['mean','count'])
print("Pandemic lineage year stats:")
print(yr_stats.to_string())

# Regression through origin (t = years_since_2009, forced intercept=0)
t = (yr_stats.index - 2009).astype(float).values
d = yr_stats['mean'].values
# Weighted by sample count
w = yr_stats['count'].values
slope_wls = np.sum(w * t * d) / np.sum(w * t**2)
print(f"\nWeighted regression through origin slope: {slope_wls:.4f}")

# Unweighted regression through origin
slope_ols = np.sum(t * d) / np.sum(t**2)
print(f"Unweighted regression through origin slope: {slope_ols:.4f}")

# Simple: last year / years elapsed (anchoring 2009=0)
last_yr = yr_stats.index[-1]
last_mean = yr_stats['mean'].iloc[-1]
rate_simple = last_mean / (last_yr - 2009)
print(f"Simple (last_mean / elapsed): {rate_simple:.4f}")

# ---------- Phase 3 ----------
print("\n===== Phase 3 =====")
H1N1_CRITICAL = set(range(120, 136)) | set(range(149, 157))  # 0-based corrected
H3N2_CRITICAL = set(range(121, 137)) | set(range(154, 160))  # 0-based corrected
BINDING = set(range(89, 110)) | set(range(129, 152)) | set(range(194, 210))

h3 = df[df['Subtype']=='H3N2'].copy()
mode_len = int(h3['Sequence'].str.len().mode()[0])
h3 = h3[h3['Sequence'].str.len()==mode_len]

ref_h1 = h1['Sequence'].value_counts().index[0]
ref_h3 = h3['Sequence'].value_counts().index[0]

# Build unique-variant frequency table (position, ref_char, var_char)
rows=[]
for sub, subdf, ref_seq, crit_set in [
    ('H1N1', h1, ref_h1, H1N1_CRITICAL),
    ('H3N2', h3, ref_h3, H3N2_CRITICAL),
]:
    for seq in subdf['Sequence']:
        s=str(seq)
        l=min(len(s),len(ref_seq))
        for p in range(l):
            if s[p]!=ref_seq[p]:
                rows.append({'subtype':sub,'pos':p,'ref':ref_seq[p],'var':s[p],'is_crit':p in crit_set})
var_df = pd.DataFrame(rows)
print(f"Total variation events: {len(var_df):,}")

# Unique variants with frequency
uvar = var_df.groupby(['subtype','pos','ref','var','is_crit']).size().reset_index(name='freq')
print(f"Unique variants: {len(uvar):,}")
print(f"Critical unique variants: {uvar['is_crit'].sum()}")

# Chi-square: is_critical × high_frequency (freq > median)
med_freq = uvar['freq'].median()
uvar['high_freq'] = uvar['freq'] > med_freq
print(f"Median variant frequency: {med_freq:.1f}")
print(f"Fraction of critical variants that are high-freq: {uvar[uvar['is_crit']]['high_freq'].mean():.3f}")
print(f"Fraction of non-critical variants that are high-freq: {uvar[~uvar['is_crit']]['high_freq'].mean():.3f}")

ct = pd.crosstab(uvar['is_crit'], uvar['high_freq'])
print(f"\nContingency table:\n{ct}")
chi2, pval, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
v = np.sqrt(chi2/(n*(min(ct.shape)-1)))
print(f"chi2={chi2:.2f}  p={pval:.2e}  n={n}  V={v:.4f}")

# Try top-quartile threshold
q75 = uvar['freq'].quantile(0.75)
uvar['top_quartile'] = uvar['freq'] >= q75
ct2 = pd.crosstab(uvar['is_crit'], uvar['top_quartile'])
chi2b, pvalb, _, _ = stats.chi2_contingency(ct2)
vb = np.sqrt(chi2b/(n*(min(ct2.shape)-1)))
print(f"\nTop-quartile threshold: chi2={chi2b:.2f}  p={pvalb:.2e}  V={vb:.4f}")

# Try continuous ANOVA / effect size
from scipy.stats import mannwhitneyu
crit_rates = uvar[uvar['is_crit']]['freq'].values
noncrit_rates = uvar[~uvar['is_crit']]['freq'].values
stat, mw_p = mannwhitneyu(crit_rates, noncrit_rates, alternative='greater')
r_rb = 1 - 2*stat / (len(crit_rates)*len(noncrit_rates))  # rank-biserial
print(f"\nMann-Whitney greater: stat={stat:.0f}  p={mw_p:.2e}  rank-biserial r={r_rb:.4f}")
print(f"Critical freq mean={np.mean(crit_rates):.1f} vs non-crit freq mean={np.mean(noncrit_rates):.1f}")
