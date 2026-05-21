"""Find Phase 1 rate and Phase 3 stats for the actual data."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('final_fixed_influenza_ha_v2ok.csv')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year', 'Sequence'])
df['Year'] = df['Year'].astype(int)

# ---- Phase 1 ----
h1 = df[(df['Subtype'] == 'H1N1') & (df['Host'].str.lower().str.contains('human', na=False))].copy()
h1 = h1.dropna(subset=['Sequence'])

ref_2009 = h1[h1['Year'] == 2009]['Sequence'].value_counts().index[0]
print(f"2009 reference (first 30): {ref_2009[:30]}")

def hd(s, r=ref_2009):
    l = min(len(str(s)), len(r))
    return sum(a != b for a, b in zip(str(s)[:l], r[:l]))

h1['dist'] = h1['Sequence'].apply(hd)

print("\nAll years summary:")
for yr, grp in h1.groupby('Year'):
    print(f"  {yr}: n={len(grp):4d}  mean={grp['dist'].mean():.1f}  "
          f"median={grp['dist'].median():.1f}  std={grp['dist'].std():.1f}")

# Post-pandemic: 2009-2017, anchor 2009=0
print("\n--- Post-pandemic rate from 2009 reference ---")
post = h1[h1['Year'] >= 2009].copy()
yr_stats = post.groupby('Year')['dist'].agg(['mean', 'median', 'std', 'count'])
print(yr_stats.to_string())

# Force 2009 baseline = 0 (reference came from 2009), rate over 2009-2017
x = np.array([2009] + list(yr_stats.index[1:]), dtype=float)
y_mean = np.array([0.0] + list(yr_stats['mean'].values[1:]))
y_median = np.array([0.0] + list(yr_stats['median'].values[1:]))

slope_mean, _, r, _, _ = stats.linregress(x, y_mean)
slope_median, _, r2, _, _ = stats.linregress(x, y_median)
print(f"\nRegression slope (mean, anchored 2009=0): {slope_mean:.4f}")
print(f"Regression slope (median, anchored 2009=0): {slope_median:.4f}")

# First-to-last rate
last_mean = yr_stats['mean'].iloc[-1]
last_yr = yr_stats.index[-1]
rate_fl = last_mean / (last_yr - 2009)
print(f"First-to-last rate (mean_2017/8): {rate_fl:.4f}")

rate_fl_med = yr_stats['median'].iloc[-1] / (last_yr - 2009)
print(f"First-to-last rate (median_2017/8): {rate_fl_med:.4f}")

# Filter: only pandemic lineage (dist < 60 from 2009 ref) for all years
print("\n--- Pandemic lineage only (dist < 60) ---")
pandemic = h1[(h1['dist'] < 60) & (h1['Year'] >= 2009)]
p_stats = pandemic.groupby('Year')['dist'].agg(['mean', 'median', 'count'])
print(p_stats.to_string())
if len(p_stats) > 1:
    x2 = np.array([2009] + list(p_stats.index[1:]), dtype=float)
    y2 = np.array([0.0] + list(p_stats['mean'].values[1:]))
    slope2, _, _, _, _ = stats.linregress(x2, y2)
    rate2 = p_stats['mean'].iloc[-1] / (p_stats.index[-1] - 2009)
    print(f"Slope: {slope2:.4f}  first-to-last: {rate2:.4f}")

# ---- Phase 3 ----
print("\n\n--- Phase 3 position-level analysis ---")
h3 = df[(df['Subtype'] == 'H3N2')].copy()
mode_len = int(h3['Sequence'].str.len().mode()[0])
h3 = h3[h3['Sequence'].str.len() == mode_len]

ref_h1 = h1['Sequence'].value_counts().index[0]
ref_h3 = h3['Sequence'].value_counts().index[0]
print(f"H1N1 overall ref len={len(ref_h1)}, H3N2 overall ref len={len(ref_h3)}")

H1N1_CRITICAL = list(range(121, 137)) + list(range(150, 158))
H3N2_CRITICAL = list(range(122, 138)) + list(range(155, 161))

def var_rate_per_pos(subdf, ref):
    n = len(subdf)
    ref_len = len(ref)
    counts = np.zeros(ref_len, dtype=int)
    for seq in subdf['Sequence']:
        s = str(seq)
        for i in range(min(len(s), ref_len)):
            if s[i] != ref[i]:
                counts[i] += 1
    return counts / n

vr_h1 = var_rate_per_pos(h1, ref_h1)
vr_h3 = var_rate_per_pos(h3, ref_h3)

h1c = set(H1N1_CRITICAL)
h3c = set(H3N2_CRITICAL)

rows = []
for pos, vr in enumerate(vr_h1):
    rows.append({'position': pos, 'subtype': 'H1N1', 'var_rate': vr,
                 'is_critical': pos in h1c})
for pos, vr in enumerate(vr_h3):
    rows.append({'position': pos, 'subtype': 'H3N2', 'var_rate': vr,
                 'is_critical': pos in h3c})

pos_df = pd.DataFrame(rows)
median_r = pos_df['var_rate'].median()
pos_df['high_var'] = pos_df['var_rate'] >= median_r
print(f"Median var rate: {median_r:.4f}")

ct = pd.crosstab(pos_df['is_critical'], pos_df['high_var'])
print(f"\nContingency table:\n{ct}")
chi2, pval, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"\nchi2={chi2:.2f}  p={pval:.4e}  n={n}  V={v:.4f}")

# Summarize critical-region variation rates
print(f"\nMean var rate in critical positions: {pos_df[pos_df['is_critical']]['var_rate'].mean():.4f}")
print(f"Mean var rate in non-critical positions: {pos_df[~pos_df['is_critical']]['var_rate'].mean():.4f}")
print(f"Ratio: {pos_df[pos_df['is_critical']]['var_rate'].mean() / pos_df[~pos_df['is_critical']]['var_rate'].mean():.2f}x")
