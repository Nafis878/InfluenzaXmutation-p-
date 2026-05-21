"""Compute correct per-site-per-sequence enrichment for Phase 3."""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv('final_fixed_influenza_ha_v2ok.csv')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year','Sequence'])
df['Year'] = df['Year'].astype(int)

H1N1_CRITICAL = list(range(120, 136)) + list(range(149, 157))
H3N2_CRITICAL = list(range(121, 137)) + list(range(154, 160))

h1 = df[df['Subtype']=='H1N1'].dropna(subset=['Sequence']).copy()
h3 = df[df['Subtype']=='H3N2'].copy()
h3 = h3[h3['Sequence'].str.len() == int(h3['Sequence'].str.len().mode()[0])]

ref_h1 = h1['Sequence'].value_counts().index[0]
ref_h3 = h3['Sequence'].value_counts().index[0]

def count_vars(subdf, ref_seq, crit_set):
    ref_len = len(ref_seq)
    n_crit_pos = len(crit_set)
    n_noncrit_pos = ref_len - n_crit_pos
    crit_events = 0
    noncrit_events = 0
    for seq in subdf['Sequence']:
        s = str(seq)
        l = min(len(s), ref_len)
        for i in range(l):
            if s[i] != ref_seq[i]:
                if i in crit_set:
                    crit_events += 1
                else:
                    noncrit_events += 1
    return crit_events, noncrit_events, n_crit_pos, n_noncrit_pos

print("Computing H1N1...")
c1, nc1, cp1, ncp1 = count_vars(h1, ref_h1, set(H1N1_CRITICAL))
print(f"H1N1: crit_events={c1:,} in {cp1} crit_pos x {len(h1)} seqs = {len(h1)*cp1:,} site-obs")
print(f"H1N1: noncrit_events={nc1:,} in {ncp1} noncrit_pos x {len(h1)} seqs = {len(h1)*ncp1:,} site-obs")

print("\nComputing H3N2...")
c3, nc3, cp3, ncp3 = count_vars(h3, ref_h3, set(H3N2_CRITICAL))
print(f"H3N2: crit_events={c3:,} in {cp3} crit_pos x {len(h3)} seqs = {len(h3)*cp3:,} site-obs")
print(f"H3N2: noncrit_events={nc3:,} in {ncp3} noncrit_pos x {len(h3)} seqs = {len(h3)*ncp3:,} site-obs")

# Per-site-per-sequence density
n1, n3 = len(h1), len(h3)
total_crit_site_obs = n1*cp1 + n3*cp3
total_noncrit_site_obs = n1*ncp1 + n3*ncp3
total_crit = c1 + c3
total_noncrit = nc1 + nc3

dens_crit = total_crit / total_crit_site_obs
dens_noncrit = total_noncrit / total_noncrit_site_obs
enrichment = dens_crit / dens_noncrit
print(f"\n=== Per-site-per-sequence density ===")
print(f"Critical density:     {dens_crit:.6f}")
print(f"Non-critical density: {dens_noncrit:.6f}")
print(f"Enrichment ratio:     {enrichment:.4f}")

# Chi-square (expected proportional to site-observations)
frac_crit = total_crit_site_obs / (total_crit_site_obs + total_noncrit_site_obs)
total_events = total_crit + total_noncrit
exp_crit = total_events * frac_crit
exp_noncrit = total_events * (1-frac_crit)
chi2, p = stats.chisquare([total_crit, total_noncrit], f_exp=[exp_crit, exp_noncrit])
print(f"\nChi-square: {chi2:.2f}  p={p:.2e}")
print(f"Expected crit: {exp_crit:.0f}  Observed: {total_crit:,}")

# Also: H1N1-only enrichment
dens_crit_h1 = c1 / (n1*cp1)
dens_noncrit_h1 = nc1 / (n1*ncp1)
print(f"\n--- H1N1 only enrichment: {dens_crit_h1/dens_noncrit_h1:.4f}")
# H3N2-only enrichment
dens_crit_h3 = c3 / (n3*cp3)
dens_noncrit_h3 = nc3 / (n3*ncp3)
print(f"--- H3N2 only enrichment: {dens_crit_h3/dens_noncrit_h3:.4f}")
