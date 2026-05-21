import pandas as pd, collections, numpy as np
from scipy import stats

df = pd.read_csv('final_fixed_influenza_ha_v2ok.csv')
h1 = df[df['Subtype']=='H1N1'].copy()
h1['Year'] = pd.to_numeric(h1['Year'], errors='coerce')
h1 = h1.dropna(subset=['Year','Sequence'])
h1['Year'] = h1['Year'].astype(int)

print("=== Sequence alphabet (first 5 H1N1 sequences) ===")
for _, r in h1.head(5).iterrows():
    seq = str(r['Sequence'])
    chars = collections.Counter(seq.upper())
    top = sorted(chars.items(), key=lambda x: -x[1])[:8]
    print(f"  Year={r['Year']} len={len(seq)} top={top}")

print()
# Check all unique characters across a sample
sample = h1.sample(min(100, len(h1)), random_state=42)
all_chars = collections.Counter()
for seq in sample['Sequence']:
    all_chars.update(str(seq).upper())
unique_chars = sorted(all_chars.keys())
print(f"Unique chars in sample: {unique_chars}")

# NT alphabet: A C G T (+ U X -)
# AA alphabet: 20 letters + X -
nt_set = set('ACGTU-N')
aa_only = set('DEFHIKLMNPQRSVWY')  # chars only in AA alphabet (not in NT)
is_aa = any(c in aa_only for c in unique_chars)
print(f"Contains AA-only chars: {is_aa} -> {'AMINO ACID' if is_aa else 'NUCLEOTIDE'}")

print()
# Compute pairwise distances for a cleaner analysis
# Use Human H1N1, 2009 onwards
human_h1 = h1[h1['Host'].str.lower().str.contains('human', na=False)]
post09 = human_h1[human_h1['Year'] >= 2009].copy()

# Best reference: most common sequence among 2009 seqs
ref = post09[post09['Year']==2009]['Sequence'].value_counts().index[0]
print(f"2009 reference len: {len(ref)}")
print(f"2009 ref first 30 chars: {ref[:30]}")

def hd(s, r):
    l = min(len(s), len(r))
    return sum(a != b for a, b in zip(s[:l], r[:l]))

post09['dist'] = post09['Sequence'].apply(lambda s: hd(str(s), ref))
yr_stats = post09.groupby('Year')['dist'].agg(['mean','std','count'])
print()
print("Post-2009 distances from 2009 reference:")
print(yr_stats.to_string())
slope, _, r2, _, _ = stats.linregress(yr_stats.index.astype(float), yr_stats['mean'])
print(f"Slope: {slope:.4f}")

# Also try: distance between consecutive year means
yr_means = yr_stats['mean'].values
years = yr_stats.index.values
print()
print("Year-to-year changes:")
for i in range(1, len(yr_means)):
    print(f"  {years[i-1]}-{years[i]}: {yr_means[i]-yr_means[i-1]:.2f}")
