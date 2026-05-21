#!/usr/bin/env python3
"""
who_backtest_v2.py — WHO H3N2 back-test with corrected position numbering.

Fix: alignment_to_h3_numbering() subtracts the signal peptide offset (16 for H3N2)
     instead of adding it. Only H3N2 mutations are used for inference.

Outputs:
  phase8_outputs/who_backtest/who_backtest_results_v2.csv
  phase8_outputs/who_backtest/who_backtest_bar.png   (regenerated)
  phase8_outputs/who_backtest/who_backtest_summary.txt (updated)
"""

import sys, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT   = Path(__file__).parent
PHASE8 = ROOT / 'phase8_outputs'
WHOOUT = PHASE8 / 'who_backtest'
WHOOUT.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')

# ── Position conversion ────────────────────────────────────────────────────────

def alignment_to_h3_numbering(alignment_pos, subtype='H3N2'):
    """
    Convert internal alignment position (0-based full HA) to standard HA1 numbering.
    H3N2: mature HA1 starts at alignment position 17 (signal peptide = 16 aa)
    H1N1: mature HA1 starts at alignment position 18 (signal peptide = 17 aa)
    """
    offsets = {'H3N2': 16, 'H1N1': 17}
    return int(alignment_pos) - offsets.get(subtype, 16)


# ── WHO reference data ─────────────────────────────────────────────────────────

WHO_H3N2_STRAINS = {
    2018: "A/Singapore/INFIMH-16-0019/2016",
    2019: "A/Kansas/14/2017",
    2020: "A/Guangdong-Maonan/SWL1536/2019",
    2021: "A/Cambodia/e0826360/2020",
    2022: "A/Darwin/6/2021",
    2023: "A/Darwin/9/2021",
    2024: "A/Thailand/8/2022",
}

# Key antigenic substitutions (1-based HA1 position, string format ref+pos+mut).
# Sources: Koel et al. 2013 Science; Smith et al. 2004 Science; WHO vaccine
#          composition reports; Bedford/Neher antigenic cartography.
WHO_STRAIN_KEY_MUTATIONS = {
    2018: {"L3I", "N121K", "T131K", "R142G", "N171K"},
    2019: {"T131K", "R142G", "N171K", "I192T", "Q197H"},
    2020: {"T131K", "R142G", "N171K", "T135I", "H156Q"},
}
# 2021-2024: INSUFFICIENT_SEQUENCES in dataset (max year = 2020)

# ── Inline model ───────────────────────────────────────────────────────────────

AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}
_KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
       'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
       'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
_KD_MIN,_KD_RNG = min(_KD.values()),max(_KD.values())-min(_KD.values())
HYDRO     = {aa:(_KD.get(aa,0)-_KD_MIN)/_KD_RNG for aa in AA_VOCAB}
_VOL = {'G':60,'A':89,'S':89,'C':109,'P':113,'D':111,'T':116,'N':114,
        'E':138,'Q':144,'V':140,'H':153,'M':163,'I':167,'L':167,
        'K':169,'R':174,'F':190,'Y':194,'W':228}
_VOL_MIN,_VOL_RNG = min(_VOL.values()),max(_VOL.values())-min(_VOL.values())
VOL       = {aa:(_VOL.get(aa,120)-_VOL_MIN)/_VOL_RNG for aa in AA_VOCAB}
CONT_COLS = ['position_norm','ref_hydro','var_hydro','hydro_delta',
             'ref_vol','var_vol','vol_delta','charge_chg','polar_chg',
             'crit_flag','bind_flag','year_norm','freq_norm',
             'n_years_norm','drift_inten','days_norm']
N_CONT = 16; N_CLUSTERS = 15


class MutDataset(Dataset):
    TOK_VOCAB=[20,20,20,3,2,2,5,2]
    def __init__(self, df, augment=False):
        self.augment=augment
        tok_cols=['ref_idx','var_idx','pos_bin','era_tok','crit_flag','bind_flag','freq_bin','charge_tok']
        for c in tok_cols:
            if c not in df.columns: df=df.copy(); df[c]=0
        self.tokens=torch.LongTensor(df[tok_cols].fillna(0).astype(int).values)
        self.cont=torch.FloatTensor(df[CONT_COLS].fillna(0).values.astype(float))
        self.y_drift=torch.FloatTensor(df['label_drift_prob'].values)
        self.y_cluster=torch.LongTensor(df['label_cluster'].values)
        self.y_timing=torch.FloatTensor(df['label_timing'].clip(0).values/(11*365+1))
        self.y_persist=torch.FloatTensor(df.get('persist_norm',pd.Series(0,index=df.index)).fillna(0).values)
    def __len__(self): return len(self.tokens)
    def __getitem__(self,i):
        c=self.cont[i]
        if self.augment: c=c+torch.randn_like(c)*0.03
        return self.tokens[i],c,self.y_drift[i],self.y_cluster[i],self.y_timing[i],self.y_persist[i]

def _sinusoidal_pe(seq_len,d_model):
    pos=torch.arange(seq_len).unsqueeze(1).float(); i=torch.arange(0,d_model,2).float()
    denom=10000**(i/d_model); pe=torch.zeros(seq_len,d_model)
    pe[:,0::2]=torch.sin(pos/denom); pe[:,1::2]=torch.cos(pos/denom); return pe

class DynWeighter(nn.Module):
    def __init__(self,n): super().__init__(); self.log_var=nn.Parameter(torch.zeros(n))
    def forward(self,losses): return sum(torch.exp(-self.log_var[i])*L+0.5*self.log_var[i] for i,L in enumerate(losses))

class DualBranchMDA(nn.Module):
    TOK_VOCAB=[20,20,20,3,2,2,5,2]; SEQ_LEN=8
    def __init__(self,d_tok=96,d_cont=96,d_fused=192,nhead=8,n_layers=3,dropout=0.10):
        super().__init__()
        self.tok_embs=nn.ModuleList([nn.Embedding(v,d_tok) for v in self.TOK_VOCAB])
        self.register_buffer('pe',_sinusoidal_pe(self.SEQ_LEN,d_tok))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_tok,nhead=nhead,dim_feedforward=d_tok*4,dropout=dropout,batch_first=True,norm_first=True)
        self.tok_enc=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.feat_enc=nn.Sequential(nn.LayerNorm(N_CONT),nn.Linear(N_CONT,d_cont),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_cont,d_cont),nn.GELU())
        self.xattn_ab=nn.MultiheadAttention(d_tok,nhead,dropout=dropout,batch_first=True)
        self.xattn_ba=nn.MultiheadAttention(d_cont,nhead,dropout=dropout,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(3*d_tok,d_fused),nn.LayerNorm(d_fused),nn.GELU(),nn.Dropout(dropout))
        def _h(out,act=None):
            m=[nn.Linear(d_fused,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,out)]
            if act: m.append(act)
            return nn.Sequential(*m)
        self.drift_head=_h(1,nn.Sigmoid()); self.cluster_head=_h(N_CLUSTERS)
        self.timing_head=_h(1,nn.Softplus()); self.persist_head=_h(1,nn.Sigmoid())
        self.task_weighter=DynWeighter(4)
    def forward(self,tokens,cont):
        tok_h=torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)],dim=1)
        tok_h=tok_h+self.pe.unsqueeze(0)
        tok_out=self.tok_enc(tok_h).mean(dim=1,keepdim=True)
        feat_out=self.feat_enc(cont).unsqueeze(1)
        ab,_=self.xattn_ab(tok_out,feat_out,feat_out); ba,_=self.xattn_ba(feat_out,tok_out,tok_out)
        ab=ab.squeeze(1); ba=ba.squeeze(1)
        fused=self.fusion(torch.cat([ab,ba,ab*ba],dim=-1))
        return self.drift_head(fused).squeeze(-1),self.cluster_head(fused),self.timing_head(fused).squeeze(-1),self.persist_head(fused).squeeze(-1)

ce_loss=nn.CrossEntropyLoss(); hub_loss=nn.HuberLoss(delta=0.1)
def bce_smooth(p,t,eps=0.05): return F.binary_cross_entropy(p,t*(1-eps)+eps/2)

def ensure_cols(df):
    df=df.copy()
    for c,src in [('ref_idx','ref_char'),('var_idx','var_char')]:
        if c not in df.columns and src in df.columns:
            df[c]=df[src].map(AA2IDX).fillna(0).astype(int)
    if 'pos_bin' not in df.columns and 'position' in df.columns:
        df['pos_bin']=(df['position'].clip(0,565)//28).clip(0,19).astype(int)
    if 'era_tok' not in df.columns:
        df['era_tok']=df.get('era',pd.Series(0,index=df.index)).fillna(0).astype(int)
    for c in ['freq_bin','charge_tok']:
        if c not in df.columns: df[c]=0
    if 'persist_norm' not in df.columns:
        df['persist_norm']=df.get('n_years_norm',pd.Series(0,index=df.index)).fillna(0)
    return df

def train_quick(train_df, seed=42, epochs=80, batch=32):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DualBranchMDA().to(DEVICE)
    train_df = ensure_cols(train_df)
    ds = MutDataset(train_df, augment=True)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(
        [{'params':[p for n,p in model.named_parameters() if 'task_weighter' not in n],'lr':3e-4},
         {'params':model.task_weighter.parameters(),'lr':1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=40,T_mult=2,eta_min=1e-5)
    step=0; opt.zero_grad()
    for epoch in range(1, epochs+1):
        model.train()
        for tok,cont,yd,yc,yt,yp in loader:
            d,cl,ti,pe = model(tok,cont)
            losses=[bce_smooth(d,yd),ce_loss(cl,yc),hub_loss(ti,yt),hub_loss(pe,yp)]
            loss=model.task_weighter(losses); (loss/2).backward(); step+=1
            if step%2==0:
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        sched.step()
    return model

def run_inference(model, df):
    df = ensure_cols(df)
    ds = MutDataset(df, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    drift_all, cluster_all, timing_all = [], [], []
    model.eval()
    with torch.no_grad():
        for tok,cont,*_ in loader:
            d,cl,ti,_ = model(tok,cont)
            drift_all.append(d.numpy())
            cluster_all.append(F.softmax(cl,dim=-1).numpy())
            timing_all.append(ti.numpy())
    drift   = np.concatenate(drift_all)
    cluster = np.concatenate(cluster_all, axis=0)
    timing  = np.clip(np.concatenate(timing_all),0,1)
    fusion_score = 0.50*drift + 0.35*cluster.max(axis=1) + 0.15*(1-timing)
    return {'drift_prob': drift, 'cluster_prob_max': cluster.max(axis=1),
            'timing_norm': timing, 'fusion_score': fusion_score}


if __name__ == '__main__':
    print('\n' + '='*62)
    print(' WHO H3N2 Back-Test v2 — Corrected Position Numbering')
    print('='*62)

    train_df = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
    val_df   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')
    test_df  = pd.read_csv(PHASE8 / 'phase8_test_data.csv')
    full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)
    year_col = 'first_year' if 'first_year' in full_df.columns else 'year'

    print(f'\n  Dataset: {len(full_df):,} mutations, subtypes: {dict(full_df["subtype"].value_counts()) if "subtype" in full_df.columns else "N/A"}')

    rows = []

    for year, strain_name in WHO_H3N2_STRAINS.items():
        print(f'\n  Year {year}: {strain_name}')

        who_muts = WHO_STRAIN_KEY_MUTATIONS.get(year)
        if who_muts is None:
            print(f'    INSUFFICIENT_SEQUENCES: dataset ends at 2020; year {year} not evaluable')
            rows.append({
                'year': year, 'who_strain': strain_name,
                'top5_predicted_mutations_ha1': 'N/A',
                'known_who_mutations': 'INSUFFICIENT_SEQUENCES',
                'precision_at_5': float('nan'), 'overlap_count': float('nan'),
                'note': f'Dataset ends 2020; prospective validation for {year} not possible'
            })
            continue

        yr   = full_df[year_col]
        tr   = full_df[yr < year].reset_index(drop=True)
        infer= full_df[yr == year].reset_index(drop=True)

        # Filter inference to H3N2 only (WHO mutations are H3N2-specific)
        if 'subtype' in infer.columns:
            infer_h3 = infer[infer['subtype']=='H3N2'].reset_index(drop=True)
        else:
            infer_h3 = infer

        print(f'    n_train={len(tr)}, n_infer_year={len(infer)}, n_infer_H3N2={len(infer_h3)}')

        if len(infer_h3) < 5:
            print(f'    SKIP: too few H3N2 mutations for year {year} (n={len(infer_h3)})')
            rows.append({
                'year': year, 'who_strain': strain_name,
                'top5_predicted_mutations_ha1': 'INSUFFICIENT_DATA',
                'known_who_mutations': '|'.join(sorted(who_muts)),
                'precision_at_5': float('nan'), 'overlap_count': float('nan'),
                'note': f'Only {len(infer_h3)} H3N2 mutations for year {year}'
            })
            continue

        # Balance train
        if len(tr) > 2000:
            pos_df = tr[tr['label_drift_prob']==1]
            neg_df = tr[tr['label_drift_prob']==0]
            n = min(len(pos_df), len(neg_df), 1000)
            tr = pd.concat([pos_df.sample(n, random_state=42),
                            neg_df.sample(n, random_state=42)]).reset_index(drop=True)

        t0 = time.perf_counter()
        model = train_quick(tr, seed=42, epochs=80)
        preds = run_inference(model, infer_h3)
        elapsed = time.perf_counter() - t0

        pred_df = infer_h3[['position','ref_char','var_char']].copy() if 'ref_char' in infer_h3.columns else infer_h3[['position']].copy()
        pred_df['fusion_score'] = preds['fusion_score']
        pred_df['drift_prob']   = preds['drift_prob']

        top5 = pred_df.nlargest(5, 'fusion_score').reset_index(drop=True)

        # Convert to HA1 numbering and build mutation strings
        top5_ha1 = []
        for _, r in top5.iterrows():
            align_pos = int(r['position'])
            ha1_pos   = alignment_to_h3_numbering(align_pos, subtype='H3N2')
            ref = r.get('ref_char', '?')
            mut = r.get('var_char', '?')
            top5_ha1.append(f'{ref}{ha1_pos}{mut}')

        print(f'    Top 5 predicted (HA1 numbering): {top5_ha1}  [{elapsed:.0f}s]')
        print(f'    WHO reference mutations:         {sorted(who_muts)}')

        # Compute precision@5: exact string match against WHO set
        matches = [s for s in top5_ha1 if s in who_muts]
        # Also check position+var_char only (tolerates ref annotation differences)
        partial_matches = []
        for s in top5_ha1:
            if len(s) >= 3:
                pos_part = s[1:-1]       # position digits
                mut_part = s[-1]         # variant residue
                for who_s in who_muts:
                    if len(who_s) >= 3 and who_s[1:-1] == pos_part and who_s[-1] == mut_part:
                        partial_matches.append(s)
                        break

        prec = len(set(matches + partial_matches)) / 5
        print(f'    Exact matches: {matches}  Position+var matches: {partial_matches}  precision@5 = {prec:.2f}')

        rows.append({
            'year': year, 'who_strain': strain_name,
            'top5_predicted_mutations_ha1': '|'.join(top5_ha1),
            'known_who_mutations': '|'.join(sorted(who_muts)),
            'precision_at_5': round(prec, 4),
            'overlap_count': len(set(matches + partial_matches)),
            'note': ''
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    results_df.to_csv(WHOOUT / 'who_backtest_results_v2.csv', index=False)
    print(f'\n  Saved: who_backtest_results_v2.csv')

    # ── Bar chart (regenerate) ─────────────────────────────────────────────────
    valid = results_df.dropna(subset=['precision_at_5'])
    fig, ax = plt.subplots(figsize=(10,6))
    random_baseline = 5/566
    if len(valid) > 0:
        mean_p = valid['precision_at_5'].mean()
        colors = ['#2471A3' if v > 0 else '#AAB7B8' for v in valid['precision_at_5']]
        bars = ax.bar(valid['year'], valid['precision_at_5'],
                      color=colors, alpha=0.85, edgecolor='white', lw=1.2)
        ax.axhline(mean_p, color='#C0392B', lw=2.5, ls='--',
                   label=f'Mean precision@5 = {mean_p:.3f}')
        for bar, val in zip(bars, valid['precision_at_5']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    ax.axhline(random_baseline, color='gray', lw=1.5, ls=':',
               label=f'Random baseline = {random_baseline:.4f}')
    ax.set_xlabel('Season Year'); ax.set_ylabel('Precision@5')
    ax.set_title('WHO H3N2 Back-Test v2 — Precision@5 (corrected HA1 numbering)',
                 fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(WHOOUT / 'who_backtest_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Saved: who_backtest_bar.png')

    # ── Summary ────────────────────────────────────────────────────────────────
    valid_rows = results_df[results_df['precision_at_5'].notna()]
    mean_prec  = valid_rows['precision_at_5'].mean() if len(valid_rows) > 0 else float('nan')

    summary = [
        'WHO H3N2 Back-Test v2 — Summary',
        '='*50,
        '',
        'POSITION NUMBERING FIX:',
        '  v1 error: added +16 offset (alignment→HA1), but should subtract.',
        '  v2 fix: alignment_to_h3_numbering(pos, H3N2) = pos - 16.',
        '  Only H3N2 mutations used for inference (WHO strains are H3N2).',
        '',
        'REFERENCE MUTATIONS UPDATED (Issue 3c):',
        '  Used WHO_STRAIN_KEY_MUTATIONS dict with published HA1 substitutions.',
        '  2021-2024: INSUFFICIENT_SEQUENCES (dataset ends at 2020).',
        '',
        f'Random baseline precision@5: {5/566:.4f} (5 out of ~566 positions)',
        f'Mean model precision@5 (evaluable years): {mean_prec:.4f}' if not np.isnan(mean_prec) else 'Mean: N/A',
        '',
        'Results by year:',
    ]
    for _, row in results_df.iterrows():
        p = row['precision_at_5']
        p_str = f'{p:.4f}' if not (isinstance(p, float) and np.isnan(p)) else 'N/A'
        summary.append(f'  {int(row["year"])}: precision@5 = {p_str}  note: {row["note"] or "ok"}')

    summary += [
        '',
        'Interpretation:',
        '  If precision@5 is still low after position correction, this reflects',
        '  that the model predicts mutations based on evolutionary frequency and',
        '  physicochemical properties, not direct antigenicity. Frame as a known',
        '  limitation: WHO vaccine strain selection integrates antigenic cartography',
        '  data not available in this training set.',
        '',
        '  Prospective validation for 2021-2024 seasons is not possible because the',
        '  dataset temporal coverage ends at 2020.',
    ]
    summary_txt = '\n'.join(summary)
    with open(WHOOUT / 'who_backtest_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_txt)
    print('  Saved: who_backtest_summary.txt')
    print()
    print(summary_txt)
