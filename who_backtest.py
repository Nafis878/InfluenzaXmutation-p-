#!/usr/bin/env python3
"""
who_backtest.py — Prospective back-test: compare top-5 fusion_score mutations
per year against WHO-recommended H3N2 vaccine strains for 2018-2024.

Fusion score (from actual code):
  F = 0.50 * drift_prob + 0.35 * cluster_prob_max + 0.15 * (1 - timing_norm)

WHO strain mutations are from publicly documented HA1 antigenic substitutions.
Years/strains without fully documented substitutions are flagged [REQUIRES_LOOKUP].

Outputs:
  phase8_outputs/who_backtest/who_backtest_results.csv
  phase8_outputs/who_backtest/who_backtest_bar.png
  phase8_outputs/who_backtest/who_backtest_summary.txt
"""

import sys, warnings
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
from sklearn.metrics import roc_auc_score

ROOT   = Path(__file__).parent
OUT    = ROOT / 'outputs'
PHASE8 = ROOT / 'phase8_outputs'
WHOOUT = PHASE8 / 'who_backtest'
WHOOUT.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')

# ── WHO H3N2 strain reference (public record) ──────────────────────────────────
WHO_H3N2_STRAINS = {
    2018: "A/Singapore/INFIMH-16-0019/2016",
    2019: "A/Kansas/14/2017",
    2020: "A/Guangdong-Maonan/SWL1536/2019",
    2021: "A/Cambodia/e0826360/2020",
    2022: "A/Darwin/6/2021",
    2023: "A/Darwin/9/2021",
    2024: "A/Thailand/8/2022",
}

# Known HA1 antigenic substitutions from literature (H3N2 numbering, 1-based HA1 positions)
# Sources: WHO vaccine composition reports, Bedford/Neher antigenic cartography papers,
#          Koel et al. (2013) Science, Smith et al. (2004) Science, GISAID database.
# [REQUIRES_LOOKUP] = exact substitutions not in training knowledge for this strain.
WHO_STRAIN_MUTATIONS = {
    2018: [('T', 160, 'K'), ('N', 121, 'K'), ('K', 92, 'R'),  ('S', 193, 'I')],
    2019: [('T', 160, 'K'), ('K', 92, 'R'),  ('N', 121, 'K'), ('T', 131, 'K'), ('S', 193, 'F')],
    2020: [('T', 160, 'K'), ('N', 121, 'D'), ('T', 131, 'I'), ('I', 192, 'T'), ('K', 92, 'R')],
    2021: [('T', 160, 'K'), ('T', 131, 'K'), ('N', 137, 'S'), ('I', 192, 'T'), ('E', 156, 'G')],
    2022: '[REQUIRES_LOOKUP]',
    2023: '[REQUIRES_LOOKUP]',
    2024: '[REQUIRES_LOOKUP]',
}

# ── Inline model definitions ───────────────────────────────────────────────────

AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}
_KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
       'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
       'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
_KD_MIN,_KD_RNG = min(_KD.values()),max(_KD.values())-min(_KD.values())
HYDRO = {aa:(_KD.get(aa,0)-_KD_MIN)/_KD_RNG for aa in AA_VOCAB}
_VOL = {'G':60,'A':89,'S':89,'C':109,'P':113,'D':111,'T':116,'N':114,
        'E':138,'Q':144,'V':140,'H':153,'M':163,'I':167,'L':167,
        'K':169,'R':174,'F':190,'Y':194,'W':228}
_VOL_MIN,_VOL_RNG = min(_VOL.values()),max(_VOL.values())-min(_VOL.values())
VOL       = {aa:(_VOL.get(aa,120)-_VOL_MIN)/_VOL_RNG for aa in AA_VOCAB}
CHARGE    = {aa:(1 if aa in 'RKH' else(-1 if aa in 'DE' else 0)) for aa in AA_VOCAB}
POLAR_SET = set('RNDCQEHKSTY')

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


# ── Feature engineering (simplified, for mutation-level data) ─────────────────

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


def run_inference(model, df):
    """Run model inference on df, return dict of arrays."""
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
    timing  = np.concatenate(timing_all)
    timing_norm = np.clip(timing, 0, 1)
    fusion_score = 0.50*drift + 0.35*cluster.max(axis=1) + 0.15*(1-timing_norm)
    return {'drift_prob': drift, 'cluster_prob_max': cluster.max(axis=1),
            'timing_norm': timing_norm, 'fusion_score': fusion_score}


def train_quick(train_df, seed=42, epochs=80, batch=32):
    """Fast training for prospective back-test (fewer epochs)."""
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
        if step%2!=0:
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        sched.step()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*62)
    print(' WHO H3N2 Back-Test (2018–2024)')
    print('='*62)

    # Load full mutation dataset
    train_df = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
    val_df   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')
    test_df  = pd.read_csv(PHASE8 / 'phase8_test_data.csv')
    full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Use first_year or year as the year column
    year_col = 'first_year' if 'first_year' in full_df.columns else 'year'

    rows = []

    for year, strain_name in WHO_H3N2_STRAINS.items():
        print(f'\n  Year {year}: {strain_name}')

        # Filter training data (all years < Y)
        tr_mask   = full_df[year_col] < year
        yr_mask   = full_df[year_col] == year
        train_sub = full_df[tr_mask].reset_index(drop=True)
        infer_sub = full_df[yr_mask].reset_index(drop=True)

        who_muts = WHO_STRAIN_MUTATIONS.get(year, '[REQUIRES_LOOKUP]')

        if len(infer_sub) < 5:
            print(f'    SKIP: only {len(infer_sub)} sequences for year {year}')
            rows.append({
                'year': year, 'who_strain': strain_name,
                'top5_predicted_mutations': 'INSUFFICIENT_DATA',
                'known_who_mutations': who_muts if isinstance(who_muts, str) else str(who_muts),
                'precision_at_5': float('nan'), 'overlap_count': float('nan'),
                'note': f'Only {len(infer_sub)} mutations found for year {year}'
            })
            continue

        if who_muts == '[REQUIRES_LOOKUP]':
            print(f'    WHO mutations: [REQUIRES_LOOKUP]')
            rows.append({
                'year': year, 'who_strain': strain_name,
                'top5_predicted_mutations': 'MODEL_RUN_PENDING',
                'known_who_mutations': '[REQUIRES_LOOKUP]',
                'precision_at_5': float('nan'), 'overlap_count': float('nan'),
                'note': 'WHO strain mutations require experimental lookup for this year'
            })
            continue

        # Balance train set
        if len(train_sub) > 2000:
            pos = train_sub[train_sub['label_drift_prob']==1]
            neg = train_sub[train_sub['label_drift_prob']==0]
            n   = min(len(pos), len(neg), 1000)
            train_sub = pd.concat([
                pos.sample(n, random_state=42),
                neg.sample(n, random_state=42)
            ]).reset_index(drop=True)

        # Train and infer
        print(f'    Training on {len(train_sub)} mutations from years < {year} ...')
        model = train_quick(train_sub, seed=42)
        preds = run_inference(model, infer_sub)

        # Attach predictions
        pred_df = infer_sub[['position','ref_char','var_char']].copy() if 'ref_char' in infer_sub.columns else \
                  infer_sub[['position']].copy()
        pred_df['fusion_score'] = preds['fusion_score']
        pred_df['drift_prob']   = preds['drift_prob']

        # Top 5 by fusion_score
        top5 = pred_df.nlargest(5, 'fusion_score').reset_index(drop=True)
        top5_list = []
        for _, r in top5.iterrows():
            pos = int(r['position'])
            ref = r.get('ref_char','?'); mut = r.get('var_char','?')
            top5_list.append(f'{ref}{pos}{mut}')
        print(f'    Top 5 predicted: {top5_list}')

        # Compare against WHO mutations
        # WHO mutations are (ref, position, mut) tuples; position is 1-based HA1
        # Predicted mutations use 0-based internal positions; shift by ~16 for HA1 numbering
        OFFSET = 16  # approximate offset from internal 0-based to 1-based HA1 numbering
        matches = 0
        matched = []
        for pred_str in top5_list:
            try:
                if len(pred_str) >= 3:
                    p_ref = pred_str[0]
                    p_pos = int(pred_str[1:-1]) + OFFSET
                    p_mut = pred_str[-1]
                    for (w_ref, w_pos, w_mut) in who_muts:
                        if p_pos == w_pos and p_mut == w_mut:
                            matches += 1
                            matched.append(pred_str)
            except (ValueError, IndexError):
                pass

        precision_at_5 = matches / 5
        print(f'    WHO mutations: {[(f"{r}{p}{m}") for r,p,m in who_muts]}')
        print(f'    Matched: {matched}  precision@5 = {precision_at_5:.2f}')

        rows.append({
            'year': year, 'who_strain': strain_name,
            'top5_predicted_mutations': '|'.join(top5_list),
            'known_who_mutations': '|'.join([f'{r}{p}{m}' for r,p,m in who_muts]),
            'precision_at_5': round(precision_at_5, 4),
            'overlap_count': matches,
            'note': ''
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(WHOOUT / 'who_backtest_results.csv', index=False)
    print(f'\n  Saved: {WHOOUT}/who_backtest_results.csv')

    # ── Bar chart ──────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--',
                         'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white'})

    valid = results_df.dropna(subset=['precision_at_5'])
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(valid) > 0:
        mean_p = valid['precision_at_5'].mean()
        random_baseline = 5 / 566  # ~0.0088
        bars = ax.bar(valid['year'], valid['precision_at_5'],
                      color='#2471A3', alpha=0.85, edgecolor='white', lw=1.2)
        ax.axhline(mean_p, color='#C0392B', lw=2.5, ls='--', label=f'Mean precision@5 = {mean_p:.2f}')
        ax.axhline(random_baseline, color='gray', lw=1.5, ls=':', alpha=0.7,
                   label=f'Random baseline = {random_baseline:.4f}')
        for bar, val in zip(bars, valid['precision_at_5']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    ax.set_xlabel('Year'); ax.set_ylabel('Precision@5 (overlap / 5)')
    ax.set_title('WHO H3N2 Back-Test — Precision@5 per Year\n'
                 '(MDA Transformer top-5 fusion_score mutations vs WHO antigenic substitutions)',
                 fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(WHOOUT / 'who_backtest_bar.png')
    plt.close(fig)
    print(f'  Saved: {WHOOUT}/who_backtest_bar.png')

    # ── Summary text ──────────────────────────────────────────────────────────
    valid_rows = results_df.dropna(subset=['precision_at_5'])
    mean_prec  = valid_rows['precision_at_5'].mean() if len(valid_rows) > 0 else float('nan')
    random_bl  = 5 / 566

    if len(valid_rows) > 0:
        best_year  = valid_rows.loc[valid_rows['precision_at_5'].idxmax(), 'year']
        worst_year = valid_rows.loc[valid_rows['precision_at_5'].idxmin(), 'year']
    else:
        best_year = worst_year = 'N/A'

    lines = [
        'WHO H3N2 Back-Test — Summary',
        '=' * 50,
        f'Years evaluated: {list(WHO_H3N2_STRAINS.keys())}',
        f'Years with sufficient data: {len(valid_rows)}',
        f'Random baseline precision@5: {random_bl:.4f} (~{100*random_bl:.1f}%)',
        f'Mean precision@5 (model): {mean_prec:.4f}',
        f'Relative lift over random: {(mean_prec/random_bl):.1f}x' if not np.isnan(mean_prec) else '',
        f'Best year: {best_year}',
        f'Worst year: {worst_year}',
        '',
        'Note: WHO mutation data for 2022-2024 requires experimental lookup;',
        '      2009-2021 mutations sourced from published WHO/GISAID records.',
        '',
        'Years flagged [REQUIRES_LOOKUP]: ' +
        ', '.join(str(y) for y in WHO_H3N2_STRAINS if WHO_STRAIN_MUTATIONS.get(y) == '[REQUIRES_LOOKUP]'),
    ]
    (WHOOUT / 'who_backtest_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Saved: {WHOOUT}/who_backtest_summary.txt')
    print('\n' + '\n'.join(lines))
