#!/usr/bin/env python3
"""
temporal_generalization.py — Temporal train/test splits + year-by-year AUC curve.

Scenarios:
  A: train < 2017, val 2017-2018, test 2019-2021
  B: train < 2019, val 2019-2020, test 2021+

Year-by-year: for each cutoff year Y from 2015 to 2022, train on first_year < Y,
              test on first_year == Y (skip if < 20 samples or 1 class).

Outputs:
  phase8_outputs/temporal/temporal_results.csv
  phase8_outputs/temporal/temporal_auc_curve.png
  phase8_outputs/temporal/temporal_summary.txt
"""

import sys, time, warnings
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
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error

ROOT   = Path(__file__).parent
OUT    = ROOT / 'outputs'
PHASE8 = ROOT / 'phase8_outputs'
TMPOUT = PHASE8 / 'temporal'
TMPOUT.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')

# ── Inline constants and classes ───────────────────────────────────────────────

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
N_CONT    = 16
N_CLUSTERS = 15


class MutDataset(Dataset):
    TOK_VOCAB = [20,20,20,3,2,2,5,2]

    def __init__(self, df, augment=False):
        self.augment = augment
        tok_cols = ['ref_idx','var_idx','pos_bin','era_tok',
                    'crit_flag','bind_flag','freq_bin','charge_tok']
        for c in tok_cols:
            if c not in df.columns:
                df = df.copy(); df[c] = 0
        self.tokens    = torch.LongTensor(df[tok_cols].fillna(0).astype(int).values)
        self.cont      = torch.FloatTensor(df[CONT_COLS].fillna(0).values.astype(float))
        self.y_drift   = torch.FloatTensor(df['label_drift_prob'].values)
        self.y_cluster = torch.LongTensor(df['label_cluster'].values)
        self.y_timing  = torch.FloatTensor(df['label_timing'].clip(0).values/(11*365+1))
        self.y_persist = torch.FloatTensor(df.get('persist_norm',pd.Series(0,index=df.index)).fillna(0).values)

    def __len__(self): return len(self.tokens)

    def __getitem__(self, i):
        cont = self.cont[i]
        if self.augment:
            cont = cont + torch.randn_like(cont)*0.03
        return self.tokens[i],cont,self.y_drift[i],self.y_cluster[i],self.y_timing[i],self.y_persist[i]


def _sinusoidal_pe(seq_len, d_model):
    pos=torch.arange(seq_len).unsqueeze(1).float()
    i=torch.arange(0,d_model,2).float()
    denom=10000**(i/d_model)
    pe=torch.zeros(seq_len,d_model)
    pe[:,0::2]=torch.sin(pos/denom); pe[:,1::2]=torch.cos(pos/denom)
    return pe


class DynamicTaskWeighter(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))
    def forward(self, losses):
        return sum(torch.exp(-self.log_var[i])*L+0.5*self.log_var[i]
                   for i,L in enumerate(losses))


class DualBranchMDA(nn.Module):
    TOK_VOCAB=[20,20,20,3,2,2,5,2]; SEQ_LEN=8
    def __init__(self, d_tok=96, d_cont=96, d_fused=192, nhead=8, n_layers=3, dropout=0.10):
        super().__init__()
        self.tok_embs=nn.ModuleList([nn.Embedding(v,d_tok) for v in self.TOK_VOCAB])
        self.register_buffer('pe',_sinusoidal_pe(self.SEQ_LEN,d_tok))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_tok,nhead=nhead,
            dim_feedforward=d_tok*4,dropout=dropout,batch_first=True,norm_first=True)
        self.tok_enc=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.feat_enc=nn.Sequential(nn.LayerNorm(N_CONT),nn.Linear(N_CONT,d_cont),
            nn.GELU(),nn.Dropout(dropout),nn.Linear(d_cont,d_cont),nn.GELU())
        self.xattn_ab=nn.MultiheadAttention(d_tok,nhead,dropout=dropout,batch_first=True)
        self.xattn_ba=nn.MultiheadAttention(d_cont,nhead,dropout=dropout,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(3*d_tok,d_fused),
            nn.LayerNorm(d_fused),nn.GELU(),nn.Dropout(dropout))
        def _head(out,act=None):
            m=[nn.Linear(d_fused,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,out)]
            if act: m.append(act)
            return nn.Sequential(*m)
        self.drift_head=_head(1,nn.Sigmoid()); self.cluster_head=_head(N_CLUSTERS)
        self.timing_head=_head(1,nn.Softplus()); self.persist_head=_head(1,nn.Sigmoid())
        self.task_weighter=DynamicTaskWeighter(4)

    def forward(self, tokens, cont):
        tok_h=torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)],dim=1)
        tok_h=tok_h+self.pe.unsqueeze(0)
        tok_out=self.tok_enc(tok_h).mean(dim=1,keepdim=True)
        feat_out=self.feat_enc(cont).unsqueeze(1)
        ab,_=self.xattn_ab(tok_out,feat_out,feat_out)
        ba,_=self.xattn_ba(feat_out,tok_out,tok_out)
        ab=ab.squeeze(1); ba=ba.squeeze(1)
        fused=self.fusion(torch.cat([ab,ba,ab*ba],dim=-1))
        return (self.drift_head(fused).squeeze(-1), self.cluster_head(fused),
                self.timing_head(fused).squeeze(-1), self.persist_head(fused).squeeze(-1))


ce_loss  = nn.CrossEntropyLoss()
hub_loss = nn.HuberLoss(delta=0.1)

def bce_smooth(pred, target, eps=0.05):
    return F.binary_cross_entropy(pred, target*(1-eps)+eps/2)


def ensure_token_cols(df):
    """Add missing token columns derived from available columns."""
    df = df.copy()
    if 'ref_idx' not in df.columns and 'ref_char' in df.columns:
        df['ref_idx'] = df['ref_char'].map(AA2IDX).fillna(0).astype(int)
    if 'var_idx' not in df.columns and 'var_char' in df.columns:
        df['var_idx'] = df['var_char'].map(AA2IDX).fillna(0).astype(int)
    if 'pos_bin' not in df.columns and 'position' in df.columns:
        df['pos_bin'] = (df['position'].clip(0,565)//28).clip(0,19).astype(int)
    if 'era_tok' not in df.columns:
        df['era_tok'] = df.get('era',pd.Series(0,index=df.index)).fillna(0).astype(int)
    if 'freq_bin' not in df.columns:
        df['freq_bin'] = 0
    if 'charge_tok' not in df.columns:
        df['charge_tok'] = df.get('charge_chg',pd.Series(0,index=df.index)).fillna(0).astype(int)
    if 'persist_norm' not in df.columns:
        df['persist_norm'] = df.get('n_years_norm',pd.Series(0,index=df.index)).fillna(0)
    return df


def train_temporal(train_df, val_df, test_df, seed=42, epochs=150, batch=32, accum=2):
    """Train from scratch and return (test_AUC, test_F1, timing_MAE, n_train, n_test)."""
    if len(test_df) < 5:
        return float('nan'), float('nan'), float('nan'), len(train_df), len(test_df)
    if len(train_df) < 10 or len(np.unique(train_df['label_drift_prob'])) < 2:
        return float('nan'), float('nan'), float('nan'), len(train_df), len(test_df)

    torch.manual_seed(seed); np.random.seed(seed)
    model = DualBranchMDA().to(DEVICE)
    train_ds = MutDataset(train_df, augment=True)
    test_ds  = MutDataset(test_df,  augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)
    opt   = torch.optim.AdamW(
        [{'params':[p for n,p in model.named_parameters() if 'task_weighter' not in n],'lr':3e-4},
         {'params':model.task_weighter.parameters(),'lr':1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=40,T_mult=2,eta_min=1e-5)

    step=0; opt.zero_grad()
    for epoch in range(1, epochs+1):
        model.train()
        for tok,cont,yd,yc,yt,yp in train_loader:
            d,cl,ti,pe_out = model(tok,cont)
            losses=[bce_smooth(d,yd),ce_loss(cl,yc),hub_loss(ti,yt),hub_loss(pe_out,yp)]
            loss=model.task_weighter(losses)
            (loss/accum).backward(); step+=1
            if step%accum==0:
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); opt.zero_grad()
        if step%accum!=0:
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); opt.zero_grad()
        sched.step()

    model.eval()
    dp_all,dy_all,ti_all,ty_all=[],[],[],[]
    with torch.no_grad():
        for tok,cont,yd,yc,yt,yp in test_loader:
            d,cl,ti,_ = model(tok,cont)
            dp_all.append(d.numpy()); dy_all.append(yd.numpy())
            ti_all.append(ti.numpy()); ty_all.append(yt.numpy())
    dp=np.concatenate(dp_all); dy=np.concatenate(dy_all).astype(int)
    ti_p=np.clip(np.concatenate(ti_all),0,None)*(11*365)
    ti_t=np.concatenate(ty_all)*(11*365+1)
    auc = roc_auc_score(dy, dp) if len(np.unique(dy))>1 else float('nan')
    f1  = f1_score(dy, (dp>=0.5).astype(int), zero_division=0)
    mae = mean_absolute_error(ti_t, ti_p)
    return float(auc), float(f1), float(mae), len(train_df), len(test_df)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*62)
    print(' Temporal Generalization Test')
    print('='*62)

    # Load the FULL aggregated mutation dataset from phase3 outputs for temporal analysis.
    # The pre-split 2000-sample balanced set has too few per-year samples.
    print('\nLoading full aggregated mutation data ...')

    full_df = None
    year_col = 'first_year'

    # Try loading the all-predictions CSV which has first_year and engineered features
    all_preds_path = PHASE8 / 'phase8_mda_all_predictions.csv'
    var_path       = OUT / 'phase3_variations_annotated.csv'

    if all_preds_path.exists():
        preds_df = pd.read_csv(all_preds_path)
        # Merge with the pre-split CSVs to get label and feature columns
        train_df_orig = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
        val_df_orig   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')
        test_df_orig  = pd.read_csv(PHASE8 / 'phase8_test_data.csv')
        # The all-predictions CSV has accession + year (renamed from first_year)
        # The pre-split CSVs have the full feature set
        # Use the pre-split CSVs as our full_df but merge year from preds
        base = pd.concat([train_df_orig, val_df_orig, test_df_orig], ignore_index=True)
        # Map first_year from all predictions via accession
        if 'accession' in preds_df.columns and 'accession' in base.columns:
            year_map = preds_df.set_index('accession')['year'].to_dict()
            if 'first_year' not in base.columns:
                base['first_year'] = base['accession'].map(year_map)
        full_df  = base
        year_col = 'first_year' if 'first_year' in full_df.columns else 'year'
        full_df  = ensure_token_cols(full_df)
        print(f'  Loaded {len(full_df):,} mutations from pre-split CSVs')
        yr_range = f'{int(full_df[year_col].dropna().min())}–{int(full_df[year_col].dropna().max())}' if full_df[year_col].notna().any() else 'unknown'
        print(f'  Year range: {yr_range}')
    else:
        train_df_orig = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
        val_df_orig   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')
        test_df_orig  = pd.read_csv(PHASE8 / 'phase8_test_data.csv')
        full_df  = pd.concat([train_df_orig, val_df_orig, test_df_orig], ignore_index=True)
        full_df  = ensure_token_cols(full_df)
        year_col = 'first_year' if 'first_year' in full_df.columns else 'year'
        print(f'  Loaded {len(full_df):,} mutations (fallback: pre-split CSVs)')

    print(f'  Full dataset: {len(full_df):,} mutations  Year range: '
          f'{int(full_df[year_col].dropna().min())}–{int(full_df[year_col].dropna().max())}')

    rows = []

    # ── Scenarios A & B ────────────────────────────────────────────────────────
    scenarios = [
        ('A', 'Conservative', lambda y: y < 2017,
               lambda y: (y >= 2017) & (y <= 2018),
               lambda y: (y >= 2019) & (y <= 2021)),
        ('B', 'Standard',    lambda y: y < 2019,
               lambda y: (y >= 2019) & (y <= 2020),
               lambda y: y >= 2021),
    ]

    for scen_id, scen_name, tr_mask, va_mask, te_mask in scenarios:
        print(f'\n  Scenario {scen_id} ({scen_name}):')
        yr = full_df[year_col]
        tr = full_df[tr_mask(yr)].reset_index(drop=True)
        va = full_df[va_mask(yr)].reset_index(drop=True)
        te = full_df[te_mask(yr)].reset_index(drop=True)
        print(f'    n_train={len(tr)}  n_val={len(va)}  n_test={len(te)}')

        # Balance train set
        if len(tr) > 2000:
            pos = tr[tr['label_drift_prob']==1]
            neg = tr[tr['label_drift_prob']==0]
            n   = min(len(pos), len(neg), 1000)
            tr  = pd.concat([pos.sample(n, random_state=42),
                             neg.sample(n, random_state=42)]).reset_index(drop=True)
            print(f'    Balanced train: {len(tr)}')

        if len(te) < 10:
            print(f'    SKIP: test set too small ({len(te)} samples)')
            continue

        auc, f1, mae, n_tr, n_te = train_temporal(tr, va, te, seed=42)
        print(f'    AUC={auc:.4f}  F1={f1:.4f}  timing_MAE={mae:.1f}d')

        test_period = '2019-2021' if scen_id == 'A' else '2021+'
        rows.append({
            'scenario': scen_id, 'scenario_name': scen_name,
            'train_cutoff': 2017 if scen_id=='A' else 2019,
            'test_period': test_period,
            'n_train': n_tr, 'n_test': n_te,
            'AUC': round(auc, 4), 'F1': round(f1, 4),
            'timing_MAE': round(mae, 2)
        })

    # ── Year-by-year AUC curve ─────────────────────────────────────────────────
    print('\n  Year-by-year AUC curve (3 seeds per year for CI) ...')
    year_aucs  = []
    test_years = list(range(2015, 2023))

    for test_year in test_years:
        yr        = full_df[year_col]
        tr_year   = full_df[yr < test_year].reset_index(drop=True)
        te_year   = full_df[yr == test_year].reset_index(drop=True)

        if len(te_year) < 5:
            print(f'  Year {test_year}: SKIP — too few test samples (n_test={len(te_year)})')
            year_aucs.append({'year': test_year, 'AUC': float('nan'), 'n_test': len(te_year)})
            continue

        # Balance train
        if len(tr_year) > 2000:
            pos = tr_year[tr_year['label_drift_prob']==1]
            neg = tr_year[tr_year['label_drift_prob']==0]
            n   = min(len(pos), len(neg), 1000)
            tr_year = pd.concat([pos.sample(n, random_state=42),
                                  neg.sample(n, random_state=42)]).reset_index(drop=True)

        seed_aucs = []
        for seed in [42, 43, 44]:
            auc, _, _, _, _ = train_temporal(tr_year, te_year, te_year, seed=seed, epochs=100)
            if not np.isnan(auc):
                seed_aucs.append(auc)

        mean_auc = float(np.mean(seed_aucs)) if seed_aucs else float('nan')
        std_auc  = float(np.std(seed_aucs))  if seed_aucs else float('nan')
        print(f'  Year {test_year}: n_train={len(tr_year)}  n_test={len(te_year)}  '
              f'AUC={mean_auc:.4f}±{std_auc:.4f}')
        year_aucs.append({'year': test_year, 'AUC': mean_auc, 'std': std_auc,
                          'n_train': len(tr_year), 'n_test': len(te_year)})

    # Save results
    results_df = pd.DataFrame(rows)
    results_df.to_csv(TMPOUT / 'temporal_results.csv', index=False)
    pd.DataFrame(year_aucs).to_csv(TMPOUT / 'year_auc_curve.csv', index=False)

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--',
                         'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white'})

    yc_df   = pd.DataFrame(year_aucs).dropna(subset=['AUC'])
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(yc_df) > 0:
        std_vals = yc_df['std'].fillna(0)
        ax.plot(yc_df['year'], yc_df['AUC'], 'o-', color='#2471A3', lw=2.5, ms=8,
                label='Mean AUC (3 seeds)')
        ax.fill_between(yc_df['year'],
                        yc_df['AUC'] - 1.96*std_vals,
                        yc_df['AUC'] + 1.96*std_vals,
                        alpha=0.2, color='#2471A3', label='95% CI')
    ax.axhline(0.80, color='#C0392B', lw=1.5, ls='--', alpha=0.7, label='AUC=0.80 threshold')
    ax.set_xlabel('Test Year (train on all data before this year)')
    ax.set_ylabel('AUC-ROC (drift probability head)')
    ax.set_title('Temporal Generalization — AUC vs Test Year\n'
                 '(train cutoff = year − 1, test = single year)', fontweight='bold')
    ax.legend(); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(TMPOUT / 'temporal_auc_curve.png')
    plt.close(fig)

    # ── Summary text ──────────────────────────────────────────────────────────
    valid_aucs = [r['AUC'] for r in year_aucs if not np.isnan(r['AUC'])]
    if valid_aucs:
        auc_trend = 'declining' if valid_aucs[-1] < valid_aucs[0] else 'stable'
    else:
        auc_trend = 'insufficient data'

    summary_lines = [
        'Temporal Generalization Test — Summary',
        '=' * 50,
        '',
        'Scenario Results:',
    ]
    for _, r in results_df.iterrows():
        summary_lines.append(
            f"  Scenario {r['scenario']} ({r['scenario_name']}): "
            f"train_cutoff={r['train_cutoff']}, test={r['test_period']}, "
            f"n_train={r['n_train']}, n_test={r['n_test']}, "
            f"AUC={r['AUC']:.4f}, F1={r['F1']:.4f}, timing_MAE={r['timing_MAE']:.1f}d")

    summary_lines += [
        '',
        f'Year-by-year AUC trend: {auc_trend}',
        f'Years with sufficient test data: {len(valid_aucs)}',
    ]
    if valid_aucs:
        summary_lines.append(f'AUC range: [{min(valid_aucs):.4f}, {max(valid_aucs):.4f}]')

    summary_lines += [
        '',
        'Interpretation:',
        f'  Model {"degrades" if auc_trend=="declining" else "maintains performance"} '
        f'over longer forecast horizons.',
        '  Scenario B (standard split) uses more training data and may be more reliable.',
    ]
    (TMPOUT / 'temporal_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
    print(f'\n  Saved outputs to {TMPOUT}')
    print('\n'.join(summary_lines))
