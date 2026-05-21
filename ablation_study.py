#!/usr/bin/env python3
"""
ablation_study.py — 5 ablated variants of DualBranchMDA, 3 seeds each.

Ablations:
  A1: Remove bidirectional cross-attention (identity fusion)
  A2: Single-task — drift BCE only (remove all other heads)
  A3: Remove continuous feature projection (zero Branch B)
  A4: Remove LayerNorm in feat_enc (Branch B)
  A5: Remove TransformerEncoder layers (identity — embeddings straight to pool)

Outputs:
  phase8_outputs/ablation/ablation_results.csv
  phase8_outputs/ablation/ablation_bar_chart.png
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
from sklearn.metrics import roc_auc_score, f1_score

ROOT   = Path(__file__).parent
PHASE8 = ROOT / 'phase8_outputs'
ABLOUT = PHASE8 / 'ablation'
ABLOUT.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')

# Baseline AUC from confirmed run
BASELINE_AUC = 0.9224

# ── Shared constants ───────────────────────────────────────────────────────────

AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}
_KD = {'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
       'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
       'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
_KD_MIN, _KD_RNG = min(_KD.values()), max(_KD.values()) - min(_KD.values())
HYDRO = {aa: (_KD.get(aa,0)-_KD_MIN)/_KD_RNG for aa in AA_VOCAB}
_VOL = {'G':60,'A':89,'S':89,'C':109,'P':113,'D':111,'T':116,'N':114,
        'E':138,'Q':144,'V':140,'H':153,'M':163,'I':167,'L':167,
        'K':169,'R':174,'F':190,'Y':194,'W':228}
_VOL_MIN,_VOL_RNG = min(_VOL.values()), max(_VOL.values())-min(_VOL.values())
VOL       = {aa:(_VOL.get(aa,120)-_VOL_MIN)/_VOL_RNG for aa in AA_VOCAB}
CHARGE    = {aa:(1 if aa in 'RKH' else(-1 if aa in 'DE' else 0)) for aa in AA_VOCAB}
POLAR_SET = set('RNDCQEHKSTY')

CONT_COLS = ['position_norm','ref_hydro','var_hydro','hydro_delta',
             'ref_vol','var_vol','vol_delta','charge_chg','polar_chg',
             'crit_flag','bind_flag','year_norm','freq_norm',
             'n_years_norm','drift_inten','days_norm']
N_CONT = 16
N_CLUSTERS = 15


# ── Dataset ────────────────────────────────────────────────────────────────────

class MutDataset(Dataset):
    TOK_VOCAB = [20, 20, 20, 3, 2, 2, 5, 2]

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
        self.y_persist = torch.FloatTensor(df.get('persist_norm', pd.Series(0,index=df.index)).fillna(0).values)

    def __len__(self): return len(self.tokens)

    def __getitem__(self, i):
        cont = self.cont[i]
        if self.augment:
            cont = cont + torch.randn_like(cont)*0.03
        return self.tokens[i], cont, self.y_drift[i], self.y_cluster[i], self.y_timing[i], self.y_persist[i]


# ── Shared model building blocks ───────────────────────────────────────────────

def _sinusoidal_pe(seq_len, d_model):
    pos   = torch.arange(seq_len).unsqueeze(1).float()
    i     = torch.arange(0, d_model, 2).float()
    denom = 10000 ** (i / d_model)
    pe    = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / denom)
    pe[:, 1::2] = torch.cos(pos / denom)
    return pe


class DynamicTaskWeighter(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))
    def forward(self, losses):
        return sum(torch.exp(-self.log_var[i])*L+0.5*self.log_var[i]
                   for i,L in enumerate(losses))


# ── Full model (baseline) ──────────────────────────────────────────────────────

class DualBranchMDA(nn.Module):
    TOK_VOCAB = [20,20,20,3,2,2,5,2]; SEQ_LEN=8

    def __init__(self, d_tok=96, d_cont=96, d_fused=192, nhead=8, n_layers=3, dropout=0.10):
        super().__init__()
        self.tok_embs = nn.ModuleList([nn.Embedding(v,d_tok) for v in self.TOK_VOCAB])
        self.register_buffer('pe', _sinusoidal_pe(self.SEQ_LEN, d_tok))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_tok, nhead=nhead,
            dim_feedforward=d_tok*4, dropout=dropout, batch_first=True, norm_first=True)
        self.tok_enc  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.feat_enc = nn.Sequential(
            nn.LayerNorm(N_CONT), nn.Linear(N_CONT,d_cont), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_cont,d_cont), nn.GELU())
        self.xattn_ab = nn.MultiheadAttention(d_tok, nhead, dropout=dropout, batch_first=True)
        self.xattn_ba = nn.MultiheadAttention(d_cont,nhead, dropout=dropout, batch_first=True)
        self.fusion   = nn.Sequential(nn.Linear(3*d_tok,d_fused),
            nn.LayerNorm(d_fused), nn.GELU(), nn.Dropout(dropout))
        def _head(out, act=None):
            m=[nn.Linear(d_fused,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,out)]
            if act: m.append(act)
            return nn.Sequential(*m)
        self.drift_head=_head(1,nn.Sigmoid()); self.cluster_head=_head(N_CLUSTERS)
        self.timing_head=_head(1,nn.Softplus()); self.persist_head=_head(1,nn.Sigmoid())
        self.task_weighter = DynamicTaskWeighter(4)

    def _branches(self, tokens, cont):
        tok_h   = torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)], dim=1)
        tok_h   = tok_h + self.pe.unsqueeze(0)
        tok_out = self.tok_enc(tok_h).mean(dim=1, keepdim=True)
        feat_out= self.feat_enc(cont).unsqueeze(1)
        return tok_out, feat_out

    def forward(self, tokens, cont):
        tok_out, feat_out = self._branches(tokens, cont)
        ab,_ = self.xattn_ab(tok_out, feat_out, feat_out)
        ba,_ = self.xattn_ba(feat_out, tok_out, tok_out)
        ab=ab.squeeze(1); ba=ba.squeeze(1)
        fused = self.fusion(torch.cat([ab, ba, ab*ba], dim=-1))
        return (self.drift_head(fused).squeeze(-1),
                self.cluster_head(fused),
                self.timing_head(fused).squeeze(-1),
                self.persist_head(fused).squeeze(-1))


# ── A1: No cross-attention (identity fusion) ───────────────────────────────────

class DualBranchMDA_NoXAttn(DualBranchMDA):
    """Replace bidirectional cross-attention with identity mean-pool."""
    def forward(self, tokens, cont):
        tok_out, feat_out = self._branches(tokens, cont)
        ab = tok_out.squeeze(1)
        ba = feat_out.squeeze(1)
        fused = self.fusion(torch.cat([ab, ba, ab*ba], dim=-1))
        return (self.drift_head(fused).squeeze(-1),
                self.cluster_head(fused),
                self.timing_head(fused).squeeze(-1),
                self.persist_head(fused).squeeze(-1))


# ── A2: Single-task (drift BCE only) ──────────────────────────────────────────

class DualBranchMDA_SingleTask(DualBranchMDA):
    """Forward is identical; training loop only uses drift loss."""
    pass  # loss zeroing handled in train loop


# ── A3: No continuous feature projection (zero Branch B) ──────────────────────

class DualBranchMDA_NoContinuousProj(DualBranchMDA):
    def forward(self, tokens, cont):
        tok_out, _ = self._branches(tokens, cont)
        ab = tok_out.squeeze(1)
        ba = torch.zeros_like(ab)  # zero-out continuous branch
        fused = self.fusion(torch.cat([ab, ba, ab*ba], dim=-1))
        return (self.drift_head(fused).squeeze(-1),
                self.cluster_head(fused),
                self.timing_head(fused).squeeze(-1),
                self.persist_head(fused).squeeze(-1))


# ── A4: No LayerNorm in feat_enc ──────────────────────────────────────────────

class DualBranchMDA_NoLayerNorm(DualBranchMDA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace LayerNorm with Identity
        d_cont  = kwargs.get('d_cont', 96)
        dropout = kwargs.get('dropout', 0.10)
        self.feat_enc = nn.Sequential(
            nn.Identity(),
            nn.Linear(N_CONT, d_cont), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_cont, d_cont), nn.GELU())


# ── A5: No Transformer Encoder (identity) ─────────────────────────────────────

class DualBranchMDA_NoEncoder(DualBranchMDA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tok_enc = nn.Identity()  # replace encoder with pass-through

    def _branches(self, tokens, cont):
        tok_h   = torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)], dim=1)
        tok_h   = tok_h + self.pe.unsqueeze(0)
        tok_out = tok_h.mean(dim=1, keepdim=True)  # skip encoder
        feat_out= self.feat_enc(cont).unsqueeze(1)
        return tok_out, feat_out


# ── Training / evaluation functions ───────────────────────────────────────────

ce_loss  = nn.CrossEntropyLoss()
hub_loss = nn.HuberLoss(delta=0.1)

def bce_smooth(pred, target, eps=0.05):
    return F.binary_cross_entropy(pred, target*(1-eps)+eps/2)

def train_and_eval(model_cls, train_df, val_df, test_df, seed, is_single_task=False,
                   epochs=150, batch_size=32, accum_steps=2):
    torch.manual_seed(seed); np.random.seed(seed)

    model = model_cls().to(DEVICE)
    train_ds = MutDataset(train_df, augment=True)
    test_ds  = MutDataset(test_df,  augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    optimiser = torch.optim.AdamW(
        [{'params': [p for n,p in model.named_parameters() if 'task_weighter' not in n], 'lr': 3e-4},
         {'params': model.task_weighter.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=40, T_mult=2, eta_min=1e-5)

    t0 = time.perf_counter()
    step_count = 0
    optimiser.zero_grad()

    for epoch in range(1, epochs+1):
        model.train()
        for tok, cont, yd, yc, yt, yp in train_loader:
            d, cl, ti, pe_out = model(tok, cont)
            l_drift = bce_smooth(d, yd)
            if is_single_task:
                loss = l_drift
            else:
                losses = [l_drift, ce_loss(cl, yc), hub_loss(ti, yt), hub_loss(pe_out, yp)]
                loss   = model.task_weighter(losses)
            (loss / accum_steps).backward()
            step_count += 1
            if step_count % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step(); optimiser.zero_grad()
        if step_count % accum_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step(); optimiser.zero_grad()
        scheduler.step()

    # Evaluate on test set
    model.eval()
    dp_all, dy_all = [], []
    with torch.no_grad():
        for tok, cont, yd, *_ in test_loader:
            d, *_ = model(tok, cont)
            dp_all.append(d.numpy()); dy_all.append(yd.numpy())
    dp = np.concatenate(dp_all); dy = np.concatenate(dy_all).astype(int)
    auc = roc_auc_score(dy, dp) if len(np.unique(dy)) > 1 else 0.5
    train_time = time.perf_counter() - t0
    return float(auc), float(train_time)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*62)
    print(' Ablation Study — DualBranchMDA v2')
    print(f' Baseline AUC (full model, seed=42): {BASELINE_AUC:.4f}')
    print('='*62)

    train_df = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
    val_df   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')
    test_df  = pd.read_csv(PHASE8 / 'phase8_test_data.csv')

    # Ensure token cols exist
    for df in [train_df, val_df, test_df]:
        if 'persist_norm' not in df.columns:
            df['persist_norm'] = df.get('n_years_norm', pd.Series(0.0, index=df.index)).fillna(0)

    ABLATIONS = [
        ('A1', 'No cross-attention (identity fusion)', DualBranchMDA_NoXAttn, False),
        ('A2', 'Single-task (drift BCE only)',          DualBranchMDA_SingleTask, True),
        ('A3', 'No continuous feature projection',     DualBranchMDA_NoContinuousProj, False),
        ('A4', 'No LayerNorm in embedding module',     DualBranchMDA_NoLayerNorm, False),
        ('A5', 'No Transformer Encoder (identity)',    DualBranchMDA_NoEncoder, False),
    ]
    SEEDS = [42, 43, 44]

    rows = []
    for abl_id, desc, ModelCls, is_single in ABLATIONS:
        print(f'\n  [{abl_id}] {desc}')
        for seed in SEEDS:
            auc, t_sec = train_and_eval(ModelCls, train_df, val_df, test_df,
                                        seed=seed, is_single_task=is_single)
            delta = auc - BASELINE_AUC
            print(f'    seed={seed}  AUC={auc:.4f}  ΔAUC={delta:+.4f}  t={t_sec:.0f}s')
            rows.append({'ablation_id': abl_id, 'description': desc,
                         'seed': seed, 'test_AUC': round(auc, 4),
                         'delta_AUC': round(delta, 4), 'train_time_sec': round(t_sec, 1)})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(ABLOUT / 'ablation_results.csv', index=False)
    print(f'\n  Saved: {ABLOUT}/ablation_results.csv')

    # Summary per ablation
    summary = results_df.groupby(['ablation_id','description']).agg(
        mean_AUC=('test_AUC','mean'), std_AUC=('test_AUC','std'),
        mean_delta=('delta_AUC','mean'), std_delta=('delta_AUC','std'),
        mean_time=('train_time_sec','mean')).reset_index()

    print('\n  ── Summary (mean ± std over 3 seeds) ──')
    for _, r in summary.iterrows():
        print(f'  {r.ablation_id}: {r.description}')
        print(f'     AUC={r.mean_AUC:.4f}±{r.std_AUC:.4f}  ΔAUC={r.mean_delta:+.4f}±{r.std_delta:.4f}  '
              f't={r.mean_time:.0f}s')

    # ── Bar chart ──────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--',
                         'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white'})

    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(len(summary))
    colors  = ['#C0392B' if d < -0.03 else '#E67E22' if d < 0 else '#27AE60'
               for d in summary['mean_delta']]
    bars    = ax.bar(x, summary['mean_delta'], color=colors, alpha=0.85,
                     edgecolor='white', lw=1.2, zorder=3)
    ax.errorbar(x, summary['mean_delta'], yerr=summary['std_delta'],
                fmt='none', color='black', capsize=6, lw=1.8, zorder=4)
    ax.axhline(0, color='gray', lw=1.5, ls='--', alpha=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.ablation_id}\n{r.description}" for _, r in summary.iterrows()],
                       fontsize=9, ha='center')
    ax.set_ylabel('ΔAUC vs Full Model')
    ax.set_title(f'Ablation Study — Component Contribution to Drift AUC\n'
                 f'(Full model baseline AUC = {BASELINE_AUC:.4f}, mean ± std, 3 seeds)',
                 fontweight='bold')

    for bar, (_, r) in zip(bars, summary.iterrows()):
        ypos = bar.get_height() + 0.002 if bar.get_height() >= 0 else bar.get_height() - 0.010
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f'{r.mean_delta:+.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.tight_layout()
    fig.savefig(ABLOUT / 'ablation_bar_chart.png')
    plt.close(fig)
    print(f'\n  Saved: {ABLOUT}/ablation_bar_chart.png')

    # ── Interpretation ─────────────────────────────────────────────────────────
    worst = summary.loc[summary['mean_delta'].idxmin()]
    best_kept = summary.loc[summary['mean_delta'].idxmax()]
    print('\n  ── Interpretation ──')
    print(f'  Most critical component: [{worst.ablation_id}] {worst.description}')
    print(f'    Removing it causes ΔAUC = {worst.mean_delta:+.4f}')
    print(f'  Least critical ablation: [{best_kept.ablation_id}] {best_kept.description}')
    print(f'    ΔAUC = {best_kept.mean_delta:+.4f}')
    print(f'\n  All results: {ABLOUT}/ablation_results.csv')
