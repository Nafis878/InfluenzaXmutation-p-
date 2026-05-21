#!/usr/bin/env python3
"""
loss_weight_sensitivity.py — Grid search over ±0.10 perturbations to the
5-task composite loss weights for the 5-head DualBranchMDA model.

Baseline weights: [0.40, 0.25, 0.15, 0.15, 0.05]
  (w_drift, w_cluster, w_timing, w_agentic/persist, w_epistasis)

NOTE: DualBranchMDA v2 uses DynamicTaskWeighter. For this sensitivity
analysis we build a fixed-weight variant that uses simple weighted sums.
The 5th "epistasis" head uses persist_norm as a proxy target (co-occurrence
proxy will be added when dedicated training data is available).

Outputs:
  phase8_outputs/sensitivity/loss_weight_grid.csv
  phase8_outputs/sensitivity/sensitivity_heatmap.png
  phase8_outputs/sensitivity/sensitivity_summary.txt
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
from sklearn.metrics import roc_auc_score

ROOT   = Path(__file__).parent
PHASE8 = ROOT / 'phase8_outputs'
SENOUT = PHASE8 / 'sensitivity'
SENOUT.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')
BASELINE_WEIGHTS = [0.40, 0.25, 0.15, 0.15, 0.05]  # 5 tasks exactly as spec

# ── Shared classes (inline) ────────────────────────────────────────────────────

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
        # persist = agentic proxy; epistasis = also persist (proxy)
        self.y_persist = torch.FloatTensor(df.get('persist_norm', pd.Series(0,index=df.index)).fillna(0).values)

    def __len__(self): return len(self.tokens)

    def __getitem__(self, i):
        cont = self.cont[i]
        if self.augment:
            cont = cont + torch.randn_like(cont)*0.03
        return self.tokens[i], cont, self.y_drift[i], self.y_cluster[i], self.y_timing[i], self.y_persist[i]


def _sinusoidal_pe(seq_len, d_model):
    pos   = torch.arange(seq_len).unsqueeze(1).float()
    i     = torch.arange(0, d_model, 2).float()
    denom = 10000**(i/d_model)
    pe    = torch.zeros(seq_len, d_model)
    pe[:,0::2] = torch.sin(pos/denom); pe[:,1::2] = torch.cos(pos/denom)
    return pe


class DualBranchMDA_FixedWeights(nn.Module):
    """DualBranchMDA with fixed composite loss weights instead of dynamic σ²."""
    TOK_VOCAB = [20,20,20,3,2,2,5,2]; SEQ_LEN=8

    def __init__(self, weights=None, d_tok=96, d_cont=96, d_fused=192,
                 nhead=8, n_layers=3, dropout=0.10):
        super().__init__()
        self.weights = weights or BASELINE_WEIGHTS

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

        self.drift_head   = _head(1, nn.Sigmoid())
        self.cluster_head = _head(N_CLUSTERS)
        self.timing_head  = _head(1, nn.Softplus())
        self.persist_head = _head(1, nn.Sigmoid())   # agentic proxy
        self.epistasis_head = _head(1, nn.Sigmoid()) # epistasis proxy (same target as persist)

    def forward(self, tokens, cont):
        tok_h   = torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)], dim=1)
        tok_h   = tok_h + self.pe.unsqueeze(0)
        tok_out = self.tok_enc(tok_h).mean(dim=1, keepdim=True)
        feat_out= self.feat_enc(cont).unsqueeze(1)
        ab,_ = self.xattn_ab(tok_out, feat_out, feat_out)
        ba,_ = self.xattn_ba(feat_out, tok_out, tok_out)
        ab=ab.squeeze(1); ba=ba.squeeze(1)
        fused = self.fusion(torch.cat([ab,ba,ab*ba], dim=-1))
        return (self.drift_head(fused).squeeze(-1),
                self.cluster_head(fused),
                self.timing_head(fused).squeeze(-1),
                self.persist_head(fused).squeeze(-1),
                self.epistasis_head(fused).squeeze(-1))

    def compute_weighted_loss(self, drift, cluster, timing, persist, epistasis,
                              yd, yc, yt, yp):
        l_drift    = F.binary_cross_entropy(drift, yd*(1-0.05)+0.05/2)
        l_cluster  = nn.CrossEntropyLoss()(cluster, yc)
        l_timing   = nn.HuberLoss(delta=0.1)(timing, yt)
        l_agentic  = nn.HuberLoss(delta=0.1)(persist, yp)
        l_epistasis= nn.HuberLoss(delta=0.1)(epistasis, yp)  # proxy same target
        w = self.weights
        return w[0]*l_drift + w[1]*l_cluster + w[2]*l_timing + w[3]*l_agentic + w[4]*l_epistasis


# ── Build 11 weight configurations ────────────────────────────────────────────

def perturb_weights(base, idx, delta):
    """Perturb base[idx] by delta, renormalize remaining to sum to 1.0."""
    w = list(base)
    w[idx] = max(0.01, w[idx] + delta)
    total_others = sum(w[i] for i in range(len(w)) if i != idx)
    target_others = 1.0 - w[idx]
    if total_others > 1e-9:
        scale = target_others / total_others
        for i in range(len(w)):
            if i != idx:
                w[i] = max(0.01, w[i] * scale)
    # Final renorm to exactly sum to 1.0
    s = sum(w)
    return [round(x/s, 6) for x in w]


def make_configs():
    configs = [('baseline', BASELINE_WEIGHTS)]
    names   = ['w_drift','w_cluster','w_timing','w_agentic','w_epistasis']
    for i, name in enumerate(names):
        for delta, sign in [(+0.10, 'plus'), (-0.10, 'minus')]:
            w = perturb_weights(BASELINE_WEIGHTS, i, delta)
            configs.append((f'{name}_{sign}0.10', w))
    return configs


# ── Training + evaluation ──────────────────────────────────────────────────────

def train_one(weights, train_df, val_df, seed=42, epochs=150, batch=32, accum=2):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DualBranchMDA_FixedWeights(weights=weights).to(DEVICE)
    train_ds = MutDataset(train_df, augment=True)
    val_ds   = MutDataset(val_df,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=128,   shuffle=False)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-5)

    step = 0; opt.zero_grad()
    for epoch in range(1, epochs+1):
        model.train()
        for tok, cont, yd, yc, yt, yp in train_loader:
            d, cl, ti, pe_out, ep_out = model(tok, cont)
            loss = model.compute_weighted_loss(d, cl, ti, pe_out, ep_out, yd, yc, yt, yp)
            (loss/accum).backward()
            step += 1
            if step % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        if step % accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        sched.step()

    # Validate
    model.eval()
    dp_all, dy_all = [], []
    with torch.no_grad():
        for tok, cont, yd, *_ in val_loader:
            d,*_ = model(tok, cont)
            dp_all.append(d.numpy()); dy_all.append(yd.numpy())
    dp = np.concatenate(dp_all); dy = np.concatenate(dy_all).astype(int)
    val_auc = roc_auc_score(dy, dp) if len(np.unique(dy)) > 1 else 0.5
    return float(val_auc)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*62)
    print(' Loss Weight Sensitivity Analysis')
    print(f' Baseline: {BASELINE_WEIGHTS}')
    print('='*62)

    train_df = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
    val_df   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')

    for df in [train_df, val_df]:
        if 'persist_norm' not in df.columns:
            df['persist_norm'] = df.get('n_years_norm', pd.Series(0.0, index=df.index)).fillna(0)

    configs = make_configs()
    names   = ['w_drift','w_cluster','w_timing','w_agentic','w_epistasis']
    rows    = []

    for cfg_id, (cfg_name, weights) in enumerate(configs):
        t0 = time.perf_counter()
        val_auc = train_one(weights, train_df, val_df, seed=42)
        elapsed = time.perf_counter() - t0
        row = {'config_id': cfg_id, 'config_name': cfg_name}
        for n, w in zip(names, weights):
            row[n] = round(w, 4)
        row['val_AUC'] = round(val_auc, 4)
        rows.append(row)
        print(f'  [{cfg_id:>2}] {cfg_name:<22} weights={[round(w,3) for w in weights]}  '
              f'val_AUC={val_auc:.4f}  t={elapsed:.0f}s')

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(SENOUT / 'loss_weight_grid.csv', index=False)
    print(f'\n  Saved: {SENOUT}/loss_weight_grid.csv')

    baseline_auc = grid_df.loc[grid_df['config_name']=='baseline','val_AUC'].values[0]

    # ── Bar chart ──────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--',
                         'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white'})

    fig, ax = plt.subplots(figsize=(14, 6))
    x       = np.arange(len(grid_df))
    colors  = ['#27AE60' if n=='baseline' else '#2471A3' for n in grid_df['config_name']]
    bars    = ax.bar(x, grid_df['val_AUC'], color=colors, alpha=0.85, edgecolor='white', lw=1.2)
    ax.axhline(baseline_auc, color='#C0392B', lw=2, ls='--', alpha=0.8, label=f'Baseline ({baseline_auc:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_df['config_name'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Validation AUC (drift probability head)')
    ax.set_title('Loss Weight Sensitivity Analysis — Val AUC per Configuration\n'
                 '(green = baseline, blue = perturbed)', fontweight='bold')
    ax.legend()
    for bar, auc in zip(bars, grid_df['val_AUC']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(SENOUT / 'sensitivity_heatmap.png')
    plt.close(fig)
    print(f'  Saved: {SENOUT}/sensitivity_heatmap.png')

    # ── Summary text ──────────────────────────────────────────────────────────
    non_base = grid_df[grid_df['config_name'] != 'baseline'].copy()
    non_base['delta'] = non_base['val_AUC'] - baseline_auc
    worst_row = non_base.loc[non_base['delta'].idxmin()]
    best_row  = non_base.loc[non_base['delta'].idxmax()]
    variance  = float(grid_df['val_AUC'].var())

    summary_lines = [
        'Loss Weight Sensitivity Analysis — Summary',
        '=' * 50,
        f'Baseline weights: {BASELINE_WEIGHTS}',
        f'Baseline val AUC: {baseline_auc:.4f}',
        '',
        f'Largest AUC DROP:  {worst_row.config_name}  ΔAUC={worst_row.delta:+.4f}  '
        f'(val_AUC={worst_row.val_AUC:.4f})',
        f'Largest AUC GAIN:  {best_row.config_name}   ΔAUC={best_row.delta:+.4f}  '
        f'(val_AUC={best_row.val_AUC:.4f})',
        f'Variance across {len(grid_df)} configs: {variance:.6f}',
        f'Std dev:  {grid_df["val_AUC"].std():.4f}',
        f'Range:    [{grid_df["val_AUC"].min():.4f}, {grid_df["val_AUC"].max():.4f}]',
        '',
        'Conclusion:',
        (f'  The baseline weighting appears {"robust" if variance < 1e-4 else "sensitive"} '
         f'to ±0.10 perturbations (std={grid_df["val_AUC"].std():.4f}).'),
        (f'  The most impactful component is {worst_row.config_name.rsplit("_",2)[0]}, '
         f'suggesting it is critical for drift prediction accuracy.'),
    ]
    summary_text = '\n'.join(summary_lines)
    (SENOUT / 'sensitivity_summary.txt').write_text(summary_text, encoding='utf-8')
    print(f'  Saved: {SENOUT}/sensitivity_summary.txt')
    print('\n' + summary_text)
