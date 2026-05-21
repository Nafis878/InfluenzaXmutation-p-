#!/usr/bin/env python3
"""
collect_metrics.py — Parse training outputs and write confirmed_metrics.json.

Loads the saved DualBranchMDA checkpoint and test split, then computes:
  - drift probability: AUC, F1, 95% CI (parsed from existing metrics file)
  - antigenic cluster: Macro-F1, ARI
  - drift timing: MAE (days), Spearman rho
  - epistasis: NaN (no dedicated head in DualBranchMDA v2; persist head is proxy)
  - dataset info: total_sequences, H1N1_count, H3N2_count, year_range

Writes: phase8_outputs/confirmed_metrics.json
"""

import sys, re, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, f1_score, mean_absolute_error,
                              adjusted_rand_score)

ROOT   = Path(__file__).parent
OUT    = ROOT / 'outputs'
PHASE8 = ROOT / 'phase8_outputs'

# ── Inline model definitions (copied from phase8_mda_transformer.py) ──────────

AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}
N_AA     = 20

def aa_to_idx(c):
    return AA2IDX.get(c, 0)

_KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
       'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
       'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
       'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2}
_KD_MIN, _KD_RNG = min(_KD.values()), max(_KD.values()) - min(_KD.values())
HYDRO = {aa: (_KD.get(aa, 0.0) - _KD_MIN) / _KD_RNG for aa in AA_VOCAB}

_VOL = {'G': 60, 'A': 89, 'S': 89, 'C': 109, 'P': 113, 'D': 111,
        'T': 116, 'N': 114, 'E': 138, 'Q': 144, 'V': 140, 'H': 153,
        'M': 163, 'I': 167, 'L': 167, 'K': 169, 'R': 174, 'F': 190,
        'Y': 194, 'W': 228}
_VOL_MIN, _VOL_RNG = min(_VOL.values()), max(_VOL.values()) - min(_VOL.values())
VOL = {aa: (_VOL.get(aa, 120) - _VOL_MIN) / _VOL_RNG for aa in AA_VOCAB}

CHARGE    = {aa: (1 if aa in 'RKH' else (-1 if aa in 'DE' else 0)) for aa in AA_VOCAB}
POLAR_SET = set('RNDCQEHKSTY')

CONT_COLS = ['position_norm', 'ref_hydro', 'var_hydro', 'hydro_delta',
             'ref_vol', 'var_vol', 'vol_delta', 'charge_chg', 'polar_chg',
             'crit_flag', 'bind_flag', 'year_norm', 'freq_norm',
             'n_years_norm', 'drift_inten', 'days_norm']
N_CONT = 16


class MutDataset(Dataset):
    TOK_VOCAB = [20, 20, 20, 3, 2, 2, 5, 2]

    def __init__(self, df):
        tok_cols = ['ref_idx', 'var_idx', 'pos_bin', 'era_tok',
                    'crit_flag', 'bind_flag', 'freq_bin', 'charge_tok']
        # Some token cols may be float — cast to int safely
        for c in tok_cols:
            if c not in df.columns:
                df[c] = 0
        self.tokens    = torch.LongTensor(df[tok_cols].fillna(0).astype(int).values)
        self.cont      = torch.FloatTensor(df[CONT_COLS].fillna(0).values.astype(float))
        self.y_drift   = torch.FloatTensor(df['label_drift_prob'].values)
        self.y_cluster = torch.LongTensor(df['label_cluster'].values)
        self.y_timing  = torch.FloatTensor(
            df['label_timing'].clip(0).values / (11 * 365 + 1))
        self.y_persist = torch.FloatTensor(df['persist_norm'].values)

    def __len__(self): return len(self.tokens)

    def __getitem__(self, i):
        return (self.tokens[i], self.cont[i],
                self.y_drift[i], self.y_cluster[i],
                self.y_timing[i], self.y_persist[i])


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
        return sum(torch.exp(-self.log_var[i]) * L + 0.5 * self.log_var[i]
                   for i, L in enumerate(losses))


class DualBranchMDA(nn.Module):
    TOK_VOCAB = MutDataset.TOK_VOCAB
    SEQ_LEN   = 8

    def __init__(self, n_clusters=15, d_tok=96, d_cont=96, d_fused=192,
                 nhead=8, n_layers=3, dropout=0.10):
        super().__init__()
        self.tok_embs = nn.ModuleList(
            [nn.Embedding(v, d_tok) for v in self.TOK_VOCAB])
        self.register_buffer('pe', _sinusoidal_pe(self.SEQ_LEN, d_tok))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=nhead,
            dim_feedforward=d_tok * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.tok_enc  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.feat_enc = nn.Sequential(
            nn.LayerNorm(N_CONT),
            nn.Linear(N_CONT, d_cont), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_cont, d_cont), nn.GELU(),
        )
        self.xattn_ab = nn.MultiheadAttention(d_tok,  nhead, dropout=dropout, batch_first=True)
        self.xattn_ba = nn.MultiheadAttention(d_cont, nhead, dropout=dropout, batch_first=True)
        self.fusion   = nn.Sequential(
            nn.Linear(3 * d_tok, d_fused),
            nn.LayerNorm(d_fused), nn.GELU(), nn.Dropout(dropout),
        )

        def _head(out_dim, final_act=None):
            mods = [nn.Linear(d_fused, 64), nn.GELU(),
                    nn.Dropout(dropout),    nn.Linear(64, out_dim)]
            if final_act: mods.append(final_act)
            return nn.Sequential(*mods)

        self.drift_head   = _head(1, nn.Sigmoid())
        self.cluster_head = _head(n_clusters)
        self.timing_head  = _head(1, nn.Softplus())
        self.persist_head = _head(1, nn.Sigmoid())
        self.task_weighter = DynamicTaskWeighter(n_tasks=4)

    def forward(self, tokens, cont):
        tok_h   = torch.stack(
            [emb(tokens[:, i]) for i, emb in enumerate(self.tok_embs)], dim=1)
        tok_h   = tok_h + self.pe.unsqueeze(0)
        tok_enc = self.tok_enc(tok_h)
        tok_out = tok_enc.mean(dim=1, keepdim=True)
        feat_out = self.feat_enc(cont).unsqueeze(1)
        ab, _  = self.xattn_ab(tok_out, feat_out, feat_out)
        ba, _  = self.xattn_ba(feat_out, tok_out, tok_out)
        ab     = ab.squeeze(1); ba = ba.squeeze(1)
        fused  = self.fusion(torch.cat([ab, ba, ab * ba], dim=-1))
        drift   = self.drift_head(fused).squeeze(-1)
        cluster = self.cluster_head(fused)
        timing  = self.timing_head(fused).squeeze(-1)
        persist = self.persist_head(fused).squeeze(-1)
        return drift, cluster, timing, persist


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            pass
    if not vals:
        v = metric_fn(y_true, y_score)
        return v, v - 0.02, v + 0.02
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ── 1. Parse existing test metrics txt ────────────────────────────────────────

metrics_txt = (PHASE8 / 'phase8_mda_test_metrics.txt').read_text(encoding='utf-8')

def extract_float(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

auc_val = extract_float(r'AUC-ROC\s*:\s*([\d.]+)', metrics_txt)
f1_val  = extract_float(r'F1-Score\s*:\s*([\d.]+)', metrics_txt)
auc_lo  = extract_float(r'AUC 95%CI\s*:\s*\[([\d.]+)', metrics_txt)
auc_hi  = extract_float(r'AUC 95%CI\s*:\s*\[[\d.]+,\s*([\d.]+)\]', metrics_txt)
f1_lo   = extract_float(r'F1 95%CI\s*:\s*\[([\d.]+)', metrics_txt)
f1_hi   = extract_float(r'F1 95%CI\s*:\s*\[[\d.]+,\s*([\d.]+)\]', metrics_txt)

print(f'Parsed from metrics txt:  AUC={auc_val:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]  '
      f'F1={f1_val:.4f} [{f1_lo:.4f}–{f1_hi:.4f}]')


# ── 2. Load model + test data, run inference ───────────────────────────────────

print('\nLoading test data and checkpoint for cluster/timing evaluation ...')
test_df = pd.read_csv(PHASE8 / 'phase8_test_data.csv')

# Derive any missing token columns
for col in ['ref_idx', 'var_idx', 'pos_bin', 'era_tok', 'freq_bin', 'charge_tok']:
    if col not in test_df.columns:
        if col == 'ref_idx':
            test_df['ref_idx'] = test_df['ref_char'].map(AA2IDX).fillna(0).astype(int)
        elif col == 'var_idx':
            test_df['var_idx'] = test_df['var_char'].map(AA2IDX).fillna(0).astype(int)
        elif col == 'pos_bin':
            test_df['pos_bin'] = (test_df['position'].clip(0, 565) // 28).clip(0, 19).astype(int)
        elif col == 'era_tok':
            test_df['era_tok'] = test_df.get('era', pd.Series(0, index=test_df.index)).fillna(0).astype(int)
        elif col == 'freq_bin':
            test_df['freq_bin'] = 0
        elif col == 'charge_tok':
            test_df['charge_tok'] = test_df.get('charge_chg', pd.Series(0, index=test_df.index)).fillna(0).astype(int)

if 'persist_norm' not in test_df.columns:
    test_df['persist_norm'] = test_df.get('n_years_norm', pd.Series(0.0, index=test_df.index)).fillna(0)

n_clusters = 15
model = DualBranchMDA(n_clusters=n_clusters)
ckpt  = PHASE8 / 'phase8_mda_model_best.pt'
state = torch.load(ckpt, map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()
print(f'  Checkpoint loaded: {sum(p.numel() for p in model.parameters()):,} params')

test_ds     = MutDataset(test_df)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

drift_preds, cluster_preds, timing_preds = [], [], []
drift_true,  cluster_true,  timing_true  = [], [], []

with torch.no_grad():
    for tok, cont, yd, yc, yt, yp in test_loader:
        d, cl, ti, _ = model(tok, cont)
        drift_preds.append(d.numpy())
        cluster_preds.append(cl.argmax(-1).numpy())
        timing_preds.append(ti.numpy())
        drift_true.append(yd.numpy())
        cluster_true.append(yc.numpy())
        timing_true.append(yt.numpy())

drift_p  = np.concatenate(drift_preds)
cluster_p = np.concatenate(cluster_preds)
timing_p  = np.concatenate(timing_preds)
drift_t   = np.concatenate(drift_true).astype(int)
cluster_t = np.concatenate(cluster_true)
timing_t_norm = np.concatenate(timing_true)

# Denormalize timing to days
timing_p_days = np.clip(timing_p, 0, None) * (11 * 365)
timing_t_days = timing_t_norm * (11 * 365 + 1)

# Cluster Macro-F1 and ARI
cluster_macro_f1 = f1_score(cluster_t, cluster_p, average='macro', zero_division=0)
cluster_ari      = adjusted_rand_score(cluster_t, cluster_p)

# Timing MAE and Spearman rho
timing_mae  = mean_absolute_error(timing_t_days, timing_p_days)
timing_rho, _ = spearmanr(timing_t_days, timing_p_days)

print(f'  Cluster Macro-F1 = {cluster_macro_f1:.4f}  ARI = {cluster_ari:.4f}')
print(f'  Timing MAE = {timing_mae:.1f} days  Spearman rho = {timing_rho:.4f}')


# ── 3. Dataset info ────────────────────────────────────────────────────────────

print('\nReading dataset for statistics ...')
raw = pd.read_csv(ROOT / 'final_fixed_influenza_ha_v2ok.csv')
total_seqs  = len(raw)
h1n1_count  = int(raw['Subtype'].str.contains('H1N1', case=False, na=False).sum())
h3n2_count  = int(raw['Subtype'].str.contains('H3N2', case=False, na=False).sum())
year_min    = int(raw['Year'].min())
year_max    = int(raw['Year'].max())
print(f'  Total: {total_seqs:,}  H1N1: {h1n1_count:,}  H3N2: {h3n2_count:,}  '
      f'Years: {year_min}–{year_max}')


# ── 4. Assemble confirmed_metrics.json ────────────────────────────────────────

metrics = {
    "drift_probability": {
        "AUC": round(float(auc_val), 4),
        "F1":  round(float(f1_val),  4),
        "CI_95": [round(float(auc_lo), 4), round(float(auc_hi), 4)]
    },
    "antigenic_cluster": {
        "MacroF1": round(float(cluster_macro_f1), 4),
        "ARI":     round(float(cluster_ari), 4)
    },
    "drift_timing": {
        "MAE_days":    round(float(timing_mae),  2),
        "SpearmanRho": round(float(timing_rho),  4)
    },
    "epistasis": {
        "SpearmanRho": None,
        "MSE":         None,
        "note": "No dedicated epistasis head in DualBranchMDA v2; persist head is proxy. "
                "Full epistasis head added in ablation/sensitivity scripts."
    },
    "dataset_info": {
        "total_sequences": int(total_seqs),
        "H1N1_count":      int(h1n1_count),
        "H3N2_count":      int(h3n2_count),
        "year_range":      [int(year_min), int(year_max)]
    }
}

out_path = PHASE8 / 'confirmed_metrics.json'
out_path.write_text(json.dumps(metrics, indent=2, default=lambda x: None), encoding='utf-8')
print(f'\nWrote: {out_path}')


# ── 5. Human-readable summary table ───────────────────────────────────────────

BOLD = '\033[1m'; RESET = '\033[0m'
SEP  = '─' * 60

print(f'\n{SEP}')
print(f'{BOLD}  CONFIRMED EXPERIMENTAL METRICS — DualBranchMDA v2{RESET}')
print(SEP)
print(f'  {"Task":<25} {"Metric":<18} {"Value"}')
print(f'  {"─"*24} {"─"*17} {"─"*12}')
print(f'  {"Drift Probability":<25} {"AUC":<18} {auc_val:.4f}  [{auc_lo:.4f}–{auc_hi:.4f}]')
print(f'  {"Drift Probability":<25} {"F1-Score":<18} {f1_val:.4f}  [{f1_lo:.4f}–{f1_hi:.4f}]')
print(f'  {"Antigenic Cluster":<25} {"Macro-F1":<18} {cluster_macro_f1:.4f}')
print(f'  {"Antigenic Cluster":<25} {"ARI":<18} {cluster_ari:.4f}')
print(f'  {"Drift Timing":<25} {"MAE (days)":<18} {timing_mae:.1f}')
print(f'  {"Drift Timing":<25} {"Spearman rho":<18} {timing_rho:.4f}')
print(f'  {"Epistasis":<25} {"SpearmanRho":<18} NaN  [no dedicated head]')
print(f'  {"Epistasis":<25} {"MSE":<18} NaN  [no dedicated head]')
print(SEP)
print(f'  {"Dataset":<25} {"Total seqs":<18} {total_seqs:,}')
print(f'  {"Dataset":<25} {"H1N1":<18} {h1n1_count:,}')
print(f'  {"Dataset":<25} {"H3N2":<18} {h3n2_count:,}')
print(f'  {"Dataset":<25} {"Year range":<18} {year_min}–{year_max}')
print(SEP)
q1_status = "YES ✓" if auc_val >= 0.80 else "BELOW THRESHOLD"
print(f'\n  Q1 Submission (AUC ≥ 0.80): {q1_status}  (AUC = {auc_val:.4f})')
print(SEP)
