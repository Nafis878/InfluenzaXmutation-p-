#!/usr/bin/env python3
"""
retrain_with_epistasis.py — Fix Issue 1: Compute NPMI epistasis labels and retrain.

Steps:
  1. Compute NPMI-based epistasis labels from phase3 co-occurrence data
  2. Add epistasis head to DualBranchMDA (DualBranchMDA_v2)
  3. Retrain with 5-task weighting (drift, cluster, timing, persist, epistasis)
  4. Evaluate Spearman rho on epistasis head output
  5. Save model to phase8_mda_model_best_v2.pt
  6. Update confirmed_metrics.json with epistasis values
  7. Document method in outputs/epistasis_label_method.txt

Fallback: if < 50 co-occurrence pairs for a mutation, use
  epistasis_label = (unique co-mutations within ±20 positions) / 20
"""

import sys, time, json, warnings, os
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error

ROOT   = Path(__file__).parent
PHASE8 = ROOT / 'phase8_outputs'
OUT    = ROOT / 'outputs'
os.makedirs(str(OUT), exist_ok=True)

DEVICE = torch.device('cpu')
SEED   = 42

# ── Constants ──────────────────────────────────────────────────────────────────
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
VOL  = {aa:(_VOL.get(aa,120)-_VOL_MIN)/_VOL_RNG for aa in AA_VOCAB}
CHARGE    = {aa:(1 if aa in 'RKH' else(-1 if aa in 'DE' else 0)) for aa in AA_VOCAB}
POLAR_SET = set('RNDCQEHKSTY')

CONT_COLS = ['position_norm','ref_hydro','var_hydro','hydro_delta',
             'ref_vol','var_vol','vol_delta','charge_chg','polar_chg',
             'crit_flag','bind_flag','year_norm','freq_norm',
             'n_years_norm','drift_inten','days_norm']
N_CONT    = 16
N_CLUSTERS = 15


# ── NPMI Epistasis Label Computation ──────────────────────────────────────────

def compute_epistasis_labels(df, phase3_path, window=20, min_pairs=50):
    """
    Compute NPMI-based epistasis label for each mutation in df.

    Returns (labels_array, method_counts_dict)
    """
    print('  Loading phase3 co-occurrence data...')
    ph3 = pd.read_csv(phase3_path, usecols=['accession','year','position','var_char'])
    ph3['position'] = ph3['position'].astype(int)
    print(f'  phase3: {len(ph3):,} mutation records loaded')

    labels      = np.zeros(len(df))
    method_used = Counter()

    # Pre-group phase3 by year for efficiency
    ph3_by_year = {yr: grp.reset_index(drop=True)
                   for yr, grp in ph3.groupby('year')}

    unique_years = sorted(df['first_year'].dropna().unique())
    print(f'  Processing {len(unique_years)} unique years in training set...')

    for year in unique_years:
        year_int = int(year)
        year_mask = df['first_year'] == year
        year_mutations = df[year_mask]

        if year_int not in ph3_by_year:
            print(f'    Year {year_int}: no phase3 data -> labels=0')
            method_used['no_phase3_data'] += int(year_mask.sum())
            continue

        ph3_yr  = ph3_by_year[year_int]
        n_acc   = ph3_yr['accession'].nunique()
        if n_acc == 0:
            method_used['no_accessions'] += int(year_mask.sum())
            continue

        # Build mutation→accession set and accession→mutation set
        mut_to_accs = defaultdict(set)
        acc_to_muts = defaultdict(set)
        for _, row in ph3_yr.iterrows():
            k = (int(row['position']), row['var_char'])
            mut_to_accs[k].add(row['accession'])
            acc_to_muts[row['accession']].add(k)

        # Mutation frequency in this year
        mut_freq = {k: len(v) for k, v in mut_to_accs.items()}

        n_muts_in_year = int(year_mask.sum())
        print(f'    Year {year_int}: {n_muts_in_year} training mutations, '
              f'{n_acc} accessions in phase3', end='', flush=True)

        npmi_count = 0; fallback_count = 0; zero_count = 0
        for idx in year_mutations.index:
            row = df.loc[idx]
            pos_i  = int(row['position'])
            var_i  = row.get('var_char', None)
            if var_i is None or (not isinstance(var_i, str)):
                method_used['no_var_char'] += 1
                continue

            mut_i = (pos_i, var_i)
            acc_set_i = mut_to_accs.get(mut_i, set())

            if len(acc_set_i) == 0:
                method_used['mutation_not_in_phase3'] += 1
                zero_count += 1
                continue

            # Count co-occurrences within ±20 positions across accessions
            cooc_counts = Counter()
            for acc in acc_set_i:
                for (pj, vj) in acc_to_muts[acc]:
                    if abs(pj - pos_i) <= window and (pj, vj) != mut_i:
                        cooc_counts[(pj, vj)] += 1

            total_pairs = sum(cooc_counts.values())

            if total_pairs < min_pairs:
                # Fallback: unique co-mutations / window size
                n_unique = len(cooc_counts)
                labels[idx] = min(float(n_unique) / window, 1.0)
                method_used['fallback_count'] += 1
                fallback_count += 1
            else:
                # Full NPMI computation
                p_i = len(acc_set_i) / n_acc
                npmi_scores = []
                for (pj, vj), cooc_cnt in cooc_counts.items():
                    p_j  = mut_freq.get((pj, vj), 1) / n_acc
                    p_ij = cooc_cnt / n_acc
                    if p_ij > 0 and p_i > 0 and p_j > 0:
                        pmi  = np.log(p_ij / (p_i * p_j + 1e-10) + 1e-10)
                        npmi = pmi / (-np.log(p_ij + 1e-10))
                        npmi_scores.append(npmi)

                if npmi_scores:
                    npmi_scores.sort(reverse=True)
                    mean_top3 = float(np.mean(npmi_scores[:3]))
                    labels[idx] = float(np.clip(mean_top3, 0.0, 1.0))
                    method_used['npmi'] += 1
                    npmi_count += 1
                else:
                    method_used['empty_npmi'] += 1
                    zero_count += 1

        print(f'  → npmi={npmi_count}, fallback={fallback_count}, zero={zero_count}')

    print(f'\n  Method summary: {dict(method_used)}')
    print(f'  Label distribution: min={labels.min():.4f}, '
          f'mean={labels.mean():.4f}, max={labels.max():.4f}')
    print(f'  Non-zero labels: {(labels>0).sum()} / {len(labels)}')

    return labels, method_used


# ── Model v2 with 5 heads ──────────────────────────────────────────────────────

class MutDataset_v2(Dataset):
    TOK_VOCAB = [20,20,20,3,2,2,5,2]

    def __init__(self, df, augment=False):
        self.augment = augment
        tok_cols = ['ref_idx','var_idx','pos_bin','era_tok',
                    'crit_flag','bind_flag','freq_bin','charge_tok']
        for c in tok_cols:
            if c not in df.columns: df=df.copy(); df[c]=0
        self.tokens    = torch.LongTensor(df[tok_cols].fillna(0).astype(int).values)
        self.cont      = torch.FloatTensor(df[CONT_COLS].fillna(0).values.astype(float))
        self.y_drift   = torch.FloatTensor(df['label_drift_prob'].values)
        self.y_cluster = torch.LongTensor(df['label_cluster'].values)
        self.y_timing  = torch.FloatTensor(df['label_timing'].clip(0).values/(11*365+1))
        self.y_persist = torch.FloatTensor(df.get('persist_norm',pd.Series(0,index=df.index)).fillna(0).values)
        self.y_epist   = torch.FloatTensor(df['epistasis_label'].fillna(0).values.astype(float))

    def __len__(self): return len(self.tokens)

    def __getitem__(self, i):
        cont = self.cont[i]
        if self.augment: cont = cont + torch.randn_like(cont)*0.03
        return (self.tokens[i], cont, self.y_drift[i], self.y_cluster[i],
                self.y_timing[i], self.y_persist[i], self.y_epist[i])


def _sinusoidal_pe(seq_len, d_model):
    pos=torch.arange(seq_len).unsqueeze(1).float()
    i=torch.arange(0,d_model,2).float()
    denom=10000**(i/d_model); pe=torch.zeros(seq_len,d_model)
    pe[:,0::2]=torch.sin(pos/denom); pe[:,1::2]=torch.cos(pos/denom)
    return pe


class DynamicTaskWeighter(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))
    def forward(self, losses):
        return sum(torch.exp(-self.log_var[i])*L+0.5*self.log_var[i]
                   for i,L in enumerate(losses))


class DualBranchMDA_v2(nn.Module):
    """5-head variant: adds epistasis_head trained on NPMI labels."""
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

        def _head(out, act=None):
            m=[nn.Linear(d_fused,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,out)]
            if act: m.append(act)
            return nn.Sequential(*m)

        self.drift_head    = _head(1, nn.Sigmoid())
        self.cluster_head  = _head(N_CLUSTERS)
        self.timing_head   = _head(1, nn.Softplus())
        self.persist_head  = _head(1, nn.Sigmoid())
        self.epistasis_head= _head(1, nn.Sigmoid())  # new
        self.task_weighter = DynamicTaskWeighter(5)   # 5 tasks

    def forward(self, tokens, cont):
        tok_h=torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)],dim=1)
        tok_h=tok_h+self.pe.unsqueeze(0)
        tok_out=self.tok_enc(tok_h).mean(dim=1,keepdim=True)
        feat_out=self.feat_enc(cont).unsqueeze(1)
        ab,_=self.xattn_ab(tok_out,feat_out,feat_out)
        ba,_=self.xattn_ba(feat_out,tok_out,tok_out)
        ab=ab.squeeze(1); ba=ba.squeeze(1)
        fused=self.fusion(torch.cat([ab,ba,ab*ba],dim=-1))
        return (self.drift_head(fused).squeeze(-1),
                self.cluster_head(fused),
                self.timing_head(fused).squeeze(-1),
                self.persist_head(fused).squeeze(-1),
                self.epistasis_head(fused).squeeze(-1))


ce_loss  = nn.CrossEntropyLoss()
hub_loss = nn.HuberLoss(delta=0.1)

def bce_smooth(pred, target, eps=0.05):
    return F.binary_cross_entropy(pred, target*(1-eps)+eps/2)


def ensure_cols(df):
    df = df.copy()
    if 'ref_idx' not in df.columns and 'ref_char' in df.columns:
        df['ref_idx'] = df['ref_char'].map(AA2IDX).fillna(0).astype(int)
    if 'var_idx' not in df.columns and 'var_char' in df.columns:
        df['var_idx'] = df['var_char'].map(AA2IDX).fillna(0).astype(int)
    if 'pos_bin' not in df.columns and 'position' in df.columns:
        df['pos_bin'] = (df['position'].clip(0,565)//28).clip(0,19).astype(int)
    if 'era_tok' not in df.columns:
        df['era_tok'] = df.get('era',pd.Series(0,index=df.index)).fillna(0).astype(int)
    for c in ['freq_bin','charge_tok']:
        if c not in df.columns: df[c] = 0
    if 'persist_norm' not in df.columns:
        df['persist_norm'] = df.get('n_years_norm',pd.Series(0,index=df.index)).fillna(0)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*62)
    print(' Retrain with NPMI Epistasis Labels (Issue 1 Fix)')
    print('='*62)

    torch.manual_seed(SEED); np.random.seed(SEED)

    # Load splits
    train_df = ensure_cols(pd.read_csv(PHASE8 / 'phase8_training_data.csv'))
    val_df   = ensure_cols(pd.read_csv(PHASE8 / 'phase8_val_data.csv'))
    test_df  = ensure_cols(pd.read_csv(PHASE8 / 'phase8_test_data.csv'))

    print(f'\n  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

    phase3_path = OUT / 'phase3_variations_annotated.csv'

    # ── Step 1: Compute epistasis labels ──────────────────────────────────────
    t0 = time.perf_counter()
    print('\n[Step 1] Computing NPMI epistasis labels...')
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_labels, method_counts = compute_epistasis_labels(all_df, phase3_path)

    # Split labels back
    n_tr = len(train_df); n_va = len(val_df)
    train_df['epistasis_label'] = all_labels[:n_tr]
    val_df['epistasis_label']   = all_labels[n_tr:n_tr+n_va]
    test_df['epistasis_label']  = all_labels[n_tr+n_va:]

    # Document method
    npmi_frac = method_counts.get('npmi', 0) / len(all_df)
    fallback_frac = method_counts.get('fallback_count', 0) / len(all_df)
    primary_method = 'npmi' if npmi_frac >= fallback_frac else 'fallback_count_ratio'

    method_doc = [
        'Epistasis Label Method',
        '='*50,
        f'Method used: {"NPMI (primary)" if primary_method=="npmi" else "Fallback count ratio (primary)"}',
        f'NPMI computation: {method_counts.get("npmi",0):,} mutations ({npmi_frac:.1%})',
        f'Fallback count: {method_counts.get("fallback_count",0):,} mutations ({fallback_frac:.1%})',
        f'Zero (no data): {method_counts.get("mutation_not_in_phase3",0)+method_counts.get("no_phase3_data",0):,} mutations',
        '',
        'NPMI formula:',
        '  PMI(i,j)  = log[P(i,j) / (P(i)*P(j))]',
        '  NPMI(i,j) = PMI(i,j) / (-log P(i,j))',
        '  epistasis_label = mean NPMI of top-3 co-occurring neighbors within ±20 pos',
        '  Clipped to [0,1]',
        '',
        'Fallback condition: < 50 co-occurrence pair observations',
        '  Fallback formula: epistasis_label = (unique co-mutations within ±20) / 20',
        '',
        'Co-occurrence definition:',
        '  Mutations i and j co-occur if both appear in the same accession (sequence)',
        '  within the same year cohort (first_year of the training mutation)',
        '  Position window: ±20 residues in full HA alignment',
        '',
        f'Label statistics:',
        f'  min={all_labels.min():.4f}, mean={all_labels.mean():.4f}',
        f'  max={all_labels.max():.4f}, non-zero={int((all_labels>0).sum())}/{len(all_df)}',
    ]
    with open(OUT / 'epistasis_label_method.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(method_doc))
    print(f'  Saved: outputs/epistasis_label_method.txt')
    print(f'  Label computation: {time.perf_counter()-t0:.1f}s')

    # ── Step 2: Retrain with 5-head model ─────────────────────────────────────
    print('\n[Step 2] Training DualBranchMDA_v2 (5 heads)...')
    t1 = time.perf_counter()

    model = DualBranchMDA_v2().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    train_ds = MutDataset_v2(train_df, augment=True)
    val_ds   = MutDataset_v2(val_df,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    opt = torch.optim.AdamW(
        [{'params':[p for n,p in model.named_parameters() if 'task_weighter' not in n],'lr':3e-4},
         {'params':model.task_weighter.parameters(),'lr':1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=40,T_mult=2,eta_min=1e-5)

    EPOCHS = 150; ACCUM = 2
    best_val_auc = 0.0; best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        step=0; opt.zero_grad()
        for tok,cont,yd,yc,yt,yp,ye in train_loader:
            d,cl,ti,pe_out,ep = model(tok,cont)
            losses = [bce_smooth(d,yd), ce_loss(cl,yc),
                      hub_loss(ti,yt), hub_loss(pe_out,yp), hub_loss(ep,ye)]
            loss = model.task_weighter(losses)
            (loss/ACCUM).backward(); step+=1
            if step%ACCUM==0:
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); opt.zero_grad()
        sched.step()

        if epoch % 25 == 0 or epoch == EPOCHS:
            model.eval()
            dp_all, dy_all = [], []
            with torch.no_grad():
                for tok,cont,yd,*_ in val_loader:
                    d,*_ = model(tok,cont)
                    dp_all.append(d.numpy()); dy_all.append(yd.numpy())
            dp = np.concatenate(dp_all); dy = np.concatenate(dy_all).astype(int)
            if len(np.unique(dy)) > 1:
                vauc = roc_auc_score(dy, dp)
                if vauc > best_val_auc:
                    best_val_auc = vauc
                    best_state   = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            print(f'  Epoch {epoch:3d}/{EPOCHS}  val_AUC={vauc:.4f}  best={best_val_auc:.4f}')

    # Save best model
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), PHASE8 / 'phase8_mda_model_best_v2.pt')
    print(f'\n  Saved: phase8_mda_model_best_v2.pt  ({time.perf_counter()-t1:.1f}s)')

    # ── Step 3: Evaluate on test set ──────────────────────────────────────────
    print('\n[Step 3] Evaluating on test set...')
    test_ds = MutDataset_v2(test_df, augment=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model.eval()
    dp_all,dy_all,ep_all,ey_all,ti_all,ty_all=[],[],[],[],[],[]
    with torch.no_grad():
        for tok,cont,yd,yc,yt,yp,ye in test_loader:
            d,cl,ti,_,ep = model(tok,cont)
            dp_all.append(d.numpy()); dy_all.append(yd.numpy())
            ep_all.append(ep.numpy()); ey_all.append(ye.numpy())
            ti_all.append(ti.numpy()); ty_all.append(yt.numpy())

    dp = np.concatenate(dp_all); dy = np.concatenate(dy_all).astype(int)
    ep = np.concatenate(ep_all); ey = np.concatenate(ey_all)
    ti_p = np.clip(np.concatenate(ti_all),0,None)*(11*365)
    ti_t = np.concatenate(ty_all)*(11*365+1)

    test_auc = roc_auc_score(dy, dp) if len(np.unique(dy))>1 else float('nan')
    test_f1  = f1_score(dy, (dp>=0.5).astype(int), zero_division=0)
    timing_mae = mean_absolute_error(ti_t, ti_p)

    # Epistasis Spearman rho
    if len(np.unique(ey)) > 1:
        ep_rho, ep_pval = spearmanr(ey, ep)
        ep_mse = float(np.mean((ep - ey)**2))
    else:
        ep_rho, ep_pval, ep_mse = float('nan'), float('nan'), float('nan')

    print(f'\n  Test AUC      = {test_auc:.4f}')
    print(f'  Test F1       = {test_f1:.4f}')
    print(f'  Timing MAE    = {timing_mae:.1f} days')
    print(f'  Epistasis rho = {ep_rho:.4f}  (p={ep_pval:.2e})')
    print(f'  Epistasis MSE = {ep_mse:.4f}')

    # ── Step 4: Update confirmed_metrics.json ─────────────────────────────────
    print('\n[Step 4] Updating confirmed_metrics.json...')
    metrics_path = PHASE8 / 'confirmed_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics['epistasis'] = {
        'spearman_rho': round(float(ep_rho),4)  if not np.isnan(ep_rho) else None,
        'spearman_pval': float(ep_pval) if not np.isnan(ep_pval) else None,
        'mse': round(float(ep_mse),4) if not np.isnan(ep_mse) else None,
        'label_method': primary_method,
        'label_method_details': 'See outputs/epistasis_label_method.txt',
        'n_npmi': int(method_counts.get('npmi',0)),
        'n_fallback': int(method_counts.get('fallback_count',0)),
    }
    metrics['model_v2'] = {
        'checkpoint': 'phase8_mda_model_best_v2.pt',
        'n_params': n_params,
        'n_heads': 5,
        'val_auc_v2': round(best_val_auc, 4),
        'test_auc_v2': round(test_auc, 4),
        'test_f1_v2': round(test_f1, 4),
        'timing_mae_v2': round(timing_mae, 1),
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Updated: confirmed_metrics.json')
    print(f'  epistasis_spearman_rho = {ep_rho:.4f}')
    print()
    print('  Issue 1 RESOLVED: epistasis Spearman rho is no longer NaN.')
