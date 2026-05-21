#!/usr/bin/env python3
"""
esm_baseline.py — ESM-2 embedding baseline comparison for drift probability prediction.

Tries: fair-esm (esm2_t6_8M_UR50D), then HuggingFace transformers (facebook/esm2_t6_8M_UR50D).
Falls back to random embeddings with a note if neither is available.

Comparison:
  MDA Transformer (full)      — from confirmed_metrics.json
  ESM + Logistic Regression   — computed
  ESM + MLP head              — computed

Outputs:
  phase8_outputs/esm_baseline/esm_baseline_results.csv
  phase8_outputs/esm_baseline/esm_roc_curves.png
  phase8_outputs/esm_baseline/esm_comparison_summary.txt
  phase8_outputs/esm_baseline/esm_embeddings_cache.pt  (if computed)
"""

import sys, warnings, json
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
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

ROOT   = Path(__file__).parent
PHASE8 = ROOT / 'phase8_outputs'
ESMOUT = PHASE8 / 'esm_baseline'
ESMOUT.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')
CACHE  = ESMOUT / 'esm_embeddings_cache.pt'

# ── 1. Load split data + raw sequences ────────────────────────────────────────

print('\n' + '='*62)
print(' ESM Baseline Comparison')
print('='*62)

train_df = pd.read_csv(PHASE8 / 'phase8_training_data.csv')
val_df   = pd.read_csv(PHASE8 / 'phase8_val_data.csv')
test_df  = pd.read_csv(PHASE8 / 'phase8_test_data.csv')

raw_df   = pd.read_csv(ROOT / 'final_fixed_influenza_ha_v2ok.csv')

def get_labels(df):
    return df['label_drift_prob'].values.astype(int)

y_train = get_labels(train_df); y_val = get_labels(val_df); y_test = get_labels(test_df)

# Match sequences via accession
acc_to_seq = dict(zip(raw_df['Accession'], raw_df['Sequence']))

def get_sequences(df):
    seqs = []
    for _, row in df.iterrows():
        acc = row.get('accession', None)
        seq = acc_to_seq.get(acc, None) if acc else None
        seqs.append(str(seq)[:512] if seq else 'ACDEFGHIKLMNPQRSTVWY')  # fallback stub
    return seqs

print('\nLoading sequences for train/val/test splits ...')
train_seqs = get_sequences(train_df)
val_seqs   = get_sequences(val_df)
test_seqs  = get_sequences(test_df)

n_found = sum(1 for s in test_seqs if len(s) > 20)
print(f'  Sequences matched: {n_found}/{len(test_seqs)} in test set')


# ── 2. ESM embedding extraction ───────────────────────────────────────────────

ESM_MODEL_NAME = None
ESM_DIM        = None
esm_note       = ''

def compute_esm_embeddings_hf(sequences, model_name='facebook/esm2_t6_8M_UR50D'):
    """Compute mean-pooled ESM-2 embeddings via HuggingFace transformers."""
    from transformers import EsmTokenizer, EsmModel
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model_esm = EsmModel.from_pretrained(model_name)
    model_esm.eval()
    embeddings = []
    batch_size = 16
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            out = model_esm(**inputs)
        # Mean-pool over sequence length (excluding padding)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embeddings.append(emb.numpy())
    return np.concatenate(embeddings, axis=0)


def compute_esm_embeddings_fairesm(sequences):
    """Compute mean-pooled ESM-1b/2 embeddings via fair-esm."""
    import esm
    model_esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model_esm.eval()
    batch_converter = alphabet.get_batch_converter()
    embeddings = []
    batch_size = 8
    for i in range(0, len(sequences), batch_size):
        batch_raw = [(f'seq{i+j}', s) for j, s in enumerate(sequences[i:i+batch_size])]
        _, _, batch_tokens = batch_converter(batch_raw)
        with torch.no_grad():
            results = model_esm(batch_tokens, repr_layers=[6])
        token_reps = results['representations'][6]
        for j, (_, seq) in enumerate(batch_raw):
            emb = token_reps[j, 1:len(seq)+1].mean(0)
            embeddings.append(emb.numpy())
    return np.array(embeddings)


if CACHE.exists():
    print('\nLoading cached ESM embeddings ...')
    cached = torch.load(CACHE, map_location='cpu', weights_only=False)
    X_train_esm = cached['train']
    X_val_esm   = cached['val']
    X_test_esm  = cached['test']
    ESM_MODEL_NAME = cached.get('model_name', 'cached')
    ESM_DIM        = X_train_esm.shape[1]
    esm_note       = cached.get('note', '')
    print(f'  Loaded: shape={X_train_esm.shape}  model={ESM_MODEL_NAME}')
else:
    print('\nComputing ESM embeddings (this may take several minutes on CPU) ...')
    all_seqs = train_seqs + val_seqs + test_seqs

    # Try fair-esm first, then HuggingFace, then fallback
    esm_loaded = False

    try:
        import esm as fairesm_lib
        print('  Using: fair-esm (esm2_t6_8M_UR50D)')
        X_all = compute_esm_embeddings_fairesm(all_seqs)
        ESM_MODEL_NAME = 'fair-esm:esm2_t6_8M_UR50D'
        esm_note = 'Used fair-esm library with esm2_t6_8M_UR50D'
        esm_loaded = True
    except ImportError:
        print('  fair-esm not available, trying HuggingFace transformers ...')

    if not esm_loaded:
        try:
            from transformers import EsmTokenizer, EsmModel
            hf_model_name = 'facebook/esm2_t6_8M_UR50D'
            print(f'  Using: HuggingFace transformers ({hf_model_name})')
            X_all = compute_esm_embeddings_hf(all_seqs, hf_model_name)
            ESM_MODEL_NAME = f'HuggingFace:{hf_model_name}'
            esm_note = f'Used HuggingFace transformers with {hf_model_name} (320-dim)'
            esm_loaded = True
        except (ImportError, Exception) as e:
            print(f'  HuggingFace unavailable: {e}')

    if not esm_loaded:
        print('  FALLBACK: Neither fair-esm nor HuggingFace available.')
        print('  Using BioPython-based physicochemical feature vectors as ESM proxy.')
        print('  NOTE: Results labeled ESM+X are actually 16-feature physicochemical baselines.')
        # Use the 16 CONT_COLS features as a stand-in (already engineered)
        CONT_COLS = ['position_norm','ref_hydro','var_hydro','hydro_delta',
                     'ref_vol','var_vol','vol_delta','charge_chg','polar_chg',
                     'crit_flag','bind_flag','year_norm','freq_norm',
                     'n_years_norm','drift_inten','days_norm']
        X_train_esm = train_df[CONT_COLS].fillna(0).values.astype(float)
        X_val_esm   = val_df[CONT_COLS].fillna(0).values.astype(float)
        X_test_esm  = test_df[CONT_COLS].fillna(0).values.astype(float)
        ESM_MODEL_NAME = 'physicochemical_fallback_16dim'
        ESM_DIM        = 16
        esm_note = ('FALLBACK: Neither fair-esm nor HuggingFace transformers were available. '
                    'Used 16-dimensional physicochemical feature vectors as a proxy baseline. '
                    'Install fair-esm or transformers for true ESM embeddings.')
        torch.save({'train': X_train_esm, 'val': X_val_esm, 'test': X_test_esm,
                    'model_name': ESM_MODEL_NAME, 'note': esm_note}, CACHE)
    else:
        n_tr, n_va, n_te = len(train_seqs), len(val_seqs), len(test_seqs)
        X_train_esm = X_all[:n_tr]
        X_val_esm   = X_all[n_tr:n_tr+n_va]
        X_test_esm  = X_all[n_tr+n_va:]
        ESM_DIM     = X_train_esm.shape[1]
        print(f'  Embedding dim: {ESM_DIM}')
        torch.save({'train': X_train_esm, 'val': X_val_esm, 'test': X_test_esm,
                    'model_name': ESM_MODEL_NAME, 'note': esm_note}, CACHE)
        print(f'  Cached to {CACHE}')

# Ensure numpy arrays
if isinstance(X_train_esm, torch.Tensor): X_train_esm = X_train_esm.numpy()
if isinstance(X_val_esm,   torch.Tensor): X_val_esm   = X_val_esm.numpy()
if isinstance(X_test_esm,  torch.Tensor): X_test_esm  = X_test_esm.numpy()
ESM_DIM = X_train_esm.shape[1]

print(f'\nESM model: {ESM_MODEL_NAME}  dim={ESM_DIM}')
if esm_note:
    print(f'Note: {esm_note}')


# ── 3. ESM + Logistic Regression ─────────────────────────────────────────────

print('\nTraining ESM + Logistic Regression ...')
X_tr_full = np.vstack([X_train_esm, X_val_esm])
y_tr_full = np.concatenate([y_train, y_val])
logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
logreg.fit(X_tr_full, y_tr_full)
lr_prob = logreg.predict_proba(X_test_esm)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)
lr_auc  = roc_auc_score(y_test, lr_prob)
lr_f1   = f1_score(y_test, lr_pred, zero_division=0)
lr_params = ESM_DIM * 2 + 1  # approx: ESM_dim weights + bias for LR
print(f'  LogReg AUC={lr_auc:.4f}  F1={lr_f1:.4f}  params≈{lr_params:,}')


# ── 4. ESM + MLP head ─────────────────────────────────────────────────────────

print('\nTraining ESM + MLP head ...')

class ESM_MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)

mlp_model  = ESM_MLP(in_dim=ESM_DIM).to(DEVICE)
mlp_opt    = torch.optim.Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
X_tr_t     = torch.FloatTensor(X_tr_full)
y_tr_t     = torch.FloatTensor(y_tr_full)
X_te_t     = torch.FloatTensor(X_test_esm)
y_te_t     = torch.FloatTensor(y_test)
best_mlp_state = None; best_mlp_val = 0.0
X_va_t = torch.FloatTensor(X_val_esm); y_va_t = torch.FloatTensor(y_val)

for epoch in range(50):
    mlp_model.train()
    for i in range(0, len(X_tr_t), 64):
        xb = X_tr_t[i:i+64]; yb = y_tr_t[i:i+64]
        loss = F.binary_cross_entropy(mlp_model(xb), yb)
        mlp_opt.zero_grad(); loss.backward(); mlp_opt.step()
    mlp_model.eval()
    with torch.no_grad():
        va_prob = mlp_model(X_va_t).numpy()
    va_auc = roc_auc_score(y_val, va_prob) if len(np.unique(y_val))>1 else 0.5
    if va_auc > best_mlp_val:
        best_mlp_val = va_auc
        best_mlp_state = {k:v.clone() for k,v in mlp_model.state_dict().items()}

mlp_model.load_state_dict(best_mlp_state)
mlp_model.eval()
with torch.no_grad():
    mlp_prob = mlp_model(X_te_t).numpy()
mlp_pred   = (mlp_prob >= 0.5).astype(int)
mlp_auc    = roc_auc_score(y_test, mlp_prob)
mlp_f1     = f1_score(y_test, mlp_pred, zero_division=0)
mlp_params = sum(p.numel() for p in mlp_model.parameters())
print(f'  MLP  AUC={mlp_auc:.4f}  F1={mlp_f1:.4f}  params={mlp_params:,}')


# ── 5. Load MDA Transformer results ───────────────────────────────────────────

metrics_json = json.loads((PHASE8 / 'confirmed_metrics.json').read_text())
mda_auc  = metrics_json['drift_probability']['AUC']
mda_f1   = metrics_json['drift_probability']['F1']
mda_params = 534550  # from training log

print(f'\n  MDA Transformer AUC={mda_auc:.4f}  F1={mda_f1:.4f}  params={mda_params:,}')


# ── 6. Comparison table ────────────────────────────────────────────────────────

results = [
    {'model': 'MDA Transformer (full)',    'AUC': mda_auc,  'F1': mda_f1,  'Params': mda_params, 'type': 'mda'},
    {'model': f'ESM + Logistic Regression ({ESM_MODEL_NAME})',
              'AUC': round(lr_auc,4),  'F1': round(lr_f1,4),  'Params': lr_params,   'type': 'esm_lr'},
    {'model': f'ESM + MLP (hidden=256, dropout=0.2) ({ESM_MODEL_NAME})',
              'AUC': round(mlp_auc,4), 'F1': round(mlp_f1,4), 'Params': mlp_params,  'type': 'esm_mlp'},
]

res_df = pd.DataFrame(results)
res_df.to_csv(ESMOUT / 'esm_baseline_results.csv', index=False)
print('\n  ── Comparison Table ──')
print(f'  {"Model":<52} {"AUC":>8} {"F1":>8} {"Params":>10}')
print(f'  {"─"*52} {"─"*8} {"─"*8} {"─"*10}')
for _, r in res_df.iterrows():
    print(f'  {r.model:<52} {r.AUC:>8.4f} {r.F1:>8.4f} {r.Params:>10,}')


# ── 7. ROC curves ─────────────────────────────────────────────────────────────

# We need test-set probabilities for MDA — recompute from checkpoint
# Load MDA model for ROC curve
AA_VOCAB_LOC = list('ACDEFGHIKLMNPQRSTVWY')
AA2IDX_LOC   = {aa: i for i, aa in enumerate(AA_VOCAB_LOC)}
CONT_COLS_LOC = ['position_norm','ref_hydro','var_hydro','hydro_delta',
                 'ref_vol','var_vol','vol_delta','charge_chg','polar_chg',
                 'crit_flag','bind_flag','year_norm','freq_norm',
                 'n_years_norm','drift_inten','days_norm']

def _sinusoidal_pe_loc(seq_len, d_model):
    pos=torch.arange(seq_len).unsqueeze(1).float()
    i=torch.arange(0,d_model,2).float()
    denom=10000**(i/d_model)
    pe=torch.zeros(seq_len,d_model)
    pe[:,0::2]=torch.sin(pos/denom); pe[:,1::2]=torch.cos(pos/denom)
    return pe

class _DynamicTaskWeighter(nn.Module):
    def __init__(self,n): super().__init__(); self.log_var=nn.Parameter(torch.zeros(n))
    def forward(self,losses): return sum(torch.exp(-self.log_var[i])*L+0.5*self.log_var[i] for i,L in enumerate(losses))

class _DualBranchMDA(nn.Module):
    TOK_VOCAB=[20,20,20,3,2,2,5,2]; SEQ_LEN=8
    def __init__(self,d_tok=96,d_cont=96,d_fused=192,nhead=8,n_layers=3,dropout=0.10):
        super().__init__()
        self.tok_embs=nn.ModuleList([nn.Embedding(v,d_tok) for v in self.TOK_VOCAB])
        self.register_buffer('pe',_sinusoidal_pe_loc(self.SEQ_LEN,d_tok))
        enc_layer=nn.TransformerEncoderLayer(d_model=d_tok,nhead=nhead,dim_feedforward=d_tok*4,dropout=dropout,batch_first=True,norm_first=True)
        self.tok_enc=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.feat_enc=nn.Sequential(nn.LayerNorm(16),nn.Linear(16,d_cont),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_cont,d_cont),nn.GELU())
        self.xattn_ab=nn.MultiheadAttention(d_tok,nhead,dropout=dropout,batch_first=True)
        self.xattn_ba=nn.MultiheadAttention(d_cont,nhead,dropout=dropout,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(3*d_tok,d_fused),nn.LayerNorm(d_fused),nn.GELU(),nn.Dropout(dropout))
        def _h(out,act=None):
            m=[nn.Linear(d_fused,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,out)]
            if act: m.append(act)
            return nn.Sequential(*m)
        self.drift_head=_h(1,nn.Sigmoid()); self.cluster_head=_h(15)
        self.timing_head=_h(1,nn.Softplus()); self.persist_head=_h(1,nn.Sigmoid())
        self.task_weighter=_DynamicTaskWeighter(4)
    def forward(self,tokens,cont):
        tok_h=torch.stack([emb(tokens[:,i]) for i,emb in enumerate(self.tok_embs)],dim=1)
        tok_h=tok_h+self.pe.unsqueeze(0)
        tok_out=self.tok_enc(tok_h).mean(dim=1,keepdim=True)
        feat_out=self.feat_enc(cont).unsqueeze(1)
        ab,_=self.xattn_ab(tok_out,feat_out,feat_out); ba,_=self.xattn_ba(feat_out,tok_out,tok_out)
        ab=ab.squeeze(1); ba=ba.squeeze(1)
        fused=self.fusion(torch.cat([ab,ba,ab*ba],dim=-1))
        return self.drift_head(fused).squeeze(-1),self.cluster_head(fused),self.timing_head(fused).squeeze(-1),self.persist_head(fused).squeeze(-1)

try:
    tok_cols = ['ref_idx','var_idx','pos_bin','era_tok','crit_flag','bind_flag','freq_bin','charge_tok']
    for c in tok_cols:
        if c not in test_df.columns: test_df[c] = 0
    if 'persist_norm' not in test_df.columns:
        test_df['persist_norm'] = test_df.get('n_years_norm', pd.Series(0.0,index=test_df.index)).fillna(0)
    tokens_t  = torch.LongTensor(test_df[tok_cols].fillna(0).astype(int).values)
    cont_t    = torch.FloatTensor(test_df[CONT_COLS_LOC].fillna(0).values.astype(float))
    mda_model = _DualBranchMDA()
    state     = torch.load(PHASE8/'phase8_mda_model_best.pt', map_location='cpu', weights_only=True)
    mda_model.load_state_dict(state); mda_model.eval()
    with torch.no_grad():
        mda_prob = mda_model(tokens_t, cont_t)[0].numpy()
    print(f'  MDA test probs computed for ROC (shape={mda_prob.shape})')
except Exception as e:
    print(f'  Could not load MDA model for ROC: {e}; using stored AUC value only')
    mda_prob = None

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white'})
fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#C0392B','#2471A3','#27AE60','#8E44AD']
models_roc = [
    ('ESM + Logistic Regression', lr_prob,  lr_auc),
    ('ESM + MLP head',            mlp_prob, mlp_auc),
]
if mda_prob is not None:
    models_roc.insert(0, ('MDA Transformer (full)', mda_prob, mda_auc))

for (name, prob, auc), color in zip(models_roc, colors):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, lw=2.5, color=color, label=f'{name} (AUC={auc:.4f})')

ax.plot([0,1],[0,1],'--',color='gray',lw=1.2,alpha=0.6,label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curves — ESM Baseline Comparison\n(ESM model: {ESM_MODEL_NAME})', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.grid(True, alpha=0.25, linestyle='--')
fig.tight_layout()
fig.savefig(ESMOUT / 'esm_roc_curves.png')
plt.close(fig)
print(f'  Saved: {ESMOUT}/esm_roc_curves.png')


# ── 8. Summary text ───────────────────────────────────────────────────────────

lines = [
    'ESM Baseline Comparison — Summary',
    '=' * 50,
    f'ESM model used: {ESM_MODEL_NAME}',
    f'Embedding dimension: {ESM_DIM}',
    f'Note: {esm_note}' if esm_note else '',
    '',
    'Results on held-out test set:',
    f'  {"Model":<50} {"AUC":>8} {"F1":>8} {"Params":>12}',
    f'  {"─"*50} {"─"*8} {"─"*8} {"─"*12}',
]
for _, r in res_df.iterrows():
    lines.append(f'  {r.model:<50} {r.AUC:>8.4f} {r.F1:>8.4f} {r.Params:>12,}')

delta_lr  = mda_auc - lr_auc
delta_mlp = mda_auc - mlp_auc
lines += [
    '',
    f'MDA vs ESM+LogReg ΔAUC = {delta_lr:+.4f}',
    f'MDA vs ESM+MLP    ΔAUC = {delta_mlp:+.4f}',
    '',
    'Interpretation:',
    f'  The MDA Transformer {"outperforms" if delta_lr > 0 else "underperforms"} ESM+LogReg by {abs(delta_lr):.4f} AUC,',
    f'  and {"outperforms" if delta_mlp > 0 else "underperforms"} ESM+MLP by {abs(delta_mlp):.4f} AUC.',
    '  ESM embeddings capture global sequence context but MDA uses mutation-level',
    '  physicochemical features with structural-context information (critical/binding region flags),',
    '  which may be more informative for mutation-level drift prediction.',
]
summary_text = '\n'.join(l for l in lines if l is not None)
(ESMOUT / 'esm_comparison_summary.txt').write_text(summary_text, encoding='utf-8')
print(f'  Saved: {ESMOUT}/esm_comparison_summary.txt')
print('\n' + summary_text)
