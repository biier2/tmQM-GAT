"""Build Ensemble + CV notebook: 5-fold × 3-seed = 15 models for XYZ GNN"""
import json

cells = []

def add_md(src, cid):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split('\n'), "id": cid})

def add_code(src, cid):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + '\n' for l in src.split('\n')], "id": cid})

# ============================================================
# Cell 1: Introduction
# ============================================================
add_md("""# XYZ GNN Ensemble + Cross-Validation

## Purpose
Response to the reviewer: a single model is unreliable → report the mean ± std over a multi-model ensemble

## Design
```
               Full dataset (95,152 complexes)
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              Train set (85%)         Test set (15%)
              ~80,879 complexes     ~14,273 complexes
                    │               (fixed, not in CV)
          ┌─────────┼─────────┐
          │         │         │
       Fold 0    Fold 1    Fold 2
      (train 2/3, val 1/3, 3 seeds each)

Total: 3 folds × 3 seeds = 15 models
Test: 9-model ensemble predictions → mean ± std
```

## Model configuration
- XYZ GNN: 12-head GAT × 384d × 5 layers
- 23-d edge features (SMILES bonds + 16 DFT-distance RBF + Mayer BO)
- Automatic metal weighting
- AMP mixed precision, CosineWarmRestarts, early stopping patience=40

## Expected time (5070 Ti)
- ~35 min per model × 15 = ~9 hours""", "cv_intro")

# ============================================================
# Cell 2: Imports + GPU configuration + data loading
# ============================================================
add_code("""import numpy as np, pandas as pd, time, gzip, os, sys, json
import matplotlib.pyplot as plt, warnings
from collections import Counter
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler

# GPU configuration
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')

from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

# ===== Load target values =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'
})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
print(f'Loaded {len(df_y)} complexes, {len(targets)} targets')
""", "cv_import")


# ============================================================
# Cell 3: XYZ+BO parsing + SMILES graph construction (reuses tmQM_Fe_XYZ)
# ============================================================
add_code("""print('='*60)
print('Step 1: XYZ+BO parsing + SMILES graph construction (full 95k)')
print('='*60)

import gzip
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

def parse_one(xyz_p, bo_p, start_idx=0):
    mols, labels = [], []
    with gzip.open(bo_p, "rt") as f:
        bo_lines = [l.strip() for l in f.readlines()]
    with gzip.open(xyz_p, "rt") as f:
        xyz_blocks = f.read().strip().split(chr(10))
    bo_blocks, cur = [], []
    for line in bo_lines:
        if line.startswith("CSD_code"):
            if cur: bo_blocks.append(cur); cur = []
        elif line: cur.append(line)
    if cur: bo_blocks.append(cur)
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break; i += 1
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]]); atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            bo_mat = None
            if mol_idx < len(bo_blocks):
                bo_mat = np.zeros((n_atoms, n_atoms), dtype=np.float32)
                for bline in bo_blocks[mol_idx]:
                    bparts = bline.split()
                    try: src_idx = int(bparts[0]) - 1
                    except ValueError: continue
                    j = 2
                    while j + 1 < len(bparts):
                        try:
                            nbr = int(bparts[j]) - 1; bo_val = float(bparts[j+1])
                            if 0 <= src_idx < n_atoms and 0 <= nbr < n_atoms:
                                bo_mat[src_idx, nbr] = bo_val
                            j += 3
                        except (ValueError, IndexError): j += 1
            mols.append({"atoms": np.array(atoms), "coords": np.array(coords, dtype=np.float32),
                         "bo": bo_mat, "csv_idx": mol_idx})
            labels.append(df_y.iloc[mol_idx][targets].astype(float).values)
    return mols, labels

all_m, all_l = [], []
for part in [1, 2, 3]:
    print(f'  Part {part}...')
    m, l = parse_one(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz", f"data_tmQM/local_vB/tmQM_X{part}.BO.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l)
    print(f'    {len(m)} TM complexes')

print(f'Total: {len(all_m)} TM complexes parsed')

# ===== SMILES graph construction (reuses tmQM_Fe_XYZ Cell 4) =====
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM = 4
rbf_c = np.linspace(1.0, 5.0, 16); rbf_sigma = 0.2
smiles_by_idx = df_y['SMILES'].tolist()

def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA_SM.get(z,(0,0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

def fast_match(sm_atoms, xyz_atoms):
    N = len(sm_atoms)
    if N == len(xyz_atoms) and all(sm_atoms[i] == xyz_atoms[i] for i in range(N)):
        return list(range(N))
    sm_to_xyz = [-1] * N; used = set()
    for i, z_sm in enumerate(sm_atoms):
        for j, z_xyz in enumerate(xyz_atoms):
            if z_sm == z_xyz and j not in used:
                sm_to_xyz[i] = j; used.add(j); break
    return sm_to_xyz

print(f'Building graphs...')
graphs, y_vals, atoms_filt, failed = [], [], [], 0
for idx, m in enumerate(tqdm(all_m, desc='Graphs')):
    csv_i = m["csv_idx"]; smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': failed += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: failed += 1; continue
    N = mol.GetNumAtoms()
    sm_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    sm_to_xyz = fast_match(sm_atoms, m['atoms'])
    coords = m['coords']; bo_mat = m.get('bo')

    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    if bo_mat is not None:
        for i in range(N):
            j = sm_to_xyz[i]
            if j >= 0 and j < bo_mat.shape[0]:
                nb = int((bo_mat[j] > 0.05).sum()) - 1
                sb = float(bo_mat[j].sum()) - bo_mat[j,j]
                mb = float(bo_mat[j].max())
                xf[i].extend([max(0,nb), max(0,sb), max(0,mb)])
            else: xf[i].extend([0,0.0,0.0])
    else:
        for i in range(N): xf[i].extend([0,0.0,0.0])
    x = torch.tensor(xf, dtype=torch.float32)

    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_atoms[i], sm_atoms[j]
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        d, bo = 2.0, 0.0
        xi, xj = sm_to_xyz[i], sm_to_xyz[j]
        if xi>=0 and xj>=0 and xi<len(coords) and xj<len(coords):
            d = float(np.linalg.norm(coords[xi]-coords[xj]))
        if bo_mat is not None and xi>=0 and xj>=0 and xi<bo_mat.shape[0] and xj<bo_mat.shape[1]:
            bo = float(bo_mat[xi,xj])
        for mu in rbf_c: bf.append(np.exp(-(d-mu)**2/rbf_sigma))
        bf.append(bo); battr.extend([bf,bf])

    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0,23)
    graphs.append({'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':N})
    y_vals.append(all_l[idx]); atoms_filt.append(m['atoms'])

y_all = torch.tensor(y_vals, dtype=torch.float32)
print(f'Built {len(graphs):,} graphs ({failed} failed)')

# NaN filtering
mask = ~(torch.isnan(y_all).any(dim=1) | torch.isinf(y_all).any(dim=1))
graphs = [g for i, g in enumerate(graphs) if mask[i]]
y_all = y_all[mask]; atoms_filt = [a for i, a in enumerate(atoms_filt) if mask[i]]
print(f'After NaN removal: {len(graphs):,} complexes')
""", "cv_data")


# ============================================================
# Cell 4: Model definition + utility functions
# ============================================================
add_code("""print('='*60)
print('Step 2: Model definition')
print('='*60)

HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10
N_TARGETS = len(targets)

def segment_softmax(scores, dst, N):
    scores_f32 = scores.float()
    max_v = torch.full((N, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(N, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__(); self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim>0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; N = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0)>0: attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)

def collate_graphs(graph_list):
    x_list, ei_list, ea_list, batch_list = [], [], [], []
    offset = 0
    for i, g in enumerate(graph_list):
        n = g['x'].size(0)
        x_list.append(g['x']); ei_list.append(g['edge_index'] + offset)
        ea_list.append(g['edge_attr'])
        batch_list.append(torch.full((n,), i, dtype=torch.long, device=g['x'].device))
        offset += n
    return {'x': torch.cat(x_list, dim=0), 'edge_index': torch.cat(ei_list, dim=1), 'edge_attr': torch.cat(ea_list, dim=0)}, torch.cat(batch_list, dim=0)

def scatter_mean_per_graph(h, batch, n_graphs):
    out = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)

def scatter_max_per_graph(h, batch, n_graphs):
    out = torch.full((n_graphs, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out

class XYZ_GNN(nn.Module):
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS,
                 output_dim=N_TARGETS, dropout=DROPOUT, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:,7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

print(f'Params: {sum(p.numel() for p in XYZ_GNN().parameters()):,}')
print(f'HIDDEN={HIDDEN} LAYERS={N_LAYERS} HEADS={GAT_HEADS} DROPOUT={DROPOUT}')
""", "cv_model")


# ============================================================
# Cell 5: Data split (fixed test set + 5-fold CV)
# ============================================================
add_code("""print('='*60)
print('Step 3: Data split (fixed 15% test set + 5-fold CV on 85%)')
print('='*60)

# Fixed 15% test set
idx_all = np.random.RandomState(42).permutation(len(graphs))
n_test = int(len(graphs) * 0.15)
te_idx = idx_all[:n_test]  # first 15% fixed as the test set
cv_idx = idx_all[n_test:]  # remaining 85% used for CV

print(f'Total: {len(graphs):,}')
print(f'Fixed test set: {len(te_idx):,} (15%)')
print(f'CV pool: {len(cv_idx):,} (85%)')

# 5-fold split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_splits = list(kf.split(cv_idx))
for fi, (tr_rel, va_rel) in enumerate(fold_splits):
    print(f'  Fold {fi}: train={len(tr_rel):,}, val={len(va_rel):,}')

# Save the split
np.savez('data_tmQM/paper_domain_vA/ensemble_split.npz',
    te_idx=te_idx, cv_idx=cv_idx,
    fold_0_train=cv_idx[fold_splits[0][0]], fold_0_val=cv_idx[fold_splits[0][1]],
    fold_1_train=cv_idx[fold_splits[1][0]], fold_1_val=cv_idx[fold_splits[1][1]],
    fold_2_train=cv_idx[fold_splits[2][0]], fold_2_val=cv_idx[fold_splits[2][1]])
print('Split saved to ensemble_split.npz')
""", "cv_split")


# ============================================================
# Cell 6: Training function
# ============================================================
add_code("""print('='*60)
print('Step 4: Define the training function + GPU preload')
print('='*60)

# ===== Upload all graphs to the GPU once (avoids re-uploading for every model) =====
print('Uploading all graphs to GPU (one-time)...')
graphs_gpu = []
for g in tqdm(graphs, desc='GPU upload'):
    graphs_gpu.append({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()})
y_all_gpu = y_all.to(device)

# ===== Create metal weights in absolute indexing (one weight per sample) =====
print('Computing per-sample metal weights...')
mc = Counter()
for atoms in atoms_filt:
    for z in set(atoms):
        if z >= 21: mc[z] += 1
total_c = len(atoms_filt)
weight_map = {z: min(5.0, total_c/(c+1)) for z, c in mc.items()}
print(f'Metal weight map: {dict(sorted(weight_map.items()))}')

sample_weights = torch.ones(len(graphs), device=device)
for i, atoms in enumerate(atoms_filt):
    w = max((weight_map.get(z, 1.0) for z in set(atoms) if z >= 21), default=1.0)
    sample_weights[i] = w
print(f'Sample weights: mean={sample_weights.mean().item():.2f}, max={sample_weights.max().item():.1f}')

def train_one_model(train_idx, val_idx, seed, fold_name):
    '''Train a single model, returning the best checkpoint and the validation metrics'''
    print(f'  [DEBUG] seed={seed}, device={device}', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'  [DEBUG] Creating model...', flush=True)
    model = XYZ_GNN().to(device)
    print(f'  [DEBUG] Model created, params={sum(p.numel() for p in model.parameters()):,}', flush=True)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    crit = nn.MSELoss(reduction='none')
    scaler = GradScaler('cuda')

    # Normalisation (based on the training set)
    print(f'  [DEBUG] Normalizing targets...', flush=True)
    y_tr = y_all_gpu[train_idx]
    y_mean_local = y_tr.mean(dim=0)
    y_std_local = y_tr.std(dim=0).clamp(min=1e-8)
    y_norm = (y_all_gpu - y_mean_local) / y_std_local

    bs = 128; best_va = float('inf'); best_state = None; patience = 0
    epochs = 200; early_stop = 40
    print(f'  [DEBUG] Starting training loop (epochs={epochs}, train={len(train_idx)}, val={len(val_idx)})', flush=True)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_idx))  # CPU
        for s in range(0, len(train_idx), bs):
            pos = perm[s:s+bs]
            bidx = train_idx[pos].tolist()
            bg = [graphs_gpu[i] for i in bidx]
            by = y_norm[bidx]
            w = sample_weights[bidx]  # take weights by batch index

            batched, batch_idx = collate_graphs(bg)
            opt.zero_grad()
            with autocast('cuda'):
                preds = model(batched, batch_idx)
                loss = (crit(preds, by).mean(dim=1) * w).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

        model.eval()
        with torch.no_grad():
            va_preds = []
            for s in range(0, len(val_idx), bs):
                bidx = val_idx[s:s+bs].tolist()
                bg = [graphs_gpu[i] for i in bidx]
                batched, batch_idx = collate_graphs(bg)
                with autocast('cuda'):
                    va_preds.append(model(batched, batch_idx).float())
            vp = torch.cat(va_preds, dim=0)
            va_loss = nn.MSELoss()(vp, y_norm[val_idx])

        sched.step(ep + va_loss.item()/10)

        if va_loss < best_va:
            best_va = va_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1

        if patience >= early_stop: break

        if (ep+1) % 50 == 0:
            print(f'  [{fold_name}] Epoch {ep+1}: VaLoss={va_loss.item():.4f} Pat={patience}')

    model.load_state_dict(best_state)
    model.eval()

    # Validation evaluation after denormalisation
    with torch.no_grad():
        vp_full = []
        for s in range(0, len(val_idx), bs):
            bidx = val_idx[s:s+bs].tolist()
            bg = [graphs_gpu[i] for i in bidx]
            batched, batch_idx = collate_graphs(bg)
            with autocast('cuda'):
                vp_full.append(model(batched, batch_idx).detach().cpu().float())
        vp_full = torch.cat(vp_full, dim=0)
        vp_np = vp_full.detach().cpu().numpy()
    vt_np = y_all[val_idx].numpy()
    ym_np = y_mean_local.detach().cpu().numpy()
    ys_np = y_std_local.detach().cpu().numpy()
    vp_orig = vp_np * ys_np + ym_np

    val_r2s = [r2_score(vt_np[:,i], vp_orig[:,i]) for i in range(N_TARGETS)]
    print(f'  [{fold_name}] Done: best_va={best_va:.4f}, val Mean R²={np.mean(val_r2s):.4f}')

    return best_state, y_mean_local.detach().cpu(), y_std_local.detach().cpu(), val_r2s

print(f'Training function ready (graphs on GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB)')
""", "cv_train_fn")


# ============================================================
# Cell 7: Run the 5-fold × 3-seed training
# ============================================================
add_code("""print('='*60)
print('Step 5: 5-fold × 3-seed training (15 models in total)')
print('='*60)

SEEDS = [42, 123, 456]
FOLDS = 5

all_models = []  # [(state_dict, y_mean, y_std, val_r2s, fold, seed)]
results_log = []

t_start = time.time()
model_count = 0

for fold_idx in range(FOLDS):
    tr_rel = fold_splits[fold_idx][0]  # indices relative to cv_idx
    va_rel = fold_splits[fold_idx][1]
    tr_abs = cv_idx[tr_rel]; va_abs = cv_idx[va_rel]

    for seed in SEEDS:
        model_count += 1
        name = f'Fold{fold_idx}_Seed{seed}'
        print(f'\\n{"="*50}')
        print(f'Model {model_count}/9: {name}')
        print(f'  Train={len(tr_abs):,} Val={len(va_abs):,} Seed={seed}')
        print(f'{"="*50}')
        t0 = time.time()

        try:
            print(f'  Starting train_one_model...', flush=True)
            state, ym, ys, val_r2s = train_one_model(tr_abs, va_abs, seed, name)
            elapsed = time.time() - t0
            all_models.append({'state': state, 'y_mean': ym, 'y_std': ys,
                               'fold': fold_idx, 'seed': seed, 'val_r2s': val_r2s})
            results_log.append({'fold': fold_idx, 'seed': seed, 'val_mean_r2': np.mean(val_r2s),
                                'val_r2s': val_r2s, 'time_min': elapsed/60})
            print(f'  Completed in {elapsed/60:.1f}min, Val Mean R²={np.mean(val_r2s):.4f}')
        except Exception as e:
            import traceback
            print(f'  FAILED: {e}')
            traceback.print_exc()
            results_log.append({'fold': fold_idx, 'seed': seed, 'error': str(e)})

    # Save intermediate results
    torch.save({'models': all_models, 'log': results_log}, 'ensemble_checkpoint.pt')
    print(f'\\nCheckpoint saved after fold {fold_idx} ({len(all_models)}/{model_count} models)')

total_time = (time.time() - t_start) / 60
print(f'\\nTraining complete: {len(all_models)}/15 models in {total_time:.0f}min')

# Print CV results
print(f'\\n{"="*60}')
print('Cross-Validation Summary')
print(f'{"="*60}')
for r in results_log:
    if 'error' in r:
        print(f'  Fold{r["fold"]}_Seed{r["seed"]}: ERROR - {r["error"]}')
    else:
        print(f'  Fold{r["fold"]}_Seed{r["seed"]}: Val Mean R²={r["val_mean_r2"]:.4f}  [{r["time_min"]:.1f}min]')

val_means = [r['val_mean_r2'] for r in results_log if 'val_mean_r2' in r]
if val_means:
    print(f'\\nCV Mean R²: {np.mean(val_means):.4f} ± {np.std(val_means):.4f}')
""", "cv_train")


# ============================================================
# Cell 8: Test-set ensemble evaluation
# ============================================================
add_code("""print('='*60)
print('Step 6: Test-set ensemble evaluation (9 models)')
print('='*60)

bs = 128
# Reuse the graphs_gpu + y_all_gpu already uploaded in Cell 4 (no re-upload)
y_test_true = y_all[te_idx]

# Collect each model's test-set predictions
if len(all_models) == 0:
    # Try to restore from the checkpoint
    if os.path.exists('ensemble_checkpoint.pt'):
        print('Loading from ensemble_checkpoint.pt...')
        ckpt = torch.load('ensemble_checkpoint.pt', map_location='cpu')
        all_models = ckpt['models']
        results_log = ckpt['log']
        print(f'Restored {len(all_models)} models from checkpoint')
    else:
        raise RuntimeError(
            'No models available! Training may have failed or not been run yet. '
            'Check that Cell 6 completed successfully.'
        )

all_test_preds = []  # [model_idx][n_test, n_targets]

for mi, mdata in enumerate(all_models):
    print(f'Model {mi+1}/{len(all_models)}: Fold{mdata["fold"]}_Seed{mdata["seed"]}...')
    model = XYZ_GNN().to(device)
    model.load_state_dict(mdata['state'])
    model.eval()

    preds = []
    with torch.no_grad():
        for s in range(0, len(te_idx), bs):
            bidx = te_idx[s:s+bs].tolist()
            bg = [graphs_gpu[i] for i in bidx]
            batched, batch_idx = collate_graphs(bg)
            with autocast('cuda'):
                p = model(batched, batch_idx).float().cpu()
            preds.append(p)
    preds = torch.cat(preds, dim=0)

    # Denormalise (using that model's training-set statistics)
    preds_orig = preds * mdata['y_std'].unsqueeze(0) + mdata['y_mean'].unsqueeze(0)
    all_test_preds.append(preds_orig.numpy())

if len(all_test_preds) == 0:
    # Try single-model evaluation from the existing xyz_best.pt as a fallback
    if os.path.exists('xyz_best.pt'):
        print('No CV models found. Falling back to xyz_best.pt for single-model evaluation...')
        fallback_state = torch.load('xyz_best.pt', map_location='cpu')
        model = XYZ_GNN().to(device)
        model.load_state_dict(fallback_state, strict=False)
        model.eval()
        # normalisation statistics of the training set are needed
        y_tr = y_all[tr_idx] if 'tr_idx' in dir() else y_all[:int(len(y_all)*0.7)]
        y_mean_fb = y_tr.mean(dim=0)
        y_std_fb = y_tr.std(dim=0).clamp(min=1e-8)
        with torch.no_grad():
            fb_preds = []
            for s in range(0, len(te_idx), bs):
                bidx = te_idx[s:s+bs].tolist()
                bg = [graphs_gpu[i] for i in bidx]
                batched, batch_idx = collate_graphs(bg)
                with autocast('cuda'):
                    p = model(batched, batch_idx).float().cpu()
                fb_preds.append(p)
        fb_preds = torch.cat(fb_preds, dim=0)
        fb_preds_orig = fb_preds * y_std_fb.unsqueeze(0) + y_mean_fb.unsqueeze(0)
        all_test_preds = [fb_preds_orig.numpy()]
        print(f'Using xyz_best.pt for single-model baseline')
    else:
        raise RuntimeError(
            'No models available! Either:\\n'
            '  1. Run Cell 6 first to train CV models, or\\n'
            '  2. Place xyz_best.pt in the working directory for single-model fallback')

all_preds = np.stack(all_test_preds, axis=0)  # [n_models, n_test, n_targets]
print(f'Stacked predictions: {all_preds.shape}')

# Ensemble statistics
ensemble_mean = all_preds.mean(axis=0)  # [n_test, n_targets]
ensemble_std = all_preds.std(axis=0)    # [n_test, n_targets]

# Per-property evaluation
print(f'\\n{"="*80}')
print('ENSEMBLE TEST SET RESULTS (15 models: 5-fold × 3-seed)')
print(f'{"="*80}')
print(f'{"Property":16s} {"Mean R²":>8s} {"R² Std":>8s} {"Mean MAE":>10s} {"MAE Std":>10s} {"Ensemble Std":>12s}')
print(f'{"-"*70}')

ensemble_results = []
for i, prop in enumerate(targets):
    yt = y_test_true[:, i].numpy()

    # R² and MAE of each model
    per_model_r2 = [r2_score(yt, all_preds[m, :, i]) for m in range(len(all_models))]
    per_model_mae = [mean_absolute_error(yt, all_preds[m, :, i]) for m in range(len(all_models))]

    # R² and MAE of the ensemble
    ens_r2 = r2_score(yt, ensemble_mean[:, i])
    ens_mae = mean_absolute_error(yt, ensemble_mean[:, i])

    # Mean standard deviation of the ensemble (predictive uncertainty)
    avg_ens_std = ensemble_std[:, i].mean()

    print(f'{prop:16s} {np.mean(per_model_r2):8.4f} {np.std(per_model_r2):8.4f} '
          f'{np.mean(per_model_mae):10.4f} {np.std(per_model_mae):10.4f} {avg_ens_std:12.4f}')

    ensemble_results.append({
        'Property': prop,
        'Ensemble_R2': ens_r2, 'Ensemble_MAE': ens_mae,
        'Mean_R2': np.mean(per_model_r2), 'Std_R2': np.std(per_model_r2),
        'Mean_MAE': np.mean(per_model_mae), 'Std_MAE': np.std(per_model_mae),
        'Avg_Ensemble_Std': avg_ens_std
    })

ens_df = pd.DataFrame(ensemble_results)
print(f'\\nEnsemble Mean R²: {ens_df["Ensemble_R2"].mean():.4f}')
print(f'Single-model Mean R²: {ens_df["Mean_R2"].mean():.4f} ± {ens_df["Std_R2"].mean():.4f}')

# Save results
ens_df.to_csv('ensemble_results.csv', index=False)
np.savez('ensemble_predictions.npz', mean=ensemble_mean, std=ensemble_std,
         per_model=all_preds, y_true=y_test_true.numpy(), te_idx=te_idx)
print('Results saved to ensemble_results.csv + ensemble_predictions.npz')
""", "cv_eval")


# ============================================================
# Cell 9: Visualisation
# ============================================================
add_code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) R² mean ± std
ax = axes[0, 0]
x = np.arange(len(targets))
means = ens_df['Mean_R2'].values
stds = ens_df['Std_R2'].values
bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', edgecolor='white')
ax.axhline(y=np.mean(means), color='darkorange', linestyle='--', linewidth=1.5, label=f'Overall Mean R²={np.mean(means):.4f}')
ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=9)
ax.set_ylabel('R²'); ax.set_title(f'5-fold × 3-seed CV: R² per Property')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# (b) Ensemble vs Single Model R²
ax = axes[0, 1]
x2 = np.arange(len(targets))
w = 0.35
ax.bar(x2 - w/2, ens_df['Ensemble_R2'].values, w, label='Ensemble (mean prediction)', color='darkorange')
ax.bar(x2 + w/2, ens_df['Mean_R2'].values, w, label='Single Model Avg', color='steelblue')
ax.set_xticks(x2); ax.set_xticklabels(targets, fontsize=9)
ax.set_ylabel('R²'); ax.set_title('Ensemble vs Single-Model R²')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# (c) Val R² distribution of each model
ax = axes[1, 0]
val_data = []
for r in results_log:
    if 'val_r2s' in r:
        val_data.append(np.mean(r['val_r2s']))
ax.boxplot([val_data])
ax.set_xticklabels(['9 Models'])
ax.scatter(np.ones(len(val_data)) + np.random.uniform(-0.1, 0.1, len(val_data)),
           val_data, color='darkorange', s=30, zorder=5)
ax.axhline(y=np.mean(val_data), color='red', linestyle='--', label=f'Mean={np.mean(val_data):.4f}')
ax.set_ylabel('Validation Mean R²'); ax.set_title('Model-to-Model Variation')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# (d) Predictive uncertainty vs error
ax = axes[1, 1]
for i, prop in enumerate(targets):
    yt = y_test_true[:, i].numpy()
    abs_err = np.abs(ensemble_mean[:, i] - yt)
    ax.scatter(ensemble_std[:, i], abs_err, alpha=0.3, s=3, label=prop)
ax.set_xlabel('Ensemble Std (Uncertainty)'); ax.set_ylabel('Absolute Error')
ax.set_title('Uncertainty vs Error (all properties)')
ax.legend(ncol=2, fontsize=7, markerscale=3); ax.grid(True, alpha=0.3)

plt.suptitle('XYZ GNN Ensemble + Cross-Validation Results', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.show()

# Key findings
print('\\n' + '='*60)
print('Key findings')
print('='*60)
print(f'1. CV Mean R²: {np.mean(val_means):.4f} ± {np.std(val_means):.4f}')
print(f'2. Ensemble R²: {ens_df["Ensemble_R2"].mean():.4f}')
print(f'3. Single Model R²: {ens_df["Mean_R2"].mean():.4f} ± {ens_df["Std_R2"].mean():.4f}')
print(f'4. Ensemble vs Single improvement: {ens_df["Ensemble_R2"].mean() - ens_df["Mean_R2"].mean():+.4f}')
print(f'5. Inter-model stability (R² std): {ens_df["Std_R2"].mean():.4f} (smaller is more stable)')
""", "cv_viz")

# ============================================================
# Cell 10: Discussion
# ============================================================
add_md("""## Ensemble + CV conclusions

### Comparison with tmGNN-XAI
| Metric | tmGNN-XAI | This work |
|------|:---:|:---:|
| CV scheme | 3-repeat × 5-fold (15 models) | 5-fold × 3-seed (9 models) |
| Ensemble | Top-5 selected by val loss | Average of all 9 models |
| Uncertainty | CV (ensemble std) | CV (ensemble std) |
| Physical constraint | ✅ HL-Gap = LUMO - HOMO | ❌ To be added (1.5) |

### Value for the manuscript
1. **R² ± std reporting**: responds to reviewer R2's concern about reproducibility
2. **Ensemble gain**: quantifies the gain of the ensemble over a single model
3. **Uncertainty estimation**: provides the basis for the later trust framework (Scenario A/B)
4. **Model stability**: small std → the architecture is well designed, not a lucky draw

### Limitations
- The test set is only 15%, and CV is run only on training+val
- 9 models is fewer than tmGNN-XAI's 15, but 3 seeds suffice to assess the variance
""", "cv_discussion")

# ============================================================
# Write the notebook
# ============================================================
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.0"}},
      "cells": cells}

with open('ensemble_cv.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

with open('ensemble_cv.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'Built {len(cells)} cells, JSON valid → ensemble_cv.ipynb')
