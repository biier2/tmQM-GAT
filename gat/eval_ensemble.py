"""独立评估脚本 — 从 checkpoint 读取模型，在测试集上评估。不依赖 Jupyter kernel 状态。"""
import numpy as np, pandas as pd, os, sys, json
import torch, torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ===== 加载 checkpoint =====
ckpt = torch.load('ensemble_checkpoint.pt', map_location='cpu')
all_models = ckpt['models']
results_log = ckpt['log']
print(f'Loaded {len(all_models)} models from checkpoint')
for m in all_models:
    print(f'  Fold{m["fold"]}_Seed{m["seed"]}')

# ===== 加载 split =====
split = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_idx = split['te_idx']
print(f'Test set: {len(te_idx):,} samples')

# ===== 加载目标值 =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'
})
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']

# ===== 加载所有数据 (与 ensemble CV 完全相同) =====
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
    return tbl.get(sym,0)

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

print('Parsing XYZ+BO files...')
all_m, all_l = [], []
for part in [1, 2, 3]:
    m, l = parse_one(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz", f"data_tmQM/local_vB/tmQM_X{part}.BO.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l)
    print(f'  Part {part}: {len(m)}')

# SMILES graph building
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
from tqdm import tqdm
graphs, y_vals = [], []
failed = 0
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
    y_vals.append(all_l[idx])

y_all = torch.tensor(y_vals, dtype=torch.float32)
mask = ~(torch.isnan(y_all).any(dim=1) | torch.isinf(y_all).any(dim=1))
graphs = [g for i,g in enumerate(graphs) if mask[i]]
y_all = y_all[mask]
print(f'Built {len(graphs):,} graphs')

# Pre-index
g_x = [g['x'] for g in graphs]
g_ei = [g['edge_index'] for g in graphs]
g_ea = [g['edge_attr'] for g in graphs]

def collate_cpu(indices):
    x_list = [g_x[i] for i in indices]
    ei_list = [g_ei[i] for i in indices]
    ea_list = [g_ea[i] for i in indices]
    batch_list = []
    offset = 0
    for k, x in enumerate(x_list):
        n = x.size(0)
        batch_list.append(torch.full((n,), k, dtype=torch.long))
        ei_list[k] = ei_list[k] + offset
        offset += n
    return {
        'x': torch.cat(x_list, dim=0).to(device),
        'edge_index': torch.cat(ei_list, dim=1).to(device),
        'edge_attr': torch.cat(ea_list, dim=0).to(device),
    }, torch.cat(batch_list, dim=0).to(device)

# ===== 模型定义 (与 ensemble CV 相同) =====
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10

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
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, dropout=DROPOUT, heads=GAT_HEADS):
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

# ===== 评估 =====
bs = 64
y_test_true = y_all[te_idx]
all_test_preds = []

for mi, mdata in enumerate(all_models):
    print(f'Model {mi+1}/{len(all_models)}: Fold{mdata["fold"]}_Seed{mdata["seed"]}...', end=' ', flush=True)
    model = XYZ_GNN().to(device)
    state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
    model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        for s in range(0, len(te_idx), bs):
            bidx = te_idx[s:s+bs].tolist()
            batched, batch_idx = collate_cpu(bidx)
            with autocast('cuda'):
                p = model(batched, batch_idx).float().cpu()
            preds.append(p)
            del batched, batch_idx
    preds = torch.cat(preds, dim=0)
    preds_orig = preds * mdata['y_std'].unsqueeze(0) + mdata['y_mean'].unsqueeze(0)
    all_test_preds.append(preds_orig.numpy())

    del model, preds, preds_orig
    torch.cuda.empty_cache()
    r2_dipole = r2_score(y_test_true[:,3].numpy(), all_test_preds[-1][:,3])
    print(f'done (dipole R²={r2_dipole:.4f})')

all_preds = np.stack(all_test_preds, axis=0)
ensemble_mean = all_preds.mean(axis=0)
ensemble_std = all_preds.std(axis=0)

print(f'\n{"="*80}')
print('ENSEMBLE TEST SET RESULTS (9 models: 3-fold x 3-seed)')
print(f'{"="*80}')
print(f'{"Property":16s} {"Ensemble R2":>12s} {"Mean R2":>10s} {"R2 Std":>10s} {"Ensemble MAE":>14s}')
print(f'{"-"*62}')

results = []
for i, prop in enumerate(targets):
    yt = y_test_true[:, i].numpy()
    per_r2 = [r2_score(yt, all_preds[m,:,i]) for m in range(len(all_models))]
    per_mae = [mean_absolute_error(yt, all_preds[m,:,i]) for m in range(len(all_models))]
    ens_r2 = r2_score(yt, ensemble_mean[:,i])
    ens_mae = mean_absolute_error(yt, ensemble_mean[:,i])
    print(f'{prop:16s} {ens_r2:12.4f} {np.mean(per_r2):10.4f} {np.std(per_r2):10.4f} {ens_mae:14.4f}')
    results.append({'Property':prop,'Ensemble_R2':ens_r2,'Ensemble_MAE':ens_mae,
                    'Mean_R2':np.mean(per_r2),'Std_R2':np.std(per_r2),
                    'Mean_MAE':np.mean(per_mae),'Std_MAE':np.std(per_mae),
                    'Avg_Ensemble_Std':ensemble_std[:,i].mean()})

ens_df = pd.DataFrame(results)
print(f'\nEnsemble Mean R2: {ens_df["Ensemble_R2"].mean():.4f}')
ens_df.to_csv('ensemble_results.csv', index=False)
np.savez('ensemble_predictions.npz', mean=ensemble_mean, std=ensemble_std,
         per_model=all_preds, y_true=y_test_true.numpy(), te_idx=te_idx)
print('Saved ensemble_results.csv + ensemble_predictions.npz')
