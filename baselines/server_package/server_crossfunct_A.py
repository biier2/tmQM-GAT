# -*- coding: utf-8 -*-
"""§3.6 跨泛函验证重跑: 新 A 变体 9 模型集成在 tmQMg (PBE0-D3BJ/def2-TZVP, 73,438 条) 上推断
数据口径与 external_validation.ipynb cell1 一致: 官方 tzvp_* 列直读, 无重算
输出 crossfunct_A_results.json + crossfunct_A_preds.npz (供 Fig6 重渲)
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from torch.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
N_TARGETS = len(targets)
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10

def segment_softmax(scores, dst, Nn):
    scores_f32 = scores.float()
    max_v = torch.full((Nn, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(Nn, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__(); self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; Nn = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0: attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, Nn)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(Nn, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
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
            mm = (x[:, 7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        mm = (x[:, 7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

# ---------- tmQMg 数据 (与 external_validation.ipynb cell1 一致) ----------
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM = 4

def atom_features(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g, int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), int(z >= 21)]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    N = x.size(0); src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml]); battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, N_BT_SM + 2)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}

print('Loading tmQMg...', flush=True)
df_ext = pd.read_csv('tmQMg_properties_and_targets.csv')
COLUMN_MAP = {'gap': 'tzvp_homo_lumo_gap', 'homo': 'tzvp_homo_energy', 'lumo': 'tzvp_lumo_energy',
              'dipole': 'tzvp_dipole_moment', 'E_elec': 'tzvp_electronic_energy',
              'E_disp': 'tzvp_dispersion_energy', 'polarizability': 'polarisability'}
targets_ext = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
valid_mask = df_ext['smiles'].apply(lambda s: isinstance(s, str) and s != '' and s != 'nan')
df_valid = df_ext[valid_mask].copy()
y_ext = np.zeros((len(df_valid), len(targets_ext)), dtype=np.float32)
for i, prop in enumerate(targets_ext):
    y_ext[:, i] = df_valid[COLUMN_MAP[prop]].values.astype(np.float32)
nan_mask = ~(np.isnan(y_ext).any(axis=1) | np.isinf(y_ext).any(axis=1))
df_valid = df_valid[nan_mask].reset_index(drop=True)
y_ext = y_ext[nan_mask]
print(f'Valid: {len(df_valid)}', flush=True)

graphs, indices = [], []
for i, smi in enumerate(df_valid['smiles']):
    g = smiles_to_graph(smi)
    if g is not None:
        graphs.append(g); indices.append(i)
y_ext_final = y_ext[indices]
print(f'Graphs: {len(graphs)}', flush=True)

# ---------- A 变体 9 模型推断 ----------
ck = torch.load('ensemble_checkpoint_A_8d6d.pt', map_location='cpu', weights_only=False)
models = ck['models']
print(f'Models: {len(models)}', flush=True)

all_preds = []
for mdata in models:
    model = XYZ_GNN(node_dim=8, edge_dim=6).to(device)
    state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
    model.load_state_dict(state); model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(graphs), 256):
            bg = graphs[s:s+256]
            x_list = [g['x'].to(device) for g in bg]
            ei_list = []; ea_list = []; bl = []; off = 0
            for k, g in enumerate(bg):
                n = g['x'].size(0)
                ei_list.append(g['edge_index'].to(device) + off)
                ea_list.append(g['edge_attr'].to(device))
                bl.append(torch.full((n,), k, dtype=torch.long, device=device))
                off += n
            batched = {'x': torch.cat(x_list, 0), 'edge_index': torch.cat(ei_list, 1),
                       'edge_attr': torch.cat(ea_list, 0)}
            with autocast('cuda'):
                p = model(batched, torch.cat(bl, 0)).float().cpu()
            preds.append(p)
    preds = torch.cat(preds, 0)
    all_preds.append((preds * mdata['y_std'] + mdata['y_mean']).numpy())
    print(f'  model done ({len(all_preds)}/9)', flush=True)
    del model; torch.cuda.empty_cache()
all_preds = np.stack(all_preds, 0)
ens = all_preds.mean(0)
smiles_idx = [0, 1, 2, 3, 5, 6, 7]  # skip metal_charge

res = {}
for j, prop in enumerate(targets_ext):
    yt = y_ext_final[:, j]; yp = ens[:, smiles_idx[j]]
    r, _ = pearsonr(yt, yp)
    res[prop] = {'pearson_r': float(r), 'r2': float(r2_score(yt, yp)),
                 'mae': float(mean_absolute_error(yt, yp))}
    print(f'{prop:14s} Pearson r={r:.4f}  R2={r2_score(yt, yp):.4f}  MAE={mean_absolute_error(yt, yp):.4f}', flush=True)
res['mean_pearson_r'] = float(np.mean([res[p]['pearson_r'] for p in targets_ext]))
print(f'Mean Pearson r = {res["mean_pearson_r"]:.4f}')

np.savez('crossfunct_A_preds.npz', y_true=y_ext_final,
         preds=all_preds[:, :, smiles_idx], ens_mean=ens[:, smiles_idx])
with open('crossfunct_A_results.json', 'w') as f:
    json.dump(res, f, indent=2)
print('SAVED crossfunct_A_results.json + crossfunct_A_preds.npz')
