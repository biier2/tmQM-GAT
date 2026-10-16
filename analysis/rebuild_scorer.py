"""Rebuild scorer notebook without torch.compile (crashes nbconvert kernel)"""
import json

cells = []

def add_code(source, cell_id):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + '\n' for l in source.split('\n')], "id": cell_id})

def add_md(source, cell_id):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split('\n'), "id": cell_id})

# Cell 0
add_md("# Dual-Model Scorer\nSMILES GAT (R2=0.887) + XYZ GNN (dipole R2=0.658)", "407e2506")

# Cell 1: Imports + graph builder
add_code("""import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, os
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES = [1,2,3,12]; B2I = {b:i for i,b in enumerate(BOND_TYPES)}; N_BT = len(BOND_TYPES)

def atom_features(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA.get(z,(0,0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g, int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    N = x.size(0); src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml]); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, N_BT+2)
    return {'x':x, 'edge_index':ei, 'edge_attr':ea}""", "2b137a1d")

# Cell 2: SMILES GAT model
add_code("""def segment_softmax(scores, dst, N):
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
        src, dst = ei[0], ei[1]; N = x.size(0)
        h = self.node_lin(x)
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

class SMILES_GAT(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden=384, n_layers=5, output_dim=8, dropout=0.10, heads=12):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            metal_mask = (x[:, 7] > 0.5)
            metal_feat = h[metal_mask].mean(dim=0) if metal_mask.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, metal_feat])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        metal_mask = (x[:, 7] > 0.5)
        metal_feat = scatter_mean_per_graph(h[metal_mask], batch[metal_mask], n_graphs) if metal_mask.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, metal_feat], dim=1))

model_smiles = SMILES_GAT(node_dim=8, edge_dim=6, hidden=384, output_dim=8, heads=12).to(device)
ckpt = 'tmqm_best.pt'
if not os.path.exists(ckpt): ckpt = 'tmqm_gat_best.pt'
if os.path.exists(ckpt):
    state = torch.load(ckpt, map_location=device)
    try:
        model_smiles.load_state_dict(state, strict=True)
    except RuntimeError:
        model_smiles.load_state_dict(state, strict=False)
        print(f'SMILES GAT loaded (partial) from {ckpt}')
    else:
        print(f'SMILES GAT loaded from {ckpt}')
else:
    print(f'No SMILES checkpoint found')
model_smiles.eval()""", "34fb7ed2")

# Cell 3: XYZ GNN model
add_code("""class XYZ_GAT(nn.Module):
    def __init__(self, node_dim=11, edge_dim=23, hidden=384, n_layers=5, output_dim=8, dropout=0.10, heads=12):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden*3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms): h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            metal_mask = (x[:, 7] > 0.5)
            metal_feat = h[metal_mask].mean(dim=0) if metal_mask.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, metal_feat])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        metal_mask = (x[:, 7] > 0.5)
        metal_feat = scatter_mean_per_graph(h[metal_mask], batch[metal_mask], n_graphs) if metal_mask.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, metal_feat], dim=1))

model_xyz = XYZ_GAT(node_dim=11, edge_dim=23, hidden=384, output_dim=8, heads=12).to(device)
ckpt_xyz = 'xyz_best.pt'
has_xyz = os.path.exists(ckpt_xyz)
if has_xyz:
    state = torch.load(ckpt_xyz, map_location=device)
    try:
        model_xyz.load_state_dict(state, strict=True)
    except RuntimeError:
        model_xyz.load_state_dict(state, strict=False)
        print(f'XYZ GNN loaded (partial) from {ckpt_xyz}')
    else:
        print(f'XYZ GNN loaded from {ckpt_xyz}')
else:
    print(f'XYZ checkpoint not found ({ckpt_xyz}), using SMILES-only scoring')
model_xyz.eval()

def smiles_to_3d_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200) != 0: return None
        AllChem.MMFFOptimizeMolecule(mol)
    except: return None
    conf = mol.GetConformer(); N = mol.GetNumAtoms()
    def pgrp(z):
        tbl = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
        return tbl.get(z,(0,0))
    xf = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum(); p,g = pgrp(z)
        xf.append([z, a.GetDegree(), a.GetFormalCharge(), p, g, int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z>=21), 0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    c = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(N)], dtype=torch.float32)
    dist = torch.norm(c.unsqueeze(0)-c.unsqueeze(1), dim=2); dist.fill_diagonal_(float('inf'))
    k = min(12, N-1); vals, nbr_idx = torch.topk(dist, k, dim=1, largest=False)
    src, dst, ef = [], [], []
    rbf_c = np.linspace(1.0, 5.0, 16); added = set()
    for i in range(N):
        for kidx in range(k):
            j = nbr_idx[i, kidx].item(); d = vals[i, kidx].item()
            if d < 5.0 and (i,j) not in added:
                added.add((i,j)); added.add((j,i))
                za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
                is_ml = float((za>=21)!=(zb>=21))
                feat = [0.0]*4; feat.extend([0.0, is_ml])
                for mu in rbf_c: feat.append(np.exp(-(d-mu)**2/0.2))
                feat.append(0.0)
                src.extend([i,j]); dst.extend([j,i]); ef.extend([feat,feat])
    ei = torch.zeros(2,0,dtype=torch.long) if not src else torch.tensor([src,dst],dtype=torch.long)
    ea = torch.zeros(0,23) if not src else torch.tensor(ef,dtype=torch.float32)
    return {'x':x, 'edge_index':ei, 'edge_attr':ea}
print('XYZ GNN model ready')""", "04b0d241")

# Cell 4: Scoring loop
add_code("""targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

candidates = {
    'cat_1': "O=[N+]1C(C)(C)CCCC1(C)C.Cl[Fe](Cl)Cl.[NO3].[NO3]",
    'cat_2': "O=[N+]1C(C)(C)CCCC1(C)C.Cl[Fe](Cl)Cl.[NO3].[Cl-]",
}

print(f'Scoring {len(candidates)} candidates...')

smiles_graphs, xyz_graphs, valid_names = [], [], []
for name, smi in candidates.items():
    g1 = smiles_to_graph(smi)
    if g1 is None: continue
    g1_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g1.items()}
    smiles_graphs.append(g1_gpu)
    if has_xyz:
        g2 = smiles_to_3d_graph(smi)
        if g2 is not None:
            g2_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g2.items()}
            xyz_graphs.append(g2_gpu)
        else: xyz_graphs.append(None)
    else: xyz_graphs.append(None)
    valid_names.append(name)

with torch.no_grad():
    if smiles_graphs:
        batched_s, batch_s = collate_graphs(smiles_graphs)
        pred_smiles_all = model_smiles(batched_s, batch_s).cpu().numpy()
    else:
        pred_smiles_all = np.zeros((0, 8))

pred_xyz_all = None
if has_xyz and any(g is not None for g in xyz_graphs):
    valid_xyz = [g for g in xyz_graphs if g is not None]
    if valid_xyz:
        with torch.no_grad():
            batched_x, batch_x = collate_graphs(valid_xyz)
            pred_xyz_all = model_xyz(batched_x, batch_x).cpu().numpy()

results = []; xyz_idx = 0
for i, name in enumerate(valid_names):
    pred_smiles = pred_smiles_all[i]
    if xyz_graphs[i] is not None and pred_xyz_all is not None:
        pred_xyz = pred_xyz_all[xyz_idx]; xyz_idx += 1
    else:
        pred_xyz = pred_smiles
    fused = pred_smiles.copy(); fused[3] = pred_xyz[3]
    results.append({'Name': name, 'gap_Ha': fused[0], 'homo_Ha': fused[1], 'lumo_Ha': fused[2], 'dipole_D(smiles)': pred_smiles[3], 'dipole_D(xyz)': pred_xyz[3], 'metal_q_e': fused[4], 'E_elec_Ha': fused[5], 'E_disp_Ha': fused[6], 'polariz_a03': fused[7]})

df = pd.DataFrame(results).sort_values('E_elec_Ha')
print(f'=== Results ({len(df)} candidates, sorted by E_elec) ===')
print(df.to_string(index=False))
if len(df) > 0:
    best = df.iloc[0]
    print(f'Best: {best[\"Name\"]}')
    print(f'  E_elec = {best[\"E_elec_Ha\"]:.1f} Ha')
    print(f'  gap    = {best[\"gap_Ha\"]:.4f} Ha = {best[\"gap_Ha\"]*27.2114:.2f} eV')
    print(f'  dipole = {best[\"dipole_D(xyz)\"]:.2f} D')""", "57648326")

# Cell 5: Chart
add_code("""import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
if len(df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    names = df['Name'].values
    for ax, col, title, color in [(axes[0], 'E_elec_Ha', 'E_elec (more stable)', 'steelblue'), (axes[1], 'gap_Ha', 'HOMO-LUMO Gap', 'darkorange'), (axes[2], 'dipole_D(xyz)', 'Dipole (XYZ GNN)', 'forestgreen')]:
        ax.barh(range(len(names)), df[col].values, color=color)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(col); ax.set_title(title); ax.invert_yaxis()
    plt.suptitle('Scoring Results', fontsize=14); plt.tight_layout(); plt.show()""", "5eecddd3")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('scorer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
with open('scorer.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'OK: {len(cells)} cells, valid JSON')
