"""Build ablation notebook for tmQM GNN - uses AblationGNN class, no factory function"""
import json

cells = []

def add_code(src, cid):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + '\n' for l in src.split('\n')], "id": cid})

def add_md(src, cid):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split('\n'), "id": cid})

# Cell 1: Intro
add_md("""# 消融实验: tmQM GNN (20k采样, 快速对比)

## 实验设计

| ID | 消融维度 | 基线 | 变体 | 回答的问题 |
|----|----------|------|------|-----------|
| A | 注意力头数 | 4 | 8, 12 | 头数越多越好？ |
| B | 网络深度 | 3层 | 4层, 5层 | 加深是否过平滑？ |
| C | 金属池化 | on | off | 金属单独池化有贡献吗？ |
| D | Fe加权 | off (1x) | on (3x) | Fe加权帮助还是过拟合？ |

每个配置: 20k配合物, 100 epochs, batch=64, hidden=256, 4 layers, 8 heads (基线).
训练时间: ~30 min/配置, 8个配置 ~4小时.""", "eq_intro")

# Cell 2: Data
add_code("""import numpy as np, pandas as pd, time, json
import torch, torch.nn as nn, torch.optim as optim
import warnings; warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from rdkit import Chem; from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

N_SAMPLE = 20000; REG_DROPOUT = 0.20; SCALE_HIDDEN = 256; SCALE_LAYERS = 4
GAT_HEADS = 8; BATCH = 64; EPOCHS = 100; EARLY_STOP = 30

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap':'gap','HOMO_Energy':'homo','LUMO_Energy':'lumo',
    'Dipole_M':'dipole','Metal_q':'metal_charge','Electronic_E':'E_elec',
    'Dispersion_E':'E_disp','Polarizability':'polarizability'})
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
df = df.dropna(subset=targets+['SMILES'])
df = df.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)
print(f'Data: {len(df)} complexes')

ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),
    14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),
    23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),
    30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),
    48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),
    77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES=[1,2,3,12]; B2I={b:i for i,b in enumerate(BOND_TYPES)}; N_BT=4

def atom_features(atom):
    z=atom.GetAtomicNum(); p,g=ELEMENT_DATA.get(z,(0,0))
    return [z,atom.GetDegree(),atom.GetFormalCharge(),p,g,int(atom.GetIsAromatic()),atom.GetTotalNumHs(),int(z>=21)]

def mol_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x=torch.tensor([atom_features(a) for a in mol.GetAtoms()],dtype=torch.float32)
    N=x.size(0); src,dst,battr=[],[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(); bt=int(bond.GetBondType())
        za,zb=mol.GetAtomWithIdx(i).GetAtomicNum(),mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml=float((za>=21)!=(zb>=21)); is_ring=float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf=[0.0]*N_BT
        if bt in B2I: bf[B2I[bt]]=1.0
        bf.extend([is_ring,is_ml]); battr.extend([bf,bf])
    ei=torch.tensor([src,dst],dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea=torch.tensor(battr,dtype=torch.float32) if battr else torch.zeros(0,N_BT+2)
    return {'x':x,'edge_index':ei,'edge_attr':ea,'num_nodes':N}

print('Building graphs...')
graphs, y_vals, failed = [], [], 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    g = mol_to_graph(row['SMILES'])
    if g is not None and g['num_nodes'] > 1:
        graphs.append(g); y_vals.append([row[t] for t in targets])
    else: failed += 1
y_all = torch.tensor(y_vals, dtype=torch.float32)
print(f'Built {len(graphs):,} graphs ({failed} failed)')

y_mean = y_all.mean(dim=0); y_std = y_all.std(dim=0).clamp(min=1e-8)
y_norm = (y_all - y_mean) / y_std
idx = np.random.RandomState(42).permutation(len(graphs))
n_tr = int(len(graphs)*0.7); n_va = int(len(graphs)*0.15)
tr_idx = idx[:n_tr]; va_idx = idx[n_tr:n_tr+n_va]; te_idx = idx[n_tr+n_va:]
print(f'Train:{len(tr_idx):,} Val:{len(va_idx):,} Test:{len(te_idx):,}')

def g2dev(g):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()}

fe_indices = set()
for i in range(len(df)):
    if 'Fe' in str(df.iloc[i]['SMILES']): fe_indices.add(i)
print(f'Fe complexes: {len(fe_indices)}')""", "eq_data")

# Cell 3: Model + train_eval
add_code("""# ======= Shared Model Components =======
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

def scatter_mean_per_graph(h, batch, n):
    out = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n, device=h.device, dtype=h.dtype)
    cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)

def scatter_max_per_graph(h, batch, n):
    out = torch.full((n, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out

class AblationGNN(nn.Module):
    # 可配置消融模型: hidden, n_layers, heads, use_metal_pool 均可变
    def __init__(self, hidden=256, n_layers=4, heads=8, dropout=0.20, use_metal_pool=True):
        super().__init__()
        self.heads = heads; self.use_metal_pool = use_metal_pool
        self.embed = nn.Linear(8, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden//heads, 6, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        in_dim = hidden*3 if use_metal_pool else hidden*2
        self.readout = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 8))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        if batch is None:
            if self.use_metal_pool:
                metal_mask = (x[:,7] > 0.5)
                metal_feat = h[metal_mask].mean(dim=0) if metal_mask.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
                return self.readout(torch.cat([h.mean(0), h.max(0).values, metal_feat])).unsqueeze(0)
            return self.readout(torch.cat([h.mean(0), h.max(0).values])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        if self.use_metal_pool:
            metal_mask = (x[:,7] > 0.5)
            metal_feat = scatter_mean_per_graph(h[metal_mask], batch[metal_mask], n_graphs) if metal_mask.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([mean_h, max_h, metal_feat], dim=1))
        return self.readout(torch.cat([mean_h, max_h], dim=1))

def train_eval(hidden, n_layers, heads, dropout, use_metal_pool, use_fe_weight):
    m = AblationGNN(hidden, n_layers, heads, dropout, use_metal_pool).to(device)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.MSELoss(reduction='none') if use_fe_weight else nn.MSELoss()
    best_va, best_state, patience = float('inf'), None, 0
    yn = y_norm.to(device)
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(tr_idx)); tr_loss = 0; nb = 0
        for s in range(0, len(tr_idx), BATCH):
            bidx = tr_idx[perm[s:s+BATCH]].tolist()
            bg = [g2dev(graphs[i]) for i in bidx]; by = yn[bidx]
            batched, batch_idx = collate_graphs(bg)
            opt.zero_grad(); preds = m(batched, batch_idx)
            if use_fe_weight:
                w = torch.tensor([3.0 if bidx[i] in fe_indices else 1.0 for i in range(len(bidx))], device=device)
                loss = (crit(preds, by).mean(dim=1) * w).mean()
            else:
                loss = crit(preds, by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); tr_loss += loss.item(); nb += 1
        m.eval()
        with torch.no_grad():
            va_preds = []
            for s in range(0, len(va_idx), BATCH):
                bidx = va_idx[s:s+BATCH].tolist(); bg = [g2dev(graphs[i]) for i in bidx]
                batched, batch_idx = collate_graphs(bg); va_preds.append(m(batched, batch_idx))
            vp = torch.cat(va_preds, dim=0); va_loss = nn.MSELoss()(vp, yn[va_idx])
        if va_loss < best_va:
            best_va = va_loss; best_state = {k: v.clone() for k, v in m.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= EARLY_STOP: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        te_preds = []
        for s in range(0, len(te_idx), BATCH):
            bidx = te_idx[s:s+BATCH].tolist(); bg = [g2dev(graphs[i]) for i in bidx]
            batched, batch_idx = collate_graphs(bg); te_preds.append(m(batched, batch_idx))
        tp = torch.cat(te_preds, dim=0).cpu() * y_std + y_mean; tt = y_all[te_idx]
    r2s = [r2_score(tt[:,i].numpy(), tp[:,i].numpy()) for i in range(8)]
    return {'r2': r2s, 'mean_r2': np.mean(r2s), 'best_va': best_va.item(), 'params': sum(p.numel() for p in m.parameters())}

results = {}
print('Ready for ablation experiments')""", "eq_model")

# Cell 4: Ablation A (Heads)
add_code("""print('\\n' + '='*60)
print('ABLATION A: GAT attention heads')
print('='*60)
for h in [4, 8, 12]:
    hidden = 256 if h in [4,8] else 252
    r = train_eval(hidden, SCALE_LAYERS, h, REG_DROPOUT, True, False)
    results[f'heads={h}'] = r
    r2 = r['r2']
    print(f'heads={h}: Mean R2={r["mean_r2"]:.4f}, gap={r2[0]:.4f}, homo={r2[1]:.4f}, lumo={r2[2]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}, params={r["params"]:,}')""", "eq_a")

# Cell 5: Ablation B (Layers)
add_code("""print('\\n' + '='*60)
print('ABLATION B: Network depth')
print('='*60)
for l in [3, 4, 5]:
    r = train_eval(SCALE_HIDDEN, l, GAT_HEADS, REG_DROPOUT, True, False)
    results[f'layers={l}'] = r
    r2 = r['r2']
    print(f'layers={l}: Mean R2={r["mean_r2"]:.4f}, gap={r2[0]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}')""", "eq_b")

# Cell 6: Ablation C (Metal Pooling)
add_code("""print('\\n' + '='*60)
print('ABLATION C: Metal-centric pooling')
print('='*60)
for use_mp in [True, False]:
    r = train_eval(SCALE_HIDDEN, SCALE_LAYERS, GAT_HEADS, REG_DROPOUT, use_mp, False)
    results[f'metal_pool={use_mp}'] = r
    r2 = r['r2']
    print(f'metal_pool={use_mp}: Mean R2={r["mean_r2"]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}')""", "eq_c")

# Cell 7: Ablation D (Fe Weight)
add_code("""print('\\n' + '='*60)
print('ABLATION D: Fe-weighted loss')
print('='*60)
r = train_eval(SCALE_HIDDEN, SCALE_LAYERS, GAT_HEADS, REG_DROPOUT, True, True)
results['fe_weight=True'] = r
r2 = r['r2']
print(f'fe_weight=True: Mean R2={r["mean_r2"]:.4f}, gap={r2[0]:.4f}, dipole={r2[3]:.4f}, metal_q={r2[4]:.4f}')""", "eq_d")

# Cell 8: Summary
add_code("""print('\\n' + '='*80)
print('ABLATION RESULTS')
print('='*80)
print(f'{"Experiment":25s} {"Mean R2":>8s} {"gap":>7s} {"homo":>7s} {"lumo":>7s} {"dipole":>7s} {"metal_q":>8s} {"params":>8s}')
print('-'*80)
for name, r in results.items():
    r2 = r['r2']
    print(f'{name:25s} {r["mean_r2"]:8.4f} {r2[0]:7.4f} {r2[1]:7.4f} {r2[2]:7.4f} {r2[3]:7.4f} {r2[4]:8.4f} {r["params"]:>8,}')

with open('ablation_results.json', 'w') as f:
    json.dump({k: {'mean_r2': v['mean_r2'], 'r2': [float(x) for x in v['r2']], 'params': v['params']} for k, v in results.items()}, f, indent=2)
print('\\nSaved to ablation_results.json')""", "eq_summary")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('ablation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
with open('ablation.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'Built {len(cells)} cells, JSON valid')
