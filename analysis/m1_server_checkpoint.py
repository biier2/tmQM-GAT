# -*- coding: utf-8 -*-
"""训练/测试 SMILES 组成重叠敏感性分析（纯推理，不动模型）
- 重推 SMILES 部署检查点 tmqm_gat_best.pt 的逐条预测（13,547 测试集, CPU）
- 3D: ensemble_predictions.npz 已有集成均值
- 口径1: CSD_code 重叠（应为 0）
- 口径2: canonical SMILES 重叠 -> 剔除重叠行后重算 dipole(及全 8 性质) R2
输出: m1_overlap_sensitivity.json
"""
import sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.metrics import r2_score

device = torch.device('cpu')
OUT_JSON = 'server_ckpt_r2.json'
OUT_NPY = 'server_ckpt_preds.npy'

# ===== 1. 数据 =====
ep = np.load('ensemble_predictions.npz')
y_true = ep['y_true']; xyz_mean = ep['mean']
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
RENAME = {'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
          'Dipole_M': 'dipole', 'Metal_q': 'metal_charge', 'Electronic_E': 'E_elec',
          'Dispersion_E': 'E_disp', 'Polarizability': 'polarizability'}
df_y = df_y.rename(columns=RENAME)
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
df_clean = df_y.dropna(subset=targets).copy()
df_vals = df_clean[targets].values.astype(np.float32)

lookup = {}
for i in range(len(df_vals)):
    lookup[df_vals[i].tobytes()] = i

matched_idx, te_rows = [], []
for r in range(len(y_true)):
    key = y_true[r].astype(np.float32).tobytes()
    csv_i = lookup.get(key)
    if csv_i is None: continue
    matched_idx.append(r); te_rows.append(csv_i)
print(f'Matched {len(te_rows):,}/{len(y_true):,}')

TM_SET = {21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80}
ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES = [1,2,3,12]; B2I = {b:i for i,b in enumerate(BOND_TYPES)}; N_BT = len(BOND_TYPES)

def atom_features(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA.get(z,(0,0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

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
    return {'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':N}

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
    for g in graph_list:
        n = g['x'].size(0)
        x_list.append(g['x']); ei_list.append(g['edge_index'] + offset)
        ea_list.append(g['edge_attr'])
        batch_list.append(torch.full((n,), len(batch_list), dtype=torch.long))
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

HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12
class SMILES_GAT(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, dropout=0.10, heads=GAT_HEADS):
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
            metal_mask = (x[:, 7] > 0.5)
            metal_feat = h[metal_mask].mean(dim=0) if metal_mask.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            return self.readout(torch.cat([h.mean(0), h.max(0).values, metal_feat])).unsqueeze(0)
        n_graphs = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n_graphs)
        max_h = scatter_max_per_graph(h, batch, n_graphs)
        metal_mask = (x[:, 7] > 0.5)
        metal_feat = scatter_mean_per_graph(h[metal_mask], batch[metal_mask], n_graphs) if metal_mask.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, metal_feat], dim=1))

model = SMILES_GAT(node_dim=8, edge_dim=6, hidden=HIDDEN, output_dim=8, heads=GAT_HEADS)
state = torch.load('d:/ml/server_check/tmqm_gat_best.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval()
print('SMILES GAT loaded from SERVER tmqm_gat_best.pt')

y_mean_smiles = np.array([0.108504, -0.197593, -0.089089, 5.799450, 0.162882,
                          -2987.288818, -0.142252, 397.104523], dtype=np.float32)
y_std_smiles  = np.array([0.033575, 0.053825, 0.054517, 3.927489, 0.782914,
                          1634.043335, 0.070115, 152.449799], dtype=np.float32)

te_smiles = [df_clean.iloc[i]['SMILES'] for i in te_rows]
print(f'Building {len(te_smiles):,} SMILES graphs...')
graphs = []; bad = []
for i, smi in enumerate(te_smiles):
    g = smiles_to_graph(smi)
    if g is None: bad.append(i)
    else: graphs.append(g)
print(f'graphs: {len(graphs):,} (bad {len(bad)})')

preds = []; bs = 128
with torch.no_grad():
    for s in range(0, len(graphs), bs):
        bg = graphs[s:s+bs]
        batched, batch_idx = collate_graphs(bg)
        pn = model(batched, batch_idx).cpu().numpy()
        preds.append(pn * y_std_smiles + y_mean_smiles)
        if (s//bs) % 10 == 0: print(f'  batch {s//bs}/{len(graphs)//bs+1}')
smiles_preds = np.concatenate(preds, axis=0)
print(f'smiles_preds: {smiles_preds.shape}')

# ===== 2. 重叠口径 =====
def canon(s):
    if not isinstance(s, str) or s == '': return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

te_canon = [canon(s) for s in te_smiles]
rest_mask = np.ones(len(df_clean), bool); rest_mask[te_rows] = False
rest_set = set()
for s in df_clean.iloc[rest_mask]['SMILES']:
    c = canon(s)
    if c: rest_set.add(c)
print(f'rest canonical set: {len(rest_set):,}')

te_csd = df_clean.iloc[te_rows]['CSD_code'].values
rest_csd = set(df_clean.iloc[rest_mask]['CSD_code'].values)
csd_overlap = [i for i in range(len(te_rows)) if te_csd[i] in rest_csd]
smi_overlap = [i for i in range(len(te_rows)) if te_canon[i] is not None and te_canon[i] in rest_set]
print(f'CSD overlap rows: {len(csd_overlap)}; SMILES overlap rows: {len(smi_overlap)}')

# ===== 3. 对齐预测与标签 =====
keep = np.array([i for i in range(len(te_smiles)) if i not in bad])
y_t = y_true[np.array(matched_idx)[keep]]
x_t = xyz_mean[np.array(matched_idx)[keep]]
s_t = smiles_preds
assert len(y_t) == len(s_t) == 13547, (len(y_t), len(s_t))
np.save(OUT_NPY, s_t)
print(f'saved {OUT_NPY}; N = {len(y_t):,}')

overlap_set = set(smi_overlap)
ov = np.array([i in overlap_set for i in range(len(y_t))])
n_ov = ov.sum()
print(f'SMILES-overlap subset: {n_ov} / {len(y_t)} ({n_ov/len(y_t)*100:.2f}%)')

res = {'N_total': int(len(y_t)), 'N_overlap': int(n_ov), 'N_nonoverlap': int(len(y_t)-n_ov)}
for prop_i, prop in enumerate(targets):
    yv = y_t[:, prop_i]
    full_s = r2_score(yv, s_t[:, prop_i]); full_x = r2_score(yv, x_t[:, prop_i])
    no_s = r2_score(yv[~ov], s_t[~ov, prop_i]); no_x = r2_score(yv[~ov], x_t[~ov, prop_i])
    o_s = r2_score(yv[ov], s_t[ov, prop_i]); o_x = r2_score(yv[ov], x_t[ov, prop_i])
    res[prop] = {'full_SMILES': round(full_s,4), 'full_3D': round(full_x,4),
                 'nonoverlap_SMILES': round(no_s,4), 'nonoverlap_3D': round(no_x,4),
                 'overlap_SMILES': round(o_s,4), 'overlap_3D': round(o_x,4)}
    print(f'{prop:14s} full S{full_s:.4f}/X{full_x:.4f} | 非重叠 S{no_s:.4f}/X{no_x:.4f} | 重叠 S{o_s:.4f}/X{o_x:.4f}')

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print('done ->', OUT_JSON)
