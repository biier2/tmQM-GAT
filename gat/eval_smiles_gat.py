"""SMILES GAT 在 3D 测试集上重评估 — 公平比较"""
import numpy as np, pandas as pd, torch, torch.nn as nn, gzip, os, json
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ===== 加载 split =====
split = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_idx = split['te_idx']
print(f'Test set: {len(te_idx):,} samples (from ensemble CV)')

# ===== 加载目标值 =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'
})
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']

# ===== SMILES GAT 模型定义 (与 tmQM_Fe.ipynb 完全相同) =====
SCALE_HIDDEN = 384; SCALE_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10

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

class TMetalGNN(nn.Module):
    def __init__(self, node_dim=8, edge_dim=6, hidden=SCALE_HIDDEN, n_layers=SCALE_LAYERS,
                 output_dim=8, dropout=DROPOUT, heads=GAT_HEADS):
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
        n = int(batch.max().item()) + 1
        mean_h = scatter_mean_per_graph(h, batch, n)
        max_h = scatter_max_per_graph(h, batch, n)
        mm = (x[:,7] > 0.5)
        mf = scatter_mean_per_graph(h[mm], batch[mm], n) if mm.any() else torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([mean_h, max_h, mf], dim=1))

# ===== 加载checkpoint =====
print('Loading SMILES GAT checkpoint...')
ckpt = torch.load('tmqm_gat_best.pt', map_location='cpu', weights_only=False)
model = TMetalGNN(node_dim=8, edge_dim=6).to(device)
model.load_state_dict(ckpt)
model.eval()
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ===== 加载 XYZ+BO (与 ensemble CV 完全相同的解析逻辑, 按 csv_idx 对齐 SMILES) =====
import gzip
from rdkit import Chem; from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym,0)
def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

def parse_one_full(xyz_p, bo_p, start_idx=0):
    """Full XYZ+BO parse — same as ensemble CV. Returns (atoms, coords, bo, csv_idx) per complex."""
    mols = []
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
            mols.append({"atoms": np.array(atoms), "csv_idx": mol_idx})
    return mols

print('Parsing XYZ+BO to get csv_idx alignment...')
all_m = []
for part in [1,2,3]:
    m = parse_one_full(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz", f"data_tmQM/local_vB/tmQM_X{part}.BO.gz", start_idx=len(all_m))
    all_m.extend(m)
    print(f'  Part {part}: {len(m)}')
print(f'Total: {len(all_m)} complexes (same as ensemble CV)')

# ===== Build SMILES-only graphs (matched by csv_idx — identical to ensemble CV logic) =====
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM = 4
smiles_by_idx = df_y['SMILES'].tolist()
csv_labels = df_y[targets].values.astype(np.float32)

def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p,g = ELEMENT_DATA_SM.get(z,(0,0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

print('Building SMILES-only graphs (matched by csv_idx)...')
graphs, y_vals, failed = [], [], 0
for m in tqdm(all_m, desc='Graphs'):
    csv_i = m['csv_idx']
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': failed += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: failed += 1; continue
    N = mol.GetNumAtoms()
    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    sm_z = [xf[i][0] for i in range(N)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml]); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0,6)
    graphs.append({'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':N})
    y_vals.append(csv_labels[csv_i])

y_all = torch.tensor(y_vals, dtype=torch.float32)
mask = ~(torch.isnan(y_all).any(dim=1) | torch.isinf(y_all).any(dim=1))
graphs = [g for i,g in enumerate(graphs) if mask[i]]
y_all = y_all[mask]
print(f'Built {len(graphs):,} graphs ({failed} failed, {len(graphs)} after NaN)')

# SMILES GAT 训练用的 y_mean/y_std (从全量 108,541 CSV 计算, 与训练时的 100,849 几乎一致)
y_mean_gat = torch.tensor([0.108505, -0.197591, -0.089089, 5.799378, 0.162882, -2987.275391, -0.142229, 397.123474])
y_std_gat  = torch.tensor([0.033575,  0.053824,  0.054516, 3.927464, 0.782902,  1634.063721,  0.070114, 152.458313])
print(f'y_mean: {y_mean_gat.tolist()}')
print(f'y_std:  {y_std_gat.tolist()}')

# Pre-index for fast collation
g_x = [g['x'] for g in graphs]
g_ei = [g['edge_index'] for g in graphs]
g_ea = [g['edge_attr'] for g in graphs]

def collate_fast(indices):
    x_list = [g_x[i] for i in indices]
    ei_list = [g_ei[i] for i in indices]
    ea_list = [g_ea[i] for i in indices]
    batch_list = []
    offset = 0
    for k, x in enumerate(x_list):
        n = x.size(0)
        batch_list.append(torch.full((n,), k, dtype=torch.long))
        ei_list[k] = ei_list[k] + offset; offset += n
    return {
        'x': torch.cat(x_list, dim=0).to(device),
        'edge_index': torch.cat(ei_list, dim=1).to(device),
        'edge_attr': torch.cat(ea_list, dim=0).to(device),
    }, torch.cat(batch_list, dim=0).to(device)

# ===== Inference on test set =====
bs = 128
y_test_true = y_all[te_idx]
preds = []

print(f'\nRunning inference on test set...')
with torch.no_grad():
    for s in tqdm(range(0, len(te_idx), bs), desc='Inference'):
        bidx = te_idx[s:s+bs].tolist()
        batched, batch_idx = collate_fast(bidx)
        with autocast('cuda'):
            p = model(batched, batch_idx).float().cpu()
        preds.append(p)
preds = torch.cat(preds, dim=0)
# 反归一化: SMILES GAT 输出的是 z-score 值, 需转回原始尺度
preds_orig = preds * y_std_gat.unsqueeze(0) + y_mean_gat.unsqueeze(0)
preds_np = preds_orig.numpy()
yt = y_test_true.numpy()

# ===== Results =====
print(f'\n{"="*60}')
print('SMILES GAT ON 3D TEST SET (95,152 sample split)')
print(f'{"="*60}')
print(f'{"Property":16s} {"R²":>8s} {"MAE":>12s}')
print('-'*40)
results = []
for i, prop in enumerate(targets):
    r2 = r2_score(yt[:,i], preds_np[:,i])
    mae = mean_absolute_error(yt[:,i], preds_np[:,i])
    print(f'{prop:16s} {r2:8.4f} {mae:12.4f}')
    results.append({'Property': prop, 'R2': r2, 'MAE': mae})

mean_r2 = np.mean([r['R2'] for r in results])
print(f'\nSMILES GAT Mean R²: {mean_r2:.4f}')

# Save
with open('smiles_gat_eval.json', 'w') as f:
    json.dump({'results': results, 'mean_r2': mean_r2}, f, indent=2)
print('Saved smiles_gat_eval.json')
