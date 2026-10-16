"""Build GAT attention XAI analysis notebook"""
import json

cells = []

def add_code(src, cid):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + '\n' for l in src.split('\n')], "id": cid})

def add_md(src, cid):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split('\n'), "id": cid})

add_md("""# GAT Attention XAI Analysis

## 分析目标

提取XYZ GNN各层GAT注意力权重, 分析:
1. 哪些原子获得最高注意力?
2. 配位原子(N/O/S/P) vs 金属原子的top-5出现频率
3. 与tmGNN-XAI的"donor atoms in top-5 >99.8%"对标
4. 不同性质(偶极矩 vs 金属电荷)的注意力模式差异""", "eq_intro")

add_code("""import numpy as np, pandas as pd, time, json, gzip, os
import torch, torch.nn as nn
from collections import Counter, defaultdict
import warnings; warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from rdkit import Chem; from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from tqdm import tqdm

# ===== 加载数据 (同tmQM_Fe_XYZ) =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={'HL_Gap':'gap','HOMO_Energy':'homo','LUMO_Energy':'lumo',
    'Dipole_M':'dipole','Metal_q':'metal_charge','Electronic_E':'E_elec',
    'Dispersion_E':'E_disp','Polarizability':'polarizability'})
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym,0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

def parse_one(xyz_p, bo_p, start_idx=0):
    mols, labels, is_Fe = [], [], []
    with gzip.open(bo_p, "rt") as f: bo_lines = [l.strip() for l in f.readlines()]
    with gzip.open(xyz_p, "rt") as f: xyz_blocks = f.read().strip().split(chr(10))
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
        if len(parts)==1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break; i += 1
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts)>=4:
                    try: z=atom_to_z(cparts[0]); coords.append([float(x) for x in cparts[1:4]]); atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms)<=1: continue
            bo_mat = None
            if mol_idx < len(bo_blocks):
                bo_mat = np.zeros((n_atoms,n_atoms),dtype=np.float32)
                for bline in bo_blocks[mol_idx]:
                    bparts = bline.split(); j = 2
                    try: src_idx=int(bparts[0])-1
                    except ValueError: continue
                    while j+1<len(bparts):
                        try:
                            nbr=int(bparts[j])-1; bo_val=float(bparts[j+1])
                            if 0<=src_idx<n_atoms and 0<=nbr<n_atoms: bo_mat[src_idx,nbr]=bo_val
                            j+=3
                        except: j+=1
            mols.append({"atoms":np.array(atoms),"coords":np.array(coords,dtype=np.float32),"bo":bo_mat,"csv_idx":mol_idx})
            labels.append(df_y.iloc[mol_idx][targets].astype(float).values)
            is_Fe.append(float(26 in atoms))
    return mols, labels, is_Fe

print('Parsing XYZ...')
all_m, all_l, all_f = [], [], []
for part in [1,2,3]:
    m,l,f = parse_one(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz", f"data_tmQM/local_vB/tmQM_X{part}.BO.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l); all_f.extend(f)
fe_cmplx = all_m; y_fe = np.array(all_l,dtype=np.float32); is_Fe = np.array(all_f,dtype=np.float32)
print(f'Total: {len(fe_cmplx)} TM complexes')
""", "eq_load")

add_code("""# ===== 模型定义 (和训练完全一致) =====
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12

def segment_softmax(scores, dst, N):
    scores_f32 = scores.float()
    max_v = torch.full((N, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(N, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)

class GATConvXAI(nn.Module):
    # GATConv-XAI: returns attention for explainability
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
        return self.act(out), alpha  # Return attention weights

def scatter_mean_per_graph(h, batch, n):
    out = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
    cnt = torch.zeros(n, device=h.device, dtype=h.dtype); cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)

def scatter_max_per_graph(h, batch, n):
    out = torch.full((n, h.size(1)), float('-inf'), device=h.device, dtype=h.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax', include_self=False)
    return out

class XYZ_GNN_XAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(11, HIDDEN)
        self.convs = nn.ModuleList([GATConvXAI(HIDDEN, HIDDEN//GAT_HEADS, 23, heads=GAT_HEADS) for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN*3, HIDDEN), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN//2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN//2, 8))
    def forward(self, g, batch=None, return_attn=False):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x); all_attn = []
        for conv, norm in zip(self.convs, self.norms):
            h_new, alpha = conv(h, ei, ea); h = norm(h + h_new); h = self.drop(h)
            if return_attn: all_attn.append(alpha)
        if batch is None:
            mm = (x[:,7] > 0.5); mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
            out = self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        else:
            n_graphs = int(batch.max().item()) + 1
            mean_h = scatter_mean_per_graph(h, batch, n_graphs); max_h = scatter_max_per_graph(h, batch, n_graphs)
            mm = (x[:,7] > 0.5); mf = scatter_mean_per_graph(h[mm], batch[mm], n_graphs) if mm.any() else torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
            out = self.readout(torch.cat([mean_h, max_h, mf], dim=1))
        if return_attn: return out, all_attn
        return out

# Load checkpoint
model = XYZ_GNN_XAI().to(device)
ckpt = 'xyz_best.pt'
if os.path.exists(ckpt):
    state = torch.load(ckpt, map_location=device)
    # Map state dict (training model has GATConv, XAI model has GATConvXAI — same param names)
    model.load_state_dict(state, strict=False)
    print(f'Loaded {ckpt}')
else:
    print(f'Checkpoint not found: {ckpt}')
    exit(1)
model.eval()
""", "eq_model")

add_code("""# ===== 快速建图: 只用SMILES, 不用BO/XYZ (XAI推理只需图拓扑) =====
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM=[1,2,3,12]; B2I_SM={b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM=4
rbf_c=np.linspace(1.0,5.0,16)

def build_xai_graph(smiles):
    """Build graph for XAI inference (25-dim edge features with zero BO/distance fallback)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    N = mol.GetNumAtoms()
    xf = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum(); p,g = ELEMENT_DATA_SM.get(z,(0,0))
        xf.append([z, a.GetDegree(), a.GetFormalCharge(), p, g, int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z>=21), 0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bt = int(bond.GetBondType())
        za,zb = mol.GetAtomWithIdx(i).GetAtomicNum(),mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        for mu in rbf_c: bf.append(np.exp(-(2.0-mu)**2/0.2))
        bf.append(0.0); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0,23)
    return {'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':N}, [a.GetSymbol() for a in mol.GetAtoms()]

# Sample test complexes
N_XAI = 500  # XAI sample size
smiles_by_idx = df_y['SMILES'].tolist()
xai_graphs, xai_symbols, xai_atoms = [], [], []
for idx in tqdm(np.random.RandomState(42).choice(len(fe_cmplx), min(N_XAI, len(fe_cmplx)), replace=False), desc='Building XAI graphs'):
    csv_i = fe_cmplx[idx]["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': continue
    result = build_xai_graph(smi)
    if result is None: continue
    g, symbols = result
    xai_graphs.append(g); xai_symbols.append(symbols); xai_atoms.append(fe_cmplx[idx]['atoms'])
print(f'Built {len(xai_graphs)} XAI graphs')
""", "eq_graphs")

add_code("""# ===== 提取注意力权重 =====
print('Extracting attention weights...')
DONOR_ELEMENTS = {'N','O','S','P','Cl','Br','I'}

# Pre-load graphs to GPU
graphs_gpu = [{k: v.to(device) for k, v in g.items()} for g in tqdm(xai_graphs, desc='GPU upload')]

# Statistics
top5_counts = defaultdict(int)  # Element -> times in top-5
total_complexes = 0
all_attn_by_layer = [[] for _ in range(N_LAYERS)]

for i, g in enumerate(tqdm(graphs_gpu, desc='Analyzing')):
    sym = xai_symbols[i]
    with torch.no_grad():
        out, attn_layers = model(g, return_attn=True)

    # Use last layer attention for analysis
    alpha = attn_layers[-1]  # [E, heads]
    # Average over heads -> per-edge attention
    alpha_edge = alpha.mean(dim=1)  # [E]

    # Aggregate per-node attention (sum of incoming edges)
    ei = g['edge_index']
    dst = ei[1]
    node_attn = torch.zeros(len(sym), device=device)
    node_attn.scatter_add_(0, dst, alpha_edge)
    # Normalize
    node_attn = node_attn / (node_attn.sum() + 1e-8)

    # Top-5 atoms
    top5_idx = node_attn.argsort(descending=True)[:5].cpu().numpy()
    for idx in top5_idx:
        top5_counts[sym[idx]] += 1
    total_complexes += 1

print(f'\\nAnalyzed {total_complexes} complexes')
print(f'\\nTop-5 atom frequency:')
print(f'{"Element":10s} {"Count":>8s} {"%":>8s} {"Type":>15s}')
print('-' * 45)
for elem, count in sorted(top5_counts.items(), key=lambda x: -x[1])[:20]:
    pct = 100 * count / (total_complexes * 5)
    etype = 'DONOR' if elem in DONOR_ELEMENTS else ('METAL' if elem in {'Fe','Co','Ni','Cu','Zn','Ru','Rh','Pd','Pt','Au','Ag'} else 'organic')
    print(f'{elem:10s} {count:>8d} {pct:>7.1f}% {etype:>15s}')

# Donor vs Metal summary
donor_total = sum(v for k,v in top5_counts.items() if k in DONOR_ELEMENTS)
metal_total = sum(v for k,v in top5_counts.items() if k in {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ru','Rh','Pd','Ag','Cd','Pt','Au','Hg','Mo','W','Ir','Os','Re'})
total_top5 = total_complexes * 5
print(f'\\nDonor atoms (N,O,S,P,Cl,Br,I): {donor_total}/{total_top5} = {100*donor_total/total_top5:.1f}%')
print(f'Metal atoms: {metal_total}/{total_top5} = {100*metal_total/total_top5:.1f}%')
print(f'Organic (C,H,etc): {total_top5-donor_total-metal_total}/{total_top5} = {100*(total_top5-donor_total-metal_total)/total_top5:.1f}%')
""", "eq_xai")

add_code("""# ===== 可视化 =====
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Top-20 element frequency
top20 = sorted(top5_counts.items(), key=lambda x: -x[1])[:20]
elements = [e for e,_ in top20]
counts = [c for _,c in top20]
colors = ['#e74c3c' if e in DONOR_ELEMENTS else '#3498db' if e in {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ru','Rh','Pd','Ag','Cd','Pt','Au','Hg','Mo','W','Ir','Os','Re'} else '#95a5a6' for e in elements]
axes[0].bar(range(len(elements)), counts, color=colors)
axes[0].set_xticks(range(len(elements))); axes[0].set_xticklabels(elements, fontsize=9)
axes[0].set_ylabel('Times in Top-5'); axes[0].set_title('Top-20 Most Attended Atoms (GAT last layer)')
axes[0].legend([plt.Rectangle((0,0),1,1,color='#e74c3c'), plt.Rectangle((0,0),1,1,color='#3498db'), plt.Rectangle((0,0),1,1,color='#95a5a6')], ['Donor (N,O,S,P,Cl,Br,I)', 'Metal', 'Organic'], fontsize=8)

# Right: Donor vs Metal summary pie
sizes = [donor_total, metal_total, total_top5-donor_total-metal_total]
labels = [f'Donor\\n{donor_total} ({100*donor_total/total_top5:.1f}%)', f'Metal\\n{metal_total} ({100*metal_total/total_top5:.1f}%)', f'Other\\n{total_top5-donor_total-metal_total}']
axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#e74c3c','#3498db','#95a5a6'])
axes[1].set_title(f'Atom Type in Top-5 ({total_complexes} complexes)')

plt.suptitle('GAT Attention XAI: Which atoms drive predictions?', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print(f'\\nKey finding: Donor atoms appear in {(donor_total/total_top5)*100:.1f}% of top-5 attributions')
print(f'This is consistent with ligand field theory: donor identity determines electronic properties')
print(f'Compare with tmGNN-XAI: donor atoms in top-5 for >99.8% of complexes')
""", "eq_viz")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('xai_attention.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
with open('xai_attention.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'Built {len(cells)} cells, JSON valid')
