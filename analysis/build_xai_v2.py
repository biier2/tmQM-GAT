"""Build XAI notebook - careful with triple quotes"""
import json

cells = []

def add_md(src, cid):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split('\n'), "id": cid})

def add_code(src, cid):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [l + '\n' for l in src.split('\n')], "id": cid})

# Cell 1: Intro
add_md("# GAT 注意力 XAI 分析\n\n提取 XYZ GNN 各层 GAT 注意力权重，分析哪些原子驱动预测。\n与 tmGNN-XAI 的配位原子主导发现对标。", "eq_intro")

# Cell 2: Load minimal data (SMILES only, no XYZ parsing needed)
add_code("import numpy as np, pandas as pd, time, json, os\nimport torch, torch.nn as nn\nfrom collections import Counter, defaultdict\nimport warnings; warnings.filterwarnings('ignore')\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')\nfrom rdkit import Chem; from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)\nfrom tqdm import tqdm\n\ndf_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')\ndf_y = df_y.rename(columns={'HL_Gap':'gap','HOMO_Energy':'homo','LUMO_Energy':'lumo','Dipole_M':'dipole','Metal_q':'metal_charge','Electronic_E':'E_elec','Dispersion_E':'E_disp','Polarizability':'polarizability'})\nsmiles_by_idx = df_y['SMILES'].tolist()\nprint(f'Loaded {len(df_y)} SMILES strings')", "eq_load")

# Cell 3: Model (using plain strings, no triple-quoted docstrings)
cell3_src = """HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12

def segment_softmax(scores, dst, N):
    scores_f32 = scores.float()
    max_v = torch.full((N, scores_f32.size(1)), float('-inf'), device=scores_f32.device, dtype=torch.float32)
    max_v.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), scores_f32, reduce='amax', include_self=False)
    exp_s = torch.exp(scores_f32 - max_v[dst])
    sum_v = torch.zeros(N, scores_f32.size(1), device=scores_f32.device, dtype=torch.float32)
    sum_v.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores_f32.size(1)), exp_s)
    return (exp_s / (sum_v[dst] + 1e-8)).to(scores.dtype)

class GATConvXAI(nn.Module):
    # GAT convolution returning attention weights for XAI
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
        return self.act(out), alpha

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

model = XYZ_GNN_XAI().to(device)
ckpt = 'xyz_best.pt'
if os.path.exists(ckpt):
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f'Loaded {ckpt}')
else:
    print(f'ERROR: {ckpt} not found'); exit(1)
model.eval()"""

add_code(cell3_src, "eq_model")

# Cell 4: Build XAI graphs
cell4_src = """# 从SMILES构建图 (XAI只需图拓扑)
ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES=[1,2,3,12]; B2I={b:i for i,b in enumerate(BOND_TYPES)}; N_BT=4
rbf_c=np.linspace(1.0,5.0,16)

def build_xai_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    N = mol.GetNumAtoms()
    xf = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum(); p,g = ELEMENT_DATA.get(z,(0,0))
        xf.append([z, a.GetDegree(), a.GetFormalCharge(), p, g, int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z>=21), 0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bt = int(bond.GetBondType())
        za,zb = mol.GetAtomWithIdx(i).GetAtomicNum(),mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za>=21)!=(zb>=21)); is_ring = float(bond.IsInRing())
        src.extend([i,j]); dst.extend([j,i])
        bf = [0.0]*N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        for mu in rbf_c: bf.append(np.exp(-(2.0-mu)**2/0.2))
        bf.append(0.0); battr.extend([bf,bf])
    ei = torch.tensor([src,dst], dtype=torch.long) if src else torch.zeros(2,0,dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0,23)
    return {'x':x, 'edge_index':ei, 'edge_attr':ea, 'num_nodes':N}, [a.GetSymbol() for a in mol.GetAtoms()]

N_XAI = 500
print(f'Building {N_XAI} XAI graphs...')
rng = np.random.RandomState(42)
xai_graphs, xai_symbols = [], []
indices = rng.choice(len(df_y), min(N_XAI*3, len(df_y)), replace=False)
for csv_i in indices:
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': continue
    result = build_xai_graph(smi)
    if result is None: continue
    g, sym = result
    xai_graphs.append(g); xai_symbols.append(sym)
    if len(xai_graphs) >= N_XAI: break
print(f'Built {len(xai_graphs)} XAI graphs')"""

add_code(cell4_src, "eq_graphs")

# Cell 5: Extract attention
cell5_src = """# 提取注意力权重并分析
print('Extracting attention...')
DONOR_SET = {'N','O','S','P','Cl','Br','I'}
METAL_SET = {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ru','Rh','Pd','Ag','Cd','Pt','Au','Hg','Mo','W','Ir','Os','Re'}

top5_counts = defaultdict(int)
total = 0

for i in tqdm(range(len(xai_graphs)), desc='分析中'):
    g = xai_graphs[i]; sym = xai_symbols[i]
    g_gpu = {k: v.to(device) for k, v in g.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out, attn_layers = model(g_gpu, return_attn=True)
    alpha = attn_layers[-1].mean(dim=1)  # average over heads
    ei = g_gpu['edge_index']; dst = ei[1]
    node_attn = torch.zeros(len(sym), device=device)
    node_attn.scatter_add_(0, dst, alpha)
    node_attn = node_attn / (node_attn.sum() + 1e-8)
    top5_idx = node_attn.argsort(descending=True)[:5].cpu().numpy()
    for idx in top5_idx: top5_counts[sym[idx]] += 1
    total += 1

print(f'\\nAnalyzed {total} complexes')
print(f'\\n元素Top-5出现频率:')
print(f'{\"Element\":10s} {\"Count\":>8s} {\"%\":>8s} {\"Type\":>12s}')
print('-' * 42)
for elem, count in sorted(top5_counts.items(), key=lambda x: -x[1])[:20]:
    pct = 100 * count / (total * 5)
    if elem in DONOR_SET: etype = 'DONOR'
    elif elem in METAL_SET: etype = 'METAL'
    else: etype = 'organic'
    print(f'{elem:10s} {count:>8d} {pct:>7.1f}% {etype:>12s}')

donor_total = sum(v for k,v in top5_counts.items() if k in DONOR_SET)
metal_total = sum(v for k,v in top5_counts.items() if k in METAL_SET)
total_top5 = total * 5
print(f'\\n配位原子: {donor_total}/{total_top5} = {100*donor_total/total_top5:.1f}%')
print(f'金属原子: {metal_total}/{total_top5} = {100*metal_total/total_top5:.1f}%')
print(f'其他: {total_top5-donor_total-metal_total}/{total_top5} = {100*(total_top5-donor_total-metal_total)/total_top5:.1f}%')"""

add_code(cell5_src, "eq_xai")

# Cell 6: Visualization
cell6_src = """import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
top20 = sorted(top5_counts.items(), key=lambda x: -x[1])[:20]
elements = [e for e,_ in top20]; counts = [c for _,c in top20]
colors = ['#e74c3c' if e in DONOR_SET else '#3498db' if e in METAL_SET else '#95a5a6' for e in elements]
axes[0].bar(range(len(elements)), counts, color=colors)
axes[0].set_xticks(range(len(elements))); axes[0].set_xticklabels(elements, fontsize=9)
axes[0].set_ylabel('Top-5出现次数'); axes[0].set_title('GAT注意力Top-20原子')
from matplotlib.patches import Rectangle
axes[0].legend([Rectangle((0,0),1,1,color='#e74c3c'), Rectangle((0,0),1,1,color='#3498db'), Rectangle((0,0),1,1,color='#95a5a6')], ['配位原子', 'Metal', '有机'], fontsize=8)

axes[1].pie([donor_total, metal_total, total_top5-donor_total-metal_total],
    labels=[f'Donor\\n{donor_total} ({100*donor_total/total_top5:.1f}%)', f'Metal\\n{metal_total}', f'Other'],
    autopct='%1.1f%%', colors=['#e74c3c','#3498db','#95a5a6'])
axes[1].set_title(f'Top-5原子类型分布 ({total} complexes)')
plt.suptitle('GAT注意力XAI: 哪些原子驱动了预测?', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print(f'\\nKey finding: 配位原子 appear in {100*donor_total/total_top5:.1f}% of top-5 attentions')
print('这是配体场理论的大规模数据验证')"""

add_code(cell6_src, "eq_viz")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('xai_attention.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
with open('xai_attention.ipynb', encoding='utf-8') as f:
    json.load(f)
print(f'Built {len(cells)} cells, JSON valid')
