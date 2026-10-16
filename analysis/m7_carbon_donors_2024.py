# -*- coding: utf-8 -*-
"""M7: 碳供体 (NHC 卡宾 C / CO 羰基 C / 其他 M-C) 扩展 donor 集合重算
协议 (复刻 xai_rigorous.ipynb):
  - XAI 子集: RandomState(42) 全量排列建图至 20,000 (11d 节点/23d 边, RBF 占位 2.0A, MBO 0)
  - 模型: xyz_best.pt = XYZ_GNN_XAI (11->384, 5xGATConv 12heads 32d, edge 23d, readout 3x384->8)
  - 注意力: 第5层 alpha.mean(12) -> scatter 到目标节点 -> 归一化 -> top-5
  - 扰动: 2000 子集 (RandomState(123).choice), 逐原子零掩码, |Δpred| 求和 8 性质
扩展: DONOR_EXT = {N,O,S,P,Cl,Br,I} ∪ 金属键合 C (与 TM 成键的 C);
      C 分类: 羰基 C (C-O 多重键) / NHC C (芳香 5 环且邻 N) / 其他 M-C
先跑经典集合 sanity check (目标 20.5% / 73.4% / 4.3% / 19.76%)
"""
import sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from tqdm import tqdm

device = torch.device('cpu')
torch.set_num_threads(16)

TM_SET = {21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80}
DONOR_CLS = {'N','O','S','P','Cl','Br','I'}
METAL_SYM = {'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ru','Rh','Pd','Ag','Cd','Pt','Au','Hg','Mo','W','Ir','Os','Re'}

ELEMENT_DATA = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES=[1,2,3,12]; B2I={b:i for i,b in enumerate(BOND_TYPES)}; N_BT=4
rbf_c = np.linspace(1.0, 5.0, 16)

def build_xai_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    xf = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum(); p, g = ELEMENT_DATA.get(z, (0, 0))
        xf.append([z, a.GetDegree(), a.GetFormalCharge(), p, g, int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z >= 21), 0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bt = int(bond.GetBondType())
        za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * N_BT
        if bt in B2I: bf[B2I[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        for mu in rbf_c: bf.append(np.exp(-(2.0 - mu) ** 2 / 0.2))
        bf.append(0.0); battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 23)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    # 碳供体分类 (按原子)
    atom_cls = [None] * len(syms)
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    nbr = [[] for _ in syms]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        nbr[i].append((j, bond.GetBondTypeAsDouble())); nbr[j].append((i, bond.GetBondTypeAsDouble()))
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if zs[i] == 6 and any(zs[j] in TM_SET for j, _ in nbr[i]):
            # 羰基: C-O 多重键
            if any(zs[j] == 8 and btype >= 2.0 for j, btype in nbr[i]):
                atom_cls[i] = 'carbonyl_C'
            elif a.GetIsAromatic() and a.IsInRingSize(5) and any(zs[j] == 7 for j, _ in nbr[i]):
                atom_cls[i] = 'nhc_C'
            else:
                atom_cls[i] = 'other_MC'
    return {'x': x, 'edge_index': ei, 'edge_attr': ea, 'num_nodes': len(syms)}, syms, atom_cls

# ===== 1. 重建 20,000 子集 =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
smiles_by_idx = df_y['SMILES'].tolist()
N_XAI = 20000
rng = np.random.RandomState(42)
indices = rng.permutation(len(df_y))
xai_graphs, xai_symbols, xai_cls = [], [], []
built = 0
for csv_i in indices:
    if built >= N_XAI: break
    smi = smiles_by_idx[csv_i]
    if not isinstance(smi, str) or smi == '': continue
    res = build_xai_graph(smi)
    if res is None: continue
    g, sym, cls = res
    xai_graphs.append(g); xai_symbols.append(sym); xai_cls.append(cls); built += 1
print(f'built {built:,} graphs')

# ===== 2. 模型 =====
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
    def __init__(self, in_dim, out_dim_per_head, edge_dim=0, heads=4):
        super().__init__(); self.heads, self.out_dim = heads, out_dim_per_head
        total_dim = out_dim_per_head * heads
        self.node_lin = nn.Linear(in_dim, total_dim, bias=False)
        self.attn_src = nn.Linear(total_dim, heads, bias=False)
        self.attn_dst = nn.Linear(total_dim, heads, bias=False)
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; N = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0: attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out), alpha

class XYZ_GNN_XAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(11, HIDDEN)
        self.convs = nn.ModuleList([GATConvXAI(HIDDEN, HIDDEN // GAT_HEADS, 23, heads=GAT_HEADS) for _ in range(N_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.drop = nn.Dropout(0.10)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HIDDEN // 2, 8))
    def forward(self, g, return_attn=False):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x); all_attn = []
        for conv, norm in zip(self.convs, self.norms):
            h_new, alpha = conv(h, ei, ea); h = norm(h + h_new); h = self.drop(h)
            if return_attn: all_attn.append(alpha)
        mm = (x[:, 7] > 0.5)
        mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
        out = self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)
        if return_attn: return out, all_attn
        return out

model = XYZ_GNN_XAI().to(device)
state = torch.load('xyz_best.pt', map_location='cpu')
missing = [k for k in model.state_dict() if k not in state]
unexpected = [k for k in state if k not in model.state_dict()]
model.load_state_dict(state, strict=False)
model.eval()
print(f'xyz_best.pt loaded; missing={missing[:3]} unexpected={unexpected[:3]}')

# ===== 3. 注意力分析 (per-complex, 同 notebook) =====
top5_counts = defaultdict(int)          # 元素 -> 次数 (经典)
top5_ext = defaultdict(int)             # 元素/类别 -> 次数 (扩展: M-C 单独计数)
layerwise_donor_frac = defaultdict(list)
layerwise_donor_ext = defaultdict(list)
n_mc_complexes = 0
mc_carbon_counts = Counter()

for i in tqdm(range(len(xai_graphs)), desc='attention'):
    g = xai_graphs[i]; sym = xai_symbols[i]; cls = xai_cls[i]
    if any(c is not None for c in cls): n_mc_complexes += 1
    for c in cls:
        if c is not None: mc_carbon_counts[c] += 1
    with torch.no_grad():
        out, attn_layers = model(g, return_attn=True)
    for layer_idx, alpha in enumerate(attn_layers):
        ei = g['edge_index']; dst = ei[1]
        node_attn = torch.zeros(len(sym), device=device)
        node_attn.scatter_add_(0, dst, alpha.mean(dim=1))
        node_attn = node_attn / (node_attn.sum() + 1e-8)
        na = node_attn.cpu().numpy()
        d_cls = sum(na[j] for j, s in enumerate(sym) if s in DONOR_CLS)
        d_ext = d_cls + sum(na[j] for j, c in enumerate(cls) if c is not None)
        layerwise_donor_frac[layer_idx].append(float(d_cls))
        layerwise_donor_ext[layer_idx].append(float(d_ext))
    alpha_l5 = attn_layers[-1].mean(dim=1)
    ei = g['edge_index']; dst = ei[1]
    node_attn_l5 = torch.zeros(len(sym), device=device)
    node_attn_l5.scatter_add_(0, dst, alpha_l5)
    node_attn_l5 = node_attn_l5 / (node_attn_l5.sum() + 1e-8)
    top5_idx = node_attn_l5.argsort(descending=True)[:5].cpu().numpy()
    for idx in top5_idx:
        top5_counts[sym[idx]] += 1
        c = cls[idx]
        top5_ext[c if c is not None else sym[idx]] += 1

total = len(xai_graphs)
donor_cls = sum(v for k, v in top5_counts.items() if k in DONOR_CLS)
metal_cls = sum(v for k, v in top5_counts.items() if k in METAL_SYM)
c_cls = top5_counts.get('C', 0)
mc_top5 = sum(v for k, v in top5_ext.items() if k in ('carbonyl_C', 'nhc_C', 'other_MC'))
backbone_c_top5 = c_cls - mc_top5
print(f'\n=== sanity check (经典集合, 期望 20.5/73.4/4.3) ===')
print(f'donor {100*donor_cls/(total*5):.2f}% | C {100*c_cls/(total*5):.2f}% | metal {100*metal_cls/(total*5):.2f}%')
print(f'\n=== 扩展集合 ===')
print(f'M-C 配合物: {n_mc_complexes:,}/{total:,} ({100*n_mc_complexes/total:.1f}%)')
print(f'M-C 碳计数: {dict(mc_carbon_counts)} (每配合物均值 {sum(mc_carbon_counts.values())/total:.2f})')
print(f'donor(经典) top-5 {100*donor_cls/(total*5):.2f}% -> donor(含 M-C) {100*(donor_cls+mc_top5)/(total*5):.2f}%')
print(f'C top-5 {100*c_cls/(total*5):.2f}% 拆分: M-C {100*mc_top5/(total*5):.2f}% (羰基 {100*top5_ext.get("carbonyl_C",0)/(total*5):.2f}% / NHC {100*top5_ext.get("nhc_C",0)/(total*5):.2f}% / 其他 {100*top5_ext.get("other_MC",0)/(total*5):.2f}%) + 骨架 C {100*backbone_c_top5/(total*5):.2f}%')
for l in range(5):
    print(f'Layer {l+1}: donor 经典 {np.mean(layerwise_donor_frac[l])*100:.2f}% | 含 M-C {np.mean(layerwise_donor_ext[l])*100:.2f}%')

# ===== 4. 扰动 (2,000 子集, 扩展 donor 集合) =====
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
N_PERTURB = 2000
perturb_idx = np.random.RandomState(123).choice(len(xai_graphs), N_PERTURB, replace=False)
perturb_donor_frac = {t: [] for t in targets}
perturb_donor_ext = {t: [] for t in targets}
for ii in tqdm(range(len(perturb_idx)), desc='perturb'):
    i = perturb_idx[ii]
    g = xai_graphs[i]; sym = xai_symbols[i]; cls = xai_cls[i]
    N = g['num_nodes']
    x_orig = g['x']
    with torch.no_grad():
        base_pred = model(g).cpu().numpy()[0]
    atom_attrs = np.zeros((N, len(targets)))
    for atom_idx in range(N):
        x_masked = x_orig.clone()
        x_masked[atom_idx] = 0.0
        with torch.no_grad():
            mp = model({'x': x_masked, 'edge_index': g['edge_index'], 'edge_attr': g['edge_attr']}).cpu().numpy()[0]
        atom_attrs[atom_idx] = base_pred - mp
    for prop_j, prop_name in enumerate(targets):
        donor_cls_idx = [j for j, s in enumerate(sym) if s in DONOR_CLS]
        donor_ext_idx = donor_cls_idx + [j for j, c in enumerate(cls) if c is not None]
        total_abs = sum(abs(atom_attrs[j, prop_j]) for j in range(N))
        if donor_cls_idx:
            perturb_donor_frac[prop_name].append(float(sum(abs(atom_attrs[j, prop_j]) for j in donor_cls_idx) / (total_abs + 1e-8)))
        if donor_ext_idx:
            perturb_donor_ext[prop_name].append(float(sum(abs(atom_attrs[j, prop_j]) for j in donor_ext_idx) / (total_abs + 1e-8)))

print('\n=== 扰动 donor fraction (经典 vs 含 M-C) ===')
pert_summary = {}
for t in targets:
    a = float(np.mean(perturb_donor_frac[t])) if perturb_donor_frac[t] else None
    b = float(np.mean(perturb_donor_ext[t])) if perturb_donor_ext[t] else None
    pert_summary[t] = {'classic': a, 'with_MC': b}
    print(f'{t:16s} classic {a*100:.1f}% -> with M-C {b*100:.1f}%' if a is not None else f'{t}: n/a')

out = {'N_attention': total, 'N_perturb': N_PERTURB,
       'sanity_donor_pct': float(100*donor_cls/(total*5)), 'sanity_C_pct': float(100*c_cls/(total*5)),
       'sanity_metal_pct': float(100*metal_cls/(total*5)),
       'n_mc_complexes': n_mc_complexes, 'mc_carbon_counts': dict(mc_carbon_counts),
       'top5_donor_classic_pct': float(100*donor_cls/(total*5)),
       'top5_donor_withMC_pct': float(100*(donor_cls+mc_top5)/(total*5)),
       'top5_MC_split_pct': {'carbonyl': float(100*top5_ext.get('carbonyl_C',0)/(total*5)),
                             'nhc': float(100*top5_ext.get('nhc_C',0)/(total*5)),
                             'other': float(100*top5_ext.get('other_MC',0)/(total*5))},
       'backbone_C_pct': float(100*backbone_c_top5/(total*5)),
       'layerwise_donor_classic': [float(np.mean(layerwise_donor_frac[l])*100) for l in range(5)],
       'layerwise_donor_withMC': [float(np.mean(layerwise_donor_ext[l])*100) for l in range(5)],
       'perturb_donor_frac': pert_summary}
with open('m7_carbon_donors_202403.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print('\nSaved -> m7_carbon_donors_202403.json')
