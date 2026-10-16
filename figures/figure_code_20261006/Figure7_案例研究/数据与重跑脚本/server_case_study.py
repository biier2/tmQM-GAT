# -*- coding: utf-8 -*-
"""§3.7 Case Study 重跑: 18 催化剂 × 新 A/D 变体集成 (9 模型均±std)
3D 路径 = ETKDG+MMFF (AddHs→Embed(42,200)→MMFF→RemoveHs) + WBO 全零 (部署约定)
DFT 参考 = Table 4 原值 (10 个已验证络合物, TPSSh/def2-SVP)
输出 case_study_new_results.json (供 Table 4 重建 + Fig7 重渲)
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import mean_absolute_error

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from torch.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import AllChem

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
N_TARGETS = len(targets)
HIDDEN = 384; N_LAYERS = 5; GAT_HEADS = 12; DROPOUT = 0.10
EV_PER_HA = 27.211386

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

# ---------- 图构建 ----------
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM = 4
rbf_c = np.linspace(1.0, 5.0, 16).astype(np.float32); rbf_sigma = 0.2

def atom_features(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g, int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), int(z >= 21)]

def smiles_graph(smiles):
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

def smiles_to_3d_graph(smiles):
    """ETKDG+MMFF → B 变体部署图 (8d 节点 + 22d 边 = 6 拓扑 + 16 RBF, 无任何 WBO)
    意见2(a): 部署一致对照 — 训练/推断均无 WBO, 避免零化引起的分布偏移"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol_h = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=200) != 0: return None
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        return None
    mol2 = Chem.RemoveHs(mol_h)
    conf = mol2.GetConformer()
    xf = [atom_features(a) for a in mol2.GetAtoms()]
    src, dst, battr = [], [], []
    for bond in mol2.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = mol2.GetAtomWithIdx(i).GetAtomicNum(), mol2.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        d = float(np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j))))
        rbf_vec = np.exp(-(d - rbf_c) ** 2 / rbf_sigma).astype(np.float32)
        bf = [0.0] * N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        src.extend([i, j]); dst.extend([j, i])
        b6 = bf + list(rbf_vec)
        battr.append(b6); battr.append(b6)
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 22)
    return {'x': torch.tensor(xf, dtype=torch.float32), 'edge_index': ei, 'edge_attr': ea}

def predict_ensemble(ckpt, node_dim, edge_dim, g):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    models = ck['models']
    outs = []
    for mdata in models:
        model = XYZ_GNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
        state = {k.replace('_orig_mod.', ''): v for k, v in mdata['state'].items()}
        model.load_state_dict(state); model.eval()
        gg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in g.items()}
        with torch.no_grad(), autocast('cuda'):
            p = model(gg).float().cpu()
        outs.append((p * mdata['y_std'] + mdata['y_mean']).numpy().flatten())
        del model
    torch.cuda.empty_cache()
    arr = np.stack(outs, 0)
    return arr.mean(0), arr.std(0)

CANDIDATES = {
    'Zr-cat1': 'Br[Zr]([CH-]1C=CC=C1)([CH-]1C=CC=C1)Br',
    'Zr-cat2': 'Br[Zr]([CH-]1C=CC=C1)([CH-]1C=CC=C1)I',
    'Zr-cat3': 'I[Zr]([CH-]1C=CC=C1)([CH-]1C=CC=C1)I',
    'Ni-cat1': 'Br[Ni]1([N]2=C3C4=[N]1C=CC(C5=CC=C(C(F)(F)F)C=C5)=C4C=CC3=C(C6=CC=C(C(F)(F)F)C=C6)C=C2)Br',
    'Ni-cat2': 'Br[Ni]1[N]2=C3C4=[N]1C=CC(C5=CC=C(C(F)(F)F)C=C5)=C4C=CC3=C(C6=CC=C(C(F)(F)F)C=C6)C=C2',
    'Ni-cat3': 'I[Ni]1[N]2=C3C4=[N]1C=CC(C5=CC=C(C(F)(F)F)C=C5)=C4C=CC3=C(C6=CC=C(C(F)(F)F)C=C6)C=C2',
    'Ni-cat4': 'Br[Ni]1([N]2=C(C3=[N]1C=CC(C(C)(C)C)=C3)C=C(C(C)(C)C)C=C2)Br',
    'Ni-cat5': 'CO1CCO2CCO(C)[Ni]21(Br)Br',
    'Pd-cat1': 'CC1(C)C2=C(OC3=C4C=CC=C13)C([P]([Pd][P](C5=CC=CC=C5)4C6=CC=CC=C6)(C7=CC=CC=C7)C8=CC=CC=C8)=CC=C2',
    'Pd-cat2': 'FC1(F)OC2=C(O1)[C@@]([C@@]3=C(OC(F)(F)O4)C4=CC=C3[P]([Pd][P](C5=CC=CC=C5)6C7=CC=CC=C7)(C8=CC=CC=C8)C9=CC=CC=C9)=C6C=C2',
    'Co-cat1': 'CC1=O[Co]2(OC(C)=O2)O1',
    'Fe-cat1': 'O=[N+]1C(C)(C)CCCC1(C)C.Cl[Fe](Cl)Cl',
    'Fe-cat2': 'O=[N+]1C(C)(C)CCCC1(C)C.Cl[Fe](Cl)Cl.O[N+](=O)[O-]',
    'cis-PdCl2(PPh3)2':   'Cl[Pd@SP1](Cl)(P(c1ccccc1)(c2ccccc2)c3ccccc3)P(c4ccccc4)(c5ccccc5)c6ccccc6',
    'trans-PdCl2(PPh3)2': 'Cl[Pd@SP2](Cl)(P(c1ccccc1)(c2ccccc2)c3ccccc3)P(c4ccccc4)(c5ccccc5)c6ccccc6',
    'Rh(CO)2(acac)':      '[C-]#[O+][Rh]1([C-]#[O+])O=C(C)[CH]C(C)=[O]1',
    'Ni(IMes)2Cl2':       'Cl[Ni](Cl)([C]1N(c2c(C)cc(C)cc2C)C=CN1c3c(C)cc(C)cc3C)[C]4N(c5c(C)cc(C)cc5C)C=CN4c6c(C)cc(C)cc6C',
    'Pt(dppe)Cl2':        'Cl[Pt](Cl)([P](c1ccccc1)(c2ccccc2)CC[P](c3ccccc3)(c4ccccc4))',
}
# Table 4 的 DFT 参考 (gap eV, dipole D) — 10 个已验证
DFT = {
    'Zr-cat3': (3.30, 5.32), 'Fe-cat1': (2.76, 9.76), 'Ni-cat3': (1.39, 5.67),
    'Pd-cat2': (2.93, 2.48), 'trans-PdCl2(PPh3)2': (3.19, 0.03), 'cis-PdCl2(PPh3)2': (3.17, 10.86),
    'Pt(dppe)Cl2': (3.95, 10.29), 'Co-cat1': (3.30, 0.00), 'Rh(CO)2(acac)': (4.03, 3.17),
    'Ni(IMes)2Cl2': (3.24, 0.00),
}

rows = {}
for name, smi in CANDIDATES.items():
    g_sm = smiles_graph(smi)
    if g_sm is None:
        print(f'  {name}: SMILES 解析失败'); continue
    sm_mean, sm_std = predict_ensemble('ensemble_checkpoint_A_8d6d.pt', 8, 6, g_sm)
    g_3d = smiles_to_3d_graph(smi)
    if g_3d is None:
        print(f'  {name}: ETKDG 失败, 3D 通道无预测')
        d3_mean = d3_std = None
    else:
        d3_mean, d3_std = predict_ensemble('ensemble_checkpoint_B_8d22d.pt', 8, 22, g_3d)
    rows[name] = {'smiles_mean': sm_mean.tolist(), 'smiles_std': sm_std.tolist(),
                  'd3_mean': (d3_mean.tolist() if d3_mean is not None else None),
                  'd3_std': (d3_std.tolist() if d3_std is not None else None)}
    # gap eV (Ha*27.211386), dipole 非负截断
    sm_gap_ev = sm_mean[0] * EV_PER_HA; sm_dip = max(0.0, sm_mean[3])
    if d3_mean is not None:
        d3_gap_ev = d3_mean[0] * EV_PER_HA; d3_dip = max(0.0, d3_mean[3])
        print(f'  {name}: gap 3D={d3_gap_ev:.2f}/SM={sm_gap_ev:.2f} eV, dipole 3D={d3_dip:.2f}/SM={sm_dip:.2f} D', flush=True)
    else:
        print(f'  {name}: gap SM={sm_gap_ev:.2f} eV, dipole SM={sm_dip:.2f} D (无3D)', flush=True)

# MAE 统计 (10 个 DFT 验证)
for model_name, key in [('3D', 'd3_mean'), ('SMILES', 'smiles_mean')]:
    gap_errs, dip_errs = [], []
    for name, (g_dft, d_dft) in DFT.items():
        r = rows.get(name)
        if r is None or r[key] is None: continue
        gap_ev = r[key][0] * EV_PER_HA
        dip = max(0.0, r[key][3])
        gap_errs.append(abs(gap_ev - g_dft)); dip_errs.append(abs(dip - d_dft))
    if gap_errs:
        print(f'{model_name}: gap MAE = {np.mean(gap_errs):.3f} eV, dipole MAE = {np.mean(dip_errs):.3f} D (n={len(gap_errs)})', flush=True)
        rows[f'__mae_{model_name}'] = {'gap_mae_ev': float(np.mean(gap_errs)),
                                       'dipole_mae_d': float(np.mean(dip_errs)), 'n': len(gap_errs),
                                       'gap_mae_ha': float(np.mean(gap_errs) / EV_PER_HA)}

json.dump(rows, open('case_study_new_results.json', 'w'), indent=2)
print('SAVED case_study_new_results.json')
