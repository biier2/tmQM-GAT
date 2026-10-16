# -*- coding: utf-8 -*-
"""S2: 几何源稳健性——训练几何(DFT/CSD xyz + Mayer BO) vs 部署几何(ETKDG+MMFF, BO=0)
不重训。图A=训练口径(逐字复刻 ensemble_cv_weighted.ipynb Step 1)；图B=scorer_3d.ipynb
cell3 部署口径。500 个测试集复合物，单模型 xyz_best_weighted.pt。输出 s2_geometry_source.json。"""
import gzip
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

torch.set_num_threads(16)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

# ===== 常量（与训练/scorer 完全一致）=====
y_mean_sm = torch.tensor([0.108505, -0.197591, -0.089089, 5.799378, 0.162882, -2987.275391, -0.142229, 397.123474])
y_std_sm = torch.tensor([0.033575, 0.053824, 0.054516, 3.927464, 0.782902, 1634.063721, 0.070114, 152.458313])
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
RBF_C = np.linspace(1.0, 5.0, 16); RBF_SIGMA = 0.2

ELEMENT_DATA_SM = {1: (1, 1), 5: (2, 13), 6: (2, 14), 7: (2, 15), 8: (2, 16), 9: (2, 17),
                   14: (3, 14), 15: (3, 15), 16: (3, 16), 17: (3, 17), 20: (4, 2), 21: (4, 3),
                   22: (4, 4), 23: (4, 5), 24: (4, 6), 25: (4, 7), 26: (4, 8), 27: (4, 9),
                   28: (4, 10), 29: (4, 11), 30: (4, 12), 35: (4, 17), 42: (5, 6), 44: (5, 8),
                   45: (5, 9), 46: (5, 10), 47: (5, 11), 48: (5, 12), 53: (5, 17), 72: (6, 4),
                   73: (6, 5), 74: (6, 6), 75: (6, 7), 76: (6, 8), 77: (6, 9), 78: (6, 10),
                   79: (6, 11), 80: (6, 12)}
BOND_TYPES_SM = [1, 2, 3, 12]; B2I_SM = {b: i for i, b in enumerate(BOND_TYPES_SM)}
TM = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
      72, 73, 74, 75, 76, 77, 78, 79, 80]

atom_to_z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
             "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
             "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
             "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Br": 35, "Ru": 44, "Rh": 45,
             "Pd": 46, "Ag": 47, "Cd": 48, "I": 53, "Pt": 78, "Au": 79, "Hg": 80,
             "Mo": 42, "W": 74, "Ir": 77, "Os": 76, "Re": 75}


def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
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


# ===== 数据 =====
df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';').rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
smiles_by_idx = df_y['SMILES'].tolist()

# ---- 第一遍：重建"训练 all_m 的 csv_idx 顺序"（含 TM 的分子块序列）----
print('Pass 1: scanning xyz files to rebuild training-order csv_idx sequence...')
seq = []        # 含 TM 块的全局 csv_idx 序列（对应训练 all_m 顺序）
collected = 0   # 跨 part 累计的含 TM 分子数（= 训练 start_idx=len(all_m) 语义）
for part in [1, 2, 3]:
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        xyz_blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, collected - 1   # 与训练一致：mol_idx 起点 = 前序 part 的 all_m 长度 - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            i += 1  # 跳过标题行
            zs = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z.get(cparts[0], 0)
                        [float(x) for x in cparts[1:4]]
                        zs.append(z)
                    except ValueError: continue
            mol_idx += 1
            if any(z in TM for z in zs) and len(zs) > 1:
                seq.append(mol_idx); collected += 1
    print(f'  Part {part} done (last mol_idx={mol_idx}, collected={collected:,})')
print(f'Total TM blocks in sequence: {len(seq):,}')

# ---- te_idx → 目标 csv_idx（训练里 graphs 过滤了 SMILES 失败 + y NaN）----
def smi_ok(csv_i):
    s = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    return isinstance(s, str) and s != '' and Chem.MolFromSmiles(s) is not None

def y_ok(csv_i):
    v = df_y.iloc[csv_i][targets].astype(float).values
    return not (np.isnan(v).any() or np.isinf(v).any())

valid_csv = [c for c in seq if smi_ok(c) and y_ok(c)]
print(f'Valid graphs (train-order, SMILES ok + y finite): {len(valid_csv):,}')

split = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_idx = split['te_idx']
print(f'te_idx size: {len(te_idx):,}')

n_cand = 900
cand_positions = np.linspace(0, len(te_idx) - 1, n_cand).astype(int)
cand_csv = set(valid_csv[int(t)] for t in te_idx[cand_positions])
print(f'Sampled {len(cand_csv)} candidate csv indices')

# ---- 第二遍：解析目标块的 coords + BO（逐字复刻训练 bo_mat 语义）----
print('Pass 2: parsing target molecules with training-identical BO logic...')
mols = {}          # csv_idx -> {atoms, coords, bo, bo_hit}
collected = 0      # 训练 all_m 长度（跨 part 累计的含 TM 分子数）
n_bo_hit = 0; n_bo_miss = 0
for part in [1, 2, 3]:
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.BO.gz', 'rt') as f:
        bo_lines = [l.strip() for l in f.readlines()]
    bo_blocks, cur = [], []
    for line in bo_lines:
        if line.startswith('CSD_code'):
            if cur: bo_blocks.append(cur); cur = []
        elif line: cur.append(line)
    if cur: bo_blocks.append(cur)
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        xyz_blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, collected - 1
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
                        z = atom_to_z.get(cparts[0], 0)  # 与训练一致：未知符号/CSD_code 行 → z=0
                        coords.append([float(x) for x in cparts[1:4]]); atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(z in TM for z in atoms) or len(atoms) <= 1: continue
            collected += 1
            if mol_idx not in cand_csv: continue
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
                            nbr = int(bparts[j]) - 1; bo_val = float(bparts[j + 1])
                            if 0 <= src_idx < n_atoms and 0 <= nbr < n_atoms:
                                bo_mat[src_idx, nbr] = bo_val
                            j += 3
                        except (ValueError, IndexError): j += 1
            if bo_mat is not None: n_bo_hit += 1
            else: n_bo_miss += 1
            mols[mol_idx] = {'atoms': np.array(atoms), 'coords': np.array(coords, dtype=np.float32),
                             'bo': bo_mat}
print(f'Targets parsed: {len(mols)}; BO hit: {n_bo_hit}, BO miss(→0): {n_bo_miss}')

# ===== 图 B：ETKDG+MMFF（scorer cell3 口径）=====
def smiles_to_3d_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200) != 0: return None
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return None
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    for i in range(N): xf[i].extend([0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(N)])
    sm_z = [xf[i][0] for i in range(N)]
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * 4
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        d = float(np.linalg.norm(coords[i] - coords[j]))
        for mu in RBF_C: bf.append(np.exp(-(d - mu) ** 2 / RBF_SIGMA))
        bf.append(0.0)
        battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 23)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}


# ===== 图 A：训练口径（xyz coords + BO）=====
def build_train_graph(smiles, m):
    mol = Chem.MolFromSmiles(smiles)
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
                sb = float(bo_mat[j].sum()) - bo_mat[j, j]
                mb = float(bo_mat[j].max())
                xf[i].extend([max(0, nb), max(0, sb), max(0, mb)])
            else: xf[i].extend([0, 0.0, 0.0])
    else:
        for i in range(N): xf[i].extend([0, 0.0, 0.0])
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_atoms[i], sm_atoms[j]
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * 4
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        d, bo = 2.0, 0.0
        xi, xj = sm_to_xyz[i], sm_to_xyz[j]
        if xi >= 0 and xj >= 0 and xi < len(coords) and xj < len(coords):
            d = float(np.linalg.norm(coords[xi] - coords[xj]))
        if bo_mat is not None and xi >= 0 and xj >= 0 and xi < bo_mat.shape[0] and xj < bo_mat.shape[1]:
            bo = float(bo_mat[xi, xj])
        for mu in RBF_C: bf.append(np.exp(-(d - mu) ** 2 / RBF_SIGMA))
        bf.append(bo); battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 23)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}


# ===== 模型 =====
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
        self.edge_attn = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.act = nn.ReLU()
    def forward(self, x, ei, ea=None):
        src, dst = ei[0], ei[1]; N = x.size(0); h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if self.edge_attn is not None and ea is not None and ea.size(0) > 0:
            attn = attn + self.edge_attn(ea)
        attn = torch.nn.functional.leaky_relu(attn, 0.2)
        alpha = segment_softmax(attn, dst, N)
        h_heads = h.view(-1, self.heads, self.out_dim)
        msg = h_heads[src] * alpha.unsqueeze(-1); msg = msg.view(-1, self.out_dim * self.heads)
        out = torch.zeros(N, self.out_dim * self.heads, device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return self.act(out)

class XYZ_GNN(nn.Module):
    def __init__(self, node_dim=11, edge_dim=23, hidden=HIDDEN, n_layers=N_LAYERS, output_dim=8, dropout=DROPOUT, heads=GAT_HEADS):
        super().__init__(); self.heads = heads
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden // heads, edge_dim, heads=heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim))
    def forward(self, g, batch=None):
        x, ei, ea = g['x'], g['edge_index'], g.get('edge_attr', None)
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei, ea)); h = self.drop(h)
        mm = (x[:, 7] > 0.5)
        mf = h[mm].mean(dim=0) if mm.any() else torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
        return self.readout(torch.cat([h.mean(0), h.max(0).values, mf])).unsqueeze(0)

st = torch.load('xyz_best_weighted.pt', map_location='cpu', weights_only=False)
st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
model = XYZ_GNN(); model.load_state_dict(st); model.eval()
print('Model loaded: xyz_best_weighted.pt')

# ===== 收集 500 个双几何样本 =====
N_SAMPLE = 500
pairs = []  # (csv_idx, smiles, y_true, gA, gB)
n_etkdg_tried = n_etkdg_ok = 0
for csv_i in sorted(mols):
    smi = smiles_by_idx[csv_i]
    n_etkdg_tried += 1
    gb = smiles_to_3d_graph(smi)
    if gb is None: continue
    n_etkdg_ok += 1
    ga = build_train_graph(smi, mols[csv_i])
    y = df_y.iloc[csv_i][targets].astype(float).values
    pairs.append((csv_i, smi, y, ga, gb))
    if len(pairs) >= N_SAMPLE: break
print(f'ETKDG tried: {n_etkdg_tried}, success: {n_etkdg_ok}, sampled: {len(pairs)}')

# ===== 推理 =====
preds_a, preds_b, yt_all = [], [], []
with torch.no_grad():
    for csv_i, smi, y, ga, gb in pairs:
        pa = model(ga).squeeze(0) * y_std_sm + y_mean_sm
        pb = model(gb).squeeze(0) * y_std_sm + y_mean_sm
        preds_a.append(pa.numpy()); preds_b.append(pb.numpy()); yt_all.append(y)
preds_a = np.array(preds_a); preds_b = np.array(preds_b); yt_all = np.array(yt_all)

np.savez('s2_preds.npz', preds_train_geom=preds_a, preds_etkdg_geom=preds_b, y_true=yt_all,
         csv_idx=[p[0] for p in pairs])

# ===== 指标 =====
def r2(y, p):
    ss_res = ((y - p) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)

out = {'n_sampled': len(pairs), 'n_candidates': n_cand,
       'etkdg_success_rate_pct': round(n_etkdg_ok / n_etkdg_tried * 100, 2),
       'bo_hit_rate_pct': round(n_bo_hit / (n_bo_hit + n_bo_miss) * 100, 2),
       'model': 'xyz_best_weighted.pt (single-model, deployment checkpoint)',
       'per_property': {}}
print(f'\n{"Property":16s} {"R2 train":>9s} {"R2 ETKDG":>9s} {"MAE train":>10s} {"MAE ETKDG":>10s} {"Spearman":>9s} {"Pearson":>9s}')
print('-' * 82)
for i, t in enumerate(targets):
    yt = yt_all[:, i]
    ra, rb = r2(yt, preds_a[:, i]), r2(yt, preds_b[:, i])
    maa = float(np.abs(preds_a[:, i] - yt).mean())
    mab = float(np.abs(preds_b[:, i] - yt).mean())
    sp = float(spearmanr(preds_a[:, i], preds_b[:, i])[0])
    pr = float(pearsonr(preds_a[:, i], preds_b[:, i])[0])
    out['per_property'][t] = {'r2_train_geom': round(ra, 4), 'r2_etkdg_geom': round(rb, 4),
                              'mae_train_geom': round(maa, 4), 'mae_etkdg_geom': round(mab, 4),
                              'spearman_cross_geom': round(sp, 4), 'pearson_cross_geom': round(pr, 4)}
    print(f'{t:16s} {ra:9.4f} {rb:9.4f} {maa:10.4f} {mab:10.4f} {sp:9.4f} {pr:9.4f}')

json.dump(out, open('s2_geometry_source.json', 'w'), indent=1)
print('\nsaved s2_geometry_source.json + s2_preds.npz')
