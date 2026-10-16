# -*- coding: utf-8 -*-
"""vA 域 (论文主域 90,317) SMILES 图数据包构建 — 意见6 方案B 数据准备 (本地, rdkit2024 环境)
背景: 服务器 R5 自检失败 (te_idx 相同 / y_true 不同) → r2b_graphs.pt 前 90,317 ≠ 论文主域行序
(服务器 rdkit 与论文 pinned 栈 RDKit 2024.03 的 ±19 解析差异导致保留序列错位)。
本脚本用论文管线 (eval_smiles_gat.py 逐字同逻辑, 已证 0.932 复现) 在本地建出
论文主域 90,317 个 SMILES 图 (node 8d / edge 6d, 与 R3/R5 协议截断特征一致) + 标签,
自检 y_all[te_idx] 与 ensemble_predictions.npz 逐位一致后打包:
  vA_domain_smiles.pt = {'graphs': [...], 'y_all': tensor(90317,8), 'atoms_filt': [...]}
上传服务器后 R5 直接加载本包训练 (不再截断 r2b_graphs.pt)。
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
RENAME = {'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
          'Dipole_M': 'dipole', 'Metal_q': 'metal_charge', 'Electronic_E': 'E_elec',
          'Dispersion_E': 'E_disp', 'Polarizability': 'polarizability'}
df_y = df_y.rename(columns=RENAME)
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
print(f'csv: {len(df_y):,} 行', flush=True)

# ===== XYZ+BO 解析 (与 ensemble CV / eval_smiles_gat.py 完全相同逻辑) =====
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

def parse_one_full(xyz_p, bo_p, start_idx=0):
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
            if i >= len(xyz_blocks): break
            i += 1
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]])
                        atoms.append(z)
                    except ValueError:
                        continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1:
                continue
            mols.append({"atoms": np.array(atoms, dtype=np.int64), "csv_idx": mol_idx})
    return mols

print('Parsing XYZ+BO (paper pipeline)...', flush=True)
all_m = []
for part in [1, 2, 3]:
    m = parse_one_full(f'{ROOT}/tmQM_X{part}.xyz.gz', f'{ROOT}/tmQM_X{part}.BO.gz', start_idx=len(all_m))
    all_m.extend(m)
    print(f'  Part {part}: {len(m):,} (累计 {len(all_m):,})', flush=True)
print(f'Total: {len(all_m):,} TM complexes', flush=True)

# ===== SMILES 建图 (node 8d / edge 6d — 与 r2b/r3/r5 截断特征逐位一致) =====
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM = 4
smiles_by_idx = df_y['SMILES'].tolist()
csv_labels = df_y[targets].values.astype(np.float32)

def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

print('Building SMILES graphs...', flush=True)
graphs, y_vals, atoms_filt, kept_tmpos, failed = [], [], [], [], 0
for idx, m in enumerate(all_m):
    csv_i = m['csv_idx']
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '':
        failed += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        failed += 1; continue
    N = mol.GetNumAtoms()
    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    sm_z = [xf[i][0] for i in range(N)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za >= 21) != (zb >= 21))
        is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 6)
    graphs.append({'x': x, 'edge_index': ei, 'edge_attr': ea, 'num_nodes': N})
    y_vals.append(csv_labels[csv_i])
    atoms_filt.append(m['atoms'])
    kept_tmpos.append(idx + 1)
    if (idx + 1) % 20000 == 0:
        print(f'  {idx+1:,}/{len(all_m):,}', flush=True)

y_all = torch.tensor(y_vals, dtype=torch.float32)
mask = ~(torch.isnan(y_all).any(dim=1) | torch.isinf(y_all).any(dim=1))
graphs = [g for i, g in enumerate(graphs) if mask[i]]
y_all = y_all[mask]
atoms_filt = [a for i, a in enumerate(atoms_filt) if mask[i]]
kept_tmpos = [p for i, p in enumerate(kept_tmpos) if mask[i]]
print(f'Built {len(graphs):,} graphs (SMILES 失败 {failed:,}, NaN/inf 剔除 {(~mask).sum():,})', flush=True)

# ===== 论文主域 90,317 = 保留序列的前 90,317 个 (README 2026-08-28 终版结论, 前缀截断) =====
# 服务器 (tmqm rdkit) 保留 95,152, 锚点第 90,317 个保留分子 = TM 块 #97,045;
# 本机 rdkit2024 保留 95,171 (+19)。截断后的逐位一致性由下方 y_true 自检实锤。
if len(graphs) > 90317:
    print(f'[truncate] 保留 {len(graphs):,} > 90,317 → 前缀截断; 本机锚点: 第 90,317 个保留分子 = TM 块 #{kept_tmpos[90316]} (服务器锚点 97,045)', flush=True)
    graphs = graphs[:90317]
    y_all = y_all[:90317]
    atoms_filt = atoms_filt[:90317]
assert len(graphs) == 90317, f'图数 {len(graphs):,} != 90,317 — 本地文件或管线与论文主域不一致'

# ===== 对齐自检: te_idx 逐位核对 (与服务器 R5 自检同一口径) =====
sp = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_idx = sp['te_idx']; cv_idx = sp['cv_idx']
ep = np.load('ensemble_predictions.npz')
same_te = np.array_equal(ep['te_idx'], te_idx)
yt_here = y_all[te_idx].numpy()
same_yt = np.allclose(yt_here, ep['y_true'], atol=1e-4)
print(f'对齐自检: te_idx 相同={same_te}, y_true 相同={same_yt}', flush=True)
assert same_te and same_yt, '!! 本地包与 3D 集成行序不一致, 中止'
all_idx = np.concatenate([te_idx, cv_idx])
assert len(set(all_idx.tolist())) == 90317 and all_idx.min() == 0 and all_idx.max() == 90316
print('划分完整性 ✓ (te+cv 恰为 0..90,316 全排列)', flush=True)

torch.save({'graphs': graphs, 'y_all': y_all, 'atoms_filt': atoms_filt}, 'vA_domain_smiles.pt')
print('saved -> vA_domain_smiles.pt', flush=True)
