# -*- coding: utf-8 -*-
"""R4: 跨泛函 BO 对照实验 — 数据准备
数据: tmQMg 74,547 (⊆ tmQM, CSD_code 对齐) | 标签: ωB97M-V (tzvp_*)
模型A: SMILES GAT 节点8d/边6d | 模型B: +TPSSh Mayer BO 3d 节点增广 (nb/sb/mb)
输出 r4_dataset.npz: 图特征 + 标签 + codes
"""
import gzip, json, re, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

sys.stdout.reconfigure(encoding='utf-8')
TQMG = 'd:/ml/chem_ml5/tmQMg_properties_and_targets.csv'
TMQM_Y = 'd:/ml/chem_ml5/data_tmQM/local_vB/tmQM_y.csv'

# ---------- 1. 读 tmQMg + tmQM_y 回退映射 ----------
df = pd.read_csv(TQMG)
y_df = pd.read_csv(TMQM_Y, sep=';')
smiles_fallback = dict(zip(y_df['CSD_code'], y_df['SMILES']))

LABEL_COLS = ['tzvp_homo_lumo_gap', 'tzvp_homo_energy', 'tzvp_lumo_energy',
              'tzvp_dipole_moment', 'tzvp_electronic_energy',
              'tzvp_dispersion_energy', 'polarisability']
PROPS = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']

codes = df['id'].tolist()
smi_used = []
for code, smi in zip(df['id'], df['smiles']):
    if isinstance(smi, str) and smi.strip():
        smi_used.append(smi)
    else:
        fb = smiles_fallback.get(code, '')
        smi_used.append(fb if isinstance(fb, str) else '')

# ---------- 2. 流式扫 xyz.gz: 目标 CSD_code -> 原子序序列 ----------
def atom_to_z(sym):
    tbl = {"H":1,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Na":11,"Mg":12,
           "Al":13,"Si":14,"P":15,"S":16,"Cl":17,"K":19,"Ca":20,"Sc":21,"Ti":22,
           "V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,
           "Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"I":53,"Mo":42,"W":74,"Ir":77,
           "Os":76,"Re":75,"Hf":72,"Ta":73,"Y":39,"Zr":40,"Nb":41,"Tc":43,"Hg":80,
           "Pt":78,"Au":79,"La":57,"As":33,"Se":34}
    return tbl.get(sym, 0)

want = set(codes)
xyz_atoms = {}   # code -> list[int]
for part in [1, 2, 3]:
    with gzip.open(f'd:/ml/chem_ml5/data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        while True:
            line = f.readline()
            if not line: break
            if not line.strip(): continue
            n = int(line.strip())
            hdr = f.readline()
            m = re.search(r'CSD_code = (\S+)', hdr or '')
            atoms = []
            for _ in range(n):
                al = f.readline()
                if not al: break
                sp = al.split()
                if len(sp) >= 4:
                    atoms.append(atom_to_z(sp[0]))
            if m and m.group(1) in want:
                xyz_atoms[m.group(1)] = atoms
print(f'xyz 匹配: {len(xyz_atoms)}/{len(want)}')

# ---------- 3. 流式扫 BO.gz: 目标 CSD_code -> per-atom (nb, sb, mb) ----------
bo_feats = {}    # code -> (N,) x 3
for part in [1, 2, 3]:
    with gzip.open(f'd:/ml/chem_ml5/data_tmQM/local_vB/tmQM_X{part}.BO.gz', 'rt') as f:
        cur_code = None
        rows = []
        for line in f:
            line = line.strip()
            if line.startswith('CSD_code'):
                if cur_code and cur_code in want and rows:
                    bo_feats[cur_code] = rows
                cur_code = re.search(r'CSD_code = (\S+)', line).group(1)
                rows = []
            elif cur_code in want and line:
                sp = line.split()
                try:
                    src = int(sp[0]) - 1
                except ValueError:
                    continue
                # 行格式: idx elem totalBO (elem nbr bo)* ; 三元组从索引 3 起
                nbrs, bos = [], []
                j = 3
                while j + 1 < len(sp):
                    try:
                        nbrs.append(int(sp[j]) - 1); bos.append(float(sp[j + 1]))
                        j += 2
                    except ValueError:
                        j += 1
                rows.append((src, nbrs, bos))
        if cur_code and cur_code in want and rows:
            bo_feats[cur_code] = rows
print(f'BO 匹配: {len(bo_feats)}/{len(want)}')

# 转 per-atom (nb, sb, mb): nb = BO>0.05 邻居数-1, sb = 行和, mb = 行最大
bo_agg = {}
for code, rows in bo_feats.items():
    n_at = len(xyz_atoms.get(code, []))
    agg = np.zeros((n_at, 3), dtype=np.float32)
    for src, nbrs, bos in rows:
        if src >= n_at: continue
        nb = sum(1 for b in bos if b > 0.05) - 1
        agg[src, 0] = max(0, nb)
        agg[src, 1] = max(0, sum(bos))
        agg[src, 2] = max(0, max(bos)) if bos else 0.0
    bo_agg[code] = agg

# ---------- 4. SMILES 建图 (论文协议: 8d 节点 / 6d 边, dative->SINGLE 槽) ----------
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),
    15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),
    25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),
    44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),
    74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}

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

xs8, xs11, eis, eas, ys, ok_codes, ok_metals = [], [], [], [], [], [], []
n_fail_smi = n_fail_bo = 0
# id -> 标签向量 dict (O(1) 查找, 避免循环内全表扫描)
label_dict = {r['id']: r[LABEL_COLS].values.astype(np.float32) for _, r in df.iterrows()}
for code, smi in zip(codes, smi_used):
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        n_fail_smi += 1; continue
    N = mol.GetNumAtoms()
    sm_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    xyz = xyz_atoms.get(code)
    if xyz is None:
        n_fail_bo += 1; continue
    sm_to_xyz = fast_match(sm_atoms, xyz)
    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    agg = bo_agg.get(code)
    for i in range(N):
        j = sm_to_xyz[i]
        if agg is not None and 0 <= j < len(agg):
            xf[i].extend([float(agg[j, 0]), float(agg[j, 1]), float(agg[j, 2])])
        else:
            xf[i].extend([0.0, 0.0, 0.0])
    x = np.array(xf, dtype=np.float32)          # (N, 11)
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        if bt == 37: bt = 1                      # DATIVE -> SINGLE 槽
        za, zb = sm_atoms[i], sm_atoms[j]
        is_ml = float((za >= 21) != (zb >= 21))
        is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * 4
        if bt in (1, 2, 3, 12):
            bf[[1, 2, 3, 12].index(bt)] = 1.0
        bf.extend([is_ring, is_ml])
        battr.extend([bf, bf])
    ei = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    ea = np.array(battr, dtype=np.float32) if battr else np.zeros((0, 6), dtype=np.float32)
    xs8.append(x[:, :8]); xs11.append(x)
    eis.append(ei); eas.append(ea)
    ys.append(label_dict[code])
    ok_codes.append(code)

y_all = np.array(ys, dtype=np.float32)
mask = ~(np.isnan(y_all).any(axis=1) | np.isinf(y_all).any(axis=1))
keep = np.where(mask)[0]
print(f'建图成功 {len(ok_codes)} (SMILES 失败 {n_fail_smi}, xyz/BO 缺失 {n_fail_bo}), NaN 标签剔除 {(~mask).sum()}')

graphs = [{'x8': xs8[i], 'x11': xs11[i], 'edge_index': eis[i], 'edge_attr': eas[i]}
          for i in keep]
import torch as _t
_t.save({'graphs': graphs, 'y': _t.tensor(y_all[keep], dtype=_t.float32),
         'codes': [ok_codes[i] for i in keep]}, 'd:/ml/chem_ml5/r4_graphs.pt')
print('Saved -> r4_graphs.pt')
