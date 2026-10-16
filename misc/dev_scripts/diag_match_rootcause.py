# -*- coding: utf-8 -*-
"""意见1 诊断: 82.7% 兜底键根因分类 (trivial/greedy/兜底-元素缺失/兜底-名额耗尽/H情况)
只读分析, 不训练。复刻 cell3/4 的 parse + fast_match, 对每个 SMILES 原子分类匹配状态。
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
TBL = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
       "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
       "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
       "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
       "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
       "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
def atom_to_z(sym): return TBL.get(sym, 0)
def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

y_csv = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
smiles_by_idx = y_csv['SMILES'].tolist()
print(f'tmQM_y.csv rows: {len(smiles_by_idx)}')

def parse_one(xyz_p, start_idx=0):
    mols = []
    with gzip.open(xyz_p, "rt") as f:
        xyz_blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break
            i += 1
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx, "atoms": atoms})
    return mols

print('parsing XYZ (3 parts)...', flush=True)
all_m = []
for part in [1, 2, 3]:
    all_m.extend(parse_one(f"{ROOT}/tmQM_X{part}.xyz.gz", start_idx=len(all_m)))
    print(f'  part {part}: cumulative {len(all_m)}', flush=True)
print(f'parsed TM blocks: {len(all_m)}')

def fast_match(sm_atoms, xyz_atoms):
    N = len(sm_atoms)
    if N == len(xyz_atoms) and all(sm_atoms[i] == xyz_atoms[i] for i in range(N)):
        return list(range(N)), 'trivial'
    sm_to_xyz = [-1] * N; used = set()
    for i, z_sm in enumerate(sm_atoms):
        for j, z_xyz in enumerate(xyz_atoms):
            if z_sm == z_xyz and j not in used:
                sm_to_xyz[i] = j; used.add(j); break
    return sm_to_xyz, 'greedy'

stats = Counter()          # 络合物级匹配类型
atom_reason = Counter()    # 未匹配 SMILES 原子的原因
xyz_has_H = 0; n_xyz_lt_sm = 0; n_xyz_gt_sm = 0; n_equal = 0
z0_elems = Counter()       # XYZ 中 z=0 的符号 (需从原文取符号, 这里先计数量)
n_z0_complexes = 0
ph_bond_counts = []        # 每个络合物兜底键占比
n_smiles_invalid = 0

for m in all_m:
    smi = smiles_by_idx[m['csv_idx']] if m['csv_idx'] < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '':
        n_smiles_invalid += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_smiles_invalid += 1; continue
    sm_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    xyz_atoms = m['atoms']
    ns, nx = len(sm_atoms), len(xyz_atoms)
    if 1 in xyz_atoms: xyz_has_H += 1
    if nx < ns: n_xyz_lt_sm += 1
    elif nx > ns: n_xyz_gt_sm += 1
    else: n_equal += 1
    nz0 = sum(1 for z in xyz_atoms if z == 0)
    if nz0: n_z0_complexes += 1
    sm_to_xyz, kind = fast_match(sm_atoms, xyz_atoms)
    stats[kind] += 1
    for i, z in enumerate(sm_atoms):
        if sm_to_xyz[i] == -1:
            cnt = sum(1 for zz in xyz_atoms if zz == z)
            atom_reason[f'z={z}:{"absent" if cnt==0 else "exhausted"}'] += 1
    n_ph = 0; n_b = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        n_b += 1
        if sm_to_xyz[i] == -1 or sm_to_xyz[j] == -1:
            n_ph += 1
    if n_b:
        ph_bond_counts.append((n_ph / n_b, n_ph, n_b))

ph = np.array([p[0] for p in ph_bond_counts])
n_any_ph = int((ph > 0).sum())
print()
print('===== 诊断结果 =====')
print(f'解析 TM 块: {len(all_m)}, SMILES 无效: {n_smiles_invalid}, 建图成功: {len(ph_bond_counts)}')
print(f'匹配类型: {dict(stats)}')
print(f'XYZ 含 H 的络合物: {xyz_has_H} ({xyz_has_H/len(all_m)*100:.1f}%)')
print(f'原子数: XYZ<SMILES {n_xyz_lt_sm} | XYZ>SMILES {n_xyz_gt_sm} | 相等 {n_equal}')
print(f'含 z=0 原子(XYZ 表外元素)的络合物: {n_z0_complexes} ({n_z0_complexes/len(all_m)*100:.1f}%)')
print(f'兜底键络合物占比: {n_any_ph}/{len(ph_bond_counts)} = {n_any_ph/len(ph_bond_counts)*100:.1f}%')
print(f'兜底键占比中位数: {np.median(ph)*100:.1f}% (均值 {ph.mean()*100:.1f}%)')
print(f'未匹配 SMILES 原子原因 TOP15:')
for k, v in atom_reason.most_common(15):
    print(f'  {k}: {v}')
