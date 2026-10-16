# -*- coding: utf-8 -*-
"""意见 13 深化: 相同连接表组内 xyz 构型差异检查
242 组 (canon 相同) → 去 csv_idx 重复 (mseq 有重复行) → 组内不同 csv 条目
对组内不同条目: 元素多集是否一致 + 金属-配体键长指纹 (排序后) 差异 → 判定构型差异
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import defaultdict
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

y_csv = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')['SMILES'].tolist()

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
            if i >= len(xyz_blocks): break; i += 1
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]])
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx, "atoms": atoms, "coords": np.array(coords)})
    return mols

print('parse...', flush=True)
all_m = []
for part in [1, 2, 3]:
    all_m.extend(parse_one(f"{ROOT}/tmQM_X{part}.xyz.gz", start_idx=len(all_m)))
pos_by_csv = {m['csv_idx']: i for i, m in enumerate(all_m)}

z = np.load('kept_cache.npz'); mseq = z['csv_idx'].copy()
rem = np.load('vA_removed_kept.npy')
keep = np.ones(len(mseq), dtype=bool); keep[rem] = False
mseq = mseq[keep]
sp = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_rows = mseq[sp['te_idx']]

canon = []
for c in te_rows:
    s = y_csv[c]
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    canon.append(Chem.MolToSmiles(m, isomericSmiles=False) if m else None)
groups = defaultdict(list)
for i, cc in enumerate(canon):
    if cc: groups[cc].append(i)

# 去 csv 重复: 组内不同 csv_idx 的 te 行位置
n_true_pairs = 0
n_same_geom = 0
n_diff_geom = 0
examples = []
for cc, v in groups.items():
    uniq_csv = {}
    for i in v:
        uniq_csv.setdefault(te_rows[i], []).append(i)
    entries = list(uniq_csv.items())
    if len(entries) < 2: continue  # 全部是同一 csv 行的重复副本
    n_true_pairs += 1
    # 元素多集 + 键长指纹
    sigs = []
    for csv_i, ii in entries:
        m = all_m[pos_by_csv[csv_i]]
        atoms = m['atoms']; coords = m['coords']
        tm = [k for k, a in enumerate(atoms) if is_TM(a)]
        if not tm: sigs.append(None); continue
        d = []
        for k in tm:
            for j in range(len(atoms)):
                if k != j:
                    d.append(np.linalg.norm(coords[k] - coords[j]))
        d = np.sort(np.array(d))
        sigs.append((tuple(sorted(atoms)), d.round(3)))
    if any(s is None for s in sigs): continue
    elems = set(s[0] for s in sigs)
    if len(elems) > 1:
        n_diff_geom += 1
        examples.append((cc[:50], len(entries)))
        continue
    dists = sigs[0][1]
    if all(np.allclose(dists, s[1], atol=0.01) for s in sigs[1:]):
        n_same_geom += 1
    else:
        n_diff_geom += 1
        examples.append((cc[:50], len(entries)))

print(f'\n相同连接表组 (canon 同): {len(groups)}')
print(f'  含 ≥2 个不同 csv 条目的组: {n_true_pairs}')
print(f'  其中 xyz 键长指纹相同 (同构型重复条目): {n_same_geom}')
print(f'  xyz 键长指纹不同 (潜在构型差异): {n_diff_geom}')
print(f'  构型差异示例 (top10):')
for e in examples[:10]: print(f'    {e}')
