# -*- coding: utf-8 -*-
"""诊断 ~995 个对齐断裂条目: 打印 te、yt_ref 内容所在 csv 行、我的 mseq 邻域
并重建 mseq_nb 候选 → 看断裂是否 = mol_idx 区域的行序差异
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, gzip
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
smiles_by_idx = df_y['SMILES'].tolist()
y_csv = df_y[targets].astype(float).values.astype(np.float32)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

def parse_one(xyz_p, start_idx=0):
    mols, labels = [], []
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
                        coords.append([float(x) for x in cparts[1:4]]); atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx})
            labels.append(y_csv[mol_idx] if mol_idx < len(y_csv) else np.full(len(targets), np.nan))
    return mols, labels

all_m, all_l = [], []
for part in [1, 2, 3]:
    m, l = parse_one(f"{ROOT}/tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l)

kept, kept_y = [], []
for idx, m in enumerate(all_m):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '':
        continue
    if Chem.MolFromSmiles(smi) is None:
        continue
    kept.append(idx); kept_y.append(all_l[idx])
y_my = np.array(kept_y, dtype=np.float32)
mseq = np.array([all_m[i]["csv_idx"] for i in kept], dtype=np.int64)   # 我的 mol_idx 序列
print(f'y_my {y_my.shape}, mseq 值域 [{mseq.min()}, {mseq.max()}]', flush=True)

ep = np.load('ensemble_predictions.npz')
te = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)

# 全局 csv 内容 → 行 (首匹配)
row_of_bytes = {}
for i in range(len(y_csv)):
    b = y_csv[i].tobytes()
    if b not in row_of_bytes:
        row_of_bytes[b] = i

# 断裂条目: yt_ref[t] 与 y_my[t..t+25] 全不同
broken = []
for i in range(len(te)):
    t = te[i]
    tb = yt_ref[i].tobytes()
    hit = -1
    for d in range(0, 26):
        if t + d < len(y_my) and y_my[t + d].tobytes() == tb:
            hit = d; break
    if hit < 0:
        r = row_of_bytes.get(tb, -1)
        broken.append((i, t, r))
print(f'断裂 {len(broken)} / {len(te)}', flush=True)
print('前 15 个: (i, te, csv首匹配行)', flush=True)
for i, t, r in broken[:15]:
    print(f'  i={i}: te={t}, csv_row={r}, mseq[t]={mseq[t]}, mseq[t±2]={mseq[t-2:t+3].tolist()}', flush=True)
print('csv首匹配行分布:', flush=True)
from collections import Counter
rs = [r for _, _, r in broken]
print('  min/max:', min(rs), max(rs), flush=True)
for lo in range(0, 108541, 4000):
    c = sum(1 for r in rs if lo <= r < lo+4000)
    if c: print(f'  [{lo},{lo+4000}): {c}', flush=True)
