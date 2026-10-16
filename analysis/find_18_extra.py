# -*- coding: utf-8 -*-
"""定位我的 90,335 kept 里比 notebook 90,317 多出的 18 行
方法: ep te_idx t → 内容在 y_my[t..t+20] 中搜索首个精确匹配 → t' = t + c(t)
c(t) 为阶梯函数, 阶梯位置 = 多出的行 (notebook 中不存在)
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
y_csv = df_y[targets].astype(float).values

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
y_my = np.array(kept_y, dtype=np.float64).astype(np.float32)
print(f'y_my: {y_my.shape}', flush=True)

ep = np.load('ensemble_predictions.npz')
te = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)

# t → t' 内容对齐 (在 t..t+25 窗口内找首个精确匹配)
c = np.full(len(te), -1, dtype=np.int64)
b = y_my.tobytes()
# 用浮点 exact 比较: 直接逐样本窗口扫描
y_rows = {y_my[i].tobytes(): i for i in range(len(y_my))}  # 首匹配行
shift = np.full(len(te), -1, dtype=np.int64)
for i in range(len(te)):
    tb = yt_ref[i].tobytes()
    r0 = y_rows.get(tb, -1)
    # 真正的行应在 te[i] 附近 (te[i] .. te[i]+20)
    if r0 >= te[i]:
        shift[i] = r0 - te[i]
    else:
        # 内容重复导致首匹配偏早, 窗口内线性扫
        for d in range(0, 26):
            j = te[i] + d
            if j < len(y_my) and y_my[j].tobytes() == tb:
                shift[i] = d
                break
print(f'shift 未定: {(shift < 0).sum()}', flush=True)
from collections import Counter
print('shift 分布:', sorted(Counter(shift).items()), flush=True)

# 阶梯: shift 递增 0→18
ss = np.unique(shift)
print('shift 唯一值:', ss.tolist(), flush=True)
# 对每个 shift 值, 取该值出现的最小 te 位置区间
for s in sorted(set(shift)):
    m = shift == s
    if m.any():
        print(f'  shift={s}: te ∈ [{te[m].min()}, {te[m].max()}], n={m.sum()}', flush=True)
# 多出行的位置: shift 从 s 跳到 s+1 的 te 区间 → 该行在 my-位置 ∈ (前区间max, 后区间min]
jumps = []
sv = sorted(set(shift))
for a2, b2 in zip(sv[:-1], sv[1:]):
    ma = (shift == a2); mb = (shift == b2)
    lo = te[ma].max(); hi = te[mb].min()
    jumps.append((a2, b2, lo, hi))
    print(f'  跳变 {a2}→{b2}: 多出 1 行, my-位置 ∈ ({lo}, {hi}]', flush=True)
