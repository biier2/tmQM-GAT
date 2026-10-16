# -*- coding: utf-8 -*-
"""按 ensemble_cv_weighted.ipynb cell 2 逐字管线重建 vA 域 (90,317)
管线: XYZ 三 part 解析 (mol_idx bug: start_idx=len(all_m)) → TM 过滤 (含 len(atoms)<=1 跳过)
     → SMILES 建图 (仅 str 非空 + MolFromSmiles 非 None 失败) → NaN/inf 剔除
     → RandomState(42).permutation[:15%] = te_idx
自检: 重建 y_all[te_idx] vs ensemble_predictions.npz y_true
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, gzip
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'  # 相对 CWD d:\ml

df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
print(f'Loaded {len(df_y)} complexes, {len(targets)} targets', flush=True)
smiles_by_idx = df_y['SMILES'].tolist()
y_csv = df_y[targets].astype(float).values   # 与 notebook 一致: float64

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
    """notebook cell 2 parse_one 的标签部分 (逐字)"""
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
    print(f'  Part {part}...', flush=True)
    m, l = parse_one(f"{ROOT}/tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l)
    print(f'    {len(m)} TM complexes', flush=True)
print(f'Total: {len(all_m)} TM complexes parsed', flush=True)

# ===== SMILES 建图 (仅失败条件: 非 str/空 或 MolFromSmiles None) =====
failed = 0
kept_idx = []   # all_m 中的索引
for idx, m in enumerate(all_m):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '':
        failed += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        failed += 1; continue
    kept_idx.append(idx)

y_vals = np.array([all_l[i] for i in kept_idx], dtype=np.float64)
print(f'Built {len(y_vals):,} graphs ({failed} failed)', flush=True)

# NaN过滤 (notebook: isnan|isinf)
mask = ~(np.isnan(y_vals).any(axis=1) | np.isinf(y_vals).any(axis=1))
y_all = y_vals[mask]
kept_idx = [kept_idx[i] for i in range(len(kept_idx)) if mask[i]]
print(f'After NaN removal: {len(y_all):,} complexes (removed {(~mask).sum()})', flush=True)

# ===== 划分 (notebook cell 4 逐字) =====
idx_all = np.random.RandomState(42).permutation(len(y_all))
n_test = int(len(y_all) * 0.15)
te_idx = idx_all[:n_test]
print(f'Total: {len(y_all):,} | Fixed test set: {len(te_idx):,} (15%) | CV pool: {len(idx_all)-n_test:,} (85%)', flush=True)

# ===== 自检 vs ensemble_predictions.npz =====
ep = np.load('ensemble_predictions.npz')
print('ep keys:', list(ep.keys()), flush=True)
te_ref = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)
print(f'te_idx 与 ep 相同: {np.array_equal(te_idx, te_ref)}  (长度 {len(te_idx)} vs {len(te_ref)})', flush=True)
if not np.array_equal(te_idx, te_ref):
    print(f'  前 10: sim={te_idx[:10].tolist()}', flush=True)
    print(f'  前 10: ref={te_ref[:10].tolist()}', flush=True)
    print(f'  前 10: ref={te_ref[:10].tolist()}', flush=True)

yt_sim = y_all[te_idx].astype(np.float32)
d = np.abs(yt_sim - yt_ref).max(axis=1)
bad = int((d > 1e-4).sum())
print(f'y_true 自检: {bad} / {len(d)} 不一致', flush=True)
if bad:
    for j in np.where(d > 1e-4)[0][:10]:
        print(f'  j={j}: te={te_idx[j]}, sim={yt_sim[j][:3]}, ref={yt_ref[j][:3]}', flush=True)

# 若 te_idx 不同, 也试 split.npz 的 te_idx
try:
    split = np.load('ensemble_split.npz')
    print(f'ensemble_split te_idx 与 ep 相同: {np.array_equal(split["te_idx"], te_ref)}', flush=True)
    print(f'ensemble_split te_idx 与 sim 相同: {np.array_equal(split["te_idx"], te_idx)}', flush=True)
except FileNotFoundError:
    print('ensemble_split.npz 未找到', flush=True)

# 保存重建结果 (成功时)
if bad == 0 and np.array_equal(te_idx, te_ref):
    np.savez('vA_rebuild_labels.npz', kept_idx=np.array(kept_idx, dtype=np.int64),
             y_all=y_all.astype(np.float32))
    print('SAVED vA_rebuild_labels.npz (kept_idx = all_m 索引, y_all = 90,317x8)', flush=True)
