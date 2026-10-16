# -*- coding: utf-8 -*-
"""精确锁定 18 个多余行: 阶梯跳变区间 → 候选 p → 删行后全量重对齐验证
输出: 18 行的 my-位置 p、csv_idx (mol_idx)、CSD_code、SMILES
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
codes = df_y['CSD_code'].tolist()
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
mseq = np.array([all_m[i]["csv_idx"] for i in kept], dtype=np.int64)

ep = np.load('ensemble_predictions.npz')
te = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)

def align(yy):
    """窗口精确对齐: shift[t] 使 yy[te[t]+shift] == yt_ref[t], 未命中 -1"""
    sh = np.full(len(te), -1, dtype=np.int64)
    for i in range(len(te)):
        t = te[i]; tb = yt_ref[i].tobytes()
        for d in range(0, 26):
            if t + d < len(yy) and yy[t + d].tobytes() == tb:
                sh[i] = d; break
    return sh

sh = align(y_my)
print(f'初始: 未命中 {(sh < 0).sum()}, 最大 shift {sh.max()}', flush=True)

# 阶梯跳变区间: shift s 与 s+1 之间, 删行位置 p ∈ (te_max_s, te_min_{s+1}]
removed = []
orig_idx = np.arange(len(kept), dtype=np.int64)   # 与 y_my 同步的原始 kept 索引
for s in range(0, 18):
    ma = (sh == s); mb = (sh == s + 1)
    lo = int(te[ma].max()) if ma.any() else -1
    hi = int(te[mb].min()) if mb.any() else -1
    # 候选区间 (lo, hi]; 每个候选 p: 删除行 p 后重新对齐, 检查是否恰好消除该跳变
    found = None
    for p in range(lo + 1, hi + 1):
        yy2 = np.delete(y_my, p, axis=0)
        sh2 = align(yy2)
        if (sh2 < 0).sum() == 0 and int(sh2.max()) <= 18 - len(removed) - 1:
            # 检查新 shift 与期望: 所有 te>=p 的 shift 减 1
            ok = True
            for i in range(len(te)):
                exp = sh[i] - (1 if te[i] >= p else 0)
                if exp < 0:
                    exp = exp  # 该样本本身在删行处, 与 yt_ref 对不上则跳过
                if sh2[i] != max(exp, 0) and not (te[i] == p):
                    ok = False; break
            if ok:
                found = p; break
    if found is None:
        print(f'  跳变 {s}->{s+1}: 候选区间 ({lo}, {hi}] 未找到', flush=True)
        continue
    k = int(orig_idx[found])
    removed.append(found)
    y_my = np.delete(y_my, found, axis=0)
    mseq = np.delete(mseq, found)
    orig_idx = np.delete(orig_idx, found)
    sh = align(y_my)
    print(f'  跳变 {s}->{s+1}: 删 my-位置 {found} (kept_idx={k}, csv_idx={all_m[k]["csv_idx"]}, '
          f'{codes[all_m[k]["csv_idx"]]}, SMILES={smiles_by_idx[all_m[k]["csv_idx"]]})', flush=True)

print(f'\n共删 {len(removed)} 行; 最终 {len(y_my)} 行; 未命中 {(sh<0).sum()}, max shift {sh.max() if len(sh) else -1}', flush=True)
if len(y_my) == 90317:
    # 最终验证: te_idx = RandomState(42).permutation(90317)[:13547] 与 ep 相同
    idx_all = np.random.RandomState(42).permutation(90317)
    te_sim = idx_all[:13547]
    print(f'te_idx 与 ep 相同: {np.array_equal(te_sim, ep["te_idx"])}', flush=True)
    yt_sim = y_my[te_sim]
    d = np.abs(yt_sim - yt_ref).max(axis=1)
    print(f'y_true 最终自检: {(d > 1e-4).sum()} / {len(d)} 不一致', flush=True)
    if (d > 1e-4).sum() == 0:
        np.savez('vA_rebuild.npz', kept_idx=orig_idx.copy())
        print('SAVED vA_rebuild.npz (kept_idx = all_m 索引, 长度 90317)', flush=True)
