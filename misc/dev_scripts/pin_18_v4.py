# -*- coding: utf-8 -*-
"""锁定 18 个多余行 (v4): 贪心, 每步区间 = 当前 0→1 跳变, 验证 d2 == sh[i]-(t>=p) 且 d2>=0
(v3 的 s=step 是错的: 每删一行, 所有层降 1, 当前跳变永远是 0→1)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, gzip, os
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

CACHE = 'kept_cache.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE)
    mseq = z['csv_idx']; y_my = z['y'].copy()
    orig = np.arange(len(mseq), dtype=np.int64)
    print(f'[cache] kept {len(mseq)} 行', flush=True)
else:
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
    mseq = np.array([all_m[i]["csv_idx"] for i in kept], dtype=np.int64)
    y_my = np.array(kept_y, dtype=np.float32)
    orig = np.arange(len(mseq), dtype=np.int64)
    np.savez(CACHE, csv_idx=mseq, y=y_my)
    print(f'[rebuilt] kept {len(mseq)} 行', flush=True)

ep = np.load('ensemble_predictions.npz')
te = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)

def align(yy):
    sh = np.full(len(te), -1, dtype=np.int64)
    for i in range(len(te)):
        t = te[i]; tb = yt_ref[i].tobytes()
        for d in range(0, 31):
            if t + d < len(yy) and yy[t + d].tobytes() == tb:
                sh[i] = d; break
    return sh

def try_drop(yy, sh, p):
    yy2 = np.delete(yy, p, axis=0)
    for i in range(len(te)):
        t = te[i]; tb = yt_ref[i].tobytes()
        d2 = -1
        for d in range(0, 31):
            if t + d < len(yy2) and yy2[t + d].tobytes() == tb:
                d2 = d; break
        exp = sh[i] - (1 if t >= p else 0)
        if d2 != exp or d2 < 0:
            return False
    return True

sh = align(y_my)
print(f'初始: 未命中 {(sh<0).sum()}, max shift {sh.max()}, n={len(y_my)}', flush=True)
removed = []
for step in range(18):
    ma = (sh == 0); mb = (sh == 1)
    lo = int(te[ma].max()) if ma.any() else -1
    hi = int(te[mb].min()) if mb.any() else -1
    found = None
    cands = range(lo + 1, hi + 1) if hi >= lo + 1 else range(lo + 1, min(lo + 201, len(y_my)))
    for p in cands:
        if try_drop(y_my, sh, p):
            found = p; break
    if found is None:
        print(f'step {step}: 区间 ({lo},{hi}] 未找到', flush=True)
        break
    c = int(mseq[found]); k = int(orig[found])
    print(f'step {step}: 删 pos={found} kept_idx={k} csv_idx={c} {codes[c]}: {smiles_by_idx[c]!r}', flush=True)
    removed.append(k)
    y_my = np.delete(y_my, found, axis=0); mseq = np.delete(mseq, found); orig = np.delete(orig, found)
    sh = align(y_my)

print(f'\n最终: {len(y_my)} 行 (删 {len(removed)}), 未命中 {(sh<0).sum()}, max shift {sh.max() if len(sh) else -1}', flush=True)
if len(y_my) == 90317 and (sh < 0).sum() == 0 and sh.max() == 0:
    idx_all = np.random.RandomState(42).permutation(90317)
    te_sim = idx_all[:13547]
    print(f'te_idx 与 ep 相同: {np.array_equal(te_sim, te)}', flush=True)
    d = np.abs(y_my[te_sim] - yt_ref).max(axis=1)
    print(f'y_true 自检: {(d > 1e-4).sum()} / {len(d)} 不一致', flush=True)
    if (d > 1e-4).sum() == 0:
        np.savez('vA_rebuild.npz', kept_idx=orig.copy())
        np.save('vA_removed_kept.npy', np.array(removed))
        print('SAVED vA_rebuild.npz + vA_removed_kept.npy', flush=True)
