# -*- coding: utf-8 -*-
"""锁定 18 个删除行 (v3): 贪心逐级, 候选区间 = 当前阶梯跳变, 全量精确验证
验证规则: 删 p 后 sh2[i] == sh[i] - (te[i] >= p) 对所有 i 成立 (align 无噪声, csv 仅 1 对重复)
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
orig = np.arange(len(kept), dtype=np.int64)

ep = np.load('ensemble_predictions.npz')
te = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)

# 预计算窗口匹配表 W[i, d]: y_my[te[i]+d] == yt_ref[i] (d=0..25)
W = np.zeros((len(te), 26), dtype=bool)
for i in range(len(te)):
    t = te[i]; tb = yt_ref[i].tobytes()
    for d in range(26):
        if t + d < len(y_my) and y_my[t + d].tobytes() == tb:
            W[i, d] = True

def sh_of(removed_set):
    """由删除集计算精确 shift: c(t) = #{r ∈ removed : r <= t}, sh = 窗口首个匹配 - t"""
    sh = np.full(len(te), -1, dtype=np.int64)
    for i in range(len(te)):
        t = te[i]
        for d in range(26):
            if W[i, d]:
                pos = t + d
                c = 0
                for r in removed_set:
                    if r <= pos: c += 1
                sh[i] = pos - c - t
                break
    return sh

# 初始 (无删除): sh = c = 0 → 全 0? 不对: W 表基于当前 y_my, 删除后 W 失效!
# 直接每轮重建 W 太慢 → 换法: 每轮维护 y_my 删行, 重建 W (13.5k×26 字节比较, 快)
def align():
    sh = np.full(len(te), -1, dtype=np.int64)
    for i in range(len(te)):
        t = te[i]; tb = yt_ref[i].tobytes()
        for d in range(26):
            if t + d < len(y_my) and y_my[t + d].tobytes() == tb:
                sh[i] = d; break
    return sh

sh = align()
print(f'初始: 未命中 {(sh<0).sum()}, max shift {sh.max()}', flush=True)
removed_kept = []
for step in range(18):
    # 当前最低未消解的跳变
    s = step
    ma = (sh == s); mb = (sh == s + 1)
    lo = int(te[ma].max()) if ma.any() else -1
    hi = int(te[mb].min()) if mb.any() else -1
    found = None
    for p in range(lo + 1, hi + 2):
        # 删 p 后的期望: sh2[i] = sh[i] - (te[i] >= p)
        yy2 = np.delete(y_my, p, axis=0)
        sh2 = np.full(len(te), -1, dtype=np.int64)
        ok = True
        for i in range(len(te)):
            t = te[i]; tb = yt_ref[i].tobytes()
            d2 = -1
            for d in range(26):
                if t + d < len(yy2) and yy2[t + d].tobytes() == tb:
                    d2 = d; break
            exp = sh[i] - (1 if t >= p else 0)
            if d2 != exp:
                ok = False; break
        if ok:
            found = p; break
    if found is None:
        print(f'step {step}: 区间 ({lo}, {hi}] 未找到', flush=True)
        # 放宽: 全范围扫 (慢但仅剩少数步)
        for p in range(0, len(y_my)):
            yy2 = np.delete(y_my, p, axis=0)
            ok = True
            for i in range(len(te)):
                t = te[i]; tb = yt_ref[i].tobytes()
                d2 = -1
                for d in range(26):
                    if t + d < len(yy2) and yy2[t + d].tobytes() == tb:
                        d2 = d; break
                exp = sh[i] - (1 if t >= p else 0)
                if d2 != exp:
                    ok = False; break
            if ok:
                found = p
                print(f'step {step}: 全范围找到 p={p}', flush=True)
                break
        if found is None:
            print(f'step {step}: 全范围也未找到, 终止', flush=True)
            break
    k = int(orig[found])
    print(f'step {step}: 删 my-位置 {found}, kept_idx={k}, csv_idx={mseq[found]}, '
          f'{codes[mseq[found]]}, SMILES={repr(smiles_by_idx[mseq[found]])[:70]}', flush=True)
    removed_kept.append(k)
    y_my = np.delete(y_my, found, axis=0); mseq = np.delete(mseq, found); orig = np.delete(orig, found)
    sh = align()

print(f'\n最终: {len(y_my)} 行 (删 {len(removed_kept)}), 未命中 {(sh<0).sum()}, max shift {sh.max() if len(sh) else -1}', flush=True)
if len(y_my) == 90317 and (sh < 0).sum() == 0 and sh.max() == 0:
    idx_all = np.random.RandomState(42).permutation(90317)
    te_sim = idx_all[:13547]
    print(f'te_idx 与 ep 相同: {np.array_equal(te_sim, ep["te_idx"])}', flush=True)
    yt_sim = y_my[te_sim]
    d = np.abs(yt_sim - yt_ref).max(axis=1)
    print(f'y_true 自检: {(d > 1e-4).sum()} / {len(d)} 不一致', flush=True)
    if (d > 1e-4).sum() == 0:
        np.savez('vA_rebuild.npz', kept_idx=orig.copy())
        np.save('vA_removed_kept.npy', np.array(removed_kept))
        print('SAVED vA_rebuild.npz + vA_removed_kept.npy', flush=True)
