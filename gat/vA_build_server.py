# -*- coding: utf-8 -*-
"""vA 域 (90,317 图 / 13,547 测试) 服务器重建 —— ensemble_cv_weighted.ipynb cell 2-4 逐字管线
用途: 生成 R5 (SMILES 9 成员集成同协议重训) 的输入 vA_domain_smiles.pt
要求: 与 ensemble_cv_weighted.ipynb 构建时【同一个 conda 环境】(RDKit 版本一致) 与【同一份数据】
用法: python vA_build_server.py
输出: vA_domain_smiles.pt + 自检报告 (te_idx / y_true 与 ensemble_predictions.npz 全等才保存)
"""
import sys, os
import numpy as np, pandas as pd, gzip
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
import torch

# ---- 配置 (按服务器实际路径修改) ----
ROOT = '.'                     # tmQM_X{1,2,3}.xyz.gz 与 tmQM_y.csv 所在目录
EP   = 'ensemble_predictions.npz'   # 3D 集成预测 (notebook 产物, 含 te_idx / y_true)
OUT  = 'vA_domain_smiles.pt'

print(f'RDKit {Chem.rdBase.rdkitVersion} | torch {torch.__version__} | python {sys.version.split()[0]}', flush=True)

df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
smiles_by_idx = df_y['SMILES'].tolist()
codes = df_y['CSD_code'].tolist()

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
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    atoms.append(atom_to_z(cparts[0]))
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx})
            labels.append(df_y.iloc[mol_idx][targets].astype(float).values)
    return mols, labels

all_m, all_l = [], []
for part in [1, 2, 3]:
    m, l = parse_one(f"{ROOT}/tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    print(f'  part{part}: {len(m)} TM complexes', flush=True)
    all_m.extend(m); all_l.extend(l)
print(f'TM: {len(all_m)} (期望 96984 = 27318+34065+35601)', flush=True)

smiles_list, labels_list, csv_list, code_list = [], [], [], []
failed = 0
for m, lab in zip(all_m, all_l):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '':
        failed += 1; continue
    if Chem.MolFromSmiles(smi) is None:
        failed += 1; continue
    smiles_list.append(smi); labels_list.append(lab)
    csv_list.append(csv_i); code_list.append(codes[csv_i] if csv_i < len(codes) else '')
print(f'Built {len(smiles_list)} graphs ({failed} failed) — 期望 90317 / 6667', flush=True)

labels = np.array(labels_list, dtype=np.float64)
nan_mask = ~np.isnan(labels).any(axis=1)
smiles_list = [s for s, k in zip(smiles_list, nan_mask) if k]
labels = labels[nan_mask]
csv_list = [c for c, k in zip(csv_list, nan_mask) if k]
code_list = [c for c, k in zip(code_list, nan_mask) if k]
print(f'After NaN removal: {len(smiles_list)} — 期望 90317', flush=True)

n = len(smiles_list)
idx_all = np.random.RandomState(42).permutation(n)
n_test = int(n * 0.15)
te_idx = idx_all[:n_test].copy()
tr_idx = idx_all[n_test:].copy()
print(f'Total {n} | Fixed test set: {n_test} (15%) | CV pool: {n - n_test} (85%)', flush=True)

# ---- 自检 vs ensemble_predictions.npz ----
ep = np.load(EP)
te_ref = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float64)
te_eq = (n == 90317) and np.array_equal(te_idx, te_ref)
maxd = float(np.abs(labels[te_idx] - yt_ref).max()) if (n == 90317 and labels.shape[1] == yt_ref.shape[1]) else float('inf')
print(f'自检: te_idx 全等 = {te_eq} | y_true max|diff| = {maxd:.3e}', flush=True)
ok = te_eq and maxd <= 1e-6

if not ok:
    print('自检未通过 → 尝试自动校正 (当前 RDKit 与 notebook 不一致时, 按 npz 对齐删除多余行)', flush=True)
    lab = labels.copy()
    sm = list(smiles_list); cs = list(csv_list); cd = list(code_list)
    def align(lb):
        sh = np.full(len(te_ref), -1, dtype=np.int64)
        for i in range(len(te_ref)):
            t = te_ref[i]; ref = yt_ref[i]
            for d in range(0, 31):
                if t + d < len(lb) and np.abs(lb[t + d] - ref).max() <= 1e-6:
                    sh[i] = d; break
        return sh
    sh = align(lab)
    print(f'校正前: n={len(lab)}, 未命中 {(sh<0).sum()}, max shift {sh.max() if len(sh) else -1}', flush=True)
    guard = 0
    while len(lab) > 90317 and guard < 60:
        guard += 1
        ma = (sh == 0); mb = (sh == 1)
        lo = int(te_ref[ma].max()) if ma.any() else -1
        hi = int(te_ref[mb].min()) if mb.any() else -1
        found = None
        cands = range(lo + 1, hi + 1) if hi >= lo + 1 else range(lo + 1, min(lo + 201, len(lab)))
        for p in cands:
            lb2 = np.delete(lab, p, axis=0)
            okp = True
            for i in range(len(te_ref)):
                t = te_ref[i]; ref = yt_ref[i]
                d2 = -1
                for d in range(0, 31):
                    if t + d < len(lb2) and np.abs(lb2[t + d] - ref).max() <= 1e-6:
                        d2 = d; break
                exp = sh[i] - (1 if t >= p else 0)
                if d2 != exp or d2 < 0:
                    okp = False; break
            if okp:
                found = p; break
        if found is None:
            print(f'校正失败: 第 {guard} 步无候选', flush=True)
            break
        print(f'  删 pos={found} csv_idx={cs[found]} {cd[found]}: {sm[found][:60]}', flush=True)
        lab = np.delete(lab, found, axis=0)
        del sm[found], cs[found], cd[found]
        sh = align(lab)
    if len(lab) == 90317 and (sh < 0).sum() == 0 and sh.max() == 0:
        idx_all = np.random.RandomState(42).permutation(90317)
        te_idx = idx_all[:13547].copy(); tr_idx = idx_all[13547:].copy()
        te_eq = np.array_equal(te_idx, te_ref)
        maxd = float(np.abs(lab[te_idx] - yt_ref).max()) if te_eq else float('inf')
        print(f'校正后自检: te_idx 全等 = {te_eq} | y_true max|diff| = {maxd:.3e}', flush=True)
        if te_eq and maxd <= 1e-6:
            print('自动校正成功 (注意: 当前 env 与 notebook 的 RDKit 不同, 已按 notebook 输出校正)', flush=True)
            smiles_list, labels, csv_list, code_list = sm, lab, cs, cd
            ok = True
    if not ok:
        print('自动校正失败。请检查: 1) 是否与 notebook 同一 conda 环境 2) 数据文件是否相同。把完整输出发回。', flush=True)
        sys.exit(1)

if ok:
    torch.save({
        'smiles': smiles_list,
        'labels': labels.astype(np.float32),
        'labels64': labels,
        'te_idx': te_idx.astype(np.int64),
        'tr_idx': tr_idx.astype(np.int64),
        'csv_idx': np.array(csv_list, dtype=np.int64),
        'csd_code': code_list,
    }, OUT)
    print(f'SAVED {OUT}', flush=True)
    print('R5 加载示例: d = torch.load("vA_domain_smiles.pt"); smis=d["smiles"]; y=d["labels"]; te=d["te_idx"]; tr=d["tr_idx"]', flush=True)
