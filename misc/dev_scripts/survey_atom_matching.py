# -*- coding: utf-8 -*-
"""意见 8 协议审计: fast_match (SMILES—XYZ 原子匹配) 实际行为量化 + te 子集复验
1) 90,317 域: trivial(逐位一致)/greedy(贪心)/unmatched(有 -1) 比例
2) te 13,547: trivial 子集 vs 全集 — R5 SMILES vs 3D 集成 mean R² 复验
数据: data_tmQM/local_vB/tmQM_X{1..3}.xyz.gz + tmQM_y.csv + kept_cache.npz + vA_removed_kept.npy
预测: r5_smiles_va_ensemble.npz + ensemble_predictions.npz
"""
import sys, gzip, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
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
    """notebook cell2 parse_one 逐字 (存 atoms)"""
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
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        atoms.append(atom_to_z(cparts[0]))
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx, "atoms": atoms})
    return mols

print('parse xyz 3 parts...', flush=True)
all_m = []
for part in [1, 2, 3]:
    m = parse_one(f"{ROOT}/tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m)
print(f'TM parsed: {len(all_m)}', flush=True)

# kept 序列: 90,335 本地 → 删 18 → 90,317
z = np.load('kept_cache.npz'); mseq = z['csv_idx'].copy()
rem = np.load('vA_removed_kept.npy')
keep = np.ones(len(mseq), dtype=bool); keep[rem] = False
mseq = mseq[keep]
assert len(mseq) == 90317, len(mseq)

# 建 csv_idx -> all_m 位置 的映射 (mseq 即 all_m 的 csv_idx 序列)
pos_by_csv = {m['csv_idx']: i for i, m in enumerate(all_m)}
kept_pos = [pos_by_csv[c] for c in mseq]

# 对 90,317 kept 行分类匹配
def fast_match(sm_atoms, xyz_atoms):
    N = len(sm_atoms)
    if N == len(xyz_atoms) and all(sm_atoms[i] == xyz_atoms[i] for i in range(N)):
        return list(range(N)), 'trivial'
    sm_to_xyz = [-1] * N; used = set(); unmatched = False
    for i, z_sm in enumerate(sm_atoms):
        for j, z_xyz in enumerate(xyz_atoms):
            if z_sm == z_xyz and j not in used:
                sm_to_xyz[i] = j; used.add(j); break
        if sm_to_xyz[i] == -1: unmatched = True
    return sm_to_xyz, ('unmatched' if unmatched else 'greedy')

print('classify 90,317...', flush=True)
n_trivial = n_greedy = n_unmatched = n_fail = 0
te_flags = np.zeros(len(mseq), dtype=np.int8)  # 0=trivial 1=greedy 2=unmatched 3=fail
for k, pos in enumerate(kept_pos):
    csv_i = mseq[k]
    smi = y_csv[csv_i] if csv_i < len(y_csv) else ''
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None:
        n_fail += 1; te_flags[k] = 3; continue
    sm_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    _, kind = fast_match(sm_atoms, all_m[pos]['atoms'])
    if kind == 'trivial': n_trivial += 1; te_flags[k] = 0
    elif kind == 'greedy': n_greedy += 1; te_flags[k] = 1
    else: n_unmatched += 1; te_flags[k] = 2

print(f'90,317 域匹配分类:')
print(f'  trivial   {n_trivial} ({n_trivial/90317*100:.2f}%)')
print(f'  greedy    {n_greedy} ({n_greedy/90317*100:.2f}%)')
print(f'  unmatched {n_unmatched} ({n_unmatched/90317*100:.2f}%)')
print(f'  fail      {n_fail}', flush=True)

# te 子集复验
sp = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_idx = sp['te_idx']
te_fl = te_flags[te_idx]
r5 = np.load('r5_smiles_va_ensemble.npz')
ep = np.load('ensemble_predictions.npz')
yt = ep['y_true']
CLIP = (0, 3)
p5 = r5['mean'].copy(); p3 = ep['mean'].copy()
for c in CLIP:
    p5[:, c] = np.maximum(p5[:, c], 0); p3[:, c] = np.maximum(p3[:, c], 0)

def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def meanr2(mask, name):
    m5 = np.mean([r2(yt[mask, i], p5[mask, i]) for i in range(8)])
    m3 = np.mean([r2(yt[mask, i], p3[mask, i]) for i in range(8)])
    print(f'  {name:14s} n={mask.sum():6d}  R5 mean={m5:.4f}  3D mean={m3:.4f}  Δ={m5-m3:+.4f}', flush=True)
    return m5, m3

print('\nte 13,547 子集复验 (clip 口径):')
print(f'  te 匹配分类: trivial {(te_fl==0).sum()} / greedy {(te_fl==1).sum()} / unmatched {(te_fl==2).sum()} / fail {(te_fl==3).sum()}')
meanr2(np.ones(len(te_fl), dtype=bool), 'full')
meanr2(te_fl == 0, 'trivial-only')
meanr2(te_fl == 1, 'greedy-only')

out = dict(n_trivial=n_trivial, n_greedy=n_greedy, n_unmatched=n_unmatched, n_fail=n_fail,
           te_trivial=int((te_fl==0).sum()), te_greedy=int((te_fl==1).sum()),
           te_unmatched=int((te_fl==2).sum()), te_fail=int((te_fl==3).sum()))
json.dump(out, open('survey_atom_matching.json', 'w'))
np.save('te_match_flags.npy', te_fl)
print('\nSAVED survey_atom_matching.json + te_match_flags.npy')
