# -*- coding: utf-8 -*-
"""意见 8 深化: -1 兜底键影响面 (te 13,547) + 匹配质量分层敏感性
每 te 行: fast_match → 键级 -1 计数 (d=2.0 兜底键) → 分层:
  A. 无 -1 键 (匹配完整, 含 trivial+greedy)
  B. -1 键占比 ≤5%
  C. -1 键占比 >5%
分层上算 R5/3D 集成 mean R² (clip 口径) + per-property 3D R²
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
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
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try: atoms.append(atom_to_z(cparts[0]))
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx, "atoms": atoms})
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

def fast_match(sm_atoms, xyz_atoms):
    N = len(sm_atoms)
    if N == len(xyz_atoms) and all(sm_atoms[i] == xyz_atoms[i] for i in range(N)):
        return list(range(N))
    sm_to_xyz = [-1] * N; used = set()
    for i, z_sm in enumerate(sm_atoms):
        for j, z_xyz in enumerate(xyz_atoms):
            if z_sm == z_xyz and j not in used:
                sm_to_xyz[i] = j; used.add(j); break
    return sm_to_xyz

frac_miss = np.zeros(len(te_rows))
n_missing_atoms = np.zeros(len(te_rows))
n_atom_diff = np.zeros(len(te_rows), dtype=int)
for k, c in enumerate(te_rows):
    smi = y_csv[c] if c < len(y_csv) else ''
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None: frac_miss[k] = -1; continue
    sm_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    xyz_atoms = all_m[pos_by_csv[c]]['atoms']
    m = fast_match(sm_atoms, xyz_atoms)
    n_missing_atoms[k] = sum(1 for x in m if x == -1)
    n_bonds = len(list(mol.GetBonds()))
    nb_miss = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if m[i] == -1 or m[j] == -1: nb_miss += 1
    frac_miss[k] = nb_miss / n_bonds if n_bonds else 0.0

print('\n-1 兜底影响面 (te 13,547):')
ok = frac_miss >= 0
print(f'  无 -1 键的行: {(frac_miss==0).sum()} ({(frac_miss==0).sum()/ok.sum()*100:.2f}%)')
print(f'  0<占比≤5%:    {((frac_miss>0)&(frac_miss<=0.05)).sum()}')
print(f'  5%<占比≤20%:  {((frac_miss>0.05)&(frac_miss<=0.20)).sum()}')
print(f'  占比>20%:     {(frac_miss>0.20).sum()}')
print(f'  受影响行中 -1 键占比中位: {np.median(frac_miss[frac_miss>0])*100:.1f}%')
print(f'  受影响行中 -1 原子数中位: {np.median(n_missing_atoms[n_missing_atoms>0]):.0f}')

r5 = np.load('r5_smiles_va_ensemble.npz')
ep = np.load('ensemble_predictions.npz')
yt = ep['y_true']
CLIP = (0, 3)
p5 = r5['mean'].copy(); p3 = ep['mean'].copy()
for cc in CLIP:
    p5[:, cc] = np.maximum(p5[:, cc], 0); p3[:, cc] = np.maximum(p3[:, cc], 0)
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print('\n匹配质量分层敏感性 (mean R², clip):')
for name, mask in [('full', ok),
                   ('A 无-1键(匹配完整)', frac_miss == 0),
                   ('B 0<-1键≤5%', (frac_miss > 0) & (frac_miss <= 0.05)),
                   ('C -1键>5%', frac_miss > 0.05)]:
    m5 = np.mean([r2(yt[mask, i], p5[mask, i]) for i in range(8)])
    m3 = np.mean([r2(yt[mask, i], p3[mask, i]) for i in range(8)])
    print(f'  {name:22s} n={mask.sum():5d}  R5={m5:.4f}  3D={m3:.4f}  Δ={m5-m3:+.4f}')

print('\n分层 per-property 3D R² (A vs C):')
names = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
mA = frac_miss == 0; mC = frac_miss > 0.05
for i, n in enumerate(names):
    rA = r2(yt[mA, i], p3[mA, i]); rC = r2(yt[mC, i], p3[mC, i])
    print(f'  {n:14s} A={rA:.4f}  C={rC:.4f}  Δ={rC-rA:+.4f}')
np.save('te_frac_miss.npy', frac_miss)
print('\nSAVED te_frac_miss.npy')
