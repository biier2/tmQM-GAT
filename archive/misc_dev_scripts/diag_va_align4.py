# -*- coding: utf-8 -*-
"""Diagnostic 4 (decisive): is the paper domain the TM kept subsequence in xyz block order?
Unified theory: the paper main-domain row order comes from the xyz file block order (TM subsequence), not from csv row order.
  - the mystery of the 348 rows = atom_to_z lacks Zr/Nb/Tc/Hf/Ta (confirmed: the blocks all contain a 28-table TM)
  - the mystery of the 2 jumps = 2 disagreements between xyz block order and csv row order
Verification: build the kept sequence in xyz block order with the complete element table -> truncate to the first 90,317 -> self-check against ep y_true.
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}
TM28 = [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
smi_list = df['SMILES'].tolist()
codes = df['CSD_code'].tolist()
row_of_code = {str(c): i for i, c in enumerate(codes)}
n = len(df)

# ===== 1. Code list in xyz block order =====
xyz_codes = []
for part in [1, 2, 3]:
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        it = iter(f)
        while True:
            try:
                ln = next(it).strip()
            except StopIteration:
                break
            if not ln or not ln.isdigit():
                continue
            n_at = int(ln)
            title = next(it).strip()
            code = title.split('CSD_code = ')[1].split('|')[0].strip() if 'CSD_code = ' in title else None
            for _ in range(n_at):
                next(it, None)
            if code is not None:
                xyz_codes.append(code)
print(f'xyz block count: {len(xyz_codes):,}', flush=True)

# ===== 2. Points of disagreement between xyz block order and csv row order =====
xyz_rows = np.array([row_of_code.get(c, -1) for c in xyz_codes], dtype=np.int64)
print(f'Unpaired codes: {(xyz_rows < 0).sum()}', flush=True)
diffs = np.where(np.diff(xyz_rows) < 0)[0]
print(f'xyz block order vs csv row order: decreasing violations {len(diffs):,}', flush=True)
print('Decreasing points (xyz block position -> csv row):', flush=True)
for d in diffs[:15]:
    print(f'  xyz block {d}: csv {xyz_rows[d]} -> xyz block {d+1}: csv {xyz_rows[d+1]}', flush=True)

# ===== 3. TM kept sequence with the complete table (xyz block order) =====
kept_rows = []
n_tm_blocks = 0
for part in [1, 2, 3]:
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        it = iter(f)
        while True:
            try:
                ln = next(it).strip()
            except StopIteration:
                break
            if not ln or not ln.isdigit():
                continue
            n_at = int(ln)
            title = next(it).strip()
            code = title.split('CSD_code = ')[1].split('|')[0].strip() if 'CSD_code = ' in title else None
            syms = []
            for _ in range(n_at):
                ln2 = next(it, None)
                if ln2 is None: break
                cparts = ln2.strip().split()
                if len(cparts) >= 4:
                    syms.append(cparts[0])
            if code is None:
                continue
            zs = [Z_OF.get(s, 0) for s in syms]
            if not any(z in TM28 for z in zs) or len(syms) <= 1:
                continue
            n_tm_blocks += 1
            r = row_of_code.get(code, -1)
            if r < 0:
                continue
            s = smi_list[r]
            if not isinstance(s, str) or s == '':
                continue
            if Chem.MolFromSmiles(s) is None:
                continue
            if not np.isfinite(y[r]).all():
                continue
            kept_rows.append(r)
kept_rows = np.array(kept_rows, dtype=np.int64)
print(f'\nComplete-table TM blocks: {n_tm_blocks:,}   kept sequence: {len(kept_rows):,}', flush=True)
print(f'csv row range of the first 90,317 of the kept sequence: [{kept_rows[:90317].min()}, {kept_rows[:90317].max()}]', flush=True)

# ===== 4. Self-check: first 90,317 truncation vs ep =====
ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)
yt = y[kept_rows[:90317]][te_idx]
d = np.abs(yt - yt_ref).max(axis=1)
bad = np.where(d > 1e-4)[0]
print(f'\nSelf-check: first 90,317 truncation y[te_idx] vs ep y_true — mismatching rows {len(bad):,} / 13,547', flush=True)
if len(bad):
    for b in bad[:10]:
        print(f'  te_idx[{b}]={te_idx[b]}: truncated y={yt[b].round(4).tolist()}  ep={yt_ref[b].round(4).tolist()}', flush=True)
else:
    print('*** All match! The paper main domain = the first 90,317 of the TM kept sequence in xyz block order ***', flush=True)

# ===== 5. Comparison with the observed f(i) curve =====
d2 = np.load('diag_va_align2.npz')
csv_sorted = d2['csv_sorted']   # csv row numbers of the 13,547 true rows obtained from the reverse lookup (sorted by te_idx)
match = sum(1 for i, c in zip(sorted(te_idx), csv_sorted) if kept_rows[i] == c)
print(f'\nxyz-block-order kept sequence vs reverse-lookup row numbers: position-by-position agreement {match:,} / 13,547', flush=True)
