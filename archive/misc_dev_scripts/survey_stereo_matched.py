# -*- coding: utf-8 -*-
"""Comment 13: build the matched set and answer "when does 3D help" directly
For the 92 groups with truly identical connectivity tables (different csv entries): pair within each group
- max bond-length fingerprint difference (strength of the configurational difference)
- y_true differences over the 8 properties vs R5/3D ensemble prediction differences -> can the models resolve configurational differences
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import defaultdict
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
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]])
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx, "atoms": atoms, "coords": np.array(coords)})
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

canon = []
for c in te_rows:
    s = y_csv[c]
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    canon.append(Chem.MolToSmiles(m, isomericSmiles=False) if m else None)
groups = defaultdict(list)
for i, cc in enumerate(canon):
    if cc: groups[cc].append(i)

def bond_fingerprint(csv_i):
    m = all_m[pos_by_csv[csv_i]]
    atoms = m['atoms']; coords = m['coords']
    tm = [k for k, a in enumerate(atoms) if is_TM(a)]
    d = []
    for k in tm:
        for j in range(len(atoms)):
            if k != j:
                d.append(np.linalg.norm(coords[k] - coords[j]))
    return np.sort(np.array(d))

pairs = []  # (i, j, maxdist_diff)
for cc, v in groups.items():
    uniq = {}
    for i in v:
        uniq.setdefault(te_rows[i], []).append(i)
    entries = list(uniq.items())
    if len(entries) < 2: continue
    for a in range(len(entries)):
        for b in range(a + 1, len(entries)):
            fa = bond_fingerprint(entries[a][0]); fb = bond_fingerprint(entries[b][0])
            if len(fa) == len(fb) and len(fa) > 0:
                dmax = float(np.max(np.abs(fa - fb)))
            else:
                dmax = float('nan')
            pairs.append((entries[a][1][0], entries[b][1][0], dmax, cc[:40]))

print(f'matched pairs (different csv entries): {len(pairs)}')
dmax = np.array([p[2] for p in pairs])
print(f'Max bond-length fingerprint difference distribution: min={np.nanmin(dmax):.4f} median={np.nanmedian(dmax):.4f} max={np.nanmax(dmax):.4f}')
print(f'  <0.05 Å (same configuration, different entries): {(dmax<0.05).sum()}')
print(f'  0.05–0.15 Å: {((dmax>=0.05)&(dmax<0.15)).sum()}')
print(f'  ≥0.15 Å (genuine configurational difference): {(dmax>=0.15).sum()}')

r5 = np.load('r5_smiles_va_ensemble.npz')
ep = np.load('ensemble_predictions.npz')
yt = ep['y_true']
CLIP = (0, 3)
p5 = r5['mean'].copy(); p3 = ep['mean'].copy()
for cc in CLIP:
    p5[:, cc] = np.maximum(p5[:, cc], 0); p3[:, cc] = np.maximum(p3[:, cc], 0)
names = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']

print('\nPair analysis (y_true difference vs model prediction difference, |·|):')
rows = []
for i, j, d, cc in pairs:
    dy = np.abs(yt[i] - yt[j])
    d5 = np.abs(p5[i] - p5[j])
    d3 = np.abs(p3[i] - p3[j])
    rows.append((d, dy, d5, d3, cc, i, j))
# Group by bond-length fingerprint difference
for label, mask_fn in [('all', lambda r: True),
                       ('strong configurational difference (dmax≥0.15Å)', lambda r: r[0] >= 0.15),
                       ('near-identical configuration (dmax<0.05Å)', lambda r: r[0] < 0.05)]:
    sub = [r for r in rows if mask_fn(r) and not np.isnan(r[0])]
    if not sub: print(f'  {label}: 0 pairs'); continue
    dy = np.array([r[1] for r in sub]); d5 = np.array([r[2] for r in sub]); d3 = np.array([r[3] for r in sub])
    print(f'  {label} (n={len(sub)}):')
    for k, n in enumerate(names):
        print(f'    {n:14s} |Δy_true| med={np.median(dy[:,k]):.4g}   |Δpred_R5| med={np.median(d5[:,k]):.4g}   |Δpred_3D| med={np.median(d3[:,k]):.4g}')

np.save('stereo_matched_rows.npy', np.array(rows, dtype=object), allow_pickle=True)
print('\nSAVED stereo_matched_rows.npy')
