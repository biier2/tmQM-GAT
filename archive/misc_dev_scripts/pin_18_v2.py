# -*- coding: utf-8 -*-
"""Pin down the remaining 10 rows: the 8 tail rows = my-position 90327..90334 (derived from c(90316)=10);
the 2 hidden rows use the criterion that a candidate deletion empties its shift layer
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

def align(yy):
    sh = np.full(len(te), -1, dtype=np.int64)
    for i in range(len(te)):
        t = te[i]; tb = yt_ref[i].tobytes()
        for d in range(0, 26):
            if t + d < len(yy) and yy[t + d].tobytes() == tb:
                sh[i] = d; break
    return sh

sh = align(y_my)
print(f'Initial: misses {(sh<0).sum()}, max shift {sh.max()}', flush=True)

# Tail-row verification: the entry with te=90316 should have shift 10
last = np.where(te == 90316)[0]
print(f'te=90316 entry shift = {sh[last][0] if len(last) else "none"}', flush=True)
# Delete the 8 tail rows (my-position 90327..90334)
tail = list(range(90327, 90335))
for p in reversed(tail):
    k = int(orig[p])
    print(f'  tail row: my-position {p}, kept_idx={k}, csv_idx={mseq[p]}, {codes[mseq[p]]}, '
          f'SMILES={repr(smiles_by_idx[mseq[p]])[:80]}', flush=True)
    y_my = np.delete(y_my, p, axis=0); mseq = np.delete(mseq, p); orig = np.delete(orig, p)
sh = align(y_my)
print(f'After deleting the 8 tail rows: misses {(sh<0).sum()}, max shift {sh.max()}', flush=True)

# The 2 hidden rows: for each jump layer, the number of samples in that layer drops to 0 after the candidate deletion
removed_total = 8
for s in range(0, removed_total + 1):
    while True:
        ma = (sh == s); mb = (sh == s + 1)
        if (mb == False).all() if not mb.any() else False:
            break
        if not ma.any() or not mb.any():
            break
        lo = int(te[ma].max()); hi = int(te[mb].min())
        found = None
        for p in range(lo + 1, hi + 1):
            yy2 = np.delete(y_my, p, axis=0)
            sh2 = align(yy2)
            if (sh2 < 0).sum() == 0:
                # Criterion: after deletion the number of samples with shift==s+1 drops markedly (the jump is removed)
                if int((sh2 == s + 1).sum()) < int(mb.sum()):
                    found = p; break
        if found is None:
            print(f'  jump {s}->{s+1}: no valid candidate found in the interval ({lo}, {hi}]', flush=True)
            break
        k = int(orig[found])
        print(f'  hidden row: my-position {found}, kept_idx={k}, csv_idx={mseq[found]}, {codes[mseq[found]]}, '
              f'SMILES={repr(smiles_by_idx[mseq[found]])[:80]}', flush=True)
        y_my = np.delete(y_my, found, axis=0); mseq = np.delete(mseq, found); orig = np.delete(orig, found)
        sh = align(y_my)
        removed_total += 1

print(f'\nFinal: {len(y_my)} rows, misses {(sh<0).sum()}, max shift {sh.max() if len(sh) else -1}, {90335 - len(y_my)} rows deleted in total', flush=True)
if len(y_my) == 90317:
    idx_all = np.random.RandomState(42).permutation(90317)
    te_sim = idx_all[:13547]
    print(f'te_idx identical to ep: {np.array_equal(te_sim, ep["te_idx"])}', flush=True)
    yt_sim = y_my[te_sim]
    d = np.abs(yt_sim - yt_ref).max(axis=1)
    print(f'y_true self-check: {(d > 1e-4).sum()} / {len(d)} mismatches', flush=True)
    if (d > 1e-4).sum() == 0:
        np.savez('vA_rebuild.npz', kept_idx=orig.copy())
        print('SAVED vA_rebuild.npz', flush=True)
