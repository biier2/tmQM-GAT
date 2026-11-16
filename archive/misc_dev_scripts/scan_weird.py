# -*- coding: utf-8 -*-
"""Scan the 90,335 kept rows for 'server RDKit rejects' candidates:
v1 already pinned 8 rows (csv_idx 862,1809,28980,39279,49275,72926,76953,87689) by their full SMILES ->
generalise the weird patterns from them -> scan all rows for candidates -> if there are exactly 18 and they cover the known 8 -> full verification after deletion
"""
import sys, re
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

# ---- Rebuild kept (with cache) ----
import os
CACHE = 'kept_cache.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE)
    mseq = z['csv_idx']; y_my = z['y']
    print(f'[cache] kept {len(mseq)} rows', flush=True)
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
    np.savez(CACHE, csv_idx=mseq, y=y_my)
    print(f'[rebuilt] kept {len(mseq)} rows → cached', flush=True)

ep = np.load('ensemble_predictions.npz')
te = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)

# ---- 1) Full SMILES of the known 8 rows ----
known_csv = [862, 1809, 28980, 39279, 49275, 72926, 76953, 87689]
print('\n== Full SMILES of the known 8 rows (v1) ==', flush=True)
for c in known_csv:
    pos = np.where(mseq == c)[0]
    pos_s = pos.tolist()
    print(f'csv_idx={c}: positions={pos_s}, {codes[c]}: {smiles_by_idx[c]!r}', flush=True)

# ---- 2) Full weird-pattern scan ----
# Patterns are grouped by category; any hit is a candidate; print the full SMILES, position and csv_idx of each candidate
def scan(pats):
    hits = {}
    for pos, c in enumerate(mseq):
        smi = smiles_by_idx[c]
        s = smi if isinstance(smi, str) else repr(smi)
        for name, rx in pats:
            if rx.search(s):
                hits.setdefault(pos, (c, name, s))
                break
    return hits

pats = [
    ('dative<-',      re.compile(r'<-')),
    ('bracket-arom',  re.compile(r'\[[a-z]')),
    ('bracket-XH-',   re.compile(r'\[[A-Z][a-z]?H-]')),      # [NH-] [CH-] [BH-]
    ('bracket-XH+',   re.compile(r'\[[A-Z][a-z]?H\+]')),     # [NH+] etc. (benign control group of candidates)
    ('nan',           re.compile(r'^nan$', re.I)),
    ('other',         re.compile(r'\[[^]]*[-+]{2,}[^]]*\]')),  # multiple charges
]
hits = scan(pats)
print(f'\n== Weird-pattern candidates: {len(hits)} rows ==', flush=True)
for pos in sorted(hits):
    c, name, s = hits[pos]
    print(f'  my-pos={pos} csv_idx={c} [{name}] {codes[c]}: {s!r}', flush=True)

# Check whether the known 8 rows are all among the candidates
known_pos = set()
for c in known_csv:
    p = np.where(mseq == c)[0]
    known_pos.update(int(x) for x in p)
miss_known = known_pos - set(hits)
print(f'\nKnown 8 rows: positions {sorted(known_pos)}', flush=True)
print(f'Known but not hit by the scan: {sorted(miss_known) if miss_known else "none"}', flush=True)

# ---- 3) If there are exactly 18 candidates and they cover the known rows -> full verification after deletion ----
cands = sorted(hits)
if len(cands) == 18 and not miss_known:
    keep = np.ones(len(mseq), dtype=bool)
    keep[cands] = False
    yy = y_my[keep]
    print(f'\nAfter deleting {cands}: {len(yy)} rows', flush=True)
    sh = np.full(len(te), -1, dtype=np.int64)
    for i in range(len(te)):
        t = te[i]; tb = yt_ref[i].tobytes()
        for d in range(0, 26):
            if t + d < len(yy) and yy[t + d].tobytes() == tb:
                sh[i] = d; break
    print(f'misses {(sh<0).sum()}, max shift {sh.max() if len(sh) else -1}', flush=True)
    if len(yy) == 90317 and (sh < 0).sum() == 0 and sh.max() == 0:
        idx_all = np.random.RandomState(42).permutation(90317)
        te_sim = idx_all[:13547]
        print(f'te_idx identical to ep: {np.array_equal(te_sim, te)}', flush=True)
        d = np.abs(yy[te_sim] - yt_ref).max(axis=1)
        print(f'y_true self-check: {(d > 1e-4).sum()} / {len(d)} mismatches', flush=True)
        if (d > 1e-4).sum() == 0:
            kept_idx = np.arange(len(mseq), dtype=np.int64)[keep]
            np.savez('vA_rebuild.npz', kept_idx=kept_idx, removed_kept=cands)
            np.save('vA_removed_kept.npy', np.array(cands))
            print('SAVED vA_rebuild.npz (kept_idx = original all_m indices) + vA_removed_kept.npy', flush=True)
        else:
            print('y_true mismatches — the candidate set is not exactly right', flush=True)
    else:
        print('Alignment did not fully converge — the candidate set is not exactly right', flush=True)
else:
    print(f'\nCandidate count {len(cands)} ≠ 18 or the known 8 rows are not covered — adjust the patterns / switch to the server dump', flush=True)
