# -*- coding: utf-8 -*-
"""Server side: dump the csv row numbers + TM block indices + labels + SMILES of the paper main-domain kept sequence
Run in the same (tmqm) environment as ensemble_cv_weighted / r2b_build.
Root cause (2026-09-02): the R5 alignment failure = after the server csv update of 8-23 the row order no longer agrees with the server's old xyz block order,
so position pairing (csv_idx=mol_idx) is scrambled -> must switch to pairing by CSD code (xyz block header code -> csv row).
At paper training time (7-18) the xyz block order = the csv row order, so position pairing was valid; it broke after the csv update.
Output: vA_kept_dump.npz {kept_csv_idx, kept_tmpos, kept_y, kept_smiles} — the full kept set (expected 95,152),
      locally only the first 90,317 are taken.
Self-check: after truncating to the first 90,317, compare y[te_idx] position by position against ensemble_predictions.npz (if the file exists), printing the number of mismatching rows.
"""
import sys, gzip, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
print('rdkit version:', Chem.rdBase.rdkitVersion, flush=True)

# ===== File path probing (the server working directory mirrors the local one) =====
def find_path(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('not found: ' + ' / '.join(cands))

CSV_P = find_path(['tmQM_y.csv', 'data_tmQM/local_vB/tmQM_y.csv'])
XYZ_P = [find_path([f'tmQM_X{p}.xyz.gz', f'data_tmQM/local_vB/tmQM_X{p}.xyz.gz']) for p in [1, 2, 3]]
BO_P = [find_path([f'tmQM_X{p}.BO.gz', f'data_tmQM/local_vB/tmQM_X{p}.BO.gz']) for p in [1, 2, 3]]
print(f'csv: {CSV_P}', flush=True)

df_y = pd.read_csv(CSV_P, sep=';')
df_y = df_y.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                            'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                            'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                            'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
csv_labels = df_y[targets].values.astype(np.float64)
smiles_by_idx = df_y['SMILES'].tolist()

# ===== Parsing character-for-character identical to r2b_build.py (keeping only csv_idx) =====
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,
           "Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,
           "Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

def parse_one(xyz_p, bo_p, start_idx=0):
    mols = []
    with gzip.open(bo_p, "rt") as f:
        bo_lines = [l.strip() for l in f.readlines()]
    with gzip.open(xyz_p, "rt") as f:
        xyz_blocks = f.read().strip().split(chr(10))
    bo_blocks, cur = [], []
    for line in bo_lines:
        if line.startswith("CSD_code"):
            if cur: bo_blocks.append(cur); cur = []
        elif line: cur.append(line)
    if cur: bo_blocks.append(cur)
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break
            title = xyz_blocks[i].strip()   # block header: CSD_code = XXX | q = ... (i already points at the title line)
            i += 1
            code = title.split('CSD_code = ')[1].split('|')[0].strip() if 'CSD_code = ' in title else None
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        [float(x) for x in cparts[1:4]]   # character-for-character with r2b_build: a bad coordinate line skips that atom
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"csv_idx": mol_idx, "code": code})
    return mols

print('Parsing XYZ (3 parts)...', flush=True)
all_m = []
for part in [0, 1, 2]:
    m = parse_one(XYZ_P[part], BO_P[part], start_idx=len(all_m))
    all_m.extend(m)
    print(f'  Part {part+1}: {len(m):,} (cumulative {len(all_m):,})', flush=True)
print(f'Total {len(all_m):,} TM blocks — expected 102,077 (matching the paper notebook)', flush=True)

# ===== Kept test (SMILES non-empty and parseable + all y finite; csv row = code pairing, not position pairing) =====
row_of_code = {str(c): i for i, c in enumerate(df_y['CSD_code'])}
kept_csv, kept_tmpos, kept_y, kept_smiles = [], [], [], []
n_smi_fail, n_nan, n_nocode = 0, 0, 0
for idx, m in enumerate(all_m):
    code = m['code']
    csv_i = row_of_code.get(code, -1) if code else -1
    if csv_i < 0:
        n_nocode += 1; continue
    smi = smiles_by_idx[csv_i]
    if not isinstance(smi, str) or smi == '':
        n_smi_fail += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_smi_fail += 1; continue
    y = csv_labels[csv_i]
    if not np.isfinite(y).all():
        n_nan += 1; continue
    kept_csv.append(csv_i); kept_tmpos.append(idx + 1); kept_y.append(y); kept_smiles.append(smi)
kept_y = np.array(kept_y, dtype=np.float32)
print(f'Kept {len(kept_y):,} (SMILES failures {n_smi_fail:,}, NaN {n_nan:,}, no code {n_nocode:,}) — expected 95,152', flush=True)
assert len(kept_y) >= 90317, f'Kept {len(kept_y)} < 90,317 — incomplete data files or wrong version'
print(f'Anchor: kept molecule 90,317 = TM block #{kept_tmpos[90316]} (csv row {kept_csv[90316]}) — expected 97,045', flush=True)

np.savez('vA_kept_dump.npz',
         kept_csv_idx=np.array(kept_csv, dtype=np.int64),
         kept_tmpos=np.array(kept_tmpos, dtype=np.int64),
         kept_y=kept_y,
         kept_smiles=np.array(kept_smiles))
print('saved -> vA_kept_dump.npz', flush=True)

# ===== Self-check: first 90,317 truncation vs ensemble_predictions.npz =====
EP = find_path(['ensemble_predictions.npz', 'data_tmQM/paper_domain_vA/ensemble_predictions.npz'])
if EP:
    ep = np.load(EP)
    te_idx = ep['te_idx']
    yt_ref = ep['y_true'].astype(np.float64)
    yt = kept_y[:90317][te_idx].astype(np.float64)
    d = np.abs(yt - yt_ref).max(axis=1)
    bad = np.where(d > 1e-4)[0]
    print(f'Self-check vs {EP}: mismatching rows of the first 90,317 truncation y[te_idx] {len(bad):,} / {len(te_idx):,}', flush=True)
    if len(bad):
        print('!! The server csv labels disagree with the paper 3D ensemble (ep) — has the server csv been updated? The old paper csv must be recovered', flush=True)
        for b in bad[:10]:
            i = int(te_idx[b])
            print(f'  te_idx[{b}]={i}: server y={yt[b].round(4).tolist()}  ep y_true={yt_ref[b].round(4).tolist()}', flush=True)
    else:
        print('Self-check passed: the first 90,317 of the server kept sequence = the paper main domain (y_true matches position by position)', flush=True)
