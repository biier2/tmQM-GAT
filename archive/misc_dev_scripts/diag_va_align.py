# -*- coding: utf-8 -*-
"""Diagnostic: divergence pattern between the truncated prefix of the local rdkit2024 kept sequence and the paper main domain (ensemble_predictions.npz)
No graph building, only the kept test + row-by-row comparison + true-label reverse lookup. Answers three questions:
  1. Do the local csv labels equal the paper 3D ensemble labels (ep['y_true'])?
  2. How many te_idx rows are misaligned, and what is the misalignment pattern (pure rdkit +-19 divergence, or larger)?
  3. Where in the local kept sequence does the true molecule of a misaligned row sit (is it pushed beyond 90,317)?
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
RENAME = {'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
          'Dipole_M': 'dipole', 'Metal_q': 'metal_charge', 'Electronic_E': 'E_elec',
          'Dispersion_E': 'E_disp', 'Polarizability': 'polarizability'}
df_y = df_y.rename(columns=RENAME)
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
csv_labels = df_y[targets].values.astype(np.float64)
smiles_by_idx = df_y['SMILES'].tolist()

# ===== XYZ parsing, character-for-character identical to vA_domain_build.py =====
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]

def parse_one_full(xyz_p, bo_p, start_idx=0):
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
            i += 1
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
    return mols

print('Parsing XYZ...', flush=True)
all_m = []
for part in [1, 2, 3]:
    m = parse_one_full(f'{ROOT}/tmQM_X{part}.xyz.gz', f'{ROOT}/tmQM_X{part}.BO.gz', start_idx=len(all_m))
    all_m.extend(m)
print(f'Total {len(all_m):,} TM blocks', flush=True)

# ===== Kept test (SMILES non-empty and parseable + all y finite) — no graph building =====
kept_csv, kept_y, kept_tmpos, kept_smiles = [], [], [], []
n_smi_fail, n_nan = 0, 0
for idx, m in enumerate(all_m):
    csv_i = m['csv_idx']
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '':
        n_smi_fail += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_smi_fail += 1; continue
    y = csv_labels[csv_i]
    if not np.isfinite(y).all():
        n_nan += 1; continue
    kept_csv.append(csv_i); kept_y.append(y); kept_tmpos.append(idx + 1); kept_smiles.append(smi)
kept_y = np.array(kept_y, dtype=np.float64)
print(f'Kept {len(kept_y):,} (SMILES failures {n_smi_fail:,}, NaN {n_nan:,})', flush=True)
print(f'Local anchor: kept molecule 90,317 = TM block #{kept_tmpos[90316]} (csv row {kept_csv[90316]}) — server anchor 97,045', flush=True)

# ===== Row-by-row comparison with ep (truncated to the first 90,317) =====
ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float64)
yt_here = kept_y[:90317][te_idx]
d = np.abs(yt_here - yt_ref).max(axis=1)
bad = np.where(d > 1e-4)[0]
print(f'\nte_idx misaligned rows: {len(bad):,} / {len(te_idx):,}', flush=True)

# ===== Reverse lookup of the true values of the misaligned rows (label -> csv row) =====
csv_pos_of = {v: i for i, v in enumerate(kept_csv)}   # csv row -> local kept position
print(f'\n{"te_idx":>6} {"kept pos":>7} {"local diff":>9} {"true csv row":>8} {"true kept pos":>8} {"true TM block":>7} {"residual":>7}')
n_pushed = 0
for b in bad[:40]:
    i = int(te_idx[b])
    dd = np.abs(csv_labels - yt_ref[b]).max(axis=1)
    j = int(np.argmin(dd))
    pos = csv_pos_of.get(j, -1)
    if 90317 <= pos < len(kept_y): n_pushed += 1
    print(f'{i:>6} {b:>7} {d[b]:>9.2e} {j:>8} {pos:>8} {kept_tmpos[pos] if pos >= 0 else -1:>7} {dd[j]:>7.1e}')
print(f'\nMisaligned rows whose true value falls after position 90,317 of the local kept sequence (pushed out): {n_pushed}/{len(bad)}', flush=True)

# If the residuals are generally > 1e-2 -> the local csv labels disagree with the paper ensemble labels (csv version problem)
max_resid = max(float(np.abs(csv_labels - yt_ref[b]).max(axis=1).min()) for b in bad) if len(bad) else 0.0
print(f'Max residual of the true-value reverse lookup for misaligned rows: {max_resid:.2e}  (≪1e-4 = labels agree, pure alignment problem; large = csv version problem)', flush=True)

np.savez('diag_va_align.npz', kept_csv=np.array(kept_csv, dtype=np.int64),
         kept_tmpos=np.array(kept_tmpos, dtype=np.int64), bad_te_pos=bad)
print('saved -> diag_va_align.npz', flush=True)
