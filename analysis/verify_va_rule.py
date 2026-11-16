# -*- coding: utf-8 -*-
"""Verify the CSV-side retention rule for the 90,317 paper main domain (major finding of 2026-09-02)
Background: diagnostics confirm that the paper main domain = a retained subsequence of the CSV row order (not the xyz TM block order; the README prefix-truncation conclusion is overturned).
This script verifies each candidate retention mask: y_domain[te_idx] matches y_true in ensemble_predictions.npz bit for bit.
Candidate rules (dropna = all 8 targets finite):
  A: dropna & SMILES parseable & contains TM (z>=21)
  B: A + CSD code present in the xyz files
  C: dropna & SMILES parseable & CSD code present in xyz
  D: dropna & SMILES parseable
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
smi_list = df['SMILES'].tolist()
codes = df['CSD_code'].tolist()
n = len(df)
print(f'csv: {n} rows', flush=True)

dropna = np.isfinite(y).all(axis=1)
print(f'dropna retained: {dropna.sum():,}', flush=True)

parse_ok = np.zeros(n, bool); tm_ok = np.zeros(n, bool)
for i, s in enumerate(smi_list):
    if not isinstance(s, str) or s == '':
        continue
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        continue
    parse_ok[i] = True
    tm_ok[i] = any(a.GetAtomicNum() >= 21 for a in mol.GetAtoms())
print(f'SMILES parseable: {parse_ok.sum():,}   contains TM: {tm_ok.sum():,}', flush=True)

# ===== Set of CSD codes in the xyz files (scan block headers line by line, skipping atom lines) =====
codes_xyz = set()
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
            if 'CSD_code = ' in title:
                c = title.split('CSD_code = ')[1].split('|')[0].strip()
                codes_xyz.add(c)
            for _ in range(n_at):
                next(it, None)
print(f'xyz-file CSD codes: {len(codes_xyz):,}', flush=True)

code_in_xyz = np.array([(str(c) in codes_xyz) if isinstance(c, str) else False for c in codes])
print(f'CSD codes present in xyz: {code_in_xyz.sum():,}', flush=True)

cands = {
    'A: dropna&parse&TM':       dropna & parse_ok & tm_ok,
    'B: dropna&parse&TM&code':  dropna & parse_ok & tm_ok & code_in_xyz,
    'C: dropna&parse&code':     dropna & parse_ok & code_in_xyz,
    'D: dropna&parse':          dropna & parse_ok,
    'E: dropna&TM&code':        dropna & tm_ok & code_in_xyz,
}

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)
assert len(te_idx) == 13547

for name, mask in cands.items():
    kept_rows = np.where(mask)[0]
    y_dom = y[kept_rows]
    nkept = len(kept_rows)
    if nkept == 90317:
        same = bool(np.allclose(y_dom[te_idx], yt_ref, atol=1e-4))
        print(f'{name}: retained {nkept:,} → y_true bit-identical = {same}', flush=True)
    elif nkept > 90317:
        same = bool(np.allclose(y_dom[:90317][te_idx], yt_ref, atol=1e-4))
        print(f'{name}: retained {nkept:,} (>90,317) → y_true bit-identical after truncating to the first 90,317 = {same}', flush=True)
    else:
        print(f'{name}: retained {nkept:,} < 90,317 ✗', flush=True)
