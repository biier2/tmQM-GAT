# -*- coding: utf-8 -*-
"""Generate vA_domain_smiles.pt locally (verified version) — bypasses the server-side csv drift
Input: kept_cache.npz (csv_idx+y of the 90,335 kept) + vA_removed_kept.npy (18 removed positions)
     + local tmQM_y.csv (3-17 version, md5 f8eeb967...f043) + ensemble_predictions.npz
Output: vA_domain_smiles.pt {smiles, labels(float32), labels64, te_idx, tr_idx, csv_idx, csd_code}
End-to-end self-check: saved only if te_idx is exactly equal and y_true is bit-identical 0/13,547
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, hashlib
import torch

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
y_csv = df_y[targets].astype(float).values  # float64 original values

z = np.load('kept_cache.npz')
mseq = z['csv_idx'].copy()          # csv_idx (mol_idx) for the 90,335 rows
rem = np.load('vA_removed_kept.npy')  # 18 removed positions (in the original 90,335 space)
print(f'cache: {len(mseq)} rows; {len(rem)} positions removed: {rem.tolist()}')

keep = np.ones(len(mseq), dtype=bool)
keep[rem] = False
mseq = mseq[keep]
assert len(mseq) == 90317, f'expected 90317, got {len(mseq)}'

smiles_list = [smiles_by_idx[c] if c < len(smiles_by_idx) else '' for c in mseq]
code_list = [codes[c] if c < len(codes) else '' for c in mseq]
labels64 = y_csv[mseq]
labels32 = labels64.astype(np.float32)

n = 90317
idx_all = np.random.RandomState(42).permutation(n)
te_idx = idx_all[:13547].copy()
tr_idx = idx_all[13547:].copy()

ep = np.load('ensemble_predictions.npz')
te_ref = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)
print('te_idx exactly equal:', np.array_equal(te_idx, te_ref))
d = np.abs(labels32[te_idx] - yt_ref).max()
print(f'y_true max|diff| (float32): {d:.3e}')
miss = sum(1 for i in range(13547)
           if labels32[te_idx[i]].tobytes() != yt_ref[i].tobytes())
print(f'y_true bit-identical mismatches: {miss} / 13547')

assert np.array_equal(te_idx, te_ref) and miss == 0, 'self-check failed, not saving'

torch.save({
    'smiles': smiles_list,
    'labels': labels32,
    'labels64': labels64,
    'te_idx': te_idx.astype(np.int64),
    'tr_idx': tr_idx.astype(np.int64),
    'csv_idx': mseq.astype(np.int64),
    'csd_code': code_list,
}, 'vA_domain_smiles.pt')
print('SAVED vA_domain_smiles.pt')
print('md5:', hashlib.md5(open('vA_domain_smiles.pt', 'rb').read()).hexdigest())
print('first 3 SMILES:', [s[:60] for s in smiles_list[:3]])
print('labels32[0]:', ['%.6g' % v for v in labels32[0]])
print('labels64[0]:', ['%.10g' % v for v in labels64[0]])
