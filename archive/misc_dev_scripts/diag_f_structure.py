# -*- coding: utf-8 -*-
"""Diagnose the structure of f(i): f(i) = row index in csv A of the ep y_true[i] label content (a sampling of the old→new csv row map m)
Key checks:
1. Is te_idx ascending? If so, f = m sampled at increasing mol_idx, and the decreasing points directly give e1/e2
2. Is f monotone apart from 2 jumps? This decides the reconstruction strategy
3. Exact values of the f decreasing points -> exact s/q boundaries
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

ROOT = 'data_tmQM/local_vB'
df = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
n = len(df)
print(f'A csv: {n} rows', flush=True)

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt = ep['y_true'].astype(np.float32)
print(f'te_idx: shape={te_idx.shape}, dtype={te_idx.dtype}', flush=True)
print(f'  te_idx[:12] = {te_idx[:12].tolist()}', flush=True)
print(f'  te_idx[-6:] = {te_idx[-6:].tolist()}', flush=True)
print(f'  te_idx ascending? {bool(np.all(np.diff(te_idx) >= 0))}  ({(np.diff(te_idx) < 0).sum()} decreasing points)', flush=True)

# ===== f(i): first exact-matching row in csv A for the content of y_true[i] =====
row_of_bytes = {}
for i in range(n):
    b = y[i].tobytes()
    if b not in row_of_bytes:
        row_of_bytes[b] = i
f = np.full(len(te_idx), -1, dtype=np.int64)
n_multi = 0
for i in range(len(te_idx)):
    b = yt[i].tobytes()
    r = row_of_bytes.get(b, -1)
    if r < 0:
        # tolerance fallback: take the first row whose max absolute difference over the 8 targets is < 1e-4
        d = np.abs(y - yt[i]).max(axis=1)
        cand = np.where(d < 1e-4)[0]
        r = cand[0] if len(cand) else -1
    f[i] = r
    if i > 0 and yt[i].tobytes() == yt[i-1].tobytes():
        n_multi += 1
print(f'f lookup: misses {(f < 0).sum()} / {len(f)}; adjacent duplicate samples in ep {n_multi}', flush=True)

# ===== Structure of f =====
desc = np.where(f[1:] < f[:-1])[0]
print(f'f decreasing points: {len(desc)}', flush=True)
for d in desc:
    lo = max(0, d - 2); hi = min(len(f), d + 4)
    print(f'  i={d}: f={f[d]} -> f={f[d+1]}  context f[{lo}..{hi}] = {f[lo:hi].tolist()}', flush=True)
mono_frac = float(np.mean(f[1:] >= f[:-1]))
print(f'f monotone fraction: {mono_frac:.4f}', flush=True)

# ===== Range and distribution of f =====
print(f'f: min={f.min()}, max={f.max()}, mean={f.mean():.1f}', flush=True)
# Distribution of adjacent differences (skipping the decreasing points)
dpos = f[1:] - f[:-1]
dpos_pos = dpos[dpos >= 0]
print(f'f adjacent positive differences: median {np.median(dpos_pos):.1f}, p90 {np.percentile(dpos_pos, 90):.1f}, max {dpos_pos.max()}', flush=True)

# ===== If te_idx is ascending: f ≈ the mol_idx sequence (after the m transform); check the relation between f and te_idx =====
if np.all(np.diff(te_idx) >= 0):
    # paper-pipeline mol_idx (buggy version) — part boundaries per A: 34,970 / 71,035
    def M(j):
        if j < 34970: return j
        if j < 71035: return 28737 + (j - 34970)
        return 64571 + (j - 71035)
    # old position p_k and mol_idx x_k of the retained block (first 90,317) (assuming xyz unchanged = case A; treated as identity for now)
    # here we only roughly look at the difference between f and the "kept-sequence mol_idx"
    print('\nf vs te_idx sampling structure (rough, assuming no reordering):', flush=True)
    for i in [0, 1, 3590, 3591, 3592, 8535, 8536, 8537, 13540, 13546]:
        print(f'  i={i}: te_idx={te_idx[i]}, f={f[i]}', flush=True)
