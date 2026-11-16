# -*- coding: utf-8 -*-
"""g(k) = f sorted by te_idx: g = the old-csv-row→new-csv-row map m sampled over the retained mol_idx sequence
Expectation: g is piecewise linear (slope 1) + a few downward points = the cut-and-insert breakpoints of m
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

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt = ep['y_true'].astype(np.float32)

row_of_bytes = {}
for i in range(n):
    b = y[i].tobytes()
    if b not in row_of_bytes:
        row_of_bytes[b] = i
f = np.full(len(te_idx), -1, dtype=np.int64)
for i in range(len(te_idx)):
    r = row_of_bytes.get(yt[i].tobytes(), -1)
    if r < 0:
        d = np.abs(y - yt[i]).max(axis=1)
        cand = np.where(d < 1e-4)[0]
        r = cand[0] if len(cand) else -1
    f[i] = r
print(f'f misses {(f < 0).sum()}', flush=True)

# ===== Sort by te_idx -> g = sampling of m(M(p_k)) =====
order = np.argsort(te_idx)
k_sorted = te_idx[order]
g = f[order]
print(f'after sorting te_idx: k range [{k_sorted[0]}, {k_sorted[-1]}], unique values {len(np.unique(k_sorted))}', flush=True)

desc = np.where(g[1:] < g[:-1])[0]
print(f'\ng decreasing points: {len(desc)}', flush=True)
for d in desc:
    lo = max(0, d - 3); hi = min(len(g), d + 5)
    print(f'  pos={d}: k={k_sorted[d]}(g={g[d]}) -> k={k_sorted[d+1]}(g={g[d+1]})', flush=True)
    print(f'     context g[{lo}..{hi}] = {g[lo:hi].tolist()}', flush=True)

# ===== Overall shape of g: the relation between g and k =====
print(f'\ng: min={g.min()}, max={g.max()}', flush=True)
dpos = g[1:] - g[:-1]
dpos_pos = dpos[dpos >= 0]
print(f'g adjacent positive differences: median {np.median(dpos_pos):.1f}, p5 {np.percentile(dpos_pos,5):.1f}, p95 {np.percentile(dpos_pos,95):.1f}', flush=True)
print(f'first 30 of g: {g[:30].tolist()}', flush=True)
print(f'last 30 of g: {g[-30:].tolist()}', flush=True)

# Piecewise-linearity check: drift of g[i] - i*slope (g ≈ a*i + b?)
import numpy.polynomial.polynomial as P
mask = np.ones(len(g), bool)
# Rough: use the median slope
slopes = dpos_pos
med = np.median(slopes)
print(f'\nmedian positive difference = {med:.2f} (if ≈6.67 then g is roughly linear in k)', flush=True)
