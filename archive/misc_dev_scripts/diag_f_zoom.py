# -*- coding: utf-8 -*-
"""Zoom in on the (k, g) values around the two decreasing points of g(k) + the global curve shape"""
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

order = np.argsort(te_idx)
k_sorted = te_idx[order]
g = f[order]

print('=== Neighbourhood of decreasing point 1 (pos 3575..3615) ===', flush=True)
for i in range(3575, 3616):
    print(f'  pos={i}: k={k_sorted[i]}, g={g[i]}, dk={k_sorted[i]-k_sorted[i-1] if i>3575 else 0}', flush=True)
print('\n=== Neighbourhood of decreasing point 2 (pos 8520..8560) ===', flush=True)
for i in range(8520, 8561):
    print(f'  pos={i}: k={k_sorted[i]}, g={g[i]}, dk={k_sorted[i]-k_sorted[i-1] if i>8520 else 0}', flush=True)

print('\n=== Global curve (every 300 samples) ===', flush=True)
for i in range(0, len(g), 300):
    print(f'  pos={i}: k={k_sorted[i]}, g={g[i]}', flush=True)

# Local slope: distribution of (g[i+1]-g[i]) / (k[i+1]-k[i])
dk = np.diff(k_sorted)
dg = np.diff(g)
slope = dg / dk
print(f'\nlocal slope dg/dk: median {np.median(slope):.3f}, p10 {np.percentile(slope,10):.3f}, p90 {np.percentile(slope,90):.3f}, negative {int((slope<0).sum())}', flush=True)
# Slope by k interval
for lo, hi in [(0, 24000), (24000, 34000), (34000, 56000), (56000, 70000), (70000, 90317)]:
    m = (k_sorted[:-1] >= lo) & (k_sorted[:-1] < hi)
    if m.sum() > 0:
        print(f'  k in [{lo},{hi}): median slope {np.median(slope[m]):.3f} (n={m.sum()})', flush=True)
