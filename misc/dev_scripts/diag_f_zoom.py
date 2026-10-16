# -*- coding: utf-8 -*-
"""放大看 g(k) 两个下降点邻域的 (k, g) 值 + 全局曲线形态"""
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

print('=== 下降点 1 邻域 (pos 3575..3615) ===', flush=True)
for i in range(3575, 3616):
    print(f'  pos={i}: k={k_sorted[i]}, g={g[i]}, dk={k_sorted[i]-k_sorted[i-1] if i>3575 else 0}', flush=True)
print('\n=== 下降点 2 邻域 (pos 8520..8560) ===', flush=True)
for i in range(8520, 8561):
    print(f'  pos={i}: k={k_sorted[i]}, g={g[i]}, dk={k_sorted[i]-k_sorted[i-1] if i>8520 else 0}', flush=True)

print('\n=== 全局曲线 (每 300 个采样) ===', flush=True)
for i in range(0, len(g), 300):
    print(f'  pos={i}: k={k_sorted[i]}, g={g[i]}', flush=True)

# 局部斜率: (g[i+1]-g[i]) / (k[i+1]-k[i]) 分布
dk = np.diff(k_sorted)
dg = np.diff(g)
slope = dg / dk
print(f'\n局部斜率 slope=dg/dk: 中位 {np.median(slope):.3f}, p10 {np.percentile(slope,10):.3f}, p90 {np.percentile(slope,90):.3f}, 负值 {int((slope<0).sum())}', flush=True)
# 按 k 区间看斜率
for lo, hi in [(0, 24000), (24000, 34000), (34000, 56000), (56000, 70000), (70000, 90317)]:
    m = (k_sorted[:-1] >= lo) & (k_sorted[:-1] < hi)
    if m.sum() > 0:
        print(f'  k∈[{lo},{hi}): 斜率中位 {np.median(slope[m]):.3f} (n={m.sum()})', flush=True)
