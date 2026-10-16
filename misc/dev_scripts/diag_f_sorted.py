# -*- coding: utf-8 -*-
"""g(k) = f 按 te_idx 排序: g = 旧csv行→新csv行映射 m 在保留序列 mol_idx 上的采样
期望: g 分段线性(斜率1) + 少量下降点 = m 的抽出-插入断点
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
print(f'f 未命中 {(f < 0).sum()}', flush=True)

# ===== 按 te_idx 排序 → g = m(M(p_k)) 采样 =====
order = np.argsort(te_idx)
k_sorted = te_idx[order]
g = f[order]
print(f'te_idx 排序后: k 范围 [{k_sorted[0]}, {k_sorted[-1]}], 唯一值 {len(np.unique(k_sorted))}', flush=True)

desc = np.where(g[1:] < g[:-1])[0]
print(f'\ng 下降点 {len(desc)} 个:', flush=True)
for d in desc:
    lo = max(0, d - 3); hi = min(len(g), d + 5)
    print(f'  pos={d}: k={k_sorted[d]}(g={g[d]}) -> k={k_sorted[d+1]}(g={g[d+1]})', flush=True)
    print(f'     上下文 g[{lo}..{hi}] = {g[lo:hi].tolist()}', flush=True)

# ===== g 的整体形态: g - k 的关系 =====
print(f'\ng: min={g.min()}, max={g.max()}', flush=True)
dpos = g[1:] - g[:-1]
dpos_pos = dpos[dpos >= 0]
print(f'g 相邻正差: 中位 {np.median(dpos_pos):.1f}, p5 {np.percentile(dpos_pos,5):.1f}, p95 {np.percentile(dpos_pos,95):.1f}', flush=True)
print(f'g 前 30: {g[:30].tolist()}', flush=True)
print(f'g 后 30: {g[-30:].tolist()}', flush=True)

# 分段线性检查: g[i] - i*斜率 的漂移 (g ≈ a*i + b?)
import numpy.polynomial.polynomial as P
mask = np.ones(len(g), bool)
# 粗略: 用中位数斜率
slopes = dpos_pos
med = np.median(slopes)
print(f'\n正差中位数 = {med:.2f} (若≈6.67 则 g 与 k 大致线性)', flush=True)
