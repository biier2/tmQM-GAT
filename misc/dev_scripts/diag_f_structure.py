# -*- coding: utf-8 -*-
"""诊断 f(i) 结构: f(i) = ep y_true[i] 标签内容在 A csv 中的行号 (old→new csv 行映射 m 的采样)
关键验证:
1. te_idx 是否升序 → 若是, f = m 在递增 mol_idx 处的采样, 下降点直接反解 e1/e2
2. f 是否除 2 处跳变外单调 → 决定重建策略
3. f 下降点精确值 → s/q 精确边界
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
print(f'A csv: {n} 行', flush=True)

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt = ep['y_true'].astype(np.float32)
print(f'te_idx: shape={te_idx.shape}, dtype={te_idx.dtype}', flush=True)
print(f'  te_idx[:12] = {te_idx[:12].tolist()}', flush=True)
print(f'  te_idx[-6:] = {te_idx[-6:].tolist()}', flush=True)
print(f'  te_idx 升序? {bool(np.all(np.diff(te_idx) >= 0))}  (下降点 {(np.diff(te_idx) < 0).sum()} 个)', flush=True)

# ===== f(i): y_true[i] 内容在 A csv 中的第一个精确匹配行 =====
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
        # 容差回退: 取 8 目标绝对差最大者 < 1e-4 的首行
        d = np.abs(y - yt[i]).max(axis=1)
        cand = np.where(d < 1e-4)[0]
        r = cand[0] if len(cand) else -1
    f[i] = r
    if i > 0 and yt[i].tobytes() == yt[i-1].tobytes():
        n_multi += 1
print(f'f 反查: 未命中 {(f < 0).sum()} / {len(f)}; ep 相邻重复样本 {n_multi}', flush=True)

# ===== f 结构 =====
desc = np.where(f[1:] < f[:-1])[0]
print(f'f 下降点 {len(desc)} 个:', flush=True)
for d in desc:
    lo = max(0, d - 2); hi = min(len(f), d + 4)
    print(f'  i={d}: f={f[d]} -> f={f[d+1]}  上下文 f[{lo}..{hi}] = {f[lo:hi].tolist()}', flush=True)
mono_frac = float(np.mean(f[1:] >= f[:-1]))
print(f'f 单调比例: {mono_frac:.4f}', flush=True)

# ===== f 值域与分布 =====
print(f'f: min={f.min()}, max={f.max()}, 均值={f.mean():.1f}', flush=True)
# 相邻差分布 (跳过下降点)
dpos = f[1:] - f[:-1]
dpos_pos = dpos[dpos >= 0]
print(f'f 相邻正差: 中位 {np.median(dpos_pos):.1f}, p90 {np.percentile(dpos_pos, 90):.1f}, max {dpos_pos.max()}', flush=True)

# ===== 若 te_idx 升序: f 近似 = mol_idx 序列 (经 m 变换), 检查 f 与 te_idx 的关系 =====
if np.all(np.diff(te_idx) >= 0):
    # 论文管线 mol_idx (bug 版) — part 边界按 A: 34,970 / 71,035
    def M(j):
        if j < 34970: return j
        if j < 71035: return 28737 + (j - 34970)
        return 64571 + (j - 71035)
    # 保留块(前90,317)的 old 位置 p_k 与 mol_idx x_k (假设 xyz 未变 = case A, 先按恒等估算)
    # 此处只粗略看 f 与 "kept 序列 mol_idx" 的差
    print('\nf 与 te_idx 采样结构 (粗略, 假设无重排):', flush=True)
    for i in [0, 1, 3590, 3591, 3592, 8535, 8536, 8537, 13540, 13546]:
        print(f'  i={i}: te_idx={te_idx[i]}, f={f[i]}', flush=True)
