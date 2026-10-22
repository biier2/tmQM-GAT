# -*- coding: utf-8 -*-
"""LEGACY (90,317 domain, 13,547 test / 11,547 evaluation). Kept for
provenance only; it does not reproduce the manuscript's conformal numbers,
which use the 90,377 domain and 13,556 / 11,556 counts. See
repro/round8_recon_20261015/recon_conformal_gating.py and
data_indices/gating_indices.npz.

S4: 标准化分裂保形预测 — 3D 集成 9 成员, 2000 校准 / 11547 评估
非一致性分数 s_i = |y - ŷ_ens| / σ_ens (σ 取 9 成员 std, 下限保护)
目标覆盖率 1-α ∈ {0.80, 0.90, 0.95}"""
import json, sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
PROPS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

d = np.load('d:/ml/chem_ml5/r3b_xyz_ensemble.npz')
pm = d['per_model']          # (9, 13547, 8)
y_true = d['y_true']         # (13547, 8)
y_mean = pm.mean(axis=0)     # (13547, 8)
y_std = pm.std(axis=0)       # (13547, 8)

# 固定划分: RandomState(42) 选 2000 校准点
rng = np.random.RandomState(42)
n = len(y_true)
cal_idx = rng.choice(n, size=2000, replace=False)
cal_mask = np.zeros(n, dtype=bool)
cal_mask[cal_idx] = True
eval_mask = ~cal_mask
print(f'校准 {cal_mask.sum()} / 评估 {eval_mask.sum()}')

eps = 1e-6
sigma = np.maximum(y_std, eps)

results = {}
print(f'\n{"prop":14s} {"cov":>5s} {"α":>6s} {"emp_cov":>8s} {"avg_width":>10s} {"med_width":>10s}')
for p, name in enumerate(PROPS):
    results[name] = {}
    for target_cov in [0.80, 0.90, 0.95]:
        alpha = 1 - target_cov
        # 非一致性分数 (校准集)
        s_cal = np.abs(y_true[cal_mask, p] - y_mean[cal_mask, p]) / sigma[cal_mask, p]
        # 有限样本校正分位数
        q_level = min(1.0, (1 - alpha) * (1 + 1.0 / len(s_cal)))
        q = np.quantile(s_cal, q_level)
        # 评估集
        s_eval = np.abs(y_true[eval_mask, p] - y_mean[eval_mask, p]) / sigma[eval_mask, p]
        cov = (s_eval <= q).mean()
        widths = 2 * q * sigma[eval_mask, p]
        results[name][str(target_cov)] = {
            'alpha': alpha, 'q': float(q), 'empirical_coverage': float(cov),
            'avg_width': float(widths.mean()), 'median_width': float(np.median(widths)),
            'n_eval': int(eval_mask.sum())}
        print(f'{name:14s} {target_cov:5.2f} {alpha:6.2f} {cov:8.4f} {widths.mean():10.3f} {np.median(widths):10.3f}')

with open('d:/ml/chem_ml5/s4_conformal.json', 'w') as f:
    json.dump({'n_cal': 2000, 'n_eval': int(eval_mask.sum()),
               'rng': 'RandomState(42)', 'results': results}, f, indent=2)
print('\nSaved -> s4_conformal.json')
