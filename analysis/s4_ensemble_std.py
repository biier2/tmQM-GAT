# -*- coding: utf-8 -*-
"""S4: 9 成员 ensemble std 不确定性校准分析（纯统计，基于已有 ensemble_predictions.npz，不重训）"""
import numpy as np
import json

d = np.load('ensemble_predictions.npz')
mean, std, per_model, y_true = d['mean'], d['std'], d['per_model'], d['y_true']
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
units = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']

print(f'{len(mean):,} test samples, {per_model.shape[0]} ensemble members\n')

out = {}
rows = []
for i, t in enumerate(targets):
    yt = y_true[:, i]
    s = std[:, i]
    err = np.abs(mean[:, i] - yt)
    q1, q2 = np.quantile(s, [1 / 3, 2 / 3])
    bins = {'low': s <= q1, 'mid': (s > q1) & (s <= q2), 'high': s > q2}
    row = {'property': t, 'unit': units[i], 'thresholds': [float(q1), float(q2)]}
    for bn, m in bins.items():
        corr = float(np.corrcoef(s[m], err[m])[0, 1]) if m.sum() > 3 else float('nan')
        row[bn] = {'n': int(m.sum()), 'mean_std': float(s[m].mean()),
                   'MAE': float(err[m].mean()), 'corr_std_err': corr}
    out[t] = row
    rows.append((t, row['low'], row['mid'], row['high']))

print(f'{"Property":16s} {"MAE low":>9s} {"MAE mid":>9s} {"MAE high":>9s} {"high/low":>9s} {"corr(std,|err|)":>15s}')
print('-' * 70)
for t, lo, mi, hi in rows:
    ratio = hi['MAE'] / lo['MAE'] if lo['MAE'] > 0 else float('nan')
    print(f'{t:16s} {lo["MAE"]:9.4f} {mi["MAE"]:9.4f} {hi["MAE"]:9.4f} {ratio:9.2f} {hi["corr_std_err"]:15.3f}')

# 全局 std-误差相关性
all_s, all_e = std.ravel(), np.abs(mean - y_true).ravel()
gcorr = float(np.corrcoef(all_s, all_e)[0, 1])
print(f'\nGlobal Spearman corr(std, |err|): {np.corrcoef(all_s, all_e)[0,1]:.3f}')
from scipy.stats import spearmanr
gspr = float(spearmanr(all_s, all_e)[0])
print(f'Global Spearman corr: {gspr:.3f}')
out['global_corr_pearson'] = gcorr
out['global_corr_spearman'] = gspr

json.dump(out, open('s4_ensemble_std.json', 'w'), indent=1)
print('\nsaved s4_ensemble_std.json')
