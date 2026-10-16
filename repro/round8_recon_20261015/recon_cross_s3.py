# -*- coding: utf-8 -*-
"""只读：复算 SI 表 S3（cross protocol transfer）候选口径。
npz（SMILES GAT 预测）逐性质算 Pearson/Spearman/MAE/slope/intercept/recal-MAE/raw R2，
与 SI 印刷值（表 11/12）及 cross_dft_metrics.json 三方对照；并打印 7 个截距。"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
NPZ = r'D:\ml\chem_ml5\cross_dft_predictions.npz'
JSONP = r'D:\ml\chem_ml5\cross_dft_metrics.json'

targets = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
# SI 表 S3 印刷值（表 11 = Pearson/Spearman/Raw MAE；表 12 = Slope/In-sample MAE）
SI = {
    'gap':            dict(p=0.877, rho=0.869, mae=0.0344, slope=0.935, recal=0.0114),
    'homo':           dict(p=0.963, rho=0.912, mae=0.0298, slope=0.981, recal=0.0097),
    'lumo':           dict(p=0.965, rho=0.949, mae=0.0099, slope=0.982, recal=0.0089),
    'dipole':         dict(p=0.870, rho=0.866, mae=1.402,  slope=1.069, recal=1.348),
    'E_elec':         dict(p=0.994, rho=0.994, mae=166.3,  slope=1.104, recal=86.3),
    'E_disp':         dict(p=0.984, rho=0.987, mae=0.0209, slope=0.876, recal=0.0040),
    'polarizability': dict(p=0.933, rho=0.952, mae=44.3,   slope=1.309, recal=27.5),
}

d = np.load(NPZ)
y_true, yp = d['y_true'], d['smiles_preds']
print('npz:', y_true.shape, '| all finite:', bool(np.isfinite(y_true).all() and np.isfinite(yp).all()))
print()

pears, spears, r2s = [], [], []
intercepts = {}
print('%-15s | %8s %6s | %8s %6s | %9s %7s | %8s %6s | %9s %7s' %
      ('property', 'Pear', 'SI', 'Spear', 'SI', 'MAE', 'SI', 'Slope', 'SI', 'RecalMAE', 'SI'))
for i, p in enumerate(targets):
    yt, yv = y_true[:, i].astype(np.float64), yp[:, i].astype(np.float64)
    slope, intercept = np.linalg.lstsq(np.vstack([yv, np.ones_like(yv)]).T, yt, rcond=None)[0]
    r = pearsonr(yt, yv)[0]
    rho = spearmanr(yt, yv)[0]
    mae = float(np.abs(yt - yv).mean())
    recal = float(np.abs(yt - (yv * slope + intercept)).mean())
    r2 = r2_score(yt, yv)
    pears.append(r); spears.append(rho); r2s.append(r2)
    intercepts[p] = float(intercept)
    s = SI[p]
    print('%-15s | %8.4f %6.3f | %8.4f %6.3f | %9.4f %7s | %8.4f %6.3f | %9.4f %7s' %
          (p, r, s['p'], rho, s['rho'], mae, s['mae'], slope, s['slope'], recal, s['recal']))
print()
print('macro Pearson %.4f (SI 0.941) | macro Spearman %.4f (SI 0.933) | macro raw R2 %.4f (SI 0.58)'
      % (np.mean(pears), np.mean(spears), np.mean(r2s)))
print()
print('7 个截距（npz 复算）:')
for p in targets:
    print('  %-15s %.6f' % (p, intercepts[p]))
print()
print('json 现值（cross_dft_metrics.json）:')
for e in json.load(open(JSONP, encoding='utf-8')):
    print('  %-15s P=%.4f rho=%.4f R2=%.4f MAE=%.4f Slope=%.4f Int=%.4f Recal=%.4f'
          % (e['Property'], e['Pearson_r'], e['Spearman_rho'], e['R2'], e['MAE'], e['Slope'], e['Intercept'], e['MAE_recal']))
