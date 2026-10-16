# -*- coding: utf-8 -*-
"""只读：用九成员集成 A（复现当前稿数据\crossfunct_A_preds.npz）复算 SI 表 S3 全量指标。
目标：逐位复现表 11（Pearson/Spearman/Raw MAE）与表 12（Slope/In-sample MAE），
并输出批注 31 所需材料：逐性质 raw R2 + 七个再校准截距（全精度）。"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
NPZ = r'D:\ml\paper\draft\round7\图表代码_20261006\Figure6_tmQMg跨协议\复现当前稿数据\crossfunct_A_preds.npz'
JS = r'D:\ml\paper\draft\round7\图表代码_20261006\Figure6_tmQMg跨协议\复现当前稿数据\crossfunct_A_results.json'

targets = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
# SI 表 S3 印刷值
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
print('keys:', d.files)
y_true = d['y_true']; preds = d['preds']; ens = d['ens_mean']
print('y_true', y_true.shape, '| preds', preds.shape, '| ens_mean', ens.shape)
print('ens == preds.mean(0):', bool(np.allclose(ens, preds.mean(0), atol=1e-5)))
print()

pears, spears, r2s = [], [], []
ints = {}
print('%-15s | %8s %6s | %8s %6s | %9s %7s | %8s %6s | %9s %7s | %8s %6s' %
      ('property', 'Pear', 'SI', 'Spear', 'SI', 'MAE', 'SI', 'Slope', 'SI', 'RecalM', 'SI', 'rawR2', 'SI'))
for i, p in enumerate(targets):
    yt = y_true[:, i].astype(np.float64)
    yv = ens[:, i].astype(np.float64)
    slope, intercept = np.linalg.lstsq(np.vstack([yv, np.ones_like(yv)]).T, yt, rcond=None)[0]
    r = pearsonr(yt, yv)[0]
    rho = spearmanr(yt, yv)[0]
    mae = float(np.abs(yt - yv).mean())
    recal = float(np.abs(yt - (yv * slope + intercept)).mean())
    r2 = float(r2_score(yt, yv))
    pears.append(r); spears.append(rho); r2s.append(r2); ints[p] = float(intercept)
    s = SI[p]
    print('%-15s | %8.4f %6.3f | %8.4f %6.3f | %9.4f %7s | %8.4f %6.3f | %9.4f %7s | %8.4f %6.2f' %
          (p, r, s['p'], rho, s['rho'], mae, s['mae'], slope, s['slope'], recal, s['recal'], r2, 0))
print()
print('macro Pearson %.4f (SI 0.941) | macro Spearman %.4f (SI 0.933) | macro raw R2 %.4f (SI 0.58)'
      % (np.mean(pears), np.mean(spears), np.mean(r2s)))
print()
print('===== 批注 31 交付材料 =====')
print('[1] 逐性质 raw R2（identity line, sklearn r2_score）:')
for p in targets:
    i = targets.index(p)
    print('  %-15s %+.4f' % (p, r2s[i]))
print('  macro mean = %.4f（圆整 0.58，与 SI P136 一致）' % np.mean(r2s))
print()
print('[2] 七个再校准截距（OLS: y_true = slope * pred + intercept）:')
for p in targets:
    print('  %-15s intercept = %+.6f' % (p, ints[p]))
print()
print('json（crossfunct_A_results.json，原分析自留 P/R2/MAE）对照:')
j = json.load(open(JS, encoding='utf-8'))
for p in targets:
    e = j[p]
    print('  %-15s P=%.6f R2=%.6f MAE=%.6f' % (p, e['pearson_r'], e['r2'], e['mae']))
print('  mean P = %.6f' % j['mean_pearson_r'])
