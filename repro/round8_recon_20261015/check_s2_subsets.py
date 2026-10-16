# -*- coding: utf-8 -*-
"""只读：核 SI 表 S2（T07 SchNet / T08 DimeNet++）印刷值究竟算在
13,556（共同子集）还是 13,557（含多出的 1 例）上；并定位多出的那 1 个络合物。"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
R = r'D:\ml\chem_ml5'
P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
# （表 7/8 印刷 Ensemble R2 行）与 MAE 行
SI_T7_R2 = [0.845, 0.838, 0.825, 0.809, 0.982, 0.977, 0.981, 0.981]
SI_T8_R2 = [0.850, 0.861, 0.856, 0.829, 0.991, 0.999, 0.998, 0.999]

za = np.load(os.path.join(R, 'test_preds_A_8d6d.npz'), allow_pickle=True)
teA = np.asarray(za['te_idx'])
print('A te len =', len(teA))

for name, si_r2 in (('test_preds_SchNet.npz', SI_T7_R2), ('test_preds_DimeNet.npz', SI_T8_R2)):
    print('\n===== %s =====' % name)
    z = np.load(os.path.join(R, name), allow_pickle=True)
    preds = np.asarray(z['preds'], np.float64)
    ens = np.asarray(z['ens_mean'], np.float64)
    yt = np.asarray(z['y_true'], np.float64)
    te = np.asarray(z['te_idx'])
    print('preds %s | ens %s | te len %d' % (preds.shape, ens.shape, len(te)))
    if len(te) == len(teA):
        eq = bool(np.array_equal(te, teA))
        print('te == A te（逐位）:', eq)
    else:
        print('te[:%d] == A te（逐位）:' % len(teA), bool(np.array_equal(te[:len(teA)], teA)))

    idx_all = np.arange(len(te))
    idx_com = np.arange(min(len(te), len(teA)))
    for label, idx in (('ALL  N=%4d' % len(te), idx_all), ('COMMON N=%4d' % len(idx_com), idx_com)):
        r2s = np.array([r2_score(yt[idx, j], ens[idx, j]) for j in range(8)])
        maes = np.array([np.abs(yt[idx, j] - ens[idx, j]).mean() for j in range(8)])
        print('  [%s] ens R2 mean = %.4f' % (label, r2s.mean()))
        print('      R2 :', ' '.join('%.4f' % v for v in r2s))
        print('      MAE:', ' '.join('%.5f' % v for v in maes))
    print('  SI R2:', ' '.join('%.3f' % v for v in si_r2), '| mean %.3f' % np.mean(si_r2))
    # 逐性质 diffs（相对 SI 两位舍入的偏差）
    r2_all = np.array([r2_score(yt[idx_all, j], ens[idx_all, j]) for j in range(8)])
    r2_com = np.array([r2_score(yt[idx_com, j], ens[idx_com, j]) for j in range(8)])
    d_all = np.abs(np.round(r2_all, 3) - np.array(si_r2))
    d_com = np.abs(np.round(r2_com, 3) - np.array(si_r2))
    print('  |round(R2_all ,3) - SI| max =', d_all.max(), ' | |round(R2_common,3) - SI| max =', d_com.max())

# 多出的络合物定位（若 DimeNet te 为 13,557）
zd = np.load(os.path.join(R, 'test_preds_DimeNet.npz'), allow_pickle=True)
ted = np.asarray(zd['te_idx'])
if len(ted) > len(teA):
    extra = int(ted[len(teA)])
    d = torch.load(os.path.join(R, 'new_domain_iso.pt'), map_location='cpu', weights_only=False)
    codes = d['csd_codes']
    csv_idx = np.asarray(d['csv_idx'])
    print('\nDimeNet 多出的 1 例：域内下标 =', extra, '| CSD =', codes[extra], '| csv_idx =', csv_idx[extra])
    print('该例在 A(13,556) 中是否存在:', extra in set(teA.tolist()))
