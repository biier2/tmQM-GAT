# -*- coding: utf-8 -*-
"""只读：补核表 S2 的 member mean R2 / member SD 两列在两个子集上是否同值。"""
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

sys.stdout.reconfigure(encoding='utf-8')
R = r'D:\ml\chem_ml5'
SI_T7_MEM = [0.753, 0.730, 0.723, 0.676, 0.952, 0.895, 0.978, 0.978]  # member mean
SI_T7_SD = [0.066, 0.036, 0.037, 0.113, 0.033, 0.195, 0.010, 0.005]
SI_T8_MEM = [0.803, 0.806, 0.802, 0.761, 0.986, 0.998, 0.997, 0.998]
SI_T8_SD = [0.008, 0.020, 0.022, 0.036, 0.002, 0.000, 0.001, 0.001]

for name, si_mem, si_sd in (('test_preds_SchNet.npz', SI_T7_MEM, SI_T7_SD),
                            ('test_preds_DimeNet.npz', SI_T8_MEM, SI_T8_SD)):
    print('=====', name, '=====')
    z = np.load(os.path.join(R, name), allow_pickle=True)
    preds = np.asarray(z['preds'], np.float64)
    yt = np.asarray(z['y_true'], np.float64)
    n_all = preds.shape[1]
    for label, idx in (('ALL  %d' % n_all, np.arange(n_all)),
                       ('COMMON 13556', np.arange(13556))):
        mem = np.zeros(8); sd = np.zeros(8)
        for j in range(8):
            per = np.array([r2_score(yt[idx, j], preds[m, idx, j]) for m in range(preds.shape[0])])
            mem[j] = per.mean(); sd[j] = per.std()  # 总体 SD（ddof=0）
        d_mem = np.abs(np.round(mem, 3) - np.array(si_mem))
        d_sd = np.abs(np.round(sd, 3) - np.array(si_sd))
        print('  [%s] member mean R2 =' % label, ' '.join('%.4f' % v for v in mem),
              '| max|ΔvsSI| = %.4f' % d_mem.max())
        print('               member SD      =', ' '.join('%.4f' % v for v in sd),
              '| max|ΔvsSI| = %.4f' % d_sd.max())
