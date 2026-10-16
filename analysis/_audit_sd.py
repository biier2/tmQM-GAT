# -*- coding: utf-8 -*-
"""P050: 'model-to-model standard deviation across the 9 ensemble members is small (0.004-0.013 in R²)'
逐个候选产物算「逐性质 9 成员 R² 的总体 SD」，看哪一个给出 0.004-0.013。"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import r2_score

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]

def rep(name, pm, yt):
    pm = np.asarray(pm, float)
    sd = np.array([np.array([r2_score(yt[:, j], pm[m, :, j]) for m in range(pm.shape[0])]).std() for j in range(8)])
    per_member_mean = np.array([[r2_score(yt[:, j], pm[m, :, j]) for j in range(8)] for m in range(pm.shape[0])]).mean(axis=1)
    print('%-34s N=%5d SD范围 %.4f-%.4f | 8性质均值SD %.4f | 逐性质 %s' % (
        name, len(yt), sd.min(), sd.max(), per_member_mean.std(),
        ' '.join('%.4f' % v for v in sd)))

for tag, lab in [('A_8d6d', 'A'), ('B_8d22d', 'B'), ('C_11d6d', 'C'), ('D_11d23d', 'D'),
                 ('E_9d6d', 'E'), ('F_10d6d', 'F'), ('G_7d6d', 'G'), ('H_7d22d', 'H')]:
    z = np.load('test_preds_%s.npz' % tag, allow_pickle=True)
    rep('当前 ' + lab, z['preds'], z['y_true'])

print()
for f, k in [('r5_smiles_va_ensemble.npz', 'per_model'), ('ensemble_predictions.npz', 'per_model'),
             ('r3_smiles_ensemble.npz', 'per_model'), ('r3b_xyz_ensemble.npz', 'per_model')]:
    try:
        z = np.load(f, allow_pickle=True)
        rep('退役 ' + f, z[k], z['y_true'])
    except Exception as e:
        print(f, 'ERR', e)
