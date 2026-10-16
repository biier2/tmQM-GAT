# -*- coding: utf-8 -*-
"""审计：保形校准（P052）在【当前】产物上重算，并与印出值对比。
印出：coverage 0.784-0.818 / 0.880-0.908 / 0.937-0.956
      90% half-width: HOMO 0.019, gap 0.020, LUMO 0.018, dipole 3.42 D,
                      metal charge 0.18 e, E_elec 406 Ha, E_disp 0.028 Ha, polar 56.9 a.u.
      半宽/MAE = 1.96-2.45（dipole 2.45）
"""
import sys, json
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import mean_absolute_error

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
PRINTED_HW = {'HL-Gap': 0.020, 'HOMO': 0.019, 'LUMO': 0.018, 'Dipole': 3.42,
              'MetalCharge': 0.18, 'E_elec': 406.0, 'E_disp': 0.028, 'Polarizability': 56.9}


def run(name, per_model, y_true):
    """per_model (9,N,8); 2000 校准 / 其余评估, RandomState(42)"""
    n = len(y_true)
    pm = np.asarray(per_model, dtype=np.float64)
    y_mean = pm.mean(axis=0)
    y_std = pm.std(axis=0)
    rng = np.random.RandomState(42)
    cal_idx = rng.choice(n, size=2000, replace=False)
    m = np.zeros(n, bool); m[cal_idx] = True
    ev = ~m
    sigma = np.maximum(y_std, 1e-6)
    print('\n' + '=' * 104)
    print('%s   N=%d  校准=%d  评估=%d' % (name, n, m.sum(), ev.sum()))
    print('=' * 104)
    out = {}
    covs = {0.8: [], 0.9: [], 0.95: []}
    for j, p in enumerate(P8):
        row = []
        for tgt in (0.8, 0.9, 0.95):
            a = 1 - tgt
            s_cal = np.abs(y_true[m, j] - y_mean[m, j]) / sigma[m, j]
            q = np.quantile(s_cal, min(1.0, (1 - a) * (1 + 1.0 / len(s_cal))))
            s_ev = np.abs(y_true[ev, j] - y_mean[ev, j]) / sigma[ev, j]
            cov = float((s_ev <= q).mean())
            w = 2 * q * sigma[ev, j]
            covs[tgt].append(cov)
            row.append((cov, w.mean() / 2, np.median(w) / 2, q))
        out[p] = row
    mae = [mean_absolute_error(y_true[:, j], y_mean[:, j]) for j in range(8)]
    print('  %-14s %-9s %-9s %-9s | %-8s %-8s | %-8s | %-7s %s' % (
        'property', 'cov@.80', 'cov@.90', 'cov@.95', 'hw90avg', 'hw90med', 'MAE(全)', 'hw/MAE', '印出hw'))
    for j, p in enumerate(P8):
        c80, c90, c95, q = out[p][0], out[p][1], out[p][2], out[p][1][3]
        hw = out[p][1][1]
        print('  %-14s %.4f    %.4f    %.4f    | %.4f   %.4f   | %-8.4g | %-7.3f %.3f' % (
            p, c80[0], c90[0], c95[0], hw, out[p][1][2], mae[j], hw / mae[j], PRINTED_HW[p]))
    for tgt in (0.8, 0.9, 0.95):
        print('  ★ 覆盖率@%.2f 范围 %.4f – %.4f' % (tgt, min(covs[tgt]), max(covs[tgt])))
    r = [out[p][1][1] / mae[j] for j, p in enumerate(P8)]
    print('  ★ hw90/MAE 范围 %.3f – %.3f   (dipole %.3f)' % (min(r), max(r), r[3]))
    return out


z = np.load('test_preds_A_8d6d.npz', allow_pickle=True)
run('当前 SMILES GAT 变体 A (test_preds_A_8d6d)', z['preds'], z['y_true'])
z = np.load('test_preds_D_11d23d.npz', allow_pickle=True)
run('当前 3D GAT 变体 D (test_preds_D_11d23d)', z['preds'], z['y_true'])
print('\n\n—— 已退役产物 ——')
d = np.load('ensemble_predictions.npz')
run('退役 3D 集成 (ensemble_predictions.npz)', d['per_model'], d['y_true'])
d = np.load('r5_smiles_va_ensemble.npz')
run('退役 SMILES 集成 (r5_smiles_va_ensemble.npz)', d['per_model'], d['y_true'])
