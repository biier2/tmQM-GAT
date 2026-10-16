# -*- coding: utf-8 -*-
"""SchNet / DimeNet++ 自助置信区间（正文 P061/P062）+ 若干计数"""
import sys
import numpy as np, pandas as pd, torch
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import r2_score

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
INT5 = [0, 1, 2, 3, 4]
zA = np.load('test_preds_A_8d6d.npz', allow_pickle=True)
teA, ytA = zA['te_idx'], zA['y_true']
zD = np.load('test_preds_D_11d23d.npz', allow_pickle=True)
enD, ytD = zD['ens_mean'], zD['y_true']

rng = np.random.RandomState(0)
B = 2000
for tag, lab, ref, ref_en in [('SchNet', 'SchNet', ytA, zA['ens_mean']),
                              ('DimeNet', 'DimeNet++', ytD, enD)]:
    z = np.load('test_preds_%s.npz' % tag, allow_pickle=True)
    common, ia, ib = np.intersect1d(teA, z['te_idx'], return_indices=True)
    yt = z['y_true'][ib]
    en = z['ens_mean'][ib]
    # 参照模型限制到同一公共子集
    _, ja, jb = np.intersect1d(teA, common, return_indices=True)
    ytr = ytA[ja]; enr = ref_en[ja]
    assert np.allclose(yt, ytr)
    n = len(common)
    print('\n===== %s  n=%d =====' % (lab, n))
    d = np.array([r2_score(yt[:, j], en[:, j]) - r2_score(ytr[:, j], enr[:, j]) for j in range(8)])
    print('  ΔR²(%s − 3D GAT): %s' % (lab, ' '.join('%s %+.3f' % (P8[j][:6], d[j]) for j in range(8))))
    print('  8性质均值 Δ = %+.4f ; intensive5 Δ = %+.4f' % (d.mean(), d[INT5].mean()))
    bs = np.empty((B, 8))
    for b in range(B):
        s = rng.randint(0, n, n)
        for j in range(8):
            bs[b, j] = r2_score(yt[s, j], en[s, j]) - r2_score(ytr[s, j], enr[s, j])
    print('  自助 95% CI（逐性质）:')
    for j in range(8):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        ex = '排除零' if lo * hi > 0 else '含零  '
        print('    %-14s Δ=%+.3f  CI [%+.3f, %+.3f]  %s' % (P8[j], d[j], lo, hi, ex))
    lo, hi = np.percentile(bs.mean(axis=1), [2.5, 97.5])
    print('  8性质均值 CI = [%+.4f, %+.4f]' % (lo, hi))

print('\n===== 计数 =====')
TARGETS = ['HL_Gap', 'HOMO_Energy', 'LUMO_Energy', 'Dipole_M', 'Metal_q',
           'Electronic_E', 'Dispersion_E', 'Polarizability']
df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
print('  列名样例:', [c for c in df.columns][:14])
df2 = df.dropna(subset=TARGETS).reset_index(drop=True)
D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
csv_idx = np.asarray(D['csv_idx']); atoms = D['atoms_filt']
te = np.load('ensemble_split_new.npz')['te_idx']
ha = np.array([sum(1 for z in np.asarray(a).tolist() if z > 1) for a in atoms])
print('  全体 90,377 平均重原子 %.2f' % ha.mean())
print('  测试集 13,556 平均重原子 %.2f  (印 34.7)' % ha[te].mean())
for name, idxs in [('全体 90,377', np.arange(90377)), ('测试 13,556', te)]:
    from collections import Counter
    c = Counter()
    for i in idxs:
        for z in set(np.asarray(atoms[i]).tolist()):
            if z >= 21: c[z] += 1
    N = len(idxs)
    top = c.most_common(4)
    cu = c.get(29, 0); zn = c.get(30, 0)
    print('  %s: 最常见 %s | Cu %d (%.1f%%) Zn %d (%.1f%%) 合计 %.1f%%' % (
        name, [(z, '%.1f%%' % (n / N * 100)) for z, n in top], cu, cu / N * 100, zn, zn / N * 100, (cu + zn) / N * 100))
