# -*- coding: utf-8 -*-
"""审计 Stage 1：正文里所有 t / p / ΔR² / 两种约定 的说法，逐条回算。"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy import stats
from sklearn.metrics import r2_score

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
INT5 = [0, 1, 2, 3, 4]
TAG = {'A': 'A_8d6d', 'B': 'B_8d22d', 'C': 'C_11d6d', 'D': 'D_11d23d',
       'E': 'E_9d6d', 'F': 'F_10d6d', 'G': 'G_7d6d', 'H': 'H_7d22d'}
D = {}
for k, t in TAG.items():
    z = np.load('test_preds_%s.npz' % t, allow_pickle=True)
    D[k] = (z['preds'], z['ens_mean'], z['y_true'])   # (9,13556,8), (13556,8)
    assert np.allclose(D[k][2], D['A'][2]), k

def r2_mem(k, j):
    pr, en, yt = D[k]
    return np.array([r2_score(yt[:, j], pr[m, :, j]) for m in range(9)])

def r2_ens(k, j):
    pr, en, yt = D[k]
    return r2_score(yt[:, j], en[:, j])

def tmem(k1, k2, js):
    """成员层面配对 t 检验：成员各自的 js 性质均值 R² 之差"""
    a = np.mean([r2_mem(k1, j) for j in js], axis=0)
    b = np.mean([r2_mem(k2, j) for j in js], axis=0)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return d.mean(), t, p

def tfold(k1, k2, js):
    """以折为单元（9 成员 = 3 折 × 3 种子）"""
    a = np.mean([r2_mem(k1, j) for j in js], axis=0).reshape(3, 3).mean(axis=1)
    b = np.mean([r2_mem(k2, j) for j in js], axis=0).reshape(3, 3).mean(axis=1)
    t, p = stats.ttest_rel(a, b)
    return a - b, t, p

def mean_mem(k, js):
    return np.mean([r2_mem(k, j) for j in js])

def mean_ens(k, js):
    return float(np.mean([r2_ens(k, j) for j in js]))

print('=' * 100)
print('成员层面均值 R²（八性质 / 五性质）——正文的「成员层面」口径')
print('=' * 100)
for k in 'ABCDEFGH':
    print('  %s  8prop %.4f   int5 %.4f   (集成层面 8prop %.4f  int5 %.4f)' % (
        k, mean_mem(k, range(8)), mean_mem(k, INT5), mean_ens(k, range(8)), mean_ens(k, INT5)))

print('\n' + '=' * 100)
print('★ P048 "the sum of the B − A and C − A mean gains, +0.0037, falls short of D − A, +0.0075"')
print('=' * 100)
for conv, f in [('成员层面', mean_mem), ('集成层面', mean_ens)]:
    a, b, c, d = (f(k, range(8)) for k in 'ABCD')
    print('  %s：B−A=%+.4f  C−A=%+.4f  和=%+.4f   D−A=%+.4f' % (conv, b - a, c - a, (b - a) + (c - a), d - a))

print('\n' + '=' * 100)
print('逐个 t 检验（成员层面，df=8）')
print('=' * 100)
CHECKS = [
    ('D−A 八性质', 'D', 'A', range(8), 0.0075, 2.80, 0.023),
    ('D−A 五性质', 'D', 'A', INT5, 0.0121, 3.47, 0.008),
    ('C−A 五性质', 'C', 'A', INT5, 0.0141, 10.7, 1e-4),
    ('C−A 八性质', 'C', 'A', range(8), 0.0025, None, 0.23),
    ('B−A 八性质', 'B', 'A', range(8), 0.0033, None, 0.049),
    ('B−A 五性质', 'B', 'A', INT5, 0.0026, None, 0.15),
    ('D−C 五性质', 'D', 'C', INT5, -0.0021, None, 0.51),
    ('E−A 八性质', 'E', 'A', range(8), 0.0042, None, 0.006),
    ('G−A 八性质', 'G', 'A', range(8), -0.108, 16.5, 1e-5),
    ('H−G 八性质', 'H', 'G', range(8), 0.018, None, 0.021),
    ('H−B 八性质', 'H', 'B', range(8), -0.093, None, 1e-5),
]
for lab, k1, k2, js, dprint, tprint, pprint in CHECKS:
    d, t, p = tmem(k1, k2, js)
    print('  %-12s Δ=%+.4f (印 %+.4f)  t=%.2f%s  p=%.4g (印 %.4g)' % (
        lab, d, dprint, t, (' (印 %.2f)' % tprint) if tprint else '', p, pprint))

print('\n  折为单元（3 折）:')
for lab, k1, k2, js, tp, pp in [('D−A 八性质', 'D', 'A', range(8), 6.56, 0.022),
                                ('D−A 五性质', 'D', 'A', INT5, 3.55, 0.071),
                                ('C−A 五性质', 'C', 'A', INT5, 15.6, 0.004)]:
    dd, t, p = tfold(k1, k2, js)
    print('    %-12s Δ折=%s  t=%.2f (印 %.2f)  p=%.4f (印 %.4f)' % (
        lab, np.round(dd, 4), t, tp, p, pp))

print('\n' + '=' * 100)
print('逐性质对比（成员层面，df=8）')
print('=' * 100)
for k1, k2, tag in [('D', 'A', 'D−A'), ('C', 'A', 'C−A'), ('B', 'A', 'B−A'), ('D', 'B', 'D−B'), ('D', 'C', 'D−C'), ('E', 'A', 'E−A'), ('F', 'A', 'F−A')]:
    line = []
    for j, p in enumerate(P8):
        a, b = r2_mem(k1, j), r2_mem(k2, j)
        t, pv = stats.ttest_rel(a, b)
        line.append('%s %+.4f(p=%.3g)' % (p[:6], (a - b).mean(), pv))
    print('  %s: %s' % (tag, '  '.join(line)))

print('\n' + '=' * 100)
print('两种约定的集成层面逐性质 Δ（供 Table 1 caption 的说法核对）')
print('=' * 100)
for k1, k2 in [('C', 'A'), ('D', 'C'), ('D', 'A'), ('B', 'A')]:
    print('  %s−%s: %s' % (k1, k2, '  '.join('%s %+.4f' % (P8[j][:6], r2_ens(k1, j) - r2_ens(k2, j)) for j in range(8))))

print('\n' + '=' * 100)
print('P050 "model-to-model SD across the 9 ensemble members is small (0.004–0.013 in R²)"')
print('=' * 100)
for k in 'ABCD':
    sds = np.array([r2_mem(k, j).std() for j in range(8)])
    print('  %s: SD 范围 %.4f–%.4f   %s' % (k, sds.min(), sds.max(), np.round(sds, 4)))
allsd = np.concatenate([np.array([r2_mem(k, j).std() for j in range(8)]) for k in 'ABCD'])
print('  四变体全部 32 个 SD：%.4f–%.4f' % (allsd.min(), allsd.max()))

print('\n' + '=' * 100)
print('P050 "3D ensemble mean outperforms the 3D single-model average ... dipole (ΔR² = +0.057)"')
print('=' * 100)
pr, en, yt = D['D']
for j, p in enumerate(P8):
    mem_mean_pred = pr[:, :, j].mean(axis=0)          # 先生成预测再平均（= ens_mean）
    single = np.mean([r2_score(yt[:, j], pr[m, :, j]) for m in range(9)])
    print('  %-14s ens %.4f  单模型均值 %.4f  Δ %+.4f' % (p, r2_score(yt[:, j], en[:, j]), single,
                                                          r2_score(yt[:, j], en[:, j]) - single))
