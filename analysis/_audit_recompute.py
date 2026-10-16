# -*- coding: utf-8 -*-
"""正文审计 · 数值回算：Table 1 / Table 2 / 摘要-结论中的基准数字，全部从当前 npz 重算。"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import r2_score, mean_absolute_error

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
INT5 = [0, 1, 2, 3, 4]          # size-intensive-5
PRINT = "%.4f"


def load(tag):
    z = np.load('test_preds_%s.npz' % tag, allow_pickle=True)
    return z['preds'], z['ens_mean'], z['y_true'], z['te_idx']


def stats(tag, label):
    pr, en, yt, te = load(tag)
    n = len(yt)
    ens_r2 = [r2_score(yt[:, j], en[:, j]) for j in range(8)]
    mem_r2 = np.array([[r2_score(yt[:, j], pr[m, :, j]) for j in range(8)] for m in range(pr.shape[0])])
    sd = mem_r2.std(axis=0)                     # ddof=0
    mm = mem_r2.mean(axis=0)                    # 成员均值
    mae = [mean_absolute_error(yt[:, j], en[:, j]) for j in range(8)]
    print('\n%-8s n=%d members=%d' % (label, n, pr.shape[0]))
    print('  %-14s %-18s %-18s %-18s' % ('property', 'R2(ens)', 'R2(ens)+-SD', 'member-mean'))
    for j, p in enumerate(P8):
        print('  %-14s %-18s %-18s %-18s' % (p, PRINT % ens_r2[j],
                                             '%s ± %s' % (PRINT % ens_r2[j], PRINT % sd[j]),
                                             PRINT % mm[j]))
    print('  %-14s %-18s %-18s %-18s' % ('MAE', ' | '.join('%.4g' % m for m in mae), '', ''))
    print('  intensive-5  ens=%.4f  8prop ens=%.4f' % (
        np.mean([ens_r2[j] for j in INT5]), np.mean(ens_r2)))
    print('  intensive-5  member-mean=%.4f  8prop member-mean=%.4f  SD(int5)=%.4f' % (
        np.mean([mm[j] for j in INT5]), np.mean(mm), np.mean([sd[j] for j in INT5])))
    return ens_r2, mae, te, yt


print('=' * 96)
print('Table 1 —— variant A/B/C/D（集成读值 + 成员 R² 的总体 SD(ddof=0)）')
print('=' * 96)
T1 = {}
for tag, lab in [('A_8d6d', 'A'), ('B_8d22d', 'B'), ('C_11d6d', 'C'), ('D_11d23d', 'D')]:
    T1[lab] = stats(tag, lab)

print('\n' + '=' * 96)
print('Table 1 印出值 vs 重算值')
print('=' * 96)
PRINTED = {  # (R², ±SD) —— 逐字取自 docx
    'A': [(0.846, 0.010), (0.947, 0.003), (0.943, 0.005), (0.707, 0.010), (0.976, 0.006), (0.965, 0.005), (0.921, 0.016), (0.934, 0.012)],
    'B': [(0.848, 0.010), (0.947, 0.006), (0.941, 0.007), (0.718, 0.008), (0.978, 0.006), (0.964, 0.009), (0.928, 0.012), (0.941, 0.009)],
    'C': [(0.867, 0.007), (0.948, 0.003), (0.949, 0.002), (0.734, 0.007), (0.978, 0.005), (0.961, 0.004), (0.896, 0.010), (0.909, 0.009)],
    'D': [(0.864, 0.013), (0.949, 0.006), (0.948, 0.006), (0.739, 0.021), (0.979, 0.007), (0.966, 0.009), (0.921, 0.013), (0.934, 0.014)],
}
PMEAN = {'A': (0.884, 0.905), 'B': (0.887, 0.908), 'C': (0.895, 0.905), 'D': (0.896, 0.912)}
bad = 0
for lab in 'ABCD':
    pr, en, yt, te = load({'A': 'A_8d6d', 'B': 'B_8d22d', 'C': 'C_11d6d', 'D': 'D_11d23d'}[lab])
    ens = [r2_score(yt[:, j], en[:, j]) for j in range(8)]
    mem = np.array([[r2_score(yt[:, j], pr[m, :, j]) for j in range(8)] for m in range(9)])
    sd = mem.std(axis=0); mm = mem.mean(axis=0)
    for j, p in enumerate(P8):
        r2p, sdp = PRINTED[lab][j]
        d1 = ens[j] - r2p
        d2 = round(sd[j], 4) - sdp
        flag = '' if abs(d1) < 5.1e-4 and abs(d2) < 5.1e-5 + 1e-9 else '  <<< '
        if flag:
            bad += 1
        print('  %s %-14s R² 印 %.3f 算 %.4f (Δ%+.4f)  ± 印 %.3f 算 %.4f (Δ%+.4f)%s' % (
            lab, p, r2p, ens[j], d1, sdp, sd[j], d2, flag))
    i5p, i8p = PMEAN[lab]
    i5 = np.mean([ens[j] for j in INT5]); i8 = np.mean(ens)
    print('  %s  Mean(int5) 印 %.3f 算 %.4f | Mean(8) 印 %.3f 算 %.4f   [成员均值 %.4f / %.4f]' % (
        lab, i5p, i5, i8p, i8, np.mean([mm[j] for j in INT5]), np.mean(mm)))
print('\n  可疑格数 %d' % bad)

# ---------------- Table 2 ----------------
print('\n' + '=' * 96)
print('Table 2 —— 基线（SchNet/DimeNet++ 取与 GAT 的公共测试子集）')
print('=' * 96)
prA, enA, ytA, teA = load('A_8d6d')
for tag, lab in [('SchNet', 'SchNet'), ('DimeNet', 'DimeNet++')]:
    z = np.load('test_preds_%s.npz' % tag, allow_pickle=True)
    te = z['te_idx']
    common, ia, ib = np.intersect1d(teA, te, return_indices=True)
    yt = z['y_true'][ib]
    en = z['ens_mean'][ib]
    same = np.allclose(yt, ytA[ia])
    r2 = [r2_score(yt[:, j], en[:, j]) for j in range(8)]
    mae = [mean_absolute_error(yt[:, j], en[:, j]) for j in range(8)]
    print('  %-9s n=%d(公共) y_true 与 GAT 一致:%s' % (lab, len(common), same))
    for j, p in enumerate(P8):
        print('    %-14s R²=%.4f  MAE=%.6g' % (p, r2[j], mae[j]))
    print('    intensive-5 mean R²=%.4f   8-prop mean R²=%.4f' % (
        np.mean([r2[j] for j in INT5]), np.mean(r2)))
