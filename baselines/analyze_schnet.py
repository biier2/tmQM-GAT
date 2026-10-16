# -*- coding: utf-8 -*-
"""
SchNet（数据修复版）vs 八个 GAT 变体：统一到 13,556 共同测试集后对比。
产出 results_SchNet_aligned.json，供论文/SI 引用。
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

FILES = {'A': 'test_preds_A_8d6d.npz', 'B': 'test_preds_B_8d22d.npz',
         'C': 'test_preds_C_11d6d.npz', 'D': 'test_preds_D_11d23d.npz',
         'E': 'test_preds_E_9d6d.npz', 'F': 'test_preds_F_10d6d.npz',
         'G': 'test_preds_G_7d6d.npz', 'H': 'test_preds_H_7d22d.npz'}
NAMES = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'Metal charge', 'E_elec', 'E_disp', 'Polarizability']

R, P, MAE = {}, {}, {}
for v, f in FILES.items():
    z = np.load(f, allow_pickle=True)
    y, em = z['y_true'], z['ens_mean']
    R[v] = [r2_score(y[:, j], em[:, j]) for j in range(8)]
    MAE[v] = [mean_absolute_error(y[:, j], em[:, j]) for j in range(8)]
    P[v] = z['preds']

# SchNet：截取到 A 的 13,556 索引（SchNet 的 te_idx 多 1 个 = round() vs int()）
zs = np.load('test_preds_SchNet.npz', allow_pickle=True)
za = np.load('test_preds_A_8d6d.npz', allow_pickle=True)
te_s, Ps, ys = np.asarray(zs['te_idx']), zs['preds'], zs['y_true']
pos = {int(v): k for k, v in enumerate(te_s)}
pa = np.array([pos[int(v)] for v in za['te_idx']])
Ps, ys = Ps[:, pa, :], ys[pa]
assert np.array_equal(ys, za['y_true']), '标签不一致'
R['SchNet'] = [r2_score(ys[:, j], Ps.mean(0)[:, j]) for j in range(8)]
MAE['SchNet'] = [mean_absolute_error(ys[:, j], Ps.mean(0)[:, j]) for j in range(8)]
P['SchNet'] = Ps
mem = np.array([[r2_score(ys[:, j], Ps[m, :, j]) for j in range(8)] for m in range(9)])

ORDER = ['A', 'E', 'F', 'C', 'G', 'H', 'B', 'D', 'SchNet']
print('=== 逐性质 ensemble R2（13,556 共同测试集）===')
print(f'{"Property":<15}' + ''.join(f'{v:>9}' for v in ORDER))
for j, n in enumerate(NAMES):
    print(f'{n:<15}' + ''.join(f'{R[v][j]:>9.3f}' for v in ORDER))
print(f'{"Mean R2":<15}' + ''.join(f'{np.mean(R[v]):>9.4f}' for v in ORDER))
print()
print('=== SchNet vs 各变体：逐性质差 (SchNet - X) ===')
for v in ['A', 'B', 'C', 'D']:
    d = [R['SchNet'][j] - R[v][j] for j in range(8)]
    print(f'  vs {v}: ' + ' '.join(f'{NAMES[j][:6]}={d[j]:+.3f}' for j in range(8))
          + f'  | mean {np.mean(R["SchNet"])-np.mean(R[v]):+.4f}')
print()
print('=== SchNet 成员级 ===')
print('  member R2 mean:', np.round(mem.mean(0), 3).tolist())
print('  member R2 std :', np.round(mem.std(0), 3).tolist())
print('  member mean(M):', round(float(mem.mean()), 4), ' ensemble(M):', round(float(np.mean(R['SchNet'])), 4))

out = {'test_set': '13,556 complexes (identical to variants A-H)',
       'ens_r2': {v: [round(float(x), 4) for x in R[v]] for v in ORDER},
       'ens_mae': {v: [round(float(x), 4) for x in MAE[v]] for v in ORDER},
       'mean_r2': {v: round(float(np.mean(R[v])), 4) for v in ORDER},
       'properties': NAMES,
       'schnet_member_r2_mean': [round(float(x), 4) for x in mem.mean(0)],
       'schnet_member_r2_std': [round(float(x), 4) for x in mem.std(0)],
       'schnet_runtime_min': 2272.0}
json.dump(out, open('results_SchNet_aligned.json', 'w'), indent=2)
print('\nsaved results_SchNet_aligned.json ✓')
