# -*- coding: utf-8 -*-
"""剩余计数类断言：金属权重范围 0.73-1.20、负偶极 1.1%/1.5%、34.7 重原子、≈11% 最常见金属"""
import sys, json
from collections import Counter
import numpy as np, torch
sys.stdout.reconfigure(encoding='utf-8')

D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
atoms = D['atoms_filt']
N = len(atoms)
mc = Counter()
for a in atoms:
    for z in set(np.asarray(a).tolist()):
        if z >= 21:
            mc[z] += 1
total_c = sum(mc.values())
print('金属计数: %d 种, 总 %d' % (len(mc), total_c))
top = mc.most_common(3)
print('最常见金属:', [(z, c, '%.1f%%' % (c / N * 100)) for z, c in top])
print('  (占全体复合物 %d 的百分比)' % N)

w = {z: max(0.5, min(3.0, (total_c / c) ** 0.5)) for z, c in mc.items()}
sw = np.ones(N)
for i, a in enumerate(atoms):
    zs = [z for z in set(np.asarray(a).tolist()) if z >= 21]
    sw[i] = max((w.get(z, 1.0) for z in zs), default=1.0)
print('原始权重范围 %.3f - %.3f' % (sw.min(), sw.max()))
n = sw / sw.mean()
print('★ 归一化后权重范围 %.4f - %.4f  (印 0.73-1.20)' % (n.min(), n.max()))
print('  归一化后均值 %.4f' % n.mean())

ha = np.array([sum(1 for z in np.asarray(a).tolist() if z > 1) for a in atoms])
print('★ 平均重原子数 %.2f  (印 34.7)  中位 %d' % (ha.mean(), np.median(ha)))

print()
zA = np.load('test_preds_A_8d6d.npz', allow_pickle=True)
zD = np.load('test_preds_D_11d23d.npz', allow_pickle=True)
for z, lab in [(zA, 'SMILES A'), (zD, '3D D')]:
    en = z['ens_mean'][:, 3]
    pr = z['preds'][:, :, 3]
    print('%-9s 集成均值负偶极 %.2f%% | 逐成员负偶极 %.2f%% | 真值负 %.2f%% | 负预测的目标均值 %.4f' % (
        lab, (en < 0).mean() * 100, (pr < 0).mean() * 100, (z['y_true'][:, 3] < 0).mean() * 100,
        z['y_true'][en < 0, 3].mean() if (en < 0).any() else float('nan')))

print()
d = json.load(open('geom_results.json', encoding='utf-8'))
def walk(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + '/' + k)
    else:
        if 'n_' in path or 'num' in path or 'count' in path or 'succ' in path:
            print('  %s = %s' % (path, o))
walk(d)
print('  keys:', list(d.keys()))
for k in d:
    if isinstance(d[k], dict):
        print('  %s -> %s' % (k, list(d[k].keys())[:6]))
