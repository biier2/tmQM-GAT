# -*- coding: utf-8 -*-
"""Remaining count-type assertions: metal weight range 0.73-1.20, negative dipole 1.1%/1.5%, 34.7 heavy atoms, ~11% most common metal"""
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
print('Metal counts: %d species, total %d' % (len(mc), total_c))
top = mc.most_common(3)
print('Most common metals:', [(z, c, '%.1f%%' % (c / N * 100)) for z, c in top])
print('  (percentage of all %d complexes)' % N)

w = {z: max(0.5, min(3.0, (total_c / c) ** 0.5)) for z, c in mc.items()}
sw = np.ones(N)
for i, a in enumerate(atoms):
    zs = [z for z in set(np.asarray(a).tolist()) if z >= 21]
    sw[i] = max((w.get(z, 1.0) for z in zs), default=1.0)
print('Raw weight range %.3f - %.3f' % (sw.min(), sw.max()))
n = sw / sw.mean()
print('★ Normalized weight range %.4f - %.4f  (printed 0.73-1.20)' % (n.min(), n.max()))
print('  normalized mean %.4f' % n.mean())

ha = np.array([sum(1 for z in np.asarray(a).tolist() if z > 1) for a in atoms])
print('★ Mean heavy-atom count %.2f  (printed 34.7)  median %d' % (ha.mean(), np.median(ha)))

print()
zA = np.load('test_preds_A_8d6d.npz', allow_pickle=True)
zD = np.load('test_preds_D_11d23d.npz', allow_pickle=True)
for z, lab in [(zA, 'SMILES A'), (zD, '3D D')]:
    en = z['ens_mean'][:, 3]
    pr = z['preds'][:, :, 3]
    print('%-9s ensemble-mean negative dipole %.2f%% | per-member negative dipole %.2f%% | negative in truth %.2f%% | target mean of negative predictions %.4f' % (
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
