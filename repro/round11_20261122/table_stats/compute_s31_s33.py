# -*- coding: utf-8 -*-
"""Recompute SI Table S31 (per-property ensemble MAE of variants A-D) and
Table S33 (per-property paired differences for the five feature contrasts).

Inputs: preds/test_preds_{A,B,C,D}.npz of this repository (9 members, 13,556
test complexes each).  Both tables are printed; nothing is written to disk.

Usage:
    python compute_s31_s33.py --repo ../../..
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

NAMES = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'MetalCharge', 'Eelec', 'Edisp', 'Polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']
FILES = {'A': 'test_preds_A_8d6d.npz', 'B': 'test_preds_B_8d22d.npz',
         'C': 'test_preds_C_11d6d.npz', 'D': 'test_preds_D_11d23d.npz'}
CONTRASTS = [('B', 'A'), ('C', 'A'), ('D', 'C'), ('D', 'B'), ('D', 'A')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   '..', '..', '..'))
    args = ap.parse_args()
    preds_dir = os.path.join(args.repo, 'preds')

    ens, members, y = {}, {}, None
    for key, fn in FILES.items():
        z = np.load(os.path.join(preds_dir, fn))
        ens[key] = z['ens_mean'].astype(np.float64)
        members[key] = z['preds'].astype(np.float64)
        y = z['y_true'].astype(np.float64)

    print('=== Table S31: per-property ensemble MAE ===')
    print('%-14s %5s %10s %10s %10s %10s' % ('Property', 'Unit', 'A', 'B', 'C', 'D'))
    for j, name in enumerate(NAMES):
        vals = [np.abs(ens[k][:, j] - y[:, j]).mean() for k in 'ABCD']
        print('%-14s %5s %10.4g %10.4g %10.4g %10.4g'
              % (name, UNITS[j], vals[0], vals[1], vals[2], vals[3]))

    print('\n=== Table S33: per-property paired differences (member level) ===')
    r2 = {}
    for k in 'ABCD':
        r2[k] = np.array([[1 - ((members[k][i, :, j] - y[:, j]) ** 2).sum()
                           / ((y[:, j] - y[:, j].mean()) ** 2).sum()
                           for j in range(8)] for i in range(members[k].shape[0])])
    print('%-14s ' % 'Property' + ' '.join('%-16s' % ('%s-%s' % c) for c in CONTRASTS))
    for j, name in enumerate(NAMES):
        cells = []
        for a, b in CONTRASTS:
            d = r2[a][:, j] - r2[b][:, j]
            cells.append('%+.4f (%d/9)' % (d.mean(), (d > 0).sum()))
        print('%-14s ' % name + ' '.join('%-16s' % c for c in cells))
    cells = []
    for a, b in CONTRASTS:
        d = r2[a][:, :5].mean(axis=1) - r2[b][:, :5].mean(axis=1)
        cells.append('%+.4f (%d/9)' % (d.mean(), (d > 0).sum()))
    print('%-14s ' % 'Intensive-5' + ' '.join('%-16s' % c for c in cells))


if __name__ == '__main__':
    main()
