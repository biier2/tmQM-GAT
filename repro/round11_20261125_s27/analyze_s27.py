# -*- coding: utf-8 -*-
"""Recompute the Table 4 and Table S27 numbers from s27_rerun_results.json.

Prints, for the six comparisons (control = the ablation reference configuration,
seeds 42/123/456):
  * per-property treatment-minus-control delta R2 (seed means),
  * eight-property mean delta, its population SD across seeds and same-sign count,
  * intensive-5 mean delta and the corresponding same-sign count,
  * the control / treatment eight-property means used in Table 4.

Usage:
    python analyze_s27.py
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 's27_rerun_results.json'), encoding='utf-8'))

SEEDS = [42, 123, 456]
CFG = ['A', 'B', 'C', 'D', 'E', 'F']
NAME = {'A': 'Metal-centric pooling', 'B': 'Attention heads', 'C': 'Metal-frequency weighting',
        'D': 'Geometry/WBO edge features', 'E': 'RBF centers', 'F': 'Wiberg bond order'}
PROPS = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'MetalCharge', 'Eelec', 'Edisp', 'Polarizability']


def arr(c, s):
    return np.array(D['%s_seed%d' % (c, s)]['r2'])


ctrl = np.stack([arr('baseline', s) for s in SEEDS])

print('%-26s %9s %7s %5s | %9s %5s | %8s %8s'
      % ('comparison', 'd8 mean', 'SD8', 'sign', 'd5 mean', 'sign', 'control', 'treat'))
for c in CFG:
    t = np.stack([arr(c, s) for s in SEEDS])
    d = t - ctrl
    d8 = d.mean(axis=1)
    d5 = d[:, :5].mean(axis=1)
    s8 = int((d8 > 0).sum()) if d8.mean() > 0 else int((d8 < 0).sum())
    s5 = int((d5 > 0).sum()) if d5.mean() > 0 else int((d5 < 0).sum())
    print('%-26s %+9.4f %7.4f %5s | %+9.4f %5s | %8.4f %8.4f'
          % (NAME[c], d8.mean(), d8.std(ddof=0), '%d/3' % s8,
             d5.mean(), '%d/3' % s5, ctrl.mean(), t.mean()))

print('\nper-property deltas (treatment minus control, seed means)')
print('%-14s ' % 'Property' + ' '.join('%16s' % NAME[c].split()[0] for c in CFG))
for j, p in enumerate(PROPS):
    print('%-14s ' % p + ' '.join('%+16.4f' % (np.stack([arr(c, s) for s in SEEDS]) - ctrl)[:, j].mean()
                                  for c in CFG))
