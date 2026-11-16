# -*- coding: utf-8 -*-
"""Atom-resolved occlusion examples (R2-4 request) for four test-set complexes."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')   # archived result files shipped with this package


def _path(name):
    """Resolve an input file: results/ first, then the script directory."""
    p = os.path.join(RES, name)
    return p if os.path.exists(p) else os.path.join(HERE, name)

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge',
           'E_elec', 'E_disp', 'polarizability']
COL = {'COORD': '#c0392b', 'REMOTE_H': '#e59866', 'MBC': '#2e86c1',
       'METAL': '#7d3c98', 'OTHER': '#7f8c8d', 'REMOTE_C': '#d5d8dc'}
LAB = {'COORD': 'Coordinated donor', 'REMOTE_H': 'Remote same element',
       'MBC': 'Metal-bonded C', 'METAL': 'Metal',
       'OTHER': 'Other heteroatom', 'REMOTE_C': 'Remote C'}

PICK = ['DURSEP', 'XEDNUQ', 'MIXROD', 'DAVZOQ']


def main():
    z = np.load(_path('examples_test_ensemble.npz'), allow_pickle=True)
    rows = {r['csd']: r for r in z['rows']}
    j = TARGETS.index('homo')

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4))
    for ax, csd in zip(axes.ravel(), PICK):
        r = rows[csd]
        sym, cls = r['sym'], np.asarray(r['cls'])
        d = np.asarray(r['dabs'])[:, j]
        share = 100 * d / (d.sum() + 1e-30)
        N = len(sym)
        xs = np.arange(N)
        cols = [COL[c] for c in cls]
        ax.bar(xs, share, color=cols, edgecolor='white', linewidth=0.3)
        # label EVERY atom: with thinned ticks the bars no longer line up with
        # their element symbols and the panel becomes unreadable
        ax.set_xticks(xs)
        ax.set_xticklabels(sym, fontsize=(5.6 if N <= 45 else 4.2), rotation=90)
        ax.set_ylabel('|attribution| share (%)', fontsize=8)
        ax.set_title(f'{csd}  (N = {N} heavy atoms)', fontsize=9.5)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.8, N - 0.2)
    present = [c for c in ['METAL', 'MBC', 'COORD', 'REMOTE_H', 'OTHER', 'REMOTE_C']]
    handles = [Patch(color=COL[c], label=LAB[c]) for c in present]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8.6,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('Atom-resolved occlusion attribution (normalized |$\\Delta$| share) '
                 'for HOMO energy — nine-member variant-A ensemble, '
                 'held-out test-set complexes', fontsize=10.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    out = os.path.join(RES, 'figS_atom_examples.png')
    fig.savefig(out, dpi=300)
    fig.savefig(os.path.join(RES, 'figS_atom_examples.pdf'))
    print('[save]', out)


if __name__ == '__main__':
    main()
