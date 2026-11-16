# -*- coding: utf-8 -*-
"""Atom-resolved occlusion examples, redrawn at the SI print width (6.0 in).

One complex per row instead of a 2x2 grid: at 6 in of print width a 2x2 grid
would shrink the per-atom element labels below legibility.
"""
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

    fig, axes = plt.subplots(4, 1, figsize=(6.0, 6.0))
    for ax, csd in zip(axes, PICK):
        r = rows[csd]
        sym, cls = r['sym'], np.asarray(r['cls'])
        d = np.asarray(r['dabs'])[:, j]
        share = 100 * d / (d.sum() + 1e-30)
        N = len(sym)
        xs = np.arange(N)
        ax.bar(xs, share, color=[COL[c] for c in cls],
               edgecolor='white', linewidth=0.25, width=0.82)
        ax.set_xticks(xs)
        ax.set_xticklabels(sym, fontsize=(5.9 if N <= 45 else 5.0), rotation=90)
        ax.tick_params(axis='x', length=1.8, pad=1.2)
        ax.tick_params(axis='y', labelsize=5.8, length=2.0, pad=1.5)
        ax.set_ylabel('Share (%)', fontsize=6.4, labelpad=2)
        ax.set_title(f'{csd}  (N = {N} heavy atoms)', fontsize=6.8, pad=2.5)
        ax.grid(axis='y', alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.9, N - 0.1)
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    handles = [Patch(color=COL[c], label=LAB[c])
               for c in ['METAL', 'MBC', 'COORD', 'REMOTE_H', 'OTHER', 'REMOTE_C']]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.4,
               frameon=False, bbox_to_anchor=(0.53, 0.002),
               handlelength=1.0, columnspacing=1.0, handletextpad=0.4,
               labelspacing=0.35)
    fig.suptitle('Normalized |$\\Delta$| attribution share per atom for the HOMO energy',
                 fontsize=7.6, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.938,
                        bottom=0.112, hspace=0.60)
    out = os.path.join(RES, 'figS_docx.png')
    fig.savefig(out, dpi=600)
    fig.savefig(os.path.join(RES, 'figS_docx.pdf'))
    print('[save]', out)


if __name__ == '__main__':
    main()
