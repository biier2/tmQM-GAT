# -*- coding: utf-8 -*-
"""Redraw Fig. 5 from the test-set recomputation.

(a) stacked attribution share by atom class, per property
(b) per-atom intensity ratio, atom-pooled (1.0 = the average atom)
(c) fraction of complexes in which each class exceeds its own size-matched
    expectation (50% = the exchangeability null)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')   # archived result files shipped with this package


def _path(name):
    """Resolve an input file: results/ first, then the script directory."""
    p = os.path.join(RES, name)
    return p if os.path.exists(p) else os.path.join(HERE, name)

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge',
           'E_elec', 'E_disp', 'polarizability']
LABEL = {'gap': 'HL-Gap', 'homo': 'HOMO', 'lumo': 'LUMO', 'dipole': 'Dipole',
         'metal_charge': 'Metal q', 'E_elec': 'E$_{elec}$',
         'E_disp': 'E$_{disp}$', 'polarizability': 'Polariz.'}
GROUPS = [('COORD', 'Coordinated donor', '#c0392b'),
          ('REMOTE_H', 'Remote same element', '#e59866'),
          ('MBC', 'Metal-bonded C', '#2e86c1'),
          ('METAL', 'Metal', '#7d3c98'),
          ('OTHER', 'Other heteroatom', '#7f8c8d'),
          ('REMOTE_C', 'Remote C', '#d5d8dc')]


def load(p):
    with open(_path(p), encoding='utf-8') as f:
        d = json.load(f)
    for sect in ('share', 'per_atom_ratio', 'frac_above_null', 'excess_pp'):
        for t in d[sect]:
            d[sect][t]['COORD'] = d[sect][t]['A_new']
    for t in d['pooled_per_atom_ratio']:
        d['pooled_per_atom_ratio'][t]['COORD'] = d['pooled_per_atom_ratio'][t]['A_new']
    return d


def main():
    d = load('fig5_test_ensemble_pred.json')
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5))
    x = np.arange(len(TARGETS))

    # ---- (a) stacked share
    ax = axes[0]
    bottom = np.zeros(len(TARGETS))
    for k, lab, col in GROUPS:
        v = np.array([d['share'][t][k]['mean'] * 100 for t in TARGETS])
        ax.bar(x, v, 0.68, bottom=bottom, label=lab, color=col,
               edgecolor='white', linewidth=0.5)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[t] for t in TARGETS], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('Share of total |attribution| (%)', fontsize=9)
    ax.set_title('(a) Attribution share by atom class', fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.0, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), ncol=3)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    # ---- (b) per-atom intensity
    ax = axes[1]
    w = 0.135
    for i, (k, lab, col) in enumerate(GROUPS):
        v = [d['pooled_per_atom_ratio'][t][k] for t in TARGETS]
        ax.bar(x + (i - 2.5) * w, v, w, label=lab, color=col,
               edgecolor='white', linewidth=0.4)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.65)
    ax.set_yscale('log')
    ax.set_ylim(0.2, 20)
    ax.set_yticks([0.25, 0.5, 1, 2, 5, 10, 15])
    ax.set_yticklabels(['0.25', '0.5', '1', '2', '5', '10', '15'])
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[t] for t in TARGETS], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('Per-atom intensity ratio (log scale)', fontsize=9)
    ax.set_title('(b) Sensitivity per atom (1.0 = average atom)', fontsize=10)
    ax.grid(axis='y', alpha=0.2, which='both')
    ax.set_axisbelow(True)

    # ---- (c) fraction above size-matched null
    ax = axes[2]
    show = [('COORD', 'Coordinated donor', '#c0392b'),
            ('REMOTE_H', 'Remote same element', '#e59866'),
            ('MBC', 'Metal-bonded C', '#2e86c1'),
            ('METAL', 'Metal', '#7d3c98')]
    for i, (k, lab, col) in enumerate(show):
        v = [100 * d['frac_above_null'][t][k]['mean'] for t in TARGETS]
        ax.bar(x + (i - 1.5) * 0.2, v, 0.2, label=lab, color=col,
               edgecolor='white', linewidth=0.4)
    ax.axhline(50, color='k', lw=1.0, ls='--', alpha=0.8)
    ax.text(len(TARGETS) - 0.45, 52, 'chance', fontsize=7.2, ha='right', alpha=0.9,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[t] for t in TARGETS], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('Complexes above size-matched null (%)', fontsize=9)
    ax.set_title('(c) Size-matched exchangeability test', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(RES, 'fig5_new.png')
    fig.savefig(out, dpi=300)
    fig.savefig(os.path.join(RES, 'fig5_new.pdf'))
    print('[save]', out)


if __name__ == '__main__':
    main()
