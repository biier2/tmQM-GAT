# -*- coding: utf-8 -*-
"""Fig. 5 redrawn at the exact print size used in the docx (5.70 in wide).

Three stacked panels so the labels stay legible at print width:
  (a) share of total |attribution| by six atom classes
  (b) attribution per atom relative to the average atom of the same complex
  (c) fraction of complexes above the size-matched exchangeability null
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

W = 5.70            # inches, matches the wp:extent of the published figure
H = 5.05
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


def style_ax(ax, title):
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels([LABEL[t] for t in TARGETS], rotation=28, ha='right',
                       fontsize=6.6)
    ax.tick_params(axis='y', labelsize=6.6, length=2.2, pad=1.5)
    ax.tick_params(axis='x', length=2.2, pad=1.5)
    ax.set_title(title, fontsize=7.5, pad=3)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.grid(axis='y', alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)


def main():
    d = load('fig5_test_ensemble_pred.json')
    fig, axes = plt.subplots(3, 1, figsize=(W, H),
                             gridspec_kw={'height_ratios': [1.16, 1.0, 1.0],
                                          'hspace': 0.60})
    x = np.arange(len(TARGETS))

    # ---- (a) stacked share
    ax = axes[0]
    bottom = np.zeros(len(TARGETS))
    handles = []
    for k, lab, col in GROUPS:
        v = np.array([d['share'][t][k]['mean'] * 100 for t in TARGETS])
        bars = ax.bar(x, v, 0.66, bottom=bottom, label=lab, color=col,
                      edgecolor='white', linewidth=0.35)
        handles.append(bars[0])
        bottom += v
    ax.set_ylabel('Share of total\n|attribution| (%)', fontsize=6.9, labelpad=2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    style_ax(ax, '(a) Attribution share by atom class')
    fig.legend(handles, [lab for _, lab, _ in GROUPS], fontsize=6.0,
               frameon=False, ncol=3, loc='upper center',
               bbox_to_anchor=(0.585, 0.998), handlelength=1.0,
               columnspacing=0.85, handletextpad=0.35, labelspacing=0.35)

    # ---- (b) per-atom intensity
    ax = axes[1]
    w = 0.128
    for i, (k, lab, col) in enumerate(GROUPS):
        v = [d['pooled_per_atom_ratio'][t][k] for t in TARGETS]
        ax.bar(x + (i - 2.5) * w, v, w, label=lab, color=col,
               edgecolor='white', linewidth=0.25)
    ax.axhline(1.0, color='k', lw=0.6, ls='--', alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylim(0.22, 18)
    ax.set_yticks([0.25, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(['0.25', '0.5', '1', '2', '5', '10'])
    ax.set_ylabel('Attribution per atom\n(1.0 = average atom)', fontsize=6.6,
                  labelpad=2)
    style_ax(ax, '(b) Sensitivity per atom (log scale)')

    # ---- (c) fraction above size-matched null
    ax = axes[2]
    show = [('COORD', 'Coordinated donor', '#c0392b'),
            ('REMOTE_H', 'Remote same element', '#e59866'),
            ('MBC', 'Metal-bonded C', '#2e86c1'),
            ('METAL', 'Metal', '#7d3c98')]
    w = 0.2
    for i, (k, lab, col) in enumerate(show):
        v = [100 * d['frac_above_null'][t][k]['mean'] for t in TARGETS]
        ax.bar(x + (i - 1.5) * w, v, w, label=lab, color=col,
               edgecolor='white', linewidth=0.25)
    ax.axhline(50, color='k', lw=0.7, ls='--', alpha=0.85)
    ax.text(-0.50, 50, 'chance', fontsize=5.5, ha='center', va='center',
            rotation=90,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Complexes above\nsize-matched null (%)', fontsize=6.6,
                  labelpad=2)
    style_ax(ax, '(c) Size-matched exchangeability test')

    fig.subplots_adjust(left=0.165, right=0.985, top=0.885, bottom=0.105)
    out = os.path.join(RES, 'fig5_docx.png')
    fig.savefig(out, dpi=600)
    fig.savefig(os.path.join(RES, 'fig5_docx.pdf'))
    print('[save]', out, f'{W}x{H} in')


if __name__ == '__main__':
    main()
