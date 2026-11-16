# -*- coding: utf-8 -*-
"""Redraw Figure 7: ten catalysts, Zr-cat3 has no variant-B bar.

Two outputs:
  fig7_old_check.png -- built from the pre-correction table (data/
                        case_study_corrected_20261114.json), to check that this
                        renderer reproduces the figure that was in the manuscript
                        before the 2026-11-15 correction
  fig7_new.png       -- built from data/case_study_corrected_20261115.json.
                        **This is the raster embedded in the manuscript.**

Style matches the shipped figure: serif font, A #3C92C7 / B #B1E5E8 /
DFT #17506B, legend in a light box between the title and the panels.

Ported from round10/_r2_6_dftgeom_20261115/make_fig7.py; only the input and
output paths changed.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')      # case-study tables shipped with this package
RES = os.path.join(HERE, 'results')    # regenerated rasters
EV = 27.211386

C_A = '#3C92C7'
C_B = '#B1E5E8'
C_D = '#17506B'

# figure order and axis labels (as in the shipped figure)
ORDER = ['Zr-cat3', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'trans-PdCl2(PPh3)2',
         'cis-PdCl2(PPh3)2', 'Pt(dppe)Cl2', 'Co-cat1', 'Rh(CO)2(acac)',
         'Ni(IMes)2Cl2']
XTICK = ['Zr-cat3', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'trans-Pd', 'cis-Pd',
         'Pt(dppe)', 'Co-cat1', 'Rh(acac)', 'Ni(IMes)$_2$']
DFT = {'Zr-cat3': (3.30, 5.32), 'Fe-cat1': (2.76, 9.76), 'Ni-cat3': (1.39, 5.67),
       'Pd-cat2': (2.93, 2.48), 'trans-PdCl2(PPh3)2': (3.19, 0.03),
       'cis-PdCl2(PPh3)2': (3.17, 10.86), 'Pt(dppe)Cl2': (3.95, 10.29),
       'Co-cat1': (3.30, 0.00), 'Rh(CO)2(acac)': (4.03, 3.17),
       'Ni(IMes)2Cl2': (3.24, 0.00)}

# Layout copied from the shipped image7.png (2400x1900 px, 8x6.3333 in @300 dpi):
#   title ink y 28-73, legend frame y 121-189, panel (a) spines 275/892,
#   panel (b) spines 1139/1757, left spine x 203, right edge x 2363.
# Fonts are 0.81x the earlier draft so that every element matches the
# published proportions (title ink height 46 px, legend frame height 68 px).
FS_TITLE = 12.0
FS_SUB = 10.8
FS_YLBL = 10.8
FS_TICK = 8.9
FS_LEG = 9.3
FS_TAG = 11.3
AX1 = [0.0848, 0.5303, 0.8998, 0.3247]
AX2 = [0.0848, 0.0750, 0.8998, 0.3253]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.0,
})
print('resolved font:', matplotlib.rcParams['font.serif'])


def arrays(tab):
    gap_a, gap_b, dip_a, dip_b = [], [], [], []
    for n in ORDER:
        a = tab[n]['smiles_mean']
        b = tab[n].get('d3_mean')
        gap_a.append(a[0] * EV)
        dip_a.append(max(0.0, a[3]))
        gap_b.append(None if b is None else b[0] * EV)
        dip_b.append(None if b is None else max(0.0, b[3]))
    return (np.array(gap_a), gap_b, np.array(dip_a), dip_b)


def panel(ax, va, vb, ref, ylab, subtitle, ylim, tag):
    x = np.arange(len(ORDER))
    w = 0.27
    ax.bar(x - w, va, w, color=C_A, label='SMILES-only scorer (variant A)')
    # variant B: draw bar by bar so a missing value leaves its slot empty
    xs = [xi for xi, v in zip(x, vb) if v is not None]
    vs = [v for v in vb if v is not None]
    ax.bar(xs, vs, w, color=C_B, label='WBO-free RBF scorer (variant B)')
    ax.bar(x + w, np.array(ref), w, color=C_D, label='DFT')
    ax.set_xticks(x)
    ax.set_xticklabels(XTICK, fontsize=FS_TICK)
    ax.set_ylabel(ylab, fontsize=FS_YLBL, fontweight='bold', labelpad=6)
    ax.set_title(subtitle, fontsize=FS_SUB, pad=5)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.tick_params(axis='x', length=6, width=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.figure.text(0.020, ax.get_position().y1, tag, ha='left', va='top',
                   fontsize=FS_TAG, fontweight='bold')


def build(path, out, title):
    tab = json.load(open(path, encoding='utf-8'))
    ga, gb, da, db = arrays(tab)
    refg = [DFT[n][0] for n in ORDER]
    refd = [DFT[n][1] for n in ORDER]

    fig = plt.figure(figsize=(8.0, 6.3333), dpi=300)
    ax1 = fig.add_axes(AX1)
    ax2 = fig.add_axes(AX2)

    panel(ax1, ga, gb, refg, 'HL-Gap (eV)', 'HL-Gap', (0, 4.5), '(a)')
    panel(ax2, da, db, refd, 'Dipole (D)', 'Dipole moment', (0, 12), '(b)')

    fig.text(0.5, 0.9849, title, ha='center', va='top', fontsize=FS_TITLE,
             fontweight='bold')

    handles = [Patch(facecolor=C_A, label='SMILES-only scorer (variant A)'),
               Patch(facecolor=C_B, label='WBO-free RBF scorer (variant B)'),
               Patch(facecolor=C_D, label='DFT')]
    leg = fig.legend(handles=handles, loc='upper center',
                     bbox_to_anchor=(0.5, 0.9460), ncol=3, frameon=True,
                     fontsize=FS_LEG, handlelength=1.55, columnspacing=2.2,
                     borderpad=0.43)
    leg.get_frame().set_facecolor('#F5F5F5')
    leg.get_frame().set_edgecolor('#8C8C8C')
    leg.get_frame().set_linewidth(0.9)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'[written] {out}')


build(os.path.join(DATA, 'case_study_corrected_20261114.json'),
      os.path.join(RES, 'fig7_old_check.png'),
      'Case study: predicted vs DFT for ten in-house catalysts')
build(os.path.join(DATA, 'case_study_corrected_20261115.json'),
      os.path.join(RES, 'fig7_new.png'),
      'Case study: predicted vs DFT for ten in-house catalysts')
