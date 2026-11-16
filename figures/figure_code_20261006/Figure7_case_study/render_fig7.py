# -*- coding: utf-8 -*-
"""Fig7 -- case study: predicted values vs DFT on ten in-house catalysts

Layout follows Fig2a (see acs_style.py); the bar-chart approach follows example3; the colours follow Fig5.

Issues fixed relative to the old version:
  · each of the two panels had its own legend; the two overlapped and the right side overflowed and was cropped ("SMILES G..." truncated)
    -> changed to a single global shared legend, centred below the overall title
  · x-axis catalyst names were rotated 40° and crowded -> after stacking the panels vertically, each group's width doubles and the labels can be placed horizontally
  · the panel letter was mixed into the title ("(a) HL-Gap") -> moved outside the upper-left of the axes area, per the Fig2a convention
  · colours changed from blue/grey/red to the light->dark blue sequential scale of Fig5, in the same visual system as Fig5

Colour note: unlike Fig5, this figure is coloured per series rather than per value -- the three bars of each group are
3D scorer (B) / SMILES GAT (A) / DFT, so three widely separated colours are taken from SEQ_PALETTE,
indices 1 / 5 / 7 (#B1E5E8 / #3C92C7 / #17506B, brightness 213.8 / 126.3 / 66.0).
Note: the caption of Figure 7 in the docx says "blue / gray / red" and must be changed in sync.

Output: Fig7_new.png (2400x1900 px @300dpi)
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import acs_style as S

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = r'D:/ml/chem_ml5/case_study_new_results.json'
DPI = 300
EV = 27.211386          # Ha -> eV
LEG_SIZE = 9

S.apply()

r = json.load(open(DATA, encoding='utf-8'))

# the ten DFT-validated catalysts; DFT[k] = (HL-Gap eV, dipole D), TPSSh/def2-SVP
ORDER = ['Zr-cat3', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'trans-PdCl2(PPh3)2', 'cis-PdCl2(PPh3)2',
         'Pt(dppe)Cl2', 'Co-cat1', 'Rh(CO)2(acac)', 'Ni(IMes)2Cl2']
NAMES = ['Zr-cat3', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'trans-Pd', 'cis-Pd',
         'Pt(dppe)', 'Co-cat1', 'Rh(acac)', 'Ni(IMes)$_2$']
DFT = {'Zr-cat3': (3.30, 5.32), 'Fe-cat1': (2.76, 9.76), 'Ni-cat3': (1.39, 5.67), 'Pd-cat2': (2.93, 2.48),
       'trans-PdCl2(PPh3)2': (3.19, 0.03), 'cis-PdCl2(PPh3)2': (3.17, 10.86), 'Pt(dppe)Cl2': (3.95, 10.29),
       'Co-cat1': (3.30, 0.00), 'Rh(CO)2(acac)': (4.03, 3.17), 'Ni(IMes)2Cl2': (3.24, 0.00)}

# series: following the old figure's within-group order (B, A, DFT); colours taken from three widely separated steps of the Fig5 scale
SERIES = [('3D scorer (B)',   S.SEQ_PALETTE[1]),   # #B1E5E8
          ('SMILES GAT (A)',  S.SEQ_PALETTE[5]),   # #3C92C7
          ('DFT',             S.SEQ_PALETTE[7])]   # #17506B

PANELS = [('a', 'HL-Gap', 'HL-Gap (eV)', 0),
          ('b', 'Dipole moment', 'Dipole (D)', 3)]

# values of each series
def series_vals(mode_idx):
    b = [r[k]['d3_mean'][mode_idx] for k in ORDER]
    a = [r[k]['smiles_mean'][mode_idx] for k in ORDER]
    if mode_idx == 0:                       # gap: Ha -> eV
        b = [v * EV for v in b]; a = [v * EV for v in a]
    else:                                   # dipole: take the magnitude, negative values clipped to 0
        b = [max(0.0, v) for v in b]; a = [max(0.0, v) for v in a]
    d = [DFT[k][0 if mode_idx == 0 else 1] for k in ORDER]
    return b, a, d


fig, axes = plt.subplots(2, 1, figsize=(2400/DPI, 1900/DPI))
# top 0.855: leaves room for the overall title (0.985) and the shared legend (0.945)
plt.subplots_adjust(hspace=0.40, left=0.085, right=0.985, top=0.855, bottom=0.075)

x = np.arange(len(ORDER))
w = 0.30                                     # offset ±w makes the three bars of a group touch (example3 approach)

for ax, (letter, title, ylab, mi) in zip(axes, PANELS):
    b_vals, a_vals, dft_vals = series_vals(mi)
    for off, (_, color), vals in zip((-w, 0.0, w), SERIES, (b_vals, a_vals, dft_vals)):
        ax.bar(x + off, vals, w, color=color, edgecolor='none')

    ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=S.CAT_SIZE)
    ax.set_ylabel(ylab, fontsize=S.LABEL_SIZE, fontweight='bold', labelpad=2)
    ax.set_title(title, fontsize=S.TITLE_SIZE, pad=5)
    ax.tick_params(labelsize=S.TICK_SIZE, pad=3)
    ax.text(-0.075, 1.02, f'({letter})', transform=ax.transAxes,
            fontsize=S.LETTER_SIZE, fontweight='bold', ha='left', va='bottom')
    ax.set_ylim(0, max(b_vals + a_vals + dft_vals) * 1.10)

# global shared legend (replacing the two duplicated and mutually overlapping legends of the old version)
handles = [Patch(facecolor=c, edgecolor='none', label=n) for n, c in SERIES]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.945),
           ncol=3, fontsize=LEG_SIZE)

fig.suptitle('Case study: predicted vs DFT for ten in-house catalysts',
             fontsize=S.SUPTITLE_SIZE, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig7_new.png'), dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig7 ✓  2400x1900  (layout follows Fig2a, colours follow Fig5)')
print('output directory →', OUT)
