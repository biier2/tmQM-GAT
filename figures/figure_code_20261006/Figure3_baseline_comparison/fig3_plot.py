# -*- coding: utf-8 -*-
"""Render Figure 3 of the manuscript: benchmark comparison (a) and per-property profile (b).

Panel (a)  seven models as horizontal bars, two per model: "All 8 properties"
           (mean R^2 over every property, same colour at alpha) and
           "Intensive-5 (ranking criterion)" (the five intensive properties).
           The seven models are exactly the seven rows of Table 3.
Panel (b)  eight properties x five series, R^2 per property.  The
           "Composition only" series must stay: without it a reader would
           attribute the high R^2 of E_elec and polarizability to the models.

    python fig3_plot.py

Input   r1_final_tables.json     (mean8 / mean5 of the seven models)
        the per-property profile is hard-coded below; it is the same array as
        Table 2 of the manuscript
Output  results/fig3_baseline_profile.png   2,859 x 3,231 px at 300 dpi

The canvas, both axes rectangles, every font size and every colour below were
measured off the published raster; they are not round numbers because they are
what reproduces the printed proportions at 300 dpi.

Both reference lines and every bar height are recomputed from the inputs, never
typed in.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, 'r1_final_tables.json')
OUTDIR = os.path.join(HERE, 'results')
OUT = os.path.join(OUTDIR, 'fig3_baseline_profile.png')

# ------------------------------------------------------------------- geometry
# fractions of the 2,859 x 3,231 canvas
AX_A = [0.231375, 0.55973, 0.735222, 0.35933]
AX_B = [0.09899, 0.04426, 0.87478, 0.36552]

TAG_A = (0.019937, 0.962582)          # left / va=bottom anchor
TAG_B = (0.015390, 0.454345)

# ---------------------------------------------------------------------- style
# Sizes are fitted to the ink boxes measured on the published raster, so the
# non-round values are deliberate.
FS_AYT = 11.0         # panel (a) y tick labels
FS_AXT = 10.3          # panel (a) x tick labels
FS_AXL = 12.0          # panel (a) x axis label
FS_ALEG = 10.2        # panel (a) legend
FS_VAL = 9.5           # "All 8 properties" value labels (grey)
FS_VALB = 10.0         # "Intensive-5" value labels (bold black)
FS_BXT = 12.0          # panel (b) x tick labels
FS_BYT = 12.0         # panel (b) y tick labels
FS_BYL = 14.6          # panel (b) y axis label
FS_BLEG = 11.0        # panel (b) legend
FS_TAG = 10.5          # panel tags (bold)

# The panel (b) property names are the one text role in this figure that is not
# DejaVu Sans: their ink widths only agree with an Arial-metric face (size-free
# width ratios fit Arial to 4 % and DejaVu to 11 %).  Everything else -- both
# legends, both axes' tick labels, both axis labels -- is DejaVu Sans.
# Arial is present on Windows/macOS; matplotlib falls back to DejaVu elsewhere.
def _tick_font(family, size):
    from matplotlib.font_manager import FontProperties, findfont
    try:
        findfont(FontProperties(family=family), fallback_to_default=False)
        return FontProperties(family=family, size=size)
    except Exception:                                   # noqa: BLE001
        return FontProperties(family='DejaVu Sans', size=size)


FP_BXT = _tick_font('Arial', FS_BXT)

C_GREY1 = '#90A4AE'
C_GREY2 = '#B0BEC5'
C_RED1 = '#C0392B'
C_RED2 = '#8E2420'
C_ORANGE = '#F39C12'
C_GREEN = '#27AE60'
C_GRID_A = '#D9D9D9'
C_GRID_B = '#E0E0E0'
C_SEP = '#B0B0B0'
BAR_ALPHA = 0.33

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# ------------------------------------------------------------------------ data
J = json.load(open(JSON, encoding='utf-8'))
M = J['models']

ROWS = [
    ('RF (ECFP4)',                   M['RF']['mean8'],           M['RF']['mean5'],           C_GREY1),
    ('XGBoost (ECFP4)',              M['XGBoost']['mean8'],      M['XGBoost']['mean5'],      C_GREY1),
    ('Composition only',             M['Composition']['mean8'],  M['Composition']['mean5'],  C_GREY2),
    ('SchNet',                       M['SchNet']['mean8'],       M['SchNet']['mean5'],       C_RED1),
    ('DimeNet++',                    M['DimeNet++']['mean8'],    M['DimeNet++']['mean5'],    C_RED2),
    ('SMILES-only GAT\n(variant A)', M['SMILES GAT']['mean8'],   M['SMILES GAT']['mean5'],   C_ORANGE),
    ('Geometry/WBO-augmented\nGAT (variant D)',
                                     M['3D GAT']['mean8'],       M['3D GAT']['mean5'],       C_GREEN),
]

PROPS = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'Metal charge',
         'E$_{elec}$', 'E$_{disp}$', 'Polarizability']
SERIES = [
    ('Composition only', C_GREY2,
     [0.310, 0.104, 0.112, 0.097, 0.659, 1.000, 0.977, 0.998]),
    ('SchNet', C_RED1,
     [0.845, 0.838, 0.825, 0.809, 0.982, 0.977, 0.981, 0.981]),
    ('DimeNet++', C_RED2,
     [0.850, 0.861, 0.856, 0.829, 0.991, 0.999, 0.998, 0.999]),
    ('SMILES-only GAT\n(variant A)', C_ORANGE,
     [0.846, 0.947, 0.943, 0.707, 0.976, 0.965, 0.921, 0.934]),
    ('Geometry/WBO-augmented\nGAT (variant D)', C_GREEN,
     [0.864, 0.949, 0.948, 0.739, 0.979, 0.966, 0.921, 0.934]),
]

os.makedirs(OUTDIR, exist_ok=True)
fig = plt.figure(figsize=(2859 / 300.0, 3231 / 300.0), dpi=300)

# ------------------------------------------------------------------ panel (a)
axa = fig.add_axes(AX_A)
y = np.arange(len(ROWS))[::-1]
H = 0.34
GAP = 0.0223
for yi, (name, m8, m5, col) in zip(y, ROWS):
    axa.barh(yi + H / 2 + GAP, m8, height=H, color=col, alpha=BAR_ALPHA,
             edgecolor='none', zorder=3)
    axa.barh(yi - H / 2 - GAP, m5, height=H, color=col,
             edgecolor='none', zorder=3)
    axa.text(m8 + 0.012, yi + H / 2 + GAP, '%.3f' % m8, va='center', ha='left',
             fontsize=FS_VAL, color='#8a8a8a', zorder=4)
    axa.text(m5 + 0.012, yi - H / 2 - GAP, '%.3f' % m5, va='center', ha='left',
             fontsize=FS_VALB, color='#1a1a1a', fontweight='bold', zorder=4)

axa.set_yticks(y)
axa.set_yticklabels([r[0] for r in ROWS], fontsize=FS_AYT)
for _t in axa.get_yticklabels():
    _t.set_multialignment('center')     # the two-line labels centre their lines
axa.set_xlim(0, 1.10)
axa.set_ylim(-0.72, len(ROWS) - 0.28)
axa.xaxis.set_major_locator(MultipleLocator(0.2))
axa.tick_params(axis='x', labelsize=FS_AXT, length=3.5, width=0.9, pad=3.78)
axa.tick_params(axis='y', length=0, pad=3.50)
axa.set_xlabel('Mean $R^2$', fontsize=FS_AXL, labelpad=4.78)
axa.xaxis.label.set_x(0.5019)   # the published label is not quite centred
axa.xaxis.grid(True, color=C_GRID_A, linewidth=0.8, zorder=0)
axa.set_axisbelow(True)
for s in axa.spines.values():
    s.set_linewidth(1.0)
    s.set_color('black')

axa.legend(handles=[Patch(facecolor='#7a7a7a', alpha=BAR_ALPHA,
                          label='All 8 properties'),
                    Patch(facecolor='#7a7a7a',
                          label='Intensive-5 (ranking criterion)')],
           loc='lower center', bbox_to_anchor=(0.5, 1.0025), ncol=2,
           frameon=False, fontsize=FS_ALEG, handlelength=1.2, handleheight=0.9,
           columnspacing=2.25, handletextpad=0.86)
for _t in axa.get_legend().get_texts():
    _t.set_multialignment('center')

# ------------------------------------------------------------------ panel (b)
axb = fig.add_axes(AX_B)
n, m = len(PROPS), len(SERIES)
BW, gap = 0.130, 0.019
x = np.arange(n, dtype=float)
off0 = -(m - 1) / 2 * (BW + gap)
for si, (name, col, vals) in enumerate(SERIES):
    axb.bar(x + off0 + si * (BW + gap), vals, width=BW, color=col,
            edgecolor='none', label=name, zorder=3)

axb.axvline(2.5, color=C_SEP, lw=0.7, ls=(0, (3.5, 3)), zorder=2)

axb.set_xticks(x)
axb.set_xticklabels(PROPS, fontproperties=FP_BXT)
axb.set_xlim(-0.60, n - 0.40)
axb.set_ylim(0, 1.03)
axb.yaxis.set_major_locator(MultipleLocator(0.2))
axb.set_ylabel('$R^2$', fontsize=FS_BYL, labelpad=6.44)
axb.tick_params(axis='y', labelsize=FS_BYT, length=3.0, width=0.8, pad=4.0)
axb.tick_params(axis='x', length=0, pad=3.9)
axb.yaxis.grid(True, color=C_GRID_B, linewidth=0.7, zorder=0)
axb.set_axisbelow(True)
for s in axb.spines.values():
    s.set_linewidth(1.0)
    s.set_color('black')

axb.legend(handles=[Patch(facecolor=c, label=l) for l, c, _ in SERIES],
           loc='lower center', bbox_to_anchor=(0.5, 0.9955), ncol=5,
           frameon=False, fontsize=FS_BLEG, handlelength=1.09,
           handleheight=0.95, columnspacing=1.167, handletextpad=0.49)
for _t in axb.get_legend().get_texts():
    _t.set_multialignment('center')

# -------------------------------------------------------------------- panel tags
for (tx, ty), tag in [(TAG_A, '(a)'), (TAG_B, '(b)')]:
    fig.text(tx, ty, tag, ha='left', va='bottom', fontsize=FS_TAG,
             fontweight='bold')

fig.savefig(OUT, dpi=300, facecolor='white')
print('[written]', OUT)

for name, m8, m5, _ in ROWS:
    print('  %-32s mean8 %.3f  mean5 %.3f' % (name.replace('\n', ' '), m8, m5))
