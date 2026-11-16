# -*- coding: utf-8 -*-
"""Render Figure 6 of the manuscript: cross-protocol transfer on tmQMg.

Seven 2-D density panels, DFT on the x axis and the SMILES-only GAT
(variant A) ensemble prediction on the y axis.  Each panel is one tmQMg
property; the eighth grid slot is left empty.  The dashed line is y = x.

    python fig6_plot.py

Input   preds/crossfunct_A_preds.npz  (y_true, ens_mean; 73,438 x 7)
Output  results/fig6_cross_protocol.png   2520 x 1380 px at 300 dpi

The density is a 100 x 100 histogram with empty bins masked out, drawn in
matplotlib's "Blues" colormap on a linear scale.  Axis limits follow the
data with a 4 % margin on each side of the combined DFT/predicted range,
so x and y always share the same limits and the identity line runs corner
to corner.

Style constants below were measured off the published raster (frame
positions, ink boxes of every text element), which is why they look
arbitrary: they reproduce the printed proportions at 300 dpi.

render_new_figs.py in this directory is the earlier draft renderer
(1320 x 760, "Cross-functional validation" title); this file supersedes it
for the values printed in the paper.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
NPZ = os.path.join(ROOT, 'preds', 'crossfunct_A_preds.npz')
OUTDIR = os.path.join(HERE, 'results')
OUT = os.path.join(OUTDIR, 'fig6_cross_protocol.png')

# tmQMg property order, with the units each panel is drawn in
TITLES = ['HOMO–LUMO Gap', 'HOMO Energy', 'LUMO Energy', 'Dipole Moment',
          'Electronic Energy', 'Dispersion Energy', 'Polarizability']
TAGS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
XLBL = ['DFT (Ha)', 'DFT (Ha)', 'DFT (Ha)', 'DFT (D)',
        'DFT ($10^4$ Ha)', 'DFT (Ha)', 'DFT (a.u.)']
YLBL = ['Predicted (Ha)', 'Predicted (Ha)', 'Predicted (Ha)', 'Predicted (D)',
        'Predicted ($10^4$ Ha)', 'Predicted (Ha)', 'Predicted (a.u.)']
SCALE = [1.0, 1.0, 1.0, 1.0, 1e-4, 1.0, 1.0]   # E_elec is drawn in units of 1e4 Ha

# axes geometry as fractions of the 8.4 x 4.6 in canvas
COLS = [0.074802, 0.320437, 0.565873, 0.811508]
ROW1_BOT, ROW2_BOT = 0.558333, 0.095290
AX_W, ROW1_H, ROW2_H = 0.173016, 0.316667, 0.317391

FS_TITLE, FS_SUB, FS_AXL = 12.0, 10.5, 10.5
FS_TICK, FS_TAG, FS_R = 9.5, 11.5, 8.75

BINS = 100
MARGIN = 0.04

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 1.0,
})

z = np.load(NPZ, allow_pickle=False)
Y = z['y_true'].astype(np.float64)
P = z['ens_mean'].astype(np.float64)
assert Y.shape[1] == len(TITLES), Y.shape

os.makedirs(OUTDIR, exist_ok=True)
fig = plt.figure(figsize=(8.4, 4.6), dpi=300)

for i, title in enumerate(TITLES):
    row, col = divmod(i, 4)
    bottom = ROW1_BOT if row == 0 else ROW2_BOT
    height = ROW1_H if row == 0 else ROW2_H
    ax = fig.add_axes([COLS[col], bottom, AX_W, height])

    yt = Y[:, i] * SCALE[i]
    yp = P[:, i] * SCALE[i]
    both = np.concatenate([yt, yp])
    lo, hi = both.min(), both.max()
    pad = MARGIN * (hi - lo)

    # density; empty bins are masked so the background stays white
    H, xe, ye = np.histogram2d(yt, yp, bins=BINS,
                               range=[[lo - pad, hi + pad], [lo - pad, hi + pad]])
    ax.pcolormesh(xe, ye, np.ma.masked_equal(H, 0).T, cmap='Blues')

    ax.plot([lo, hi], [lo, hi], ls='--', color='0.25', lw=1.1, zorder=5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(XLBL[i], fontsize=FS_AXL, fontweight='bold', labelpad=2.4)
    ax.set_ylabel(YLBL[i], fontsize=FS_AXL, fontweight='bold', labelpad=1.9)
    ax.tick_params(labelsize=FS_TICK, length=3.5, width=1.0, pad=2.9)
    ax.set_title(title, fontsize=FS_SUB, pad=5.1)

    r = np.corrcoef(yt, yp)[0, 1]
    ax.text(0.9404, 0.0790, r'$r = %.3f$' % r, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=FS_R,
            bbox=dict(boxstyle='round,pad=0.333', mutation_aspect=1.19,
                      facecolor='white', edgecolor='0.5', linewidth=0.8))
    print('(%s) %-18s r = %.4f' % (TAGS[i][1], title, r))

    fig.text(COLS[col] - 0.0292, bottom + height + 0.0138, TAGS[i],
             ha='left', va='bottom', fontsize=FS_TAG, fontweight='bold')

fig.text(0.5, 0.9826,
         'Cross-protocol transfer: SMILES GAT (variant A) on tmQMg, '
         r'$N$ = ' + '{:,}'.format(len(Y)),
         ha='center', va='top', fontsize=FS_TITLE, fontweight='bold')

fig.savefig(OUT, dpi=300, facecolor='white')
print('[written]', OUT)
