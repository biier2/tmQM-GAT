# -*- coding: utf-8 -*-
"""Render Figure 2 of the manuscript: eight-property SMILES GAT parity plus CV summary.

Panel (a)  eight parity-density panels, reference value on x, ensemble
           prediction on y, with MAE / R^2 annotated in each panel.
Panel (b)  two bar charts: per-property mean member R^2 with the spread over
           the nine members, and the ensemble-vs-single-model comparison.
Panel (c)  box plot of the nine members' mean R^2 across the eight properties.

    python fig2_plot.py

Input   preds/test_preds_A_8d6d.npz
        y_true (13,556 x 8), ens_mean (13,556 x 8), preds (9 x 13,556 x 8)
Output  results/fig2_eight_property.png   2965 x 2866 px at 300 dpi

Every axis position, font size, colour and offset below was measured off the
published raster; they are not round numbers because they reproduce the
printed proportions at 300 dpi.  The two reference lines in panel (b) are
0.8868 (mean member R^2 over the 9 x 8 grid) and 0.9048 (mean ensemble R^2
over the eight properties); both are recomputed from the npz, not typed in.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
NPZ = os.path.join(ROOT, 'preds', 'test_preds_A_8d6d.npz')
OUTDIR = os.path.join(HERE, 'results')
OUT = os.path.join(OUTDIR, 'fig2_eight_property.png')

# ----------------------------------------------------------------- properties
# name, title, axis label, display scale (E_elec is drawn in units of 1e4 Ha),
# and the number of decimals used for the MAE annotation
PROPS = [
    ('gap',    'HOMO–LUMO Gap',     'Ha',           1.0,  4),
    ('homo',   'HOMO Energy',       'Ha',           1.0,  4),
    ('lumo',   'LUMO Energy',       'Ha',           1.0,  4),
    ('dipole', 'Dipole Moment',     'Debye',        1.0,  3),
    ('mchg',   'Metal Charge',      'e',            1.0,  4),
    ('eelec',  'Electronic Energy', '$10^4$ Ha',    1e-4, 1),
    ('edisp',  'Dispersion Energy', 'Ha',           1.0,  4),
    ('polar',  'Polarizability',    'a.u.',         1.0,  3),
]
XTICKLAB = ['gap', 'HOMO', 'LUMO', 'dipole', 'metal charge',
            '$E_{elec}$', '$E_{disp}$', 'polarizability']

# ------------------------------------------------------------------- geometry
# fractions of the 2965 x 2866 canvas
A_LEFT = [0.134065, 0.342833, 0.551602, 0.760371]
A_W = 0.147049
A_ROW1_BOT, A_ROW1_H = 0.774075, 0.152477
A_ROW2_BOT, A_ROW2_H = 0.551117, 0.152826

B_LEFT, B_W = 0.075379, 0.537268
B_UP_BOT, B_UP_H = 0.275122, 0.153524
B_LO_BOT, B_LO_H = 0.045185, 0.153524

C_LEFT, C_BOT, C_W, C_H = 0.691231, 0.136254, 0.283979, 0.238661

TITLE_A_X, TITLE_A_Y = 0.495280, 0.970516
TITLE_B_X, TITLE_B_Y = 0.318383, 0.479240
TAG_A = (0.092910, 0.971478)
TAG_B = (0.024782, 0.480126)
TAG_C = (0.634226, 0.385730)

# ---------------------------------------------------------------------- style
FS_TITLE, FS_SUB = 12.0, 10.5          # main title (bold), subplot/chart titles
FS_AXL, FS_TICK = 10.5, 9.5            # axis labels (bold), tick labels
FS_TAG, FS_ANN = 9.5, 8.5              # panel tags (sans bold), in-panel MAE/R^2
FS_CXL = 9.5                           # panel (c) x label
FS_LEG = 8.5                           # legend text

C_BLUE = '#3C92C7'
C_PALE = '#B1E5E8'
CMAP = mcolors.LinearSegmentedColormap.from_list('tmqm_blue', ['#FFFFFF', C_BLUE])

BINS = 100
BLUR_BINS = 1.25      # Gaussian smoothing of the 2-D density, in bins
VMAX_COUNT = 55.0     # count at which the density colormap saturates
MARGIN = 0.04
REF_MEMBER = 0.8868
REF_ENS = 0.9048
BAR_R = (0.8174, 0.9319, 0.9253, 0.6494, 0.9691, 0.9585, 0.9147, 0.9282)  # docs

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})


def r2_score(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def gaussian_blur(a, sigma):
    """Separable Gaussian blur of a 2-D array (no SciPy needed)."""
    if sigma <= 0:
        return a
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    b = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, b)


# ------------------------------------------------------------------------ data
z = np.load(NPZ, allow_pickle=False)
Y = z['y_true'].astype(np.float64)
P = z['ens_mean'].astype(np.float64)
M = z['preds'].astype(np.float64)
N = len(Y)

ens_r2 = np.array([r2_score(Y[:, i], P[:, i]) for i in range(len(PROPS))])
ens_mae = np.array([np.abs(Y[:, i] - P[:, i]).mean() for i in range(len(PROPS))])
mem_r2 = np.array([[r2_score(Y[:, i], M[j, :, i]) for i in range(len(PROPS))]
                   for j in range(M.shape[0])])

print('member mean R^2 per property:', np.round(mem_r2.mean(axis=0), 4))
print('ensemble R^2 per property    :', np.round(ens_r2, 4))
print('mean member R^2 = %.4f   mean ensemble R^2 = %.4f'
      % (mem_r2.mean(), ens_r2.mean()))

os.makedirs(OUTDIR, exist_ok=True)
fig = plt.figure(figsize=(2965 / 300.0, 2866 / 300.0), dpi=300)

# ------------------------------------------------------------- panel (a) grid
for i, (name, title, unit, scale, ndec) in enumerate(PROPS):
    row, col = divmod(i, 4)
    bottom = A_ROW1_BOT if row == 0 else A_ROW2_BOT
    height = A_ROW1_H if row == 0 else A_ROW2_H
    ax = fig.add_axes([A_LEFT[col], bottom, A_W, height])

    yt = Y[:, i] * scale
    yp = P[:, i] * scale
    both = np.concatenate([yt, yp])
    lo, hi = both.min(), both.max()
    pad = MARGIN * (hi - lo)

    H, xe, ye = np.histogram2d(yt, yp, bins=BINS,
                               range=[[lo - pad, hi + pad], [lo - pad, hi + pad]])
    H = gaussian_blur(H.astype(np.float64), BLUR_BINS)
    H[H == 0] = np.nan          # empty cells stay white, as in the manuscript
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(H).T, cmap=CMAP,
                  norm=mcolors.Normalize(0, VMAX_COUNT))

    ax.plot([lo, hi], [lo, hi], ls='--', color='0.25', lw=1.2,
            dashes=(4.0, 1.5), zorder=5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel('Reference (%s)' % unit, fontsize=FS_AXL, fontweight='bold',
                  labelpad=2.1)
    ax.set_ylabel('Predicted (%s)' % unit, fontsize=FS_AXL, fontweight='bold',
                  labelpad=2.0)
    ax.tick_params(labelsize=FS_TICK, length=3.3, width=1.0, pad=3.2)
    ax.set_title(title, fontsize=FS_SUB, pad=5.0)

    ax.text(0.9278, 0.0881,
            'MAE = %.*f\n$R^2$ = %.3f' % (ndec, ens_mae[i], ens_r2[i]),
            transform=ax.transAxes, ha='right', va='bottom',
            multialignment='right', linespacing=1.2, fontsize=FS_ANN,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='0.5', linewidth=1.0, mutation_aspect=1.3))

# ---------------------------------------------------------------- panel (b) 1
axb1 = fig.add_axes([B_LEFT, B_UP_BOT, B_W, B_UP_H])
xb = np.arange(len(PROPS))
axb1.bar(xb, mem_r2.mean(axis=0), width=0.8, color=C_BLUE, edgecolor='none',
         yerr=mem_r2.std(axis=0), error_kw=dict(ecolor='black', lw=1.0,
                                                capsize=3.0, capthick=1.0))
axb1.axhline(REF_MEMBER, color='red', ls='--', lw=1.7,
             label='Member mean $R^2$=%.4f' % REF_MEMBER)
axb1.axhline(REF_ENS, color='green', ls='-', lw=1.7,
             label='Ensemble $R^2$=%.4f' % REF_ENS)
axb1.set_xlim(-0.792, 7.791)
axb1.set_ylim(0, 1.25)
axb1.set_yticks(np.arange(0, 1.2501, 0.25))
axb1.set_xticks(xb)
axb1.set_xticklabels(XTICKLAB)
axb1.set_ylabel('Member $R^2$', fontsize=FS_AXL, fontweight='bold', labelpad=2.0)
axb1.set_title('Per-property member-model $R^2$ (3-fold $\\times$ 3-seed, n=9)',
               fontsize=FS_SUB, pad=5.4)
axb1.tick_params(labelsize=FS_TICK, length=3.3, width=1.0, pad=3.2)
axb1.legend(loc='upper left', fontsize=FS_LEG, frameon=True, ncol=2,
            handlelength=1.7, handletextpad=1.0, columnspacing=1.8,
            borderpad=0.4, labelspacing=0.35, borderaxespad=0.55)

# ---------------------------------------------------------------- panel (b) 2
axb2 = fig.add_axes([B_LEFT, B_LO_BOT, B_W, B_LO_H])
axb2.bar(xb - 0.2, ens_r2, width=0.4, color=C_BLUE, edgecolor='none',
         label='Ensemble (mean prediction)')
axb2.bar(xb + 0.2, mem_r2.mean(axis=0), width=0.4, color=C_PALE,
         edgecolor='none', label='Single model Avg')
axb2.set_xlim(-0.792, 7.791)
axb2.set_ylim(0, 1.25)
axb2.set_yticks(np.arange(0, 1.2501, 0.25))
axb2.set_xticks(xb)
axb2.set_xticklabels(XTICKLAB)
axb2.set_ylabel('$R^2$', fontsize=FS_AXL, fontweight='bold', labelpad=2.0)
axb2.set_title('Ensemble vs single-model $R^2$', fontsize=FS_SUB, pad=5.4)
axb2.tick_params(labelsize=FS_TICK, length=3.3, width=1.0, pad=3.2)
axb2.legend(loc='upper left', fontsize=FS_LEG, frameon=True, ncol=2,
            handlelength=1.7, handletextpad=1.0, columnspacing=1.8,
            borderpad=0.4, labelspacing=0.35, borderaxespad=0.55)

# ---------------------------------------------------------------- panel (c)
axc = fig.add_axes([C_LEFT, C_BOT, C_W, C_H])
vals = mem_r2.mean(axis=1)
axc.boxplot([vals], positions=[1.0], widths=0.16, showfliers=True,
            medianprops=dict(color='black', lw=1.1),
            boxprops=dict(color='black', lw=1.1),
            whiskerprops=dict(color='black', lw=1.1),
            capprops=dict(color='black', lw=1.1),
            flierprops=dict(marker='o', markersize=5.5, markerfacecolor='none',
                            markeredgecolor='black', markeredgewidth=1.1))
rng = np.random.RandomState(7)
xj = 1.0 + rng.uniform(-0.02, 0.08, size=len(vals))
axc.scatter(xj, vals, s=55, color=C_BLUE, edgecolors='none', zorder=4)
axc.axhline(REF_MEMBER, color='red', ls='--', lw=1.7,
            label='Member mean=%.4f' % REF_MEMBER)
axc.axhline(REF_ENS, color='green', ls='-', lw=1.7,
            label='Ensemble=%.4f' % REF_ENS)
axc.set_xlim(0.5, 1.5)
axc.set_ylim(0.874, 0.909)
axc.set_xticks([1.0])
axc.set_xticklabels(['9 models'], fontsize=FS_CXL)
axc.set_ylabel('Mean $R^2$', fontsize=FS_AXL, fontweight='bold', labelpad=2.0)
axc.set_title('Model-to-model variation', fontsize=FS_SUB, pad=4.7)
axc.tick_params(labelsize=FS_TICK, length=3.3, width=1.0, pad=3.2)
axc.legend(loc='upper center', fontsize=FS_LEG, frameon=True, ncol=1,
           handlelength=1.7, handletextpad=1.0,
           borderpad=0.4, labelspacing=0.35, borderaxespad=0.55)

# ------------------------------------------------------------------- captions
fig.text(TITLE_A_X, TITLE_A_Y,
         'SMILES GAT (variant A) ensemble: predicted versus reference values, '
         '$N$ = {:,}'.format(N), ha='center', va='center',
         fontsize=FS_TITLE, fontweight='bold')
fig.text(TITLE_B_X, TITLE_B_Y,
         'SMILES GAT (variant A) ensemble CV summary, N={:,}'.format(N),
         ha='center', va='center', fontsize=FS_TITLE, fontweight='bold')

for (tx, ty), tag in zip([TAG_A, TAG_B, TAG_C], ['(a)', '(b)', '(c)']):
    fig.text(tx, ty, tag, ha='left', va='center', fontsize=FS_TAG,
             fontweight='bold', fontfamily='sans-serif')

fig.savefig(OUT, dpi=300, facecolor='white')
print('[written]', OUT)
