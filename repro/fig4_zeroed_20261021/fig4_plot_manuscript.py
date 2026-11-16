# -*- coding: utf-8 -*-
"""Render Figure 4 in the layout used in the revised manuscript.

Panels (a)-(f) are drawn from the two archived result files:
    results/fig4_zeroed_results.npz          11d/23d zero-padded model
    results/fig4_smiles_results_subset.npz   SMILES-only checkpoint, same subset

Layout differences from fig4_plot.py (the design rendering):
  * panel (e) compliance is evaluated at the <0.001 Ha tolerance for both
    models, which is where the 99.7% / 100% bars come from;
  * the published RGCN bar is drawn at its literature value (100% at a
    0.01 Ha tolerance) and labelled "(literature value, <0.01 Ha)";
  * the canvas is 6,506 x 4,173 px at 300 dpi, the pixel size of the image
    embedded in the manuscript, so the drawing needs no rescaling in the docx.

Everything positional below -- the six axes rectangles, every font size, the
title pad, the tick and spine widths, the legend anchor and frame width, the
marker sizes and edge widths, the reference-line widths, and the two panel-(e)
axis limits -- was measured off the published raster.  They are not round
numbers because they are what reproduces the printed proportions at 300 dpi.
A fresh render matches 96.79% of the published raster's pixels exactly and
differs only in marker antialiasing; see README.md for the per-panel breakdown.

Output: results/fig4_manuscript.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'results')
OUT = os.path.join(SRC, 'fig4_manuscript.png')

DPI = 300
CANVAS = (6506, 4173)
FIGW, FIGH = CANVAS[0] / DPI, CANVAS[1] / DPI

# ------------------------------------------------------------------- geometry
# [left, bottom, width, height] in figure fractions, from the spine centres of
# the published raster.
_W, _H = 1552 / 6506, 1180 / 4173
_B1, _B2 = 1 - 1829.5 / 4173, 1 - 3625.5 / 4173
AX_A = [314.5 / 6506, _B1, _W, _H]
AX_B = [2619.5 / 6506, _B1, 1551 / 6506, _H]
AX_C = [4923.5 / 6506, _B1, _W, _H]
AX_D = [314.5 / 6506, _B2, _W, _H]
AX_E = [2619.5 / 6506, _B2, 1551 / 6506, _H]
AX_F = [4923.5 / 6506, _B2, _W, _H]

# ---------------------------------------------------------------------- style
FS = 17.0             # every tick label, axis label, title, legend entry
FS_SUPTITLE = 21.15   # bold
TITLE_PAD = 2.64      # points between the axes top and the title block
LEGEND_ANCHOR = (0.52190, 0.00623)
LW_SPINE = 1.92
TICK_LEN = 5.0
TICK_W = 1.92

MINUS = '−'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.unicode_minus': False,
    'font.size': FS,
    'axes.titlesize': FS,
    'axes.labelsize': FS,
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'legend.fontsize': FS,
    'axes.linewidth': LW_SPINE,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': TICK_LEN,
    'ytick.major.size': TICK_LEN,
    'xtick.major.width': TICK_W,
    'ytick.major.width': TICK_W,
})


def _style(ax):
    for s in ax.spines.values():
        s.set_linewidth(LW_SPINE)


def main():
    xz = np.load(os.path.join(SRC, 'fig4_zeroed_results.npz'))
    sm = np.load(os.path.join(SRC, 'fig4_smiles_results_subset.npz'))
    xyz = {k: xz[k] for k in xz.files}
    smi = {k: sm[k] for k in sm.files}

    r2_xyz = np.corrcoef(xyz['gap_derived'], xyz['gap_pred'])[0, 1] ** 2
    r2_sm = np.corrcoef(smi['gap_derived'], smi['gap_pred'])[0, 1] ** 2
    mean_xyz = xyz['deviation'].mean() * 1000
    mean_sm = smi['deviation'].mean() * 1000
    viol = (~(xyz['homo_pred'] < xyz['lumo_pred'])).mean() * 100
    print(f'R2 {r2_xyz:.6f} / {r2_sm:.6f}; mean {mean_xyz:.4f} / {mean_sm:.4f} mHa; '
          f'violation {viol:.4f}%', flush=True)

    fig = plt.figure(figsize=(FIGW, FIGH), dpi=DPI, facecolor='white')

    # (a)
    ax = fig.add_axes(AX_A)
    ax.scatter(xyz['gap_derived'], xyz['gap_pred'], alpha=0.3, s=6,
               c='darkorange', linewidths=0.5)
    lims = [min(xyz['gap_pred'].min(), xyz['gap_derived'].min()),
            max(xyz['gap_pred'].max(), xyz['gap_derived'].max())]
    ax.plot(lims, lims, 'r--', lw=2.0)
    ax.set_title(f'(a) 11d/23d-layout GAT: gap vs\n(LUMO{MINUS}HOMO)\n'
                 f'R² = {r2_xyz:.4f}', pad=TITLE_PAD)
    ax.set_xlabel(f'LUMO$_{{pred}}$ {MINUS} HOMO$_{{pred}}$ (Ha)')
    ax.set_ylabel(f'gap$_{{pred}}$ (Ha)')
    _style(ax)

    # (b)
    ax = fig.add_axes(AX_B)
    ax.hist(xyz['deviation'] * 1000, bins=80, color='darkorange',
            edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(x=mean_xyz, color='red', linestyle='--', lw=2.0,
               label=f'Mean = {mean_xyz:.3f} mHa')
    ax.set_title(f'(b) 11d/23d-layout GAT: |gap {MINUS}\n'
                 f'(LUMO{MINUS}HOMO)| distribution', pad=TITLE_PAD)
    ax.set_xlabel('Deviation (mHa)')
    ax.set_ylabel('Count')
    leg_b = ax.legend()
    leg_b.get_frame().set_linewidth(2.0)
    _style(ax)

    # (c)
    ax = fig.add_axes(AX_C)
    ax.scatter(smi['gap_derived'], smi['gap_pred'], alpha=0.3, s=6,
               c='steelblue', linewidths=0.5)
    lims = [min(smi['gap_pred'].min(), smi['gap_derived'].min()),
            max(smi['gap_pred'].max(), smi['gap_derived'].max())]
    ax.plot(lims, lims, 'r--', lw=2.0)
    ax.set_title(f'(c) SMILES GAT: gap vs (LUMO{MINUS}HOMO)\n'
                 f'R² = {r2_sm:.4f}', pad=TITLE_PAD)
    ax.set_xlabel(f'LUMO$_{{pred}}$ {MINUS} HOMO$_{{pred}}$ (Ha)')
    ax.set_ylabel(f'gap$_{{pred}}$ (Ha)')
    _style(ax)

    # (d)
    ax = fig.add_axes(AX_D)
    homo_ok = xyz['homo_pred'] < xyz['lumo_pred']
    colors = ['#2ecc71' if ok else '#e74c3c' for ok in homo_ok]
    ax.scatter(xyz['homo_pred'][:500], xyz['lumo_pred'][:500],
               c=colors[:500], alpha=0.5, s=9, linewidths=0.5)
    ax.plot([-0.5, 0.1], [-0.5, 0.1], 'k--', lw=1.5)
    ax.set_xlabel('HOMO$_{pred}$ (Ha)')
    ax.set_ylabel('LUMO$_{pred}$ (Ha)')
    ax.set_title('(d) 11d/23d-layout GAT: HOMO < LUMO\nordering\n'
                 f'Violation rate: {viol:.2f}% (green = OK, red =\nviolation)',
                 pad=TITLE_PAD)
    _style(ax)

    # (e)
    ax = fig.add_axes(AX_E)
    methods = ['11d/23d-layout\nGAT', 'SMILES GAT', 'Published\nRGCN']
    mean_devs = [mean_xyz, mean_sm, 0.1]
    # models: fraction of the 5,000-complex subset below 0.001 Ha;
    # literature: 100% below 0.01 Ha (tmGNN-XAI, reported at 0.01 Ha).
    comply_models = [(xyz['deviation'] < 0.001).mean() * 100,
                     (smi['deviation'] < 0.001).mean() * 100]
    comply = comply_models + [100.0]
    print(f'compliance <0.001 Ha: 11d/23d {comply_models[0]:.2f}%, '
          f'SMILES {comply_models[1]:.2f}%', flush=True)
    x = np.arange(len(methods))
    w = 0.25
    ax.bar(x - w, mean_devs, w, color=['steelblue', 'steelblue', '#9E9E9E'],
           edgecolor=['none', 'none', '#5F5F5F'], linewidth=2.0,
           hatch=['', '', '//'], label='Mean deviation (mHa)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean deviation (mHa)')
    ax.set_title('(e) Literature comparison (not directly\ncomparable)',
                 pad=TITLE_PAD)
    # headroom for the italic annotation above the tallest compliance bar
    ax.set_ylim(0, max(mean_devs) * 1.45)
    ax2 = ax.twinx()
    ax2.bar(x + w, comply, w, color=['darkorange', 'darkorange', '#9E9E9E'],
            edgecolor=['none', 'none', '#5F5F5F'], linewidth=2.0,
            hatch=['', '', '//'], label='Compliance <0.001 Ha (%)')
    ax2.set_ylabel('Compliance <0.001 Ha (%)')
    ax2.set_ylim(0, max(comply) * 1.18)
    ax2.text(1.5, 105.5, '(literature value, <0.01 Ha)', ha='center',
             va='bottom', fontsize=FS, style='italic', color='#4A4A4A')
    handles = [Patch(facecolor='steelblue', label='Mean deviation (mHa)'),
               Patch(facecolor='darkorange', label='Compliance <0.001 Ha (%)')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=LEGEND_ANCHOR, fontsize=FS, frameon=True,
               borderaxespad=0.0, borderpad=0.47)
    _style(ax)
    _style(ax2)

    # (f)
    ax = fig.add_axes(AX_F)
    ax.scatter(xyz['gap_true'], xyz['deviation'] * 1000, alpha=0.2, s=6,
               c='darkorange', edgecolors='face', linewidths=0.5)
    ax.set_xlabel('True gap (Ha)')
    ax.set_ylabel('Deviation (mHa)')
    ax.set_title('(f) 11d/23d-layout GAT: Deviation vs True\nGap',
                 pad=TITLE_PAD)
    ax.axhline(y=mean_xyz, color='red', linestyle='--', alpha=0.7, lw=2.0)
    _style(ax)

    fig.text(0.5, 1 - 29.0 / 4173, 'Internal algebraic consistency',
             ha='center', va='top', fontsize=FS_SUPTITLE, fontweight='bold')

    fig.savefig(OUT, dpi=DPI, facecolor='white')
    plt.close(fig)

    from PIL import Image
    im = Image.open(OUT)
    print(f'rendered {im.size[0]}x{im.size[1]}', flush=True)


if __name__ == '__main__':
    main()
