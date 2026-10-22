# -*- coding: utf-8 -*-
"""Render Figure 4 in the layout used in the revised manuscript (plan B of the
2026-10-22 figure review).

Panels (a)-(f) are drawn from the two archived result files:
    results/fig4_zeroed_results.npz          11d/23d zero-padded model
    results/fig4_smiles_results_subset.npz   SMILES-only checkpoint, same subset

Layout differences from fig4_plot.py (the design rendering):
  * panel (e) compliance is evaluated at the <0.001 Ha tolerance for both
    models, which is where the 99.7% / 100% bars come from;
  * the published RGCN bar is drawn at its literature value (100% at a
    0.01 Ha tolerance) and labelled "(literature value, <0.01 Ha)";
  * the canvas is 6506 x 4173 px at 300 dpi, the pixel size of the image
    embedded in the manuscript (display extent 5,486,400 x 3,519,020 EMU),
    so the drawing needs no rescaling in the docx.

The script reproduces every plotted quantity from the archived arrays. The
raster embedded in the manuscript (md5 64e1f4dddea078f63af75e44eeeb15a1) was
assembled from this layout by the authors; a fresh render is visually
equivalent but not guaranteed to be byte-identical.

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
TARGET = (6506, 4173)

DPI = 300
FIGW, FIGH = 22.0, 15.0
FS_BASE, FS_TITLE, FS_AXLABEL = 13, 13, 13
FS_TICK, FS_LEGEND, FS_LEGEND_E, FS_SUPTITLE = 12, 12, 11, 16

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': FS_BASE, 'axes.titlesize': FS_TITLE,
                     'axes.labelsize': FS_AXLABEL, 'xtick.labelsize': FS_TICK,
                     'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND})

MINUS = '−'


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

    fig, axes = plt.subplots(2, 3, figsize=(FIGW, FIGH))

    # (a)
    ax = axes[0, 0]
    ax.scatter(xyz['gap_derived'], xyz['gap_pred'], alpha=0.3, s=6, c='darkorange')
    lims = [min(xyz['gap_pred'].min(), xyz['gap_derived'].min()),
            max(xyz['gap_pred'].max(), xyz['gap_derived'].max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_title(f'(a) 11d/23d-layout GAT: gap vs\n(LUMO{MINUS}HOMO)\nR$^2$ = {r2_xyz:.4f}')
    ax.set_xlabel(f'LUMO_pred {MINUS} HOMO_pred (Ha)')
    ax.set_ylabel('gap_pred (Ha)')

    # (b)
    ax = axes[0, 1]
    ax.hist(xyz['deviation'] * 1000, bins=80, color='darkorange', edgecolor='white', alpha=0.8)
    ax.axvline(x=mean_xyz, color='red', linestyle='--',
               label=f'Mean = {mean_xyz:.3f} mHa')
    ax.set_title(f'(b) 11d/23d-layout GAT: |gap {MINUS}\n(LUMO{MINUS}HOMO)| distribution')
    ax.set_xlabel('Deviation (mHa)')
    ax.set_ylabel('Count')
    ax.legend()

    # (c)
    ax = axes[0, 2]
    ax.scatter(smi['gap_derived'], smi['gap_pred'], alpha=0.3, s=6, c='steelblue')
    lims = [min(smi['gap_pred'].min(), smi['gap_derived'].min()),
            max(smi['gap_pred'].max(), smi['gap_derived'].max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_title(f'(c) SMILES GAT: gap vs (LUMO{MINUS}HOMO)\nR$^2$ = {r2_sm:.4f}')
    ax.set_xlabel(f'LUMO_pred {MINUS} HOMO_pred (Ha)')
    ax.set_ylabel('gap_pred (Ha)')

    # (d)
    ax = axes[1, 0]
    homo_ok = xyz['homo_pred'] < xyz['lumo_pred']
    colors = ['#2ecc71' if ok else '#e74c3c' for ok in homo_ok]
    ax.scatter(xyz['homo_pred'][:500], xyz['lumo_pred'][:500],
               c=colors[:500], alpha=0.5, s=9)
    ax.plot([-0.5, 0.1], [-0.5, 0.1], 'k--', lw=0.5)
    ax.set_xlabel('HOMO_pred (Ha)')
    ax.set_ylabel('LUMO_pred (Ha)')
    ax.set_title('(d) 11d/23d-layout GAT: HOMO < LUMO\nordering\n'
                 f'Violation rate: {viol:.2f}% (green = OK, red =\nviolation)')

    # (e)
    ax = axes[1, 1]
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
    ax.bar(x - w, mean_devs, w, color=['steelblue', 'steelblue', 'lightgray'],
           edgecolor=['steelblue', 'steelblue', 'gray'],
           hatch=['', '', '//'], label='Mean deviation (mHa)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean deviation (mHa)')
    ax.set_title('(e) Literature comparison (not directly\ncomparable)')
    ax.set_ylim(0, max(mean_devs) * 1.12)
    ax2 = ax.twinx()
    ax2.bar(x + w, comply, w, color=['darkorange', 'darkorange', 'lightgray'],
            edgecolor=['darkorange', 'darkorange', 'gray'],
            hatch=['', '', '//'], label='Compliance <0.001 Ha (%)')
    ax2.set_ylabel('Compliance <0.001 Ha (%)')
    ax2.set_ylim(0, 105)
    ax2.text(x[2] + w, 101, '(literature value, <0.01 Ha)', ha='center', va='bottom',
             fontsize=FS_LEGEND_E, style='italic', color='gray')
    handles = [Patch(facecolor='steelblue', label='Mean deviation (mHa)'),
               Patch(facecolor='darkorange', label='Compliance <0.001 Ha (%)')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.018), fontsize=FS_LEGEND_E, frameon=True)

    # (f)
    ax = axes[1, 2]
    ax.scatter(xyz['gap_true'], xyz['deviation'] * 1000, alpha=0.2, s=6, c='darkorange')
    ax.set_xlabel('True gap (Ha)')
    ax.set_ylabel(f'|gap_pred {MINUS} (LUMO{MINUS}HOMO)| (mHa)')
    ax.set_title('(f) 11d/23d-layout GAT: Deviation vs True\nGap')
    ax.axhline(y=mean_xyz, color='red', linestyle='--', alpha=0.7)

    fig.suptitle('Physics Consistency: HL-Gap = LUMO ' + MINUS + ' HOMO Verification',
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    from PIL import Image
    im = Image.open(OUT).convert('RGB')
    print(f'rendered {im.size[0]}x{im.size[1]}', flush=True)
    if im.size != TARGET:
        scale = min(TARGET[0] / im.size[0], TARGET[1] / im.size[1])
        new = (round(im.size[0] * scale), round(im.size[1] * scale))
        im = im.resize(new, Image.LANCZOS)
        canvas = Image.new('RGB', TARGET, 'white')
        canvas.paste(im, ((TARGET[0] - new[0]) // 2, (TARGET[1] - new[1]) // 2))
        canvas.save(OUT, dpi=(DPI, DPI))
        print(f'padded to {TARGET} (content {new})', flush=True)
    else:
        im.save(OUT, dpi=(DPI, DPI))


if __name__ == '__main__':
    main()
