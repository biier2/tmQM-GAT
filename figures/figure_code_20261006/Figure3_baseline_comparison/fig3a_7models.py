# -*- coding: utf-8 -*-
"""Figure 3(a) -- keeping only the 7 models of Table 3 (advisor comment of 2026-10-02:
   delete the three bars for 2D GCN, GIN, 3D GCN, so that Figure 3(a) corresponds exactly to Table 3;
   the data of the three deleted models go instead into the new SI table Table S7, and this figure no longer needs their data files).

   The only difference from the previous version fig3a_intensive5.py: rows reduced from 10 to 7. Everything else is unchanged
   (canvas 9.2 x 5.0 in, dpi 300, colour semantics, legend, axes, fonts), so the output is still
   2760 x 1500 px and the wp:extent of this figure in the main text needs no change; just swap the figure directly.

   Numerical source: r1_final_tables.json in the same directory (mean8 / mean5 of the seven models).
   Run:
       python fig3a_7models.py
   Output:
       fig3a_7models.png   (2760 x 1500 px, used to replace word/media/image5.png in the main text)
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, 'r1_final_tables.json'), encoding='utf-8'))
M = J['models']

# ---- the seven models of Table 3, in the same order as the original figure (no-graph baselines → 3D baselines → the GAT of this work) ----
rows = [
    ('RF (ECFP4)',              M['RF']['mean8'],              M['RF']['mean5'],              '#90A4AE'),
    ('XGBoost (ECFP4)',         M['XGBoost']['mean8'],         M['XGBoost']['mean5'],         '#90A4AE'),
    ('Composition only',        M['Composition']['mean8'],     M['Composition']['mean5'],     '#B0BEC5'),
    ('SchNet',                  M['SchNet']['mean8'],          M['SchNet']['mean5'],          '#C0392B'),
    ('DimeNet++',               M['DimeNet++']['mean8'],       M['DimeNet++']['mean5'],       '#8E2420'),
    ('SMILES GAT (variant A)',  M['SMILES GAT']['mean8'],      M['SMILES GAT']['mean5'],      '#F39C12'),
    ('3D GAT (variant D)',      M['3D GAT']['mean8'],          M['3D GAT']['mean5'],          '#27AE60'),
]

fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=300)
y = np.arange(len(rows))[::-1]
H = 0.34

for yi, (name, m8, m5, col) in zip(y, rows):
    ax.barh(yi + H / 2 + 0.02, m8, height=H, color=col, alpha=0.32,
            edgecolor='none', zorder=3)
    ax.barh(yi - H / 2 - 0.02, m5, height=H, color=col,
            edgecolor='none', zorder=3)
    ax.text(m8 + 0.012, yi + H / 2 + 0.02, f'{m8:.3f}', va='center', ha='left',
            fontsize=9.2, color='#8a8a8a', zorder=4)
    ax.text(m5 + 0.012, yi - H / 2 - 0.02, f'{m5:.3f}', va='center', ha='left',
            fontsize=9.8, color='#1a1a1a', fontweight='bold', zorder=4)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
ax.set_xlim(0, 1.10)
ax.set_ylim(-0.72, len(rows) - 0.28)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.tick_params(axis='x', labelsize=10.5, length=3.5, width=0.9)
ax.tick_params(axis='y', length=0)
ax.set_xlabel('Mean $R^2$', fontsize=12, labelpad=6)
ax.xaxis.grid(True, color='#D9D9D9', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(1.0); s.set_color('black')

ax.legend(handles=[Patch(facecolor='#7a7a7a', alpha=0.32, label='All 8 properties'),
                   Patch(facecolor='#7a7a7a', label='Intensive-5 (ranking criterion)')],
          loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
          fontsize=10.2, handlelength=1.2, handleheight=0.9, columnspacing=2.2)

fig.tight_layout()
out = os.path.join(HERE, 'fig3a_7models.png')
fig.savefig(out, dpi=300, facecolor='white')
from PIL import Image
w, h = Image.open(out).size
print(f'saved -> {out}   {w}x{h} px   ratio {w/h:.4f}  '
      f'(6.000 in wide -> {6.0*h/w:.3f} in high)')
print()
print(f'{"method":26s}{"mean8":>9s}{"mean5":>9s}   rank(5)')
o5 = sorted(rows, key=lambda r: -r[2])
rank = {r[0]: i + 1 for i, r in enumerate(o5)}
for name, m8, m5, _ in rows:
    print(f'{name:26s}{m8:9.4f}{m5:9.4f}   {rank[name]}')
