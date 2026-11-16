# -*- coding: utf-8 -*-
"""R1 companion: Figure 3(a) changed to a grouped bar chart -- each method shows side by side the "mean over all 8 properties" and the "intensive-5 mean".
   Purpose: to draw the very fact that "the extensive properties dilute the headline metric", rather than only giving the corrected ranking.
   The canvas stays 9.2 x 5.0 in (no cropping), exactly matching the 2760x1500 px / 1.840 of the existing image5.png,
   so wp:extent needs no change.
   Colour semantics consistent with the old figure: grey = no-graph baseline, blue = 2D message passing, red family = 3D message passing, orange/green = the GAT of this work.
   All values are taken from round6/r1_final_tables.json (produced by build_r1_tables.py).
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

# baseline_results_new.json (server_baselines.py, 13,556 test set, target order as below)
BASE = json.load(open(r'D:/ml/chem_ml5/baseline_results_new.json'))
PROPS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
BASE_M5 = {k: float(np.mean(v['r2'][:5])) for k, v in BASE.items()}
BASE_M8 = {k: v['mean_r2'] for k, v in BASE.items()}

M = J['models']
rows = [
    ('RF (ECFP4)',              M['RF']['mean8'],              M['RF']['mean5'],              '#90A4AE'),
    ('XGBoost (ECFP4)',         M['XGBoost']['mean8'],         M['XGBoost']['mean5'],         '#90A4AE'),
    ('Composition only',        M['Composition']['mean8'],     M['Composition']['mean5'],     '#B0BEC5'),
    ('2D GCN',                  BASE_M8['gcn'],                BASE_M5['gcn'],                '#5DADE2'),
    ('GIN',                     BASE_M8['gin'],                BASE_M5['gin'],                '#5DADE2'),
    ('3D GCN',                  BASE_M8['gcn3d'],              BASE_M5['gcn3d'],              '#E74C3C'),
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
out = os.path.join(HERE, 'fig3a_intensive5.png')
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
