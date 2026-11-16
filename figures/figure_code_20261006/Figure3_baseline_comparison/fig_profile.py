# -*- coding: utf-8 -*-
"""Per-property profile v2 (SchNet / DimeNet++ / SMILES GAT(A) / 3D GAT(D) + composition-baseline reference).
   Colours continue the semantics of Figure 3: grey = no-graph baseline, red family = 3D message passing, orange/green = the GAT of this work.
   The composition baseline must be drawn -- otherwise readers would think that E_elec 0.999 / polarizability 0.999 is due to the models.
   v2 relative to v1: (i) 1.03 headroom left above the tallest bars; (ii) narrower bars, wider within-group gaps;
                      (iii) X labels unified to a single line; (iv) canvas changed to the actual print width of a two-column layout, so the font sizes are true values.
"""
import sys
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

props = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'Metal charge',
         'E_elec', 'E_disp', 'Polarizability']
series = [
    ('Composition only', '#B0BEC5',
     [0.310, 0.104, 0.112, 0.097, 0.659, 1.000, 0.977, 0.998]),
    ('SchNet',           '#C0392B',
     [0.845, 0.838, 0.825, 0.809, 0.982, 0.977, 0.981, 0.981]),
    ('DimeNet++',        '#8E2420',
     [0.850, 0.861, 0.856, 0.829, 0.991, 0.999, 0.998, 0.999]),
    ('SMILES GAT (variant A)', '#F39C12',
     [0.846, 0.947, 0.943, 0.707, 0.976, 0.965, 0.921, 0.934]),
    ('3D GAT (variant D)',     '#27AE60',
     [0.864, 0.949, 0.948, 0.739, 0.979, 0.966, 0.921, 0.934]),
]

n, m = len(props), len(series)
BW, gap = 0.130, 0.019           # single bar width / within-group gap
x = np.arange(n) * 1.0

fig, ax = plt.subplots(figsize=(7.2, 3.7), dpi=300)

off0 = -(m - 1) / 2 * (BW + gap)
for si, (name, col, vals) in enumerate(series):
    ax.bar(x + off0 + si * (BW + gap), vals, width=BW, color=col,
           edgecolor='none', label=name, zorder=3)

# between-group separator: 3 orbital-energy descriptors | 5 response/extensive quantities
ax.axvline(2.5, color='#B0B0B0', lw=0.7, ls=(0, (3.5, 3)), zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(props, fontsize=9.0)
ax.set_xlim(-0.60, n - 0.40)
ax.set_ylim(0, 1.03)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_ylabel('$R^2$', fontsize=11, labelpad=5)
ax.tick_params(axis='y', labelsize=9.5, length=3, width=0.8)
ax.tick_params(axis='x', length=0, pad=3)

ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(0.9); s.set_color('black')

ax.legend(handles=[Patch(facecolor=c, label=l) for l, c, _ in series],
          loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5,
          frameon=False, fontsize=8.8, handlelength=1.0,
          handleheight=0.95, columnspacing=1.1, handletextpad=0.45)

fig.tight_layout()
fig.savefig(r'D:/ml/paper/draft/round5/fig_profile.png',
            dpi=300, facecolor='white', bbox_inches='tight')
print('saved -> fig_profile.png')
