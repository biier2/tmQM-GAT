# -*- coding: utf-8 -*-
"""Figure 3(a) —— 只保留 Table 3 的 7 个模型（2026-10-02 老师意见：
   删除 2D GCN、GIN、3D GCN 三根条，使 Figure 3(a) 与 Table 3 完全对应；
   被删的三个模型的数据改放 SI 新表 Table S7，本图不再需要它们的数据文件）。

   与上一版 fig3a_intensive5.py 的唯一区别：rows 从 10 行减为 7 行。其余全部保持原样
   （画幅 9.2 x 5.0 in、dpi 300、配色语义、图例、坐标轴、字体），因此输出仍为
   2760 x 1500 px，正文里该图的 wp:extent 不需要改动，直接换图即可。

   数值来源：同目录 r1_final_tables.json（七模型的 mean8 / mean5）。
   运行：
       python fig3a_7models.py
   输出：
       fig3a_7models.png   （2760 x 1500 px，用于替换正文 word/media/image5.png）
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

# ---- Table 3 的七个模型，顺序与原图一致（无图基线 → 3D 基线 → 本文 GAT）----
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
print(f'saved -> {out}   {w}x{h} px   比例 {w/h:.4f}  '
      f'(6.000 in 宽 -> {6.0*h/w:.3f} in 高)')
print()
print(f'{"method":26s}{"mean8":>9s}{"mean5":>9s}   名次(5)')
o5 = sorted(rows, key=lambda r: -r[2])
rank = {r[0]: i + 1 for i, r in enumerate(o5)}
for name, m8, m5, _ in rows:
    print(f'{name:26s}{m8:9.4f}{m5:9.4f}   {rank[name]}')
