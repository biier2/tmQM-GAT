# -*- coding: utf-8 -*-
"""Fig7 — case study: 十个自研催化剂上预测值 vs DFT

版式以 Fig2a 为准(见 acs_style.py); 柱状图做法参照 example3; 配色沿用 Fig5。

相对旧版修正的问题:
  · 两个面板各挂一个图例, 二者互相重叠且右侧溢出被裁("SMILES G..." 截断)
    -> 改为全局共享一个图例, 置于总标题下方居中
  · x 轴催化剂名旋转 40°, 拥挤 -> 面板改为上下排列后每组宽度翻倍, 标签可水平放置
  · 面板编号混在标题里("(a) HL-Gap") -> 按 Fig2a 约定移到坐标区左上外侧
  · 配色由 蓝/灰/红 改为 Fig5 的浅->深蓝顺序色阶, 与 Fig5 同一视觉体系

配色说明: 与 Fig5 不同, 本图是按「系列」而非按「数值」上色 —— 每组三根柱子分别是
3D scorer (B) / SMILES GAT (A) / DFT, 因此从 SEQ_PALETTE 中取三档间隔较大的颜色
索引 1 / 5 / 7 (#B1E5E8 / #3C92C7 / #17506B, 亮度 213.8 / 126.3 / 66.0)。
注意: docx 中 Figure 7 的 caption 写的是 "blue / gray / red", 需同步改。

输出: Fig7_new.png (2400x1900 px @300dpi)
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

# 十个 DFT 验证过的催化剂; DFT[k] = (HL-Gap eV, dipole D), TPSSh/def2-SVP
ORDER = ['Zr-cat3', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'trans-PdCl2(PPh3)2', 'cis-PdCl2(PPh3)2',
         'Pt(dppe)Cl2', 'Co-cat1', 'Rh(CO)2(acac)', 'Ni(IMes)2Cl2']
NAMES = ['Zr-cat3', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'trans-Pd', 'cis-Pd',
         'Pt(dppe)', 'Co-cat1', 'Rh(acac)', 'Ni(IMes)$_2$']
DFT = {'Zr-cat3': (3.30, 5.32), 'Fe-cat1': (2.76, 9.76), 'Ni-cat3': (1.39, 5.67), 'Pd-cat2': (2.93, 2.48),
       'trans-PdCl2(PPh3)2': (3.19, 0.03), 'cis-PdCl2(PPh3)2': (3.17, 10.86), 'Pt(dppe)Cl2': (3.95, 10.29),
       'Co-cat1': (3.30, 0.00), 'Rh(CO)2(acac)': (4.03, 3.17), 'Ni(IMes)2Cl2': (3.24, 0.00)}

# 系列: 沿用旧图的组内顺序(B, A, DFT), 颜色取自 Fig5 色阶中间隔较大的三档
SERIES = [('3D scorer (B)',   S.SEQ_PALETTE[1]),   # #B1E5E8
          ('SMILES GAT (A)',  S.SEQ_PALETTE[5]),   # #3C92C7
          ('DFT',             S.SEQ_PALETTE[7])]   # #17506B

PANELS = [('a', 'HL-Gap', 'HL-Gap (eV)', 0),
          ('b', 'Dipole moment', 'Dipole (D)', 3)]

# 各系列的数值
def series_vals(mode_idx):
    b = [r[k]['d3_mean'][mode_idx] for k in ORDER]
    a = [r[k]['smiles_mean'][mode_idx] for k in ORDER]
    if mode_idx == 0:                       # gap: Ha -> eV
        b = [v * EV for v in b]; a = [v * EV for v in a]
    else:                                   # dipole: 取模长, 负值截为 0
        b = [max(0.0, v) for v in b]; a = [max(0.0, v) for v in a]
    d = [DFT[k][0 if mode_idx == 0 else 1] for k in ORDER]
    return b, a, d


fig, axes = plt.subplots(2, 1, figsize=(2400/DPI, 1900/DPI))
# top 0.855: 给总标题(0.985)和共享图例(0.945)留位
plt.subplots_adjust(hspace=0.40, left=0.085, right=0.985, top=0.855, bottom=0.075)

x = np.arange(len(ORDER))
w = 0.30                                     # 偏移 ±w 使同组三柱相接(example3 做法)

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

# 全局共享图例(替代旧版两个重复且互相重叠的图例)
handles = [Patch(facecolor=c, edgecolor='none', label=n) for n, c in SERIES]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.945),
           ncol=3, fontsize=LEG_SIZE)

fig.suptitle('Case study: predicted vs DFT for ten in-house catalysts',
             fontsize=S.SUPTITLE_SIZE, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig7_new.png'), dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig7 ✓  2400x1900  (版式以 Fig2a 为准, 配色沿用 Fig5)')
print('输出目录 →', OUT)
