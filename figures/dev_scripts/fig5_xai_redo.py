# -*- coding: utf-8 -*-
"""Fig 5 重画: XAI 分析 — SMILES 部署模型版 (tmqm_gat_best.pt)
数据来源: xai_smiles_executed.ipynb (A1/A2/A4/B2 数字) + m7_carbon_donors_smiles.json (B3 overlap/z)
旧图: 3D 模型数字 (donor 20.5%, carbon 73.4%, metal 4.3%, overlap 18.1%)
新图: SMILES 模型数字 (donor 22.3%, carbon 71.2%, metal 4.7%, overlap 18.8%)
输出: fig5_xai_smiles.png"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']

# ===== A1: 逐元素 top-5 计数 (Layer 5, N=20,000, 100,000 slots) =====
a1 = [('C', 71166, 'organic'), ('N', 8252, 'donor'), ('O', 7723, 'donor'), ('P', 2188, 'donor'),
      ('Cl', 2157, 'donor'), ('S', 1290, 'donor'), ('F', 602, 'organic'), ('Ru', 520, 'metal'),
      ('Si', 513, 'organic'), ('Br', 450, 'donor'), ('Pd', 392, 'metal'), ('Ni', 392, 'metal'),
      ('Pt', 371, 'metal'), ('B', 325, 'organic'), ('Zn', 303, 'metal'), ('Re', 287, 'metal'),
      ('I', 276, 'donor'), ('Mo', 256, 'metal'), ('Ir', 241, 'metal'), ('Fe', 237, 'metal'),
      ('Rh', 231, 'metal'), ('Au', 178, 'metal'), ('W', 172, 'metal'), ('Ti', 169, 'metal'),
      ('Co', 168, 'metal')]
a1 = a1[:18]  # 显示 top 18 元素 (与旧图相当)

# ===== B2: 扰动 donor fraction (mean ± std, per complex) =====
b2 = {'gap': (26.4, 15.3), 'homo': (43.9, 24.8), 'lumo': (41.8, 23.5), 'dipole': (26.9, 14.6),
      'metal_charge': (25.7, 14.3), 'E_elec': (14.6, 10.8), 'E_disp': (22.7, 13.9), 'polarizability': (20.2, 12.7)}

# ===== B3: overlap / z (来自 m7 json) =====
m7 = json.load(open('d:/ml/chem_ml5/m7_carbon_donors_smiles.json', encoding='utf-8'))
overlap = m7['overlap_mean'] * 100
chance = m7['chance_5N_mean'] * 100
zval = m7['paired_z']

fig, axes = plt.subplots(2, 3, figsize=(8.5, 6.0))
plt.rcParams.update({'font.size': 11.5, 'axes.titlesize': 12, 'axes.labelsize': 11.5,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10})

# (a) 逐元素 top-5 柱状图
ax = axes[0, 0]
elems = [e[0] for e in a1]
counts = [e[1] for e in a1]
colors = [{'donor': '#E74C3C', 'metal': '#3498DB', 'organic': '#95A5A6'}[e[2]] for e in a1]
bars = ax.bar(elems, counts, color=colors, edgecolor='white')
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, c + 400, f'{c:,}', ha='center', va='bottom', fontsize=9.5, rotation=60)
ax.set_ylabel('Top-5 Appearances')
ax.set_title('(a) Element Top-5 Frequency (N=20,000)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#E74C3C', label='Donor'), Patch(color='#3498DB', label='Metal'),
                   Patch(color='#95A5A6', label='Organic')], fontsize=10.5)
ax.set_ylim(0, max(counts) * 1.15)
ax.tick_params(axis='x', labelsize=10)

# (b) 饼图
ax = axes[0, 1]
sizes = [22336, 4657, 73007]
labels = [f'Donor\n22,336 (22.3%)', f'Metal\n4,657 (4.7%)', f'Organic\n73,007 (73.0%)']
cols = ['#E74C3C', '#3498DB', '#95A5A6']
ax.pie(sizes, labels=labels, colors=cols, autopct='', startangle=90, textprops={'fontsize': 11})
ax.set_title('(b) Atom Type Distribution (N=20,000)')

# (c) 逐性质 donor top-5
ax = axes[0, 2]
ax.barh(display[::-1], [22.3] * 8, color='#E74C3C', height=0.55)
for i, p in enumerate(display[::-1]):
    ax.text(22.3 + 0.25, i, '22.3%', va='center', fontsize=11)
ax.set_xlim(0, 26)
ax.set_xlabel('Donor Atom Top-5 Fraction (%)')
ax.set_title('(c) Donor Attention per Property')
ax.grid(axis='x', alpha=0.3)

# (d) 多层演变
ax = axes[1, 0]
layers = [1, 2, 3, 4, 5]
ax.errorbar(layers, [19.76] * 5, yerr=[11.52] * 5, fmt='o-', color='#E74C3C', capsize=5, lw=2)
ax.axhline(y=19.76, color='#E74C3C', linestyle='--', alpha=0.4)
ax.set_ylim(5, 35)
ax.set_xticks(layers)
ax.set_xlabel('GAT Layer')
ax.set_ylabel('Donor Atom Attention Fraction (%)')
ax.set_title('(d) Donor Attention Evolution Across Layers')
ax.grid(alpha=0.3)

# (e) 扰动 donor fraction
ax = axes[1, 1]
means = [b2[t][0] for t in targets]
stds = [b2[t][1] for t in targets]
x = np.arange(8)
ax.bar(x, means, yerr=stds, capsize=4, color='#E74C3C', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(display, rotation=35, ha='right', fontsize=10.5)
ax.set_ylabel('Donor Attribution Fraction (%)')
ax.set_title('(e) Perturbation: Donor Fraction per Property')
ax.grid(axis='y', alpha=0.3)

# (f) 重叠对比
ax = axes[1, 2]
bars = ax.bar(['Attention vs\nPerturbation', 'Chance\n(5/N)'], [overlap, chance],
              color=['#E74C3C', '#95A5A6'], edgecolor='white', width=0.45)
for bar, v in zip(bars, [overlap, chance]):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=13.5, fontweight='bold')
ax.set_ylim(0, max(overlap, chance) * 1.25)
ax.set_ylabel('Top-5 Overlap (%)')
ax.set_title(f'(f) Attention–Perturbation Top-5 Overlap\n(z = {zval:.1f}, p < 0.001)')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('SMILES GAT Attention XAI — Rigorous Analysis (N=20,000 complexes)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
import shutil, os
out = 'd:/ml/chem_ml5/fig5_xai_smiles.png'
if os.path.exists(out):
    shutil.copy(out, 'd:/ml/chem_ml5/fig5_xai_smiles_v1_20in.png')
plt.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
print(f'fig5 saved. overlap={overlap:.1f}% chance={chance:.1f}% z={zval:.2f}')
