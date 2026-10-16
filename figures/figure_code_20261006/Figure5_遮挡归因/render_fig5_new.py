# -*- coding: utf-8 -*-
"""Fig5: 扰动(遮挡)归因 — 最终版渲染脚本 (新 A 变体 9 模型集成, 2,000 络合物)
数字来自 chem_ml5/perturbation_A_results.json (reval_perturbation_A.py 产出), 此处硬编码:
(a) donor 原子 {N,O,S,P,Cl,Br,I} 归因占比; (b) donor + 金属键合碳
输出: Fig5_new.png (1255x864 px, 300 dpi)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

donor_only = {'HOMO': 47.6, 'LUMO': 44.1, 'gap': 31.3, 'dipole': 26.1, 'metal charge': 30.5,
              'E_elec': 14.0, 'E_disp': 23.5, 'polarizability': 21.5}
with_mc = {'gap': 39.1, 'HOMO': 62.0, 'LUMO': 58.2, 'dipole': 32.3, 'metal charge': 37.4,
           'E_elec': 18.9, 'E_disp': 33.8, 'polarizability': 30.7}

fig, axes = plt.subplots(1, 2, figsize=(1255/300, 864/300))
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.86, bottom=0.16)
for ax, data, title in [(axes[0], donor_only, '(a) Donor-atom attribution fraction'),
                        (axes[1], with_mc, '(b) With metal-bonded carbons included')]:
    names = list(data.keys()); vals = list(data.values())
    ax.bar(names, vals, color='#5DADE2', edgecolor='white')
    for k, v in zip(names, vals):
        ax.text(k, v + 1.5, f'{v:.1f}%', ha='center', fontsize=6.5)
    ax.set_ylabel('Attribution fraction (%)', fontsize=7)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(axis='x', rotation=30, labelsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
fig.suptitle('Perturbation (occlusion) attribution, SMILES GAT (variant A) ensemble, 2,000 complexes',
             fontsize=8.5, fontweight='bold', y=0.985)
fig.savefig(r'D:/ml/paper/draft/round5/Fig5_new.png', dpi=300, facecolor='white')
print('Fig5_new.png saved')
