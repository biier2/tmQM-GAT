# -*- coding: utf-8 -*-
"""
生成终稿.docx 的新 Figure 7:
(a) HL-Gap, (b) Dipole —— 3D GAT / SMILES GAT / DFT 三系列分组柱状图
10 个 DFT 验证催化剂 (TPSSh-D3BJ/def2-SVP), 开壳层 gap 取较小 alpha/beta
输出尺寸与旧图一致 (1320x787 @100dpi), 直接替换 docx 内嵌图片
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DFT = {
    'Zr-cat3':            {'gap': 3.297484, 'dipole': 5.3175},
    'Co-cat1':            {'gap': 3.304110, 'dipole': 0.0016},
    'Fe-cat1':            {'gap': 2.763144, 'dipole': 9.7600},
    'Ni-cat3':            {'gap': 1.393083, 'dipole': 5.6682},
    'Pd-cat2':            {'gap': 2.925023, 'dipole': 2.4821},
    'cis-PdCl2(PPh3)2':   {'gap': 3.169948, 'dipole': 10.8571},
    'trans-PdCl2(PPh3)2': {'gap': 3.193102, 'dipole': 0.0322},
    'Rh(CO)2(acac)':      {'gap': 4.027482, 'dipole': 3.1738},
    'Ni(IMes)2Cl2':       {'gap': 3.235281, 'dipole': 0.0003},   # m1 单重态 (log 复核 2026-08-24)
    'Pt(dppe)Cl2':        {'gap': 3.953475, 'dipole': 10.2881},
}
ORDER = ['Zr-cat3', 'Co-cat1', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2',
         'cis-PdCl2(PPh3)2', 'trans-PdCl2(PPh3)2', 'Rh(CO)2(acac)',
         'Ni(IMes)2Cl2', 'Pt(dppe)Cl2']

df = pd.read_csv('scorer_3d_results.csv').set_index('Name')

fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), dpi=100)
C_3D, C_SM, C_DFT = '#1f77b4', '#9e9e9e', '#c1272d'

for ax, key, ymax, unit in [(axes[0], 'gap', 6.0, 'eV'), (axes[1], 'dipole', 14.0, 'D')]:
    col_3d = 'gap_eV' if key == 'gap' else 'dipole_D'
    col_sm = 'gap_SMILES_eV' if key == 'gap' else 'dipole_SMILES_D'
    v3d = [df.loc[n, col_3d] for n in ORDER]
    vsm = [df.loc[n, col_sm] for n in ORDER]
    vdft = [DFT[n][key] for n in ORDER]
    x = range(len(ORDER))
    w = 0.26
    b1 = ax.bar([i - w for i in x], v3d, w, color=C_3D, label='3D GAT', zorder=3)
    b2 = ax.bar([i for i in x], vsm, w, color=C_SM, label='SMILES GAT', zorder=3)
    b3 = ax.bar([i + w for i in x], vdft, w, color=C_DFT, label='DFT (TPSSh-D3BJ/def2-SVP)', zorder=3)
    for i, v in enumerate(v3d):
        ax.text(i - w, v + ymax * 0.035, f'{v:.2f}', ha='center', fontsize=11.5, fontweight='bold', color=C_3D)
    for i, v in enumerate(vsm):
        ax.text(i, v + ymax * 0.035, f'{v:.2f}', ha='center', fontsize=10, color='#555555')
    for i, v in enumerate(vdft):
        ax.text(i + w, v + ymax * 0.035, f'{v:.2f}', ha='center', fontsize=11.5, color=C_DFT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER, rotation=25, ha='right', fontsize=11)
    ax.set_ylabel(f'{"HL-Gap" if key == "gap" else "Dipole moment"} ({unit})', fontsize=12.5)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='black', framealpha=1)
    # MAE box
    if key == 'gap':
        import numpy as np
        mae_3d = np.mean([abs(a - b) for a, b in zip(v3d, vdft)])
        mae_sm = np.mean([abs(a - b) for a, b in zip(vsm, vdft)])
        txt = f'3D GAT MAE = {mae_3d:.2f} eV ({mae_3d/27.2114:.3f} Ha)\nSMILES GAT MAE = {mae_sm:.2f} eV ({mae_sm/27.2114:.3f} Ha)'
    else:
        import numpy as np
        mae_3d = np.mean([abs(a - b) for a, b in zip(v3d, vdft)])
        mae_sm = np.mean([abs(a - b) for a, b in zip(vsm, vdft)])
        txt = f'3D GAT MAE = {mae_3d:.2f} D\nSMILES GAT MAE = {mae_sm:.2f} D'
    ax.text(0.985, 0.97, txt, transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', fc='white', ec='black', lw=0.8))
    ax.text(-0.005, 1.02, '(a) HL-Gap' if key == 'gap' else '(b) Dipole moment',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')

fig.suptitle('Case Study: Predicted vs DFT', fontsize=15, fontweight='bold', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig('fig7_hd.png', dpi=600)
print('saved fig7_hd.png')
