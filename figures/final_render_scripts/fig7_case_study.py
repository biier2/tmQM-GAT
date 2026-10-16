# -*- coding: utf-8 -*-
"""Fig 7 (服务器版): Case Study — (a) HL-Gap, (b) Dipole
3D GAT / SMILES GAT / DFT 三系列分组柱状图, 10 个 DFT 验证催化剂
DFT 数据 (TPSSh/def2-SVP, 优化+单点同层级, 无 D3BJ — 日志核验 2026-09-02) 硬编码,
与正文 Table 4 一致 (勿改);
模型预测读 data/scorer_3d_results.csv
输出: fig7_hd.png"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 布局参数 (notebook 大画布风格) ============
DPI = 600
FIGW, FIGH = 18.0, 11.0       # 2行×1列
FS_VAL_3D, FS_VAL_SM, FS_XTICK = 12, 12, 15
FS_YLABEL, FS_LEGEND, FS_MAEBOX, FS_PANEL, FS_SUPTITLE = 17, 15, 15, 18, 20
# ======================================================

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

df = pd.read_csv('data/scorer_3d_results.csv').set_index('Name')

fig, axes = plt.subplots(2, 1, figsize=(FIGW, FIGH), dpi=100)
C_3D, C_SM, C_DFT = '#1f77b4', '#9e9e9e', '#c1272d'

for ax, key, ymin, ymax, unit in [(axes[0], 'gap', 0.0, 6.0, 'eV'),
                                  (axes[1], 'dipole', -2.2, 14.0, 'D')]:
    col_3d = 'gap_eV' if key == 'gap' else 'dipole_D'
    col_sm = 'gap_SMILES_eV' if key == 'gap' else 'dipole_SMILES_D'
    v3d = [df.loc[n, col_3d] for n in ORDER]
    vsm = [df.loc[n, col_sm] for n in ORDER]
    if key == 'dipole':   # 意见7 (2026-09-02): 偶极矩统一 max(0,·) 非负截断, 负柱变 0.00
        v3d = [max(0.0, v) for v in v3d]
        vsm = [max(0.0, v) for v in vsm]
    vdft = [DFT[n][key] for n in ORDER]
    x = range(len(ORDER))
    w = 0.26
    ax.bar([i - w for i in x], v3d, w, color=C_3D, label='3D GAT', zorder=3)
    ax.bar([i for i in x], vsm, w, color=C_SM, label='SMILES GAT', zorder=3)
    ax.bar([i + w for i in x], vdft, w, color=C_DFT, label='DFT (TPSSh/def2-SVP)', zorder=3)
    def _lbl(x, v, **kw):
        # 竖排数值: 正值放柱顶上方, 负值放柱底下方 (避免穿过柱体与 x 轴标签重叠)
        if v >= 0:
            ax.text(x, v + ymax * 0.025, f'{v:.2f}', ha='center', va='bottom', rotation=90, **kw)
        else:
            ax.text(x, v - ymax * 0.025, f'{v:.2f}', ha='center', va='top', rotation=90, **kw)
    for i, v in enumerate(v3d):
        _lbl(i - w, v, fontsize=FS_VAL_3D, fontweight='bold', color=C_3D)
    for i, v in enumerate(vsm):
        _lbl(i, v, fontsize=FS_VAL_SM, color='#555555')
    for i, v in enumerate(vdft):
        _lbl(i + w, v, fontsize=FS_VAL_3D, color=C_DFT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER, rotation=25, ha='right', fontsize=FS_XTICK)
    ax.set_ylabel(f'{"HL-Gap" if key == "gap" else "Dipole moment"} ({unit})', fontsize=FS_YLABEL)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.grid(True, color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    # MAE box
    if key == 'gap':
        mae_3d = np.mean([abs(a - b) for a, b in zip(v3d, vdft)])
        mae_sm = np.mean([abs(a - b) for a, b in zip(vsm, vdft)])
        txt = f'3D GAT MAE = {mae_3d:.2f} eV ({mae_3d/27.2114:.3f} Ha)\nSMILES GAT MAE = {mae_sm:.2f} eV ({mae_sm/27.2114:.3f} Ha)'
    else:
        mae_3d = np.mean([abs(a - b) for a, b in zip(v3d, vdft)])
        mae_sm = np.mean([abs(a - b) for a, b in zip(vsm, vdft)])
        txt = f'3D GAT MAE = {mae_3d:.2f} D\nSMILES GAT MAE = {mae_sm:.2f} D'
    ax.text(0.015, 0.97, txt, transform=ax.transAxes, ha='left', va='top', fontsize=FS_MAEBOX,
            bbox=dict(boxstyle='round', fc='white', ec='black', lw=0.8))
    ax.text(-0.005, 1.02, '(a) HL-Gap' if key == 'gap' else '(b) Dipole moment',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_PANEL, fontweight='bold')

# 图例移出数据区: 放图顶部右侧 (2026-09-02 用户反馈图例挡住柱顶数值)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.985), ncol=3,
           fontsize=FS_LEGEND, frameon=True, edgecolor='black', framealpha=1)
fig.suptitle('Case Study: Predicted vs DFT', fontsize=FS_SUPTITLE, fontweight='bold',
             x=0.02, y=0.985, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig('fig7_hd.png', dpi=DPI)
print('fig7 saved.')
print('[verify] DFT 数值硬编码与 Table 4 一致 (10 催化剂), 勿改')
