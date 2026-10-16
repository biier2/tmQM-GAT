# -*- coding: utf-8 -*-
"""Fig2b 拆分为两张图:
Fig2b_cols.png (1800x840, 6.0x2.8in): 两个柱状图 1x2 横排 (左 per-property, 右 ensemble vs single)
Fig2b_box.png  (1320x900, 4.4x3.0in): 箱线图单独一张 (仿参考图: 白底黑边/橙中位线/橙散点/红均值虚线/图例右上角)
x 标签 5.5pt 45°, 柱状图图例轴下方 -0.22
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
OUT = r'D:/ml/paper/draft/round4/figure'
targets_disp = ['gap','homo','lumo','dipole','M charge','E_elec','E_disp','polariz.']

zA = np.load(r'D:/ml/chem_ml5/test_preds_A_8d6d.npz', allow_pickle=False)
yt = zA['y_true']; ensA = zA['ens_mean']; predsA = zA['preds']

per_prop_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for m in range(9)]) for i in range(8)])
per_prop_std = np.array([np.std([r2_score(yt[:, i], predsA[m, :, i]) for m in range(9)]) for i in range(8)])
ens_r2 = np.array([r2_score(yt[:, i], ensA[:, i]) for i in range(8)])
val_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for i in range(8)]) for m in range(9)])
x = np.arange(8)

# ================= 柱状图 x2: 1x2 横排 =================
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
plt.subplots_adjust(wspace=0.40, left=0.09, right=0.99, top=0.88, bottom=0.28)

ax = axes[0]
ax.bar(x, per_prop_mean, yerr=per_prop_std, capsize=3, color='steelblue', edgecolor='white')
ax.axhline(np.mean(per_prop_mean), color='darkorange', ls='--', lw=1.2,
           label=f'Overall mean R$^2$={np.mean(per_prop_mean):.4f}')
ax.set_xticks(x); ax.set_xticklabels(targets_disp, fontsize=5.5, rotation=45, ha='right')
ax.set_ylabel('Member R$^2$', fontsize=8.5)
ax.set_title('Per-property member-model R$^2$ (3-fold × 3-seed, n=9)', fontsize=9)
ax.set_ylim(0, None)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7.5, frameon=False)
ax.tick_params(labelsize=7)

ax = axes[1]
w = 0.35
ax.bar(x - w/2, ens_r2, w, label='Ensemble mean', color='darkorange')
ax.bar(x + w/2, per_prop_mean, w, label='Single model avg', color='steelblue')
ax.set_xticks(x); ax.set_xticklabels(targets_disp, fontsize=5.5, rotation=45, ha='right')
ax.set_ylabel('R$^2$', fontsize=8.5)
ax.set_title('Ensemble vs single-model R$^2$', fontsize=9)
ax.set_ylim(0, None)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.5, frameon=False)
ax.tick_params(labelsize=7)

fig.suptitle('SMILES GAT (variant A) ensemble CV summary, N=13,556', fontsize=10, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig2b_cols.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig2b_cols ✓ 1800x840')

# ================= 箱线图单张 (仿 example_box.png: 1.4:1 横向, y 0.870-0.902 不压缩) =================
fig, ax = plt.subplots(figsize=(4.4, 3.15))
plt.subplots_adjust(left=0.15, right=0.96, top=0.86, bottom=0.14)
ax.boxplot([val_mean], widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='white', edgecolor='black'),
           medianprops=dict(color='darkorange', lw=1.5),
           whiskerprops=dict(color='black'), capprops=dict(color='black'))
ax.scatter(np.ones(9) + np.random.RandomState(0).uniform(-0.15, 0.15, 9), val_mean,
           color='darkorange', s=30, zorder=5)
ax.axhline(np.mean(val_mean), color='red', ls='--', lw=1.2, label=f'Mean={np.mean(val_mean):.4f}')
ax.set_xticklabels(['9 Models'], fontsize=8); ax.set_ylabel('Validation Mean R$^2$', fontsize=9)
ax.set_title('Model-to-Model Variation', fontsize=10)
ax.legend(loc='upper right', fontsize=8, frameon=False)
ax.tick_params(labelsize=8)
ax.set_ylim(0.868, 0.904)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
fig.savefig(os.path.join(OUT, 'Fig2b_box.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig2b_box ✓ 1320x945')
