# -*- coding: utf-8 -*-
"""Fig2c 修复版 — 口径双线 (回应审稿致命问题 2):
成员级均值 0.8868 (红虚线, 箱图统计口径) + ensemble 0.9050 (绿实线, 与 Table 1 Mean R² 行一致)
尺寸 1080x900 与用户上传版一致, 风格沿用用户版 (标题/轴标签小写)
数据: chem_ml5/test_preds_A_8d6d.npz (与新评测 ensemble 0.905 同源, 无污染)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

zA = np.load(r'D:/ml/chem_ml5/test_preds_A_8d6d.npz', allow_pickle=False)
yt = zA['y_true']; ensA = zA['ens_mean']; predsA = zA['preds']
val_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for i in range(8)]) for m in range(9)])
ens_mean = np.mean([r2_score(yt[:, i], ensA[:, i]) for i in range(8)])
print(f'member mean = {val_mean.mean():.4f} (line)')
print(f'ensemble mean = {ens_mean:.4f} (line)')

fig, ax = plt.subplots(figsize=(1080/300, 900/300))
plt.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.13)
ax.boxplot([val_mean], widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='white', edgecolor='black'),
           medianprops=dict(color='darkorange', lw=1.5),
           whiskerprops=dict(color='black'), capprops=dict(color='black'))
ax.scatter(np.ones(9) + np.random.RandomState(0).uniform(-0.15, 0.15, 9), val_mean,
           color='darkorange', s=30, zorder=5)
ax.axhline(val_mean.mean(), color='red', ls='--', lw=1.2, label=f'Member mean={val_mean.mean():.4f}')
ax.axhline(ens_mean, color='green', ls='-', lw=1.4, label='Ensemble=0.9050')  # 与 Table 1 Mean R² 行的 0.905 一致
ax.set_xticklabels(['9 models'], fontsize=8); ax.set_ylabel('Mean R$^2$', fontsize=9)
ax.set_title('Model-to-model variation', fontsize=10)
ax.legend(loc='upper right', fontsize=7.5, frameon=False)
ax.tick_params(labelsize=8)
ax.set_ylim(0.872, 0.910)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
fig.savefig(r'D:/ml/paper/draft/round5/Fig2c_v2.png', dpi=300, facecolor='white')
plt.close(fig)
print('Fig2c_v2 ✓ 1080x900')
