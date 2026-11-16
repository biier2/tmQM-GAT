# -*- coding: utf-8 -*-
"""Fig2c fixed version -- dual reference lines (response to review fatal issue 2):
member-level mean 0.8868 (red dashed line, the box-plot statistic) + ensemble 0.9050 (green solid line, consistent with the Table 1 Mean R² row)
size 1080x900 identical to the user-uploaded version; style follows the user version (lowercase titles/axis labels)
data: chem_ml5/test_preds_A_8d6d.npz (same source as the new-evaluation ensemble 0.905, no contamination)
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
ax.axhline(ens_mean, color='green', ls='-', lw=1.4, label='Ensemble=0.9050')  # consistent with the 0.905 of the Table 1 Mean R² row
ax.set_xticklabels(['9 models'], fontsize=8); ax.set_ylabel('Mean R$^2$', fontsize=9)
ax.set_title('Model-to-model variation', fontsize=10)
ax.legend(loc='upper right', fontsize=7.5, frameon=False)
ax.tick_params(labelsize=8)
ax.set_ylim(0.872, 0.910)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
fig.savefig(r'D:/ml/paper/draft/round5/Fig2c_v2.png', dpi=300, facecolor='white')
plt.close(fig)
print('Fig2c_v2 ✓ 1080x900')
