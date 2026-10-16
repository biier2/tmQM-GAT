# -*- coding: utf-8 -*-
"""Fig2a/2b 重渲 v2 — 解决拥挤问题:
Fig2a: 画布 1320x940, wspace 0.5, hspace 0.75, top 0.885 / bottom 0.095 防 x 标签裁剪,
       不用科学计数法偏移(避免 1e4 悬在标题旁), 长刻度靠宽间距容纳
Fig2b: 画布 1315x1150, hspace 0.8, bottom 0.085 防底部图例裁剪,
       三个图例移到坐标轴下方, x 轴长词缩写(M charge / polariz.)防粘连
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
OUT = r'D:/ml/paper/draft/round4/figure'
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
targets_disp = ['gap','homo','lumo','dipole','M charge','E_elec','E_disp','polariz.']

zA = np.load(r'D:/ml/chem_ml5/test_preds_A_8d6d.npz', allow_pickle=False)
yt = zA['y_true']; ensA = zA['ens_mean']; predsA = zA['preds']

def scatter_panel(ax, y, p, annot, scale=1.0):
    # 大数值属性整体除以 10 的幂(量级写入 x 轴标签), 短刻度避免长标签(-20000)挤入相邻面板
    y = y / scale; p = p / scale
    ax.scatter(y, p, alpha=0.15, s=2.5, c='#E74C3C', edgecolors='none', rasterized=True)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.text(0.04, 0.96, annot, transform=ax.transAxes, fontsize=6.5, va='top',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, lw=0.4))
    ax.tick_params(labelsize=6, pad=1.5)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)

# ===== Fig 2a: SMILES (A) 集成 parity, 1320x940 =====
fig, axes = plt.subplots(2, 4, figsize=(1320/300, 940/300))
plt.subplots_adjust(wspace=0.5, hspace=0.75, left=0.06, right=0.99, top=0.875, bottom=0.095)
axes = axes.flatten()
SCALE = {'E_elec': 1e4}  # 仅 E_elec 量级过大; polarizability(10³)原始刻度标签已够短, 不缩放
SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹'
for i, prop in enumerate(targets):
    ax = axes[i]
    y = yt[:, i]; p = ensA[:, i]
    r2 = r2_score(y, p); mae = mean_absolute_error(y, p)
    scale = SCALE.get(prop, 1.0)
    scatter_panel(ax, y, p, f'R$^2$ = {r2:.3f}\nMAE = {mae:.3f}', scale=scale)
    if i >= 4:
        # Unicode 上标直接写入, 不用 mathtext(避免 E_elec 下划线被解析成下标)
        unit = f' (×10{SUP[int(round(np.log10(scale)))]})' if scale != 1.0 else ''
        ax.set_xlabel(f'DFT {prop}{unit}', fontsize=7)
    if i % 4 == 0:
        ax.set_ylabel('Predicted', fontsize=7)
    ax.set_title(prop, fontweight='bold', fontsize=8)
fig.suptitle('SMILES GAT (variant A) ensemble: predicted vs DFT, N=13,556',
             fontsize=9, fontweight='bold', y=0.995)
fig.savefig(os.path.join(OUT, 'Fig2a_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig2a ✓')

# ===== Fig 2b: SMILES (A) CV 汇总, 1315x1150 =====
per_prop_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for m in range(9)]) for i in range(8)])
per_prop_std = np.array([np.std([r2_score(yt[:, i], predsA[m, :, i]) for m in range(9)]) for i in range(8)])
ens_r2 = np.array([r2_score(yt[:, i], ensA[:, i]) for i in range(8)])
val_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for i in range(8)]) for m in range(9)])
x = np.arange(8)

fig, axes = plt.subplots(3, 1, figsize=(1315/300, 1150/300))
plt.subplots_adjust(hspace=0.8, left=0.10, right=0.97, top=0.885, bottom=0.085)

ax = axes[0]
ax.bar(x, per_prop_mean, yerr=per_prop_std, capsize=3, color='steelblue', edgecolor='white')
ax.axhline(np.mean(per_prop_mean), color='darkorange', ls='--', lw=1.2,
           label=f'Overall mean R$^2$={np.mean(per_prop_mean):.4f}')
ax.set_xticks(x); ax.set_xticklabels(targets_disp, fontsize=7)
ax.set_ylabel('Member R$^2$', fontsize=8)
ax.set_title('Per-property member-model R$^2$ (3-fold × 3-seed, n=9)', fontsize=8.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=7, frameon=False)
ax.tick_params(labelsize=6.5)

ax = axes[1]
w = 0.35
ax.bar(x - w/2, ens_r2, w, label='Ensemble (mean prediction)', color='darkorange')
ax.bar(x + w/2, per_prop_mean, w, label='Single model avg', color='steelblue')
ax.set_xticks(x); ax.set_xticklabels(targets_disp, fontsize=7)
ax.set_ylabel('R$^2$', fontsize=8)
ax.set_title('Ensemble vs single-model R$^2$', fontsize=8.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7, frameon=False)
ax.tick_params(labelsize=6.5)

ax = axes[2]
ax.boxplot([val_mean], widths=0.4)
ax.scatter(np.ones(9) + np.random.RandomState(0).uniform(-0.05, 0.05, 9), val_mean,
           color='darkorange', s=28, zorder=5)
ax.axhline(np.mean(val_mean), color='red', ls='--', lw=1.2, label=f'Mean={np.mean(val_mean):.4f}')
ax.set_xticklabels(['9 models'], fontsize=7); ax.set_ylabel('Mean R$^2$', fontsize=8)
ax.set_title('Model-to-model variation', fontsize=8.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=7, frameon=False)
ax.tick_params(labelsize=6.5)
ax.set_ylim(np.floor(val_mean.min()*1000)/1000 - 0.002, np.ceil(val_mean.max()*1000)/1000 + 0.002)

fig.suptitle('SMILES GAT (variant A) ensemble CV summary, N=13,556', fontsize=9, fontweight='bold', y=0.995)
fig.savefig(os.path.join(OUT, 'Fig2b_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig2b ✓')
