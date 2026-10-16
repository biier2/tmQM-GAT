# -*- coding: utf-8 -*-
"""Fig 2 (SMILES 主体) — 服务器版
(a) SMILES GAT 部署 checkpoint parity (13,547 测试, 90,317 域)
(b) SMILES GAT 集成 CV 汇总 (9 成员, 14,272 测试, 95,152 域 R3 协议)
数据: data/ensemble_predictions.npz + data/smiles_test_preds.npy (a);
      data/r3_smiles_ensemble.npz + data/r3_smiles_results.json (b)
输出: fig2a_smiles_parity.png / fig2b_smiles_cv.png"""
import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 布局参数 (notebook 大画布风格: 大 figsize + 大字号) ============
DPI = 600
FIG2A_W, FIG2A_H = 20.0, 12.0   # (a) 4行×2列 竖排 8 面板; 每面板约 5in 宽
FIG2B_W, FIG2B_H = 16.0, 12.0   # (b) 2×2 四面板
FS_A_TITLE, FS_A_TICK, FS_A_SUPTITLE = 16, 12, 20
FS_B_XTICK, FS_B_LEG, FS_B_SUPTITLE = 11, 12, 20
# ============================================================

def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def mae(y, p):
    return float(np.abs(y - p).mean())

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'axes.titlesize': 16,
                     'legend.fontsize': 13})

# ================= (a) SMILES parity =================
ep = np.load('data/ensemble_predictions.npz')
y_true = ep['y_true']                          # (13547, 8)
preds = np.load('data/smiles_test_preds.npy')  # (13547, 8)

fig, axes = plt.subplots(4, 2, figsize=(FIG2A_W, FIG2A_H))
for i, ax in enumerate(axes.ravel()):
    yt, yp = y_true[:, i], preds[:, i]
    ax.scatter(yt, yp, s=5, alpha=0.25, color='steelblue', rasterized=True)
    lo, hi = float(yt.min()), float(yt.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_title(f'{display[i]}\nR² = {r2(yt, yp):.3f}   MAE = {mae(yt, yp):.3g} {UNITS[i]}',
                 fontsize=FS_A_TITLE)
    ax.tick_params(labelsize=FS_A_TICK)
plt.suptitle('SMILES GAT (deployment checkpoint): Predicted vs DFT (N = 13,547)',
             fontsize=FS_A_SUPTITLE, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('fig2a_smiles_parity.png', dpi=DPI, bbox_inches='tight', facecolor='white')
per_prop = {t: round(r2(y_true[:, i], preds[:, i]), 4) for i, t in enumerate(targets)}
mean_r2 = float(np.mean(list(per_prop.values())))
print('fig2a saved.')
print('逐性质 R²:', per_prop)
print(f'[verify] mean R² = {mean_r2:.4f} (正文定稿: 0.932)')

# ================= (b) SMILES 集成 CV 汇总 =================
r3 = np.load('data/r3_smiles_ensemble.npz')
per_model = r3['per_model']   # (9, 14272, 8)
ens_mean, ens_std, y3 = r3['mean'], r3['std'], r3['y_true']

fig, axes = plt.subplots(2, 2, figsize=(FIG2B_W, FIG2B_H))

# 1. per-property 测试 R² (9 成员 mean±std)
ax = axes[0, 0]
x = np.arange(len(targets))
means, stds = [], []
for i in range(8):
    r2s = [r2(y3[:, i], per_model[m, :, i]) for m in range(9)]
    means.append(np.mean(r2s)); stds.append(np.std(r2s))
means, stds = np.array(means), np.array(stds)
ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', edgecolor='white')
ax.axhline(y=float(np.mean(means)), color='darkorange', linestyle='--', linewidth=1.5,
           label=f'Overall Mean R² = {np.mean(means):.4f}')
ax.set_xticks(x); ax.set_xticklabels(display, rotation=35, ha='right', fontsize=FS_B_XTICK)
ax.set_ylabel('R²'); ax.set_title('3-fold × 3-seed CV: R² per Property')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# 2. Ensemble vs Single
ax = axes[0, 1]
x2 = np.arange(len(targets)); w = 0.35
ens_r2s = np.array([r2(y3[:, i], ens_mean[:, i]) for i in range(8)])
ax.bar(x2 - w / 2, ens_r2s, w, label='Ensemble (mean prediction)', color='darkorange')
ax.bar(x2 + w / 2, means, w, label='Single Model Avg', color='steelblue')
ax.set_xticks(x2); ax.set_xticklabels(display, rotation=35, ha='right', fontsize=FS_B_XTICK)
ax.set_ylabel('R²'); ax.set_title('Ensemble vs Single-Model R²')
ax.legend(loc='upper left', ncol=2); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.25)

# 3. Model-to-model variation (val mean R², 9 模型)
ax = axes[1, 0]
res = json.load(open('data/r3_smiles_results.json', encoding='utf-8'))
val_data = [r['val_mean_r2'] for r in res['log']]
ax.boxplot([val_data])
ax.set_xticklabels(['9 Models'])
ax.scatter(np.ones(len(val_data)) + np.random.RandomState(0).uniform(-0.1, 0.1, len(val_data)),
           val_data, color='darkorange', s=30, zorder=5)
ax.axhline(y=float(np.mean(val_data)), color='red', linestyle='--',
           label=f'Mean = {np.mean(val_data):.4f}')
ax.set_ylabel('Validation Mean R²'); ax.set_title('Model-to-Model Variation')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# 4. Uncertainty vs Error
ax = axes[1, 1]
for i, prop in enumerate(display):
    ax.scatter(ens_std[:, i], np.abs(ens_mean[:, i] - y3[:, i]), alpha=0.3, s=3, label=prop)
ax.set_xlabel('Ensemble Std (Uncertainty)'); ax.set_ylabel('Absolute Error')
ax.set_title('Uncertainty vs Error (all properties)')
ax.legend(ncol=2, fontsize=FS_B_LEG, markerscale=3, columnspacing=3.0); ax.grid(True, alpha=0.3)

plt.suptitle('SMILES GAT Ensemble + Cross-Validation Results (3-fold × 3-seed, N = 14,272)',
             fontsize=FS_B_SUPTITLE, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2b_smiles_cv.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('fig2b saved.')
print(f'[verify] Ensemble R² = {ens_r2s.mean():.4f}   Single R² = {means.mean():.4f} ± {stds.mean():.4f}')
print(f'[verify] CV val Mean R² = {np.mean(val_data):.4f} ± {np.std(val_data):.4f}')
