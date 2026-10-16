# -*- coding: utf-8 -*-
"""Fig 2 重画 (SMILES 主体):
(a) SMILES GAT 部署 checkpoint parity (13,547 测试, 90,317 域) — smiles_test_preds.npy
(b) SMILES GAT 集成 CV 汇总 (9 成员, 14,272 测试, 95,152 域 R3 协议) — r3_smiles_ensemble.npz
输出: fig2a_smiles_parity.png / fig2b_smiles_cv.png"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']

# ================= (a) SMILES parity =================
ep = np.load('d:/ml/chem_ml5/ensemble_predictions.npz')
y_true = ep['y_true']                      # (13547, 8) — 与论文 3D 集成同标签口径
preds = np.load('d:/ml/chem_ml5/smiles_test_preds.npy')  # (13547, 8)

fig, axes = plt.subplots(4, 2, figsize=(7, 8))
for i, ax in enumerate(axes.ravel()):
    yt = y_true[:, i]
    yp = preds[:, i]
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    ax.scatter(yt, yp, s=2, alpha=0.25, color='steelblue', rasterized=True)
    lo, hi = float(yt.min()), float(yt.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_title(f'{display[i]}\nR² = {r2:.3f}   MAE = {mae:.3g} {UNITS[i]}', fontsize=11)
    ax.tick_params(labelsize=8.5)
plt.suptitle('SMILES GAT (deployment checkpoint): Predicted vs DFT (N = 13,547)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig2a_smiles_parity.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig2a saved. 逐性质:', {t: round(r2_score(y_true[:, i], preds[:, i]), 4) for i, t in enumerate(targets)})
print('mean R² =', round(np.mean([r2_score(y_true[:, i], preds[:, i]) for i in range(8)]), 4))

# ================= (b) SMILES 集成 CV 汇总 =================
r3 = np.load('d:/ml/chem_ml5/r3_smiles_ensemble.npz')
per_model = r3['per_model']   # (9, 14272, 8)
ens_mean = r3['mean']
ens_std = r3['std']
y3 = r3['y_true']

fig, axes = plt.subplots(2, 2, figsize=(8, 6.2))

# 1. per-property 测试 R² (9 成员 mean±std)
ax = axes[0, 0]
x = np.arange(len(targets))
means, stds = [], []
for i in range(8):
    r2s = [r2_score(y3[:, i], per_model[m, :, i]) for m in range(9)]
    means.append(np.mean(r2s))
    stds.append(np.std(r2s))
means = np.array(means); stds = np.array(stds)
ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', edgecolor='white')
ax.axhline(y=float(np.mean(means)), color='darkorange', linestyle='--', linewidth=1.5,
           label=f'Overall Mean R² = {np.mean(means):.4f}')
ax.set_xticks(x); ax.set_xticklabels(display, fontsize=10.5)
ax.set_ylabel('R²'); ax.set_title('3-fold × 3-seed CV: R² per Property')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# 2. Ensemble vs Single
ax = axes[0, 1]
x2 = np.arange(len(targets))
w = 0.35
ens_r2s = np.array([r2_score(y3[:, i], ens_mean[:, i]) for i in range(8)])
ax.bar(x2 - w / 2, ens_r2s, w, label='Ensemble (mean prediction)', color='darkorange')
ax.bar(x2 + w / 2, means, w, label='Single Model Avg', color='steelblue')
ax.set_xticks(x2); ax.set_xticklabels(display, fontsize=10.5)
ax.set_ylabel('R²'); ax.set_title('Ensemble vs Single-Model R²')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# 3. Model-to-model variation (val mean R², 9 模型)
ax = axes[1, 0]
res = json.load(open('d:/ml/chem_ml5/r3_smiles_results.json', encoding='utf-8'))
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
    abs_err = np.abs(ens_mean[:, i] - y3[:, i])
    ax.scatter(ens_std[:, i], abs_err, alpha=0.3, s=3, label=prop)
ax.set_xlabel('Ensemble Std (Uncertainty)'); ax.set_ylabel('Absolute Error')
ax.set_title('Uncertainty vs Error (all properties)')
ax.legend(ncol=2, fontsize=8.5, markerscale=3); ax.grid(True, alpha=0.3)

plt.suptitle('SMILES GAT Ensemble + Cross-Validation Results (3-fold × 3-seed, N = 14,272)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig2b_smiles_cv.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig2b saved.')
print(f'Ensemble R²: {ens_r2s.mean():.4f}   Single R²: {means.mean():.4f} ± {stds.mean():.4f}')
print(f'CV val Mean R²: {np.mean(val_data):.4f} ± {np.std(val_data):.4f}')
