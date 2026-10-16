# -*- coding: utf-8 -*-
"""Fig 2b 重渲 — R5 同协议版 (vA 域 90,317 / 13,547 测试, 3-fold × 3-seed 9 成员):
版式 = hd_figs_server/fig2_smiles.py (b) 段 (16×12in 大画布 + X标签rotation35),
数据源换 r5_smiles_va_ensemble.npz + r5_smiles_va_results.json
标题 N = 13,547 (原 95,152 域 N = 14,272)
输出: fig2b_smiles_cv_r5.png"""
import numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']

r5 = np.load('d:/ml/chem_ml5/r5_smiles_va_ensemble.npz')
per_model = r5['per_model']   # (9, 13547, 8)
ens_mean = r5['mean']
ens_std = r5['std']
y3 = r5['y_true']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
FS_B_XTICK, FS_B_LEG, FS_B_SUPTITLE = 11, 12, 20

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
ax.set_xticks(x); ax.set_xticklabels(display, rotation=35, ha='right', fontsize=FS_B_XTICK)
ax.set_ylabel('R²'); ax.set_title('3-fold × 3-seed CV: R² per Property')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# 2. Ensemble vs Single
ax = axes[0, 1]
x2 = np.arange(len(targets))
w = 0.35
ens_r2s = np.array([r2_score(y3[:, i], ens_mean[:, i]) for i in range(8)])
ax.bar(x2 - w / 2, ens_r2s, w, label='Ensemble (mean prediction)', color='darkorange')
ax.bar(x2 + w / 2, means, w, label='Single Model Avg', color='steelblue')
ax.set_xticks(x2); ax.set_xticklabels(display, rotation=35, ha='right', fontsize=FS_B_XTICK)
ax.set_ylabel('R²'); ax.set_title('Ensemble vs Single-Model R²')
ax.legend(loc='upper left', ncol=2); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.25)

# 3. Model-to-model variation (val mean R², 9 模型)
ax = axes[1, 0]
res = json.load(open('d:/ml/chem_ml5/r5_smiles_va_results.json', encoding='utf-8'))
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
ax.legend(ncol=2, fontsize=FS_B_LEG, markerscale=3, columnspacing=3.0); ax.grid(True, alpha=0.3)

plt.suptitle('SMILES GAT Ensemble + Cross-Validation Results (3-fold × 3-seed, N = 13,547)',
             fontsize=FS_B_SUPTITLE, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig2b_smiles_cv_r5.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig2b saved.')
print(f'Ensemble R²: {ens_r2s.mean():.4f}   Single R²: {means.mean():.4f} ± {stds.mean():.4f}')
print(f'CV val Mean R²: {np.mean(val_data):.4f} ± {np.std(val_data):.4f}')
