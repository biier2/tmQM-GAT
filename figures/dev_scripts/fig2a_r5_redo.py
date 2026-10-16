# -*- coding: utf-8 -*-
"""Fig 2a 重渲 (2×4 版式, 复刻 docx 内嵌 v1_wide 版) — R5 同协议版:
SMILES GAT 9 成员集成 parity (13,547 测试, vA 域 90,317, 3-fold × 3-seed)
意见7: dipole/gap 预测统一 max(0,·) 非负截断
输出: fig2a_parity_2x4_r5.png
注: 与 fig2a_redo_2x4.py 同版式, 仅数据源换 r5_smiles_va_ensemble.npz (同协议重训)"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']
CLIP_COLS = {0, 3}   # 意见7 非负截断

r5 = np.load('d:/ml/chem_ml5/r5_smiles_va_ensemble.npz')
y_true = r5['y_true']                       # (13547, 8)
preds = r5['mean'].copy()                   # (13547, 8) — 9 成员集成均值
for c in CLIP_COLS:
    preds[:, c] = np.maximum(preds[:, c], 0)

fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.12))
for i, ax in enumerate(axes.ravel()):
    yt = y_true[:, i]
    yp = preds[:, i]
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    ax.scatter(yt, yp, s=4, alpha=0.3, color='steelblue', rasterized=True)
    lo, hi = float(yt.min()), float(yt.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_title(f'{display[i]}\nR² = {r2:.3f}   MAE = {mae:.3g} {UNITS[i]}', fontsize=12)
    ax.tick_params(labelsize=10)
plt.suptitle('SMILES GAT (9-model ensemble): Predicted vs DFT (N = 13,547)',
             fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig2a_parity_2x4_r5.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig2a saved.')
for i, t in enumerate(targets):
    print(f'  {t:14s} R²={r2_score(y_true[:, i], preds[:, i]):.4f}  MAE={mean_absolute_error(y_true[:, i], preds[:, i]):.4g}')
print(f'  mean R² = {np.mean([r2_score(y_true[:, i], preds[:, i]) for i in range(8)]):.4f}')
