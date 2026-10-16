# -*- coding: utf-8 -*-
"""Fig 2a 重渲 (2×4 版式, 复刻 docx 内嵌 v1_wide 版):
SMILES GAT 部署 checkpoint parity (13,547 测试, 90,317 域)
意见7: dipole/gap 预测统一 max(0,·) 非负截断 → dipole 面板 R²=0.819, MAE=1.11 D
输出: fig2a_parity_2x4_nonneg.png
注: docx 内嵌图是后处理版本, 无本地哈希匹配; 本脚本按 docx 版式 (2×4, 同标题格式) 复刻"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']
# 意见7 非负截断: 偶极矩 (|μ|) 与 HL-Gap 物理上非负 (其余为有符号物理量, 不截断)
CLIP_COLS = {0, 3}

ep = np.load('d:/ml/chem_ml5/ensemble_predictions.npz')
y_true = ep['y_true']                      # (13547, 8) — 与论文 3D 集成同标签口径
preds = np.load('d:/ml/chem_ml5/smiles_test_preds.npy').copy()  # (13547, 8)
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
plt.suptitle('SMILES GAT (deployment checkpoint): Predicted vs DFT (N = 13,547)',
             fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig2a_parity_2x4_nonneg.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig2a saved.')
for i, t in enumerate(targets):
    print(f'  {t:14s} R²={r2_score(y_true[:, i], preds[:, i]):.4f}  MAE={mean_absolute_error(y_true[:, i], preds[:, i]):.4g}')
print(f'  mean R² = {np.mean([r2_score(y_true[:, i], preds[:, i]) for i in range(8)]):.4f}')
