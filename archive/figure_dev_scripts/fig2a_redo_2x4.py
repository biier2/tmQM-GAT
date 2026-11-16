# -*- coding: utf-8 -*-
"""Fig 2a re-render (2×4 layout, replicating the v1_wide version embedded in the docx):
SMILES GAT deployment checkpoint parity (13,547 test, 90,317 domain)
Comment 7: dipole/gap predictions uniformly clipped to max(0,·) non-negative → dipole panel R²=0.819, MAE=1.11 D
Output: fig2a_parity_2x4_nonneg.png
Note: the docx-embedded figure is a post-processed version with no local hash match; this script replicates it following the docx layout (2×4, same title format)"""
import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'metal charge', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'e', 'Ha', 'Ha', 'a.u.']
# Comment 7 non-negative clipping: dipole moment (|μ|) and HL-Gap are physically non-negative (the rest are signed quantities, not clipped)
CLIP_COLS = {0, 3}

ep = np.load('d:/ml/chem_ml5/ensemble_predictions.npz')
y_true = ep['y_true']                      # (13547, 8) — same label convention as the paper's 3D ensemble
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
