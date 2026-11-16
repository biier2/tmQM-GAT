# -*- coding: utf-8 -*-
"""Fig 6 redraw: cross-functional validation (fixes the old figure's wB97xd spelling -> PBE0-D3BJ/def2-TZVP, round4 comment 4)
Data: cross_dft_predictions.npz (73,438 × 7, SMILES GAT deployment checkpoint)
Old figure problems: wrong wB97xd spelling in the title; panel annotations not matching the main text convention; wrong level label ωB97M-V/def2-TZVPP
New figure: panel annotations Pearson r + ρ (main text convention), title PBE0-D3BJ/def2-TZVP (the official tmQMg target level)
Output: fig6_crossdft.png"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('d:/ml/chem_ml5/cross_dft_predictions.npz')
y_true = data['y_true']        # (73438, 7)
preds = data['smiles_preds']   # (73438, 7)
metrics = json.load(open('d:/ml/chem_ml5/cross_dft_metrics.json', encoding='utf-8'))

targets = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'Ha', 'Ha', 'a.u.']
met = {m['Property']: m for m in metrics}

fig, axes = plt.subplots(2, 4, figsize=(12, 8))
for i, ax in enumerate(axes.ravel()):
    if i >= 7:
        ax.axis('off')
        continue
    yt = y_true[:, i]
    yp = preds[:, i]
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    m = met[targets[i]]
    ax.scatter(yt, yp, s=3, alpha=0.25, color='steelblue', rasterized=True)
    lo, hi = float(yt.min()), float(yt.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_title(f"{display[i]}\nr = {m['Pearson_r']:.3f}   ρ = {m['Spearman_rho']:.3f}\n"
                 f"MAE = {m['MAE']:.3g} {UNITS[i]}", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_xlabel(f'DFT {display[i]} (PBE0)', fontsize=14)
    ax.set_ylabel(f'Predicted {display[i]}', fontsize=14)

plt.suptitle('Cross-DFT Validation: tmQM (TPSSh-D3BJ) → tmQMg (PBE0-D3BJ/def2-TZVP), N = 73,438',
             fontsize=20, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig6_crossdft.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig6 saved.')

# Summary verification (cross-checked against main text P88)
prs = [m['Pearson_r'] for m in metrics]
rhos = [m['Spearman_rho'] for m in metrics]
print(f'Pearson mean = {np.mean(prs):.4f} (main text 0.943)')
print(f'Spearman mean = {np.mean(rhos):.4f} (main text 0.934)')
print(f'gap Pearson = {met["gap"]["Pearson_r"]:.4f} (main text 0.884 HL-Gap)')
