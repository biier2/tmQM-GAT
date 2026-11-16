# -*- coding: utf-8 -*-
"""Fig 6 (server version): cross-functional validation tmQM (TPSSh-D3BJ) → tmQMg (PBE0-D3BJ/def2-TZVP)
Data: data/cross_dft_predictions.npz (73,438 × 7, SMILES GAT deployment checkpoint)
      + data/cross_dft_metrics.json (Pearson r / Spearman ρ / MAE)
Panel annotations r + ρ + MAE are all read from the json, consistent with the main text P88 convention (Pearson mean 0.943, Spearman mean 0.934)
Output: fig6_crossdft.png"""
import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # fallback for the Windows GBK console; harmless on Linux
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ layout parameters (2 rows × 4 columns, tuned for a 6in Word page width) ============
DPI = 600
FIGW, FIGH = 12.0, 8.0       # 2 rows × 4 columns horizontal, 8 cells (7 panels + 1 empty), each panel 3×4in
# User request (2026-09-02): 4 per row + enlarged. The 6in page text width is a hard cap:
# 12in-wide canvas → Word scales by 0.5, so a 16pt title is effectively 8pt (the old 7in canvas gave effectively 9pt, almost the same);
# panels are 1.5×2in in Word, 63% larger than the old 1.23in panel height
FS_TITLE, FS_TICK, FS_AXLABEL, FS_SUPTITLE = 16, 13, 14, 20
# ======================================================

data = np.load('data/cross_dft_predictions.npz')
y_true = data['y_true']        # (73438, 7)
preds = data['smiles_preds']   # (73438, 7)
metrics = json.load(open('data/cross_dft_metrics.json', encoding='utf-8'))

targets = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
display = ['gap', 'homo', 'lumo', 'dipole', 'E_elec', 'E_disp', 'polarizability']
UNITS = ['Ha', 'Ha', 'Ha', 'D', 'Ha', 'Ha', 'a.u.']
met = {m['Property']: m for m in metrics}

fig, axes = plt.subplots(2, 4, figsize=(FIGW, FIGH))
for i, ax in enumerate(axes.ravel()):
    if i >= 7:
        ax.axis('off')
        continue
    yt = y_true[:, i]
    yp = preds[:, i]
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    m = met[targets[i]]
    ax.scatter(yt, yp, s=6, alpha=0.25, color='steelblue', rasterized=True)
    lo, hi = float(yt.min()), float(yt.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_title(f"{display[i]}\nr = {m['Pearson_r']:.3f}   ρ = {m['Spearman_rho']:.3f}\n"
                 f"MAE = {m['MAE']:.3g} {UNITS[i]}", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel(f'DFT {display[i]} (PBE0)', fontsize=FS_AXLABEL)
    ax.set_ylabel(f'Predicted {display[i]}', fontsize=FS_AXLABEL)

plt.suptitle('Cross-DFT Validation: tmQM (TPSSh-D3BJ) → tmQMg (PBE0-D3BJ/def2-TZVP), N = 73,438',
             fontsize=FS_SUPTITLE, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('fig6_crossdft.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('fig6 saved.')

# Summary verification (cross-checked against main text P88)
prs = [m['Pearson_r'] for m in metrics]
rhos = [m['Spearman_rho'] for m in metrics]
print(f'[verify] Pearson mean = {np.mean(prs):.4f} (main text 0.943)')
print(f'[verify] Spearman mean = {np.mean(rhos):.4f} (main text 0.934)')
print(f'[verify] gap Pearson = {met["gap"]["Pearson_r"]:.4f} (main text 0.884 HL-Gap)')
