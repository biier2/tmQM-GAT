# -*- coding: utf-8 -*-
"""Fig 6 (服务器版): 跨泛函验证 tmQM (TPSSh-D3BJ) → tmQMg (PBE0-D3BJ/def2-TZVP)
数据: data/cross_dft_predictions.npz (73,438 × 7, SMILES GAT 部署 checkpoint)
      + data/cross_dft_metrics.json (Pearson r / Spearman ρ / MAE)
面板标注 r + ρ + MAE 均读 json, 与正文 P88 口径一致 (Pearson mean 0.943, Spearman mean 0.934)
输出: fig6_crossdft.png"""
import json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 布局参数 (2行×4列, 按 Word 6in 页宽调优) ============
DPI = 600
FIGW, FIGH = 12.0, 8.0       # 2行×4列 横排 8 格 (7 面板 + 1 空), 每面板 3×4in
# 用户要求 (2026-09-02): 每行4张 + 放大。页面文本宽 6in 是硬上限:
# 画布 12in 宽 → Word 缩放 0.5, 16pt 标题等效 8pt (旧 7in 画布等效 9pt, 几乎持平);
# 面板在 Word 中 1.5×2in, 比旧版面板高 1.23in 放大 63%
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

# 汇总验证 (与正文 P88 对照)
prs = [m['Pearson_r'] for m in metrics]
rhos = [m['Spearman_rho'] for m in metrics]
print(f'[verify] Pearson mean = {np.mean(prs):.4f} (正文 0.943)')
print(f'[verify] Spearman mean = {np.mean(rhos):.4f} (正文 0.934)')
print(f'[verify] gap Pearson = {met["gap"]["Pearson_r"]:.4f} (正文 0.884 HL-Gap)')
