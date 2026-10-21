# -*- coding: utf-8 -*-
"""Fig. 4 (revised): pure plotting script, no torch/rdkit required.
Reads results/fig4_zeroed_results.npz and results/fig4_smiles_results_subset.npz,
produced by eval_fig4_zeroed.py and eval_smiles_same_subset.py.
Output: results/fig4_physics_hd.png"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 布局参数 (与提交版图 4 一致, notebook 大画布风格) ============
DPI = 600
FIGW, FIGH = 22.0, 15.0        # 2×3 六面板, 每面板约 7.3in 宽
FS_BASE, FS_TITLE, FS_AXLABEL = 13, 13, 13
FS_TICK, FS_LEGEND, FS_LEGEND_E, FS_SUPTITLE = 12, 12, 11, 16
# =============================================================================

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': FS_BASE, 'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_AXLABEL,
                     'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND})

HERE = os.path.dirname(os.path.abspath(__file__))
xz = np.load(os.path.join(HERE, 'results', 'fig4_zeroed_results.npz'))
sm = np.load(os.path.join(HERE, 'results', 'fig4_smiles_results_subset.npz'))
xyz_results = {k: xz[k] for k in xz.files}
smiles_results = {k: sm[k] for k in sm.files}
print('npz loaded')

fig, axes = plt.subplots(2, 3, figsize=(FIGW, FIGH))

ax = axes[0, 0]
ax.scatter(xyz_results['gap_derived'], xyz_results['gap_pred'], alpha=0.3, s=6, c='darkorange')
lims = [min(xyz_results['gap_pred'].min(), xyz_results['gap_derived'].min()),
        max(xyz_results['gap_pred'].max(), xyz_results['gap_derived'].max())]
ax.plot(lims, lims, 'r--', lw=1)
r2_xyz_gap = np.corrcoef(xyz_results['gap_derived'], xyz_results['gap_pred'])[0, 1] ** 2
ax.set_title(f'(a) 11d/23d GAT: gap vs (LUMO−HOMO)\nR² = {r2_xyz_gap:.4f}')
ax.set_xlabel('LUMO_pred − HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

ax = axes[0, 1]
ax.hist(xyz_results['deviation'] * 1000, bins=80, color='darkorange', edgecolor='white', alpha=0.8)
ax.axvline(x=xyz_results['deviation'].mean() * 1000, color='red', linestyle='--',
           label=f'Mean = {xyz_results["deviation"].mean()*1000:.2f} mHa')
ax.set_title('(b) 11d/23d GAT: |gap − (LUMO−HOMO)| distribution')
ax.set_xlabel('Deviation (mHa)'); ax.set_ylabel('Count'); ax.legend()

ax = axes[0, 2]
ax.scatter(smiles_results['gap_derived'], smiles_results['gap_pred'], alpha=0.3, s=6, c='steelblue')
lims = [min(smiles_results['gap_pred'].min(), smiles_results['gap_derived'].min()),
        max(smiles_results['gap_pred'].max(), smiles_results['gap_derived'].max())]
ax.plot(lims, lims, 'r--', lw=1)
r2_sm_gap = np.corrcoef(smiles_results['gap_derived'], smiles_results['gap_pred'])[0, 1] ** 2
ax.set_title(f'(c) SMILES GAT: gap vs (LUMO−HOMO)\nR² = {r2_sm_gap:.4f}')
ax.set_xlabel('LUMO_pred − HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

ax = axes[1, 0]
homo_ok = xyz_results['homo_pred'] < xyz_results['lumo_pred']
violation_rate = (~homo_ok).mean() * 100
colors = ['#2ecc71' if ok else '#e74c3c' for ok in homo_ok]
ax.scatter(xyz_results['homo_pred'][:500], xyz_results['lumo_pred'][:500],
           c=colors[:500], alpha=0.5, s=9)
ax.plot([-0.5, 0.1], [-0.5, 0.1], 'k--', lw=0.5)
ax.set_xlabel('HOMO_pred (Ha)'); ax.set_ylabel('LUMO_pred (Ha)')
ax.set_title(f'(d) 11d/23d GAT: HOMO < LUMO ordering\nViolation rate: {violation_rate:.2f}% (green = OK, red = violation)')

ax = axes[1, 1]
methods = ['11d/23d GAT', 'SMILES GAT', 'Published\nRGCN']
mean_devs = [xyz_results['deviation'].mean() * 1000, smiles_results['deviation'].mean() * 1000, 0.1]
comply_010 = [(xyz_results['deviation'] < 0.010).mean() * 100,
              (smiles_results['deviation'] < 0.010).mean() * 100, 100]
x = np.arange(len(methods)); w = 0.25
ax.bar(x - w, mean_devs, w, label='Mean deviation (mHa)', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + w, comply_010, w, label='Compliance <0.01 Ha (%)', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('Mean |gap − (LUMO−HOMO)| (mHa)')
ax2.set_ylabel('Compliance <0.01 Ha (%)'); ax.set_title('(e) Physics Consistency Comparison')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
          bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=FS_LEGEND_E)

ax = axes[1, 2]
ax.scatter(xyz_results['gap_true'], xyz_results['deviation'] * 1000, alpha=0.2, s=6, c='darkorange')
ax.set_xlabel('True gap (Ha)'); ax.set_ylabel('|gap_pred − (LUMO−HOMO)| (mHa)')
ax.set_title('(f) 11d/23d GAT: Deviation vs True Gap')
ax.axhline(y=xyz_results['deviation'].mean() * 1000, color='red', linestyle='--', alpha=0.7)

fig.suptitle('Physics Consistency: HL-Gap = LUMO − HOMO Verification', fontsize=FS_SUPTITLE, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0.09, 1, 1])
out = os.path.join(HERE, 'results', 'fig4_physics_hd.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f'fig4 saved: {out}')

# 验证数字 vs 正文图注（Main P104）
print(f'[verify] (a) R²={r2_xyz_gap:.4f} (caption: 1.000)')
print(f'[verify] (b) mean={xyz_results["deviation"].mean()*1000:.3f} mHa (caption: 0.100)')
print(f'[verify] (c) mean={smiles_results["deviation"].mean()*1000:.3f} mHa (caption: 0.219)')
print(f'[verify] (d) violation={violation_rate:.3f}% (caption: <1%)')
from PIL import Image
im = Image.open(out)
print(f'[size] {im.size[0]}x{im.size[1]}, extent cy = {round(5486400 * im.size[1] / im.size[0])}')
