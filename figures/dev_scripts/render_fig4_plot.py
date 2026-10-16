# -*- coding: utf-8 -*-
"""Fig 4 画图专用 (独立进程, 不 import torch — 规避推理后同进程画图 access violation):
读 render_fig4_hd.py 存下的 fig4_xyz_results.npz / fig4_smiles_results.npz, 渲染高清版"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 11.5, 'axes.titlesize': 12, 'axes.labelsize': 11.5,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})

xz = np.load('d:/ml/chem_ml5/fig4_xyz_results.npz')
sm = np.load('d:/ml/chem_ml5/fig4_smiles_results.npz')
xyz_results = {k: xz[k] for k in xz.files}
smiles_results = {k: sm[k] for k in sm.files}
print('npz loaded')

fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.8))

ax = axes[0, 0]
ax.scatter(xyz_results['gap_derived'], xyz_results['gap_pred'], alpha=0.3, s=4, c='darkorange')
lims = [min(xyz_results['gap_pred'].min(), xyz_results['gap_derived'].min()),
        max(xyz_results['gap_pred'].max(), xyz_results['gap_derived'].max())]
ax.plot(lims, lims, 'r--', lw=1)
r2_xyz_gap = np.corrcoef(xyz_results['gap_derived'], xyz_results['gap_pred'])[0,1]**2
ax.set_title(f'(a) 3D GAT: gap vs (LUMO−HOMO)\nR² = {r2_xyz_gap:.4f}')
ax.set_xlabel('LUMO_pred − HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

ax = axes[0, 1]
ax.hist(xyz_results['deviation']*1000, bins=80, color='darkorange', edgecolor='white', alpha=0.8)
ax.axvline(x=xyz_results['deviation'].mean()*1000, color='red', linestyle='--',
           label=f'Mean = {xyz_results["deviation"].mean()*1000:.2f} mHa')
ax.set_title('(b) 3D GAT: |gap − (LUMO−HOMO)| distribution')
ax.set_xlabel('Deviation (mHa)'); ax.set_ylabel('Count'); ax.legend()

ax = axes[0, 2]
ax.scatter(smiles_results['gap_derived'], smiles_results['gap_pred'], alpha=0.3, s=4, c='steelblue')
lims = [min(smiles_results['gap_pred'].min(), smiles_results['gap_derived'].min()),
        max(smiles_results['gap_pred'].max(), smiles_results['gap_derived'].max())]
ax.plot(lims, lims, 'r--', lw=1)
r2_sm_gap = np.corrcoef(smiles_results['gap_derived'], smiles_results['gap_pred'])[0,1]**2
ax.set_title(f'(c) SMILES GAT: gap vs (LUMO−HOMO)\nR² = {r2_sm_gap:.4f}')
ax.set_xlabel('LUMO_pred − HOMO_pred (Ha)'); ax.set_ylabel('gap_pred (Ha)')

ax = axes[1, 0]
homo_ok = xyz_results['homo_pred'] < xyz_results['lumo_pred']
violation_rate = (~homo_ok).mean() * 100
colors = ['#2ecc71' if ok else '#e74c3c' for ok in homo_ok]
ax.scatter(xyz_results['homo_pred'][:500], xyz_results['lumo_pred'][:500],
           c=colors[:500], alpha=0.5, s=6)
ax.plot([-0.5, 0.1], [-0.5, 0.1], 'k--', lw=0.5)
ax.set_xlabel('HOMO_pred (Ha)'); ax.set_ylabel('LUMO_pred (Ha)')
ax.set_title(f'(d) 3D GAT: HOMO < LUMO ordering\nViolation rate: {violation_rate:.2f}% (green = OK, red = violation)')

ax = axes[1, 1]
methods = ['3D GAT', 'SMILES GAT', 'Published\nRGCN']
mean_devs = [xyz_results['deviation'].mean()*1000, smiles_results['deviation'].mean()*1000, 0.1]
comply_010 = [(xyz_results['deviation'] < 0.010).mean()*100,
              (smiles_results['deviation'] < 0.010).mean()*100, 100]
x = np.arange(len(methods)); w = 0.25
ax.bar(x - w, mean_devs, w, label='Mean deviation (mHa)', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + w, comply_010, w, label='Compliance <0.01 Ha (%)', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('Mean |gap − (LUMO−HOMO)| (mHa)')
ax2.set_ylabel('Compliance <0.01 Ha (%)'); ax.set_title('(e) Physics Consistency Comparison')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

ax = axes[1, 2]
ax.scatter(xyz_results['gap_true'], xyz_results['deviation']*1000, alpha=0.2, s=4, c='darkorange')
ax.set_xlabel('True gap (Ha)'); ax.set_ylabel('|gap_pred − (LUMO−HOMO)| (mHa)')
ax.set_title('(f) 3D GAT: Deviation vs True Gap')
ax.axhline(y=xyz_results['deviation'].mean()*1000, color='red', linestyle='--', alpha=0.7)

fig.suptitle('Physics Consistency: HL-Gap = LUMO − HOMO Verification', fontsize=13.5, fontweight='bold', y=1.00)
plt.tight_layout()
out = 'd:/ml/chem_ml5/fig4_physics_hd.png'
plt.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
print(f'fig4 saved: {out}')

print(f'[verify] (a) R²={r2_xyz_gap:.4f} (caption: 0.999)')
print(f'[verify] (b) mean={xyz_results["deviation"].mean()*1000:.3f} mHa (caption: 1.25)')
print(f'[verify] (c) mean={smiles_results["deviation"].mean()*1000:.3f} mHa (caption: 0.20)')
print(f'[verify] (d) violation={violation_rate:.3f}% (caption: <1%)')
from PIL import Image
im = Image.open(out)
print(f'[size] {im.size[0]}x{im.size[1]}, extent cy = {round(5486400 * im.size[1] / im.size[0])}')
