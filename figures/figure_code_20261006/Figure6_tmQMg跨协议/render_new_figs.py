# -*- coding: utf-8 -*-
"""新数字重渲 Fig2a/2b/5/6/7 — 紧凑布局, 与旧图同像素尺寸"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
OUT = r'D:/ml/paper/draft/round4/figure'
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']

zA = np.load(r'D:/ml/chem_ml5/test_preds_A_8d6d.npz', allow_pickle=False)
yt = zA['y_true']; ensA = zA['ens_mean']; predsA = zA['preds']
zD = np.load(r'D:/ml/chem_ml5/test_preds_D_11d23d.npz', allow_pickle=False)
ensD = zD['ens_mean']

def scatter_panel(ax, y, p, annot):
    ax.scatter(y, p, alpha=0.15, s=2, c='#E74C3C', edgecolors='none', rasterized=True)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lims, lims, 'k--', lw=0.7, alpha=0.5)
    ax.text(0.04, 0.94, annot, transform=ax.transAxes, fontsize=5.5, va='top',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', alpha=0.85, lw=0.4))
    ax.tick_params(labelsize=5)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)

# ===== Fig 2a: SMILES (A) 集成 parity, 1320x679 =====
fig, axes = plt.subplots(2, 4, figsize=(1320/300, 679/300))
plt.subplots_adjust(wspace=0.32, hspace=0.55, left=0.055, right=0.985, top=0.86, bottom=0.075)
axes = axes.flatten()
for i, prop in enumerate(targets):
    ax = axes[i]
    y = yt[:, i]; p = ensA[:, i]
    r2 = r2_score(y, p); mae = mean_absolute_error(y, p)
    scatter_panel(ax, y, p, f'R$^2$ = {r2:.3f}\nMAE = {mae:.3f}')
    if i >= 4:
        ax.set_xlabel(f'DFT {prop}', fontsize=6)
    if i % 4 == 0:
        ax.set_ylabel('Predicted', fontsize=6)
    ax.set_title(prop, fontweight='bold', fontsize=7)
fig.suptitle('SMILES GAT (variant A) ensemble: predicted vs DFT, N=13,556',
             fontsize=8.5, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig2a_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig2a ✓')

# ===== Fig 2b: SMILES (A) CV 汇总, 1315x1011 =====
fig, axes = plt.subplots(3, 1, figsize=(1315/300, 1011/300))
plt.subplots_adjust(hspace=0.55, left=0.08, right=0.97, top=0.86, bottom=0.06)
per_prop_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for m in range(9)]) for i in range(8)])
per_prop_std = np.array([np.std([r2_score(yt[:, i], predsA[m, :, i]) for m in range(9)]) for i in range(8)])
ens_r2 = np.array([r2_score(yt[:, i], ensA[:, i]) for i in range(8)])
val_mean = np.array([np.mean([r2_score(yt[:, i], predsA[m, :, i]) for i in range(8)]) for m in range(9)])
x = np.arange(8)
ax = axes[0]
ax.bar(x, per_prop_mean, yerr=per_prop_std, capsize=3, color='steelblue', edgecolor='white')
ax.axhline(np.mean(per_prop_mean), color='darkorange', ls='--', lw=1.2,
           label=f'Overall mean R$^2$={np.mean(per_prop_mean):.4f}')
ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=6.5)
ax.set_ylabel('Member R$^2$', fontsize=7)
ax.set_title('Per-property member-model R$^2$ (3-fold × 3-seed, n=9)', fontsize=7.5)
ax.legend(fontsize=6.5); ax.tick_params(labelsize=6)
ax = axes[1]
w = 0.35
ax.bar(x - w/2, ens_r2, w, label='Ensemble (mean prediction)', color='darkorange')
ax.bar(x + w/2, per_prop_mean, w, label='Single model avg', color='steelblue')
ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=6.5)
ax.set_ylabel('R$^2$', fontsize=7)
ax.set_title('Ensemble vs single-model R$^2$', fontsize=7.5)
ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9); ax.tick_params(labelsize=6)
ax = axes[2]
ax.boxplot([val_mean], widths=0.4)
ax.scatter(np.ones(9) + np.random.RandomState(0).uniform(-0.05, 0.05, 9), val_mean, color='darkorange', s=22, zorder=5)
ax.axhline(np.mean(val_mean), color='red', ls='--', lw=1.2, label=f'Mean={np.mean(val_mean):.4f}')
ax.set_xticklabels(['9 models'], fontsize=7); ax.set_ylabel('Mean R$^2$', fontsize=7)
ax.set_title('Model-to-model variation', fontsize=7.5)
ax.legend(fontsize=6.5, framealpha=0.9); ax.tick_params(labelsize=6)
ax.set_ylim(np.floor(val_mean.min()*1000)/1000 - 0.002, np.ceil(val_mean.max()*1000)/1000 + 0.002)
fig.suptitle('SMILES GAT (variant A) ensemble CV summary, N=13,556', fontsize=8.5, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig2b_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig2b ✓')

# ===== Fig 5: 扰动归因, 1255x864 =====
fig, axes = plt.subplots(1, 2, figsize=(1255/300, 864/300))
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.86, bottom=0.14)
donor_only = {'HOMO': 43.9, 'LUMO': 41.8, 'E_elec': 9.0, 'Polarizability': 9.2}
with_mc = {'gap': 33.6, 'HOMO': 56.2, 'LUMO': 54.0, 'dipole': 34.1, 'metal charge': 32.1,
           'E_elec': 19.7, 'E_disp': 33.1, 'polarizability': 29.5}
for ax, data, title in [(axes[0], donor_only, '(a) Donor-atom attribution fraction'),
                        (axes[1], with_mc, '(b) With metal-bonded carbons included')]:
    names = list(data.keys()); vals = list(data.values())
    ax.bar(names, vals, color='#5DADE2', edgecolor='white')
    for k, v in zip(names, vals):
        ax.text(k, v + 1.5, f'{v:.1f}%', ha='center', fontsize=6.5)
    ax.set_ylabel('Attribution fraction (%)', fontsize=7)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(axis='x', rotation=30, labelsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
fig.suptitle('Perturbation (occlusion) attribution, SMILES GAT, 2,000 complexes', fontsize=8.5, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig5_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig5 ✓')

# ===== Fig 6: 跨泛函, 1320x760 =====
zc = np.load(r'D:/ml/chem_ml5/crossfunct_A_preds.npz', allow_pickle=False)
yc = zc['y_true']; ec = zc['ens_mean']
ext = ['gap','homo','lumo','dipole','E_elec','E_disp','polarizability']
fig, axes = plt.subplots(2, 4, figsize=(1320/300, 760/300))
plt.subplots_adjust(wspace=0.34, hspace=0.58, left=0.06, right=0.985, top=0.86, bottom=0.075)
axes = axes.flatten()
for j, prop in enumerate(ext):
    ax = axes[j]
    y = yc[:, j]; p = ec[:, j]
    r, _ = pearsonr(y, p)
    ax.scatter(y, p, alpha=0.10, s=2, c='#3498DB', edgecolors='none', rasterized=True)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lims, lims, 'k--', lw=0.7, alpha=0.5)
    ax.text(0.04, 0.94, f'r = {r:.3f}', transform=ax.transAxes, fontsize=6, va='top',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', alpha=0.85, lw=0.4))
    ax.tick_params(labelsize=5)
    ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=4)
    if j >= 4:
        ax.set_xlabel(f'DFT {prop}', fontsize=6)
    if j % 4 == 0:
        ax.set_ylabel('Predicted', fontsize=6)
    ax.set_title(prop, fontweight='bold', fontsize=7)
axes[7].axis('off')
fig.suptitle('Cross-functional validation: SMILES GAT (variant A) on tmQMg, N=73,438',
             fontsize=8.5, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig6_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig6 ✓')

# ===== Fig 7: case study, 1320x807 =====
r = json.load(open(r'D:/ml/chem_ml5/case_study_new_results.json', encoding='utf-8'))
EV = 27.211386
DFT = {'Zr-cat3': (3.30, 5.32), 'Fe-cat1': (2.76, 9.76), 'Ni-cat3': (1.39, 5.67), 'Pd-cat2': (2.93, 2.48),
       'trans-PdCl2(PPh3)2': (3.19, 0.03), 'cis-PdCl2(PPh3)2': (3.17, 10.86), 'Pt(dppe)Cl2': (3.95, 10.29),
       'Co-cat1': (3.30, 0.00), 'Rh(CO)2(acac)': (4.03, 3.17), 'Ni(IMes)2Cl2': (3.24, 0.00)}
order = ['Zr-cat3','Fe-cat1','Ni-cat3','Pd-cat2','trans-PdCl2(PPh3)2','cis-PdCl2(PPh3)2',
         'Pt(dppe)Cl2','Co-cat1','Rh(CO)2(acac)','Ni(IMes)2Cl2']
names = ['Zr-cat3','Fe-cat1','Ni-cat3','Pd-cat2','trans-Pd','cis-Pd','Pt(dppe)','Co-cat1','Rh(acac)','Ni(IMes)$_2$']
fig, axes = plt.subplots(1, 2, figsize=(1320/300, 807/300))
plt.subplots_adjust(wspace=0.32, left=0.095, right=0.98, top=0.86, bottom=0.24)
for ax, mode in [(axes[0], 'gap'), (axes[1], 'dipole')]:
    dft_vals = [DFT[k][0 if mode == 'gap' else 1] for k in order]
    b_vals = []; a_vals = []
    for k in order:
        v = r[k]
        if mode == 'gap':
            b_vals.append(v['d3_mean'][0] * EV); a_vals.append(v['smiles_mean'][0] * EV)
        else:
            b_vals.append(max(0, v['d3_mean'][3])); a_vals.append(max(0, v['smiles_mean'][3]))
    x = np.arange(len(order)); w = 0.26
    ax.bar(x - w, b_vals, w, label='3D scorer (B)', color='#3498DB')
    ax.bar(x, a_vals, w, label='SMILES GAT (A)', color='#7F8C8D')
    ax.bar(x + w, dft_vals, w, label='DFT', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=6)
    ax.set_ylabel('HL-Gap (eV)' if mode == 'gap' else 'Dipole (D)', fontsize=7)
    ax.set_title('(a) HL-Gap' if mode == 'gap' else '(b) Dipole moment', fontsize=8)
    ax.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(0.0, -0.30), ncol=3)
    ax.tick_params(labelsize=6)
    if mode == 'dipole':
        ax.set_ylim(-0.6, 12.4)
fig.suptitle('Case study: predicted vs DFT for ten in-house catalysts',
             fontsize=8.5, fontweight='bold', y=0.985)
fig.savefig(os.path.join(OUT, 'Fig7_new.png'), dpi=300, facecolor='white')
plt.close(fig)
print('Fig7 ✓')
print('全部渲染完成 →', OUT)
