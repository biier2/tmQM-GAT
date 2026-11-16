# -*- coding: utf-8 -*-
"""Fig 3 (server version): benchmark comparison figure — rewritten following the figure format in benchmark_comparison_v3.ipynb
Outputs 3 figures (notebook cell 3/4/5 layout, dpi=600):
  figure_benchmark_r2.png        grouped bar chart (7 methods x 8 properties R2, value labelled on each bar)
  figure_benchmark_mean_r2.png   Mean R2 overview
  figure_benchmark_dipole.png    Dipole head-to-head horizontal bar chart
Numerical convention (final, do not change): RF/XGB = data/benchmark_results.csv (updated 2026-09 to the re-evaluated convention:
train=cv_idx 76,770 / test=te_idx 13,547, same set as the GNNs, mean 0.357/0.632),
GCN/GIN = data/baseline_results.json,
3D GCN = data/gcn_3d_results.json, 3D GAT = data/ensemble_results.csv,
SMILES GAT = deployment checkpoint per property (mean 0.932, consistent with the final Fig3;
the notebook's original hardcoded 0.8775 was from the old CV checkpoint and has been replaced with the final values)"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd, json, os, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============ layout parameters (original notebook layout) ============
DPI = 600
FIG1_W, FIG1_H = 20, 8    # grouped bar chart
FIG2_W, FIG2_H = 12, 6    # Mean R2 overview
FIG3_W, FIG3_H = 10, 6    # Dipole head-to-head
# ==================================================

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

# ===== 1. RF + XGB =====
bm = pd.read_csv('data/benchmark_results.csv')
if 'RF_R2' in bm.columns:
    rf_r2 = dict(zip(bm['Property'], bm['RF_R2']))
    xgb_r2 = dict(zip(bm['Property'], bm['XGB_R2']))
else:
    rf_r2 = dict(zip(bm['Property'], bm['RF']))
    xgb_r2 = dict(zip(bm['Property'], bm['XGBoost']))

# ===== 2. GCN / GIN =====
with open('data/baseline_results.json') as f: bl = json.load(f)
gcn_r2 = {t: bl['gcn']['r2'][i] for i, t in enumerate(targets)}
gin_r2 = {t: bl['gin']['r2'][i] for i, t in enumerate(targets)}

# ===== 3. SMILES GAT =====
# deployment checkpoint per property (final convention, mean 0.932; the notebook's old hardcoded 0.8775 is deprecated)
smiles_r2 = {'gap': 0.8767, 'homo': 0.9677, 'lumo': 0.9641, 'dipole': 0.8177,
             'metal_charge': 0.9492, 'E_elec': 0.9762, 'E_disp': 0.9436, 'polarizability': 0.9598}

# ===== 4. 3D GCN =====
with open('data/gcn_3d_results.json') as f: g3d = json.load(f)
gcn3d_r2 = {r['Property']: r['R2'] for r in g3d['results']}

# ===== 5. XYZ GAT Ensemble =====
ens = pd.read_csv('data/ensemble_results.csv')
xyz_r2 = dict(zip(ens['Property'], ens['Ensemble_R2']))

methods = [
    ('RF',         rf_r2,     '#F1C40F'),
    ('XGBoost',    xgb_r2,    '#E67E22'),
    ('GCN (2D)',   gcn_r2,    '#95A5A6'),
    ('3D GCN',     gcn3d_r2,  '#E8DAEF'),
    ('GIN (2D)',   gin_r2,    '#7F8C8D'),
    ('SMILES GAT', smiles_r2, '#3498DB'),
    ('3D GAT',     xyz_r2,    '#E74C3C'),
]

# ===== Print comparison table =====
sep = '=' * 100
print(sep)
print("R2 COMPARISON (all methods on same test set)")
print(sep)
header = f"{'Property':16s}"
for name, _, _ in methods:
    header += f' {name:>12s}'
print(header)
print('-' * (16 + 13 * len(methods)))
for prop in targets:
    line = f'{prop:16s}'
    for _, data, _ in methods:
        line += f' {data.get(prop, 0):12.4f}'
    print(line)
print('-' * (16 + 13 * len(methods)))
line = f"{'Mean R2':16s}"
mean_vals = []
for name, data, _ in methods:
    m = np.mean(list(data.values()))
    mean_vals.append((name, m))
    line += f' {m:12.4f}'
print(line)
comp = []
for prop in targets:
    row = {"Property": prop}
    for name, data, _ in methods:
        row[name] = data.get(prop, None)
    comp.append(row)
pd.DataFrame(comp).to_csv("benchmark_comparison_v3.csv", index=False)

# ===== Figure 1: grouped bar chart (notebook cell 3 layout) =====
fig, ax = plt.subplots(figsize=(FIG1_W, FIG1_H))
x = np.arange(len(targets))
n_methods = len(methods)
width = 0.8 / n_methods
for j, (name, data, color) in enumerate(methods):
    vals = [data.get(t, 0) for t in targets]
    bars = ax.bar(x + j * width, vals, width, label=name, color=color, edgecolor='white', lw=0.5)
    for bar, val in zip(bars, vals):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
ax.set_ylabel('R2', fontsize=13)
ax.set_title('External Benchmark: R2 Comparison Across Methods (higher is better)', fontsize=15, fontweight='bold')
ax.set_xticks(x + width * (n_methods - 1) / 2)
ax.set_xticklabels(targets, fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.savefig('figure_benchmark_r2.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figure_benchmark_r2.png')

# ===== Figure 2: Mean R2 overview (notebook cell 4 layout) =====
fig, ax = plt.subplots(figsize=(FIG2_W, FIG2_H))
names = []; vals = []; colors = []
for name, data, color in methods:
    names.append(name)
    vals.append(np.mean(list(data.values())))
    colors.append(color)
bars = ax.bar(names, vals, color=colors, edgecolor='white', lw=1.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean R2 (8 properties)', fontsize=13)
ax.set_title('Mean R2 Comparison Across Methods', fontsize=15, fontweight='bold')
ax.set_ylim(0, max(vals) * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig('figure_benchmark_mean_r2.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figure_benchmark_mean_r2.png')

# ===== Figure 3: Dipole head-to-head (notebook cell 5 layout) =====
fig, ax = plt.subplots(figsize=(FIG3_W, FIG3_H))
dipole_vals = []; dipole_names = []; dipole_colors = []
for name, data, color in methods:
    dipole_names.append(name)
    dipole_vals.append(data["dipole"])
    dipole_colors.append(color)
bars = ax.barh(dipole_names, dipole_vals, color=dipole_colors, edgecolor='white', lw=1.5)
for bar, val in zip(bars, dipole_vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Dipole R2', fontsize=13)
ax.set_title('Dipole Moment Prediction: 3D vs Topology-Only Models', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(dipole_vals) * 1.2)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figure_benchmark_dipole.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figure_benchmark_dipole.png')

# Verify numbers
print(f'[verify] Mean R2: RF={vals[0]:.4f} XGB={vals[1]:.4f} GCN={vals[2]:.4f} 3DGCN={vals[3]:.4f} '
      f'GIN={vals[4]:.4f} SMILES={vals[5]:.4f} (final 0.932) 3DGAT={vals[6]:.4f} (final 0.903)')
