# -*- coding: utf-8 -*-
"""按 benchmark_comparison_v3.ipynb 的图代码重跑 benchmark 三图 (dpi 300 -> 600)
图 1: figure_benchmark_r2.png        分组柱状图 (7 方法 x 8 性质 R2, 每柱标值)
图 2: figure_benchmark_mean_r2.png   Mean R2 总览
图 3: figure_benchmark_dipole.png    Dipole head-to-head 横向条形图
数据与 notebook 完全一致: benchmark_results.csv / baseline_results.json /
gcn_3d_results.json / ensemble_results.csv + notebook 内 SMILES GAT 硬编码值"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np, pandas as pd, json, os, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

DPI = 600  # 用户要求: notebook 原为 300

targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

# ===== 1. Load RF + XGB results =====
if os.path.exists('benchmark_results.csv'):
    bm = pd.read_csv('benchmark_results.csv')
    if 'RF_R2' in bm.columns:
        rf_r2 = dict(zip(bm['Property'], bm['RF_R2']))
        xgb_r2 = dict(zip(bm['Property'], bm['XGB_R2']))
    else:
        rf_r2 = dict(zip(bm['Property'], bm['RF']))
        xgb_r2 = dict(zip(bm['Property'], bm['XGBoost']))
    print(f'RF Mean R2: {np.mean(list(rf_r2.values())):.4f}')
    print(f'XGB Mean R2: {np.mean(list(xgb_r2.values())):.4f}')
else:
    rf_r2 = {}; xgb_r2 = {}

# ===== 2. Load GCN/GIN results =====
gcn_r2, gin_r2 = {}, {}
if os.path.exists('baseline_results.json'):
    with open('baseline_results.json') as f: bl = json.load(f)
    gcn_r2 = {t: bl['gcn']['r2'][i] for i, t in enumerate(targets)}
    gin_r2 = {t: bl['gin']['r2'][i] for i, t in enumerate(targets)}
    print(f'GCN Mean R2: {bl["gcn"]["mean_r2"]:.4f}')
    print(f'GIN Mean R2: {bl["gin"]["mean_r2"]:.4f}')

# ===== 3. SMILES GAT (notebook 硬编码值, 原样保留) =====
smiles_r2 = {'gap': 0.809, 'homo': 0.919, 'lumo': 0.908, 'dipole': 0.606,
             'metal_charge': 0.960, 'E_elec': 0.954, 'E_disp': 0.926, 'polarizability': 0.938}
print(f'SMILES GAT Mean R2: {np.mean(list(smiles_r2.values())):.4f}')

# ===== 4. 3D GCN =====
gcn3d_r2 = {}
if os.path.exists('gcn_3d_results.json'):
    with open('gcn_3d_results.json') as f: g3d = json.load(f)
    for r in g3d['results']:
        gcn3d_r2[r['Property']] = r['R2']
    print(f'3D GCN Mean R2: {g3d["mean_r2"]:.4f}')
else:
    gcn3d_r2 = {'gap': 0.644, 'homo': 0.780, 'lumo': 0.773, 'dipole': 0.392,
                'metal_charge': 0.875, 'E_elec': 0.712, 'E_disp': 0.195, 'polarizability': 0.192}
    print(f'3D GCN Mean R2: {np.mean(list(gcn3d_r2.values())):.4f} (hardcoded)')

# ===== 5. XYZ GAT Ensemble =====
xyz_r2, xyz_mae = {}, {}
if os.path.exists('ensemble_results.csv'):
    ens = pd.read_csv('ensemble_results.csv')
    xyz_r2 = dict(zip(ens['Property'], ens['Ensemble_R2']))
    xyz_mae = dict(zip(ens['Property'], ens['Ensemble_MAE']))
    print(f'XYZ GAT Ensemble Mean R2: {ens["Ensemble_R2"].mean():.4f}')

print('\nAll data loaded.\n')

# ===== Build comparison table (notebook cell 2, 修 f-string 语法) =====
methods = [
    ('RF',        rf_r2,     '#F1C40F'),
    ('XGBoost',   xgb_r2,    '#E67E22'),
    ('GCN (2D)',  gcn_r2,    '#95A5A6'),
    ('3D GCN',    gcn3d_r2,  '#E8DAEF'),
    ('GIN (2D)',  gin_r2,    '#7F8C8D'),
    ('SMILES GAT', smiles_r2, '#3498DB'),
    ('3D GAT',    xyz_r2,    '#E74C3C'),
]

sep = '=' * 100
print(sep)
print("R2 COMPARISON (all GNNs on same test set)")
print(sep)
header = f"{'Property':16s}"
for name, _, _ in methods:
    header += f' {name:>12s}'
print(header)
print('-' * (16 + 13 * len(methods)))
for prop in targets:
    line = f'{prop:16s}'
    for _, data, _ in methods:
        v = data.get(prop, 0)
        line += f' {v:12.4f}'
    print(line)
print('-' * (16 + 13 * len(methods)))
line = f"{'Mean R2':16s}"
mean_vals = []
for name, data, _ in methods:
    if data:
        m = np.mean(list(data.values()))
        mean_vals.append((name, m))
        line += f' {m:12.4f}'
    else:
        line += f' {"N/A":>12s}'
print(line)
comp = []
for prop in targets:
    row = {"Property": prop}
    for name, data, _ in methods:
        row[name] = data.get(prop, None)
    comp.append(row)
pd.DataFrame(comp).to_csv("benchmark_comparison_v3.csv", index=False)
print('\nSaved benchmark_comparison_v3.csv')

# ===== 图 1: grouped bar chart (notebook cell 3) =====
fig, ax = plt.subplots(figsize=(20, 8))
x = np.arange(len(targets))
n_methods = len(methods)
width = 0.8 / n_methods
method_names = [m[0] for m in methods]
method_colors = [m[2] for m in methods]
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
ax.legend(loc="lower right", fontsize=9, ncol=3)
ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig('figure_benchmark_r2.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figure_benchmark_r2.png')

# ===== 图 2: Mean R2 overview (notebook cell 4) =====
fig, ax = plt.subplots(figsize=(12, 6))
names = []; vals = []; colors = []
for name, data, color in methods:
    if data:
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

# ===== 图 3: Dipole head-to-head (notebook cell 5) =====
fig, ax = plt.subplots(figsize=(10, 6))
dipole_vals = []; dipole_names = []; dipole_colors = []
for name, data, color in methods:
    if 'dipole' in data:
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
