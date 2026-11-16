# -*- coding: utf-8 -*-
"""Fig 3 redraw: benchmark bar chart — aligned with the Table 2 conventions
- all 8 methods on the same 13,547 test set (90,317 domain, RandomState(42) 15% split)
- SMILES GAT bar = same-protocol 9-member ensemble 0.898 (R5 retrain, same convention as the 3D GAT bar)
- RF 0.357 / XGB 0.632 (re-evaluated convention: train=cv_idx 76,770, test=te_idx 13,547, benchmark_reval_results.json), GCN 0.6397 / GIN 0.8420 (baseline_results.json),
  3D GCN 0.5704 (gcn_3d_results.json), 3D GAT 0.903 (9-model ensemble)
- DimeNet++ 0.0874 (controlled-protocol baseline, R4 best checkpoint evaluated on the test set, same 13,547 test set; same convention as Table 2)
Output: fig3_benchmark_mean_r2.png"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['RF', 'XGBoost', 'GCN', '3D GCN', 'GIN', 'SMILES GAT', '3D GAT', 'DimeNet++']
values  = [0.3573, 0.6321, 0.6397, 0.5704, 0.8420, 0.898, 0.903, 0.0874]
colors  = ['#F1C40F', '#E67E22', '#95A5A6', '#5D6D7E', '#7F8C8D', '#3498DB', '#E74C3C', '#8E44AD']
# highlighted outlines for the two GAT bars + DimeNet++ (newly added controlled baseline)
edge = ['none'] * 8
edge[5] = '#1F618D'; edge[6] = '#922B21'; edge[7] = '#4A235A'

fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(methods))
bars = ax.bar(x, values, color=colors, edgecolor=edge, linewidth=[0, 0, 0, 0, 0, 2, 2, 2])
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f'{val:.3f}',
            ha='center', va='bottom', fontsize=13.5, fontweight='bold')
ax.axhline(y=0.898, color='#3498DB', linestyle='--', linewidth=1, alpha=0.6,
           label='SMILES GAT 9-model ensemble (0.898)')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=13)
ax.set_ylabel('Mean R² (8 properties)', fontsize=13)
ax.set_title('Benchmark Comparison on the 13,547-Complex Test Set', fontsize=14.5, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=10.5, loc='upper left')
plt.tight_layout()
plt.savefig('d:/ml/chem_ml5/fig3_benchmark_mean_r2.png', dpi=600, bbox_inches='tight', facecolor='white')
print('fig3 saved. values:', dict(zip(methods, values)))
