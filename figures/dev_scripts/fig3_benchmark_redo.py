# -*- coding: utf-8 -*-
"""Fig 3 重画: benchmark 柱状图 — Table 2 口径对齐
- 全部 8 方法在相同 13,547 测试集 (90,317 域, RandomState(42) 15% split)
- SMILES GAT 柱 = 同协议 9 成员集成 0.898 (R5 重训, 与 3D GAT 柱同口径)
- RF 0.357 / XGB 0.632 (重评口径: 训练=cv_idx 76,770, 测试=te_idx 13,547, benchmark_reval_results.json), GCN 0.6397 / GIN 0.8420 (baseline_results.json),
  3D GCN 0.5704 (gcn_3d_results.json), 3D GAT 0.903 (9-model ensemble)
- DimeNet++ 0.0874 (受控协议基线, R4 best 检查点测试集评估, 同 13,547 测试集; 与 Table 2 同口径)
输出: fig3_benchmark_mean_r2.png"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['RF', 'XGBoost', 'GCN', '3D GCN', 'GIN', 'SMILES GAT', '3D GAT', 'DimeNet++']
values  = [0.3573, 0.6321, 0.6397, 0.5704, 0.8420, 0.898, 0.903, 0.0874]
colors  = ['#F1C40F', '#E67E22', '#95A5A6', '#5D6D7E', '#7F8C8D', '#3498DB', '#E74C3C', '#8E44AD']
# 两组 GAT 高亮描边 + DimeNet++ (新增受控基线) 描边
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
