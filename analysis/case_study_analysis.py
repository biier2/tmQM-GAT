# -*- coding: utf-8 -*-
"""
案例验证分析: scorer 预测 vs DFT (TPSSh-D3BJ/def2-SVP) 对比
数据源:
  - scorer_3d_results.csv (18 催化剂, 3D GAT + SMILES GAT 双模型预测)
  - 案例验证.md (10 个 DFT job 的权威 gap / dipole, 开壳层取较小 alpha/beta gap)
映射 (电子数法, 见项目交接文档 §4.1; ⚠ 标记为待 gjf 确认):
  Co-cat3_sp→Co-cat1, Fe-cat1_sp→Fe-cat1, Ni-cat3_sp→Ni-cat3,
  Pd-cat2_sp→Pd-cat2(⚠), Zr-cat3-2_sp→Zr-cat3(⚠),
  cis-PdCl2-PPh32→cis-PdCl2(PPh3)2, trans-Pd-PPh_sp→trans-PdCl2(PPh3)2,
  Pt-dppe_sp→Pt(dppe)Cl2, Rh-CO2_sp→Rh(CO)2(acac), Ni-NHC_sp→(旧 51 原子分子, 已弃用)
(Ni-NHC 原优化结构 = C2-H 保留的 51 原子分子; 重新建模 Ni(IMes)2Cl2 m1/m3 后,
 单重态 m1 为基态, log 复核 2026-08-24, 采用 m1)
"""
import pandas as pd

# DFT 参考值: gap (eV, 开壳层=min(alpha,beta)), dipole (D)
DFT = {
    # 旧 5
    'Co-cat1':            {'gap': 3.304110, 'dipole': 0.0016},
    'Fe-cat1':            {'gap': 2.763144, 'dipole': 9.7600},
    'Ni-cat3':            {'gap': 1.393083, 'dipole': 5.6682},
    'Pd-cat2':            {'gap': 2.925023, 'dipole': 2.4821},
    'Zr-cat3':            {'gap': 3.297484, 'dipole': 5.3175},
    # 新 5
    'cis-PdCl2(PPh3)2':   {'gap': 3.169948, 'dipole': 10.8571},
    'trans-PdCl2(PPh3)2': {'gap': 3.193102, 'dipole': 0.0322},
    'Rh(CO)2(acac)':      {'gap': 4.027482, 'dipole': 3.1738},
    'Ni(IMes)2Cl2':       {'gap': 3.235281, 'dipole': 0.0003},   # ✓ m1 单重态 (log 复核 2026-08-24)
    'Pt(dppe)Cl2':        {'gap': 3.953475, 'dipole': 10.2881},
}

OLD5 = ['Co-cat1', 'Fe-cat1', 'Ni-cat3', 'Pd-cat2', 'Zr-cat3']
NEW5 = ['cis-PdCl2(PPh3)2', 'trans-PdCl2(PPh3)2', 'Rh(CO)2(acac)', 'Ni(IMes)2Cl2', 'Pt(dppe)Cl2']

df = pd.read_csv('scorer_3d_results.csv')
df['Metal'] = df['Name'].str.extract(r'(Zr|Ni|Pd|Co|Fe|Rh|Pt)')

rows = []
for name in OLD5 + NEW5:
    r = df[df['Name'] == name].iloc[0]
    d = DFT[name]
    rows.append({
        'Catalyst': name,
        'Metal': r['Metal'],
        'Model': r['Model'],
        'gap_3D': r['gap_eV'],      'gap_DFT': d['gap'],      'd_gap': abs(r['gap_eV'] - d['gap']),
        'gap_SM': r['gap_SMILES_eV'], 'd_gap_SM': abs(r['gap_SMILES_eV'] - d['gap']),
        'dip_3D': r['dipole_D'],    'dip_DFT': d['dipole'],   'd_dip': abs(r['dipole_D'] - d['dipole']),
        'dip_SM': r['dipole_SMILES_D'], 'd_dip_SM': abs(r['dipole_SMILES_D'] - d['dipole']),
    })
cmp = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', lambda v: f'{v:.3f}')

for grp, names in [('旧5 (回退路径候选)', OLD5), ('新5 (本次新算 DFT)', NEW5)]:
    g = cmp[cmp['Catalyst'].isin(names)]
    print(f'\n===== {grp} =====')
    print(g[['Catalyst', 'gap_3D', 'gap_DFT', 'd_gap', 'gap_SM', 'd_gap_SM',
             'dip_3D', 'dip_DFT', 'd_dip', 'dip_SM', 'd_dip_SM']].to_string(index=False))
    print(f'--- {grp} MAE ---')
    print(f"  gap  3D={g['d_gap'].mean():.3f} eV | SMILES={g['d_gap_SM'].mean():.3f} eV")
    print(f"  dip  3D={g['d_dip'].mean():.3f} D  | SMILES={g['d_dip_SM'].mean():.3f} D")

g_all = cmp
print('\n===== 全部 10 个 MAE =====')
print(f"  gap  3D={g_all['d_gap'].mean():.3f} eV | SMILES={g_all['d_gap_SM'].mean():.3f} eV")
print(f"  dip  3D={g_all['d_dip'].mean():.3f} D  | SMILES={g_all['d_dip_SM'].mean():.3f} D")

# 全部 10 个映射均已 log 确认 (见 项目交接文档 §4.1)，无需再做子集敏感性分析
print('\n===== 使用情况统计 =====')
print(df['Model'].value_counts().to_string())
