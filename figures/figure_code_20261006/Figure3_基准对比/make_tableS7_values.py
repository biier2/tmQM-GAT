# -*- coding: utf-8 -*-
"""由 baseline_results_new.json 现算 SI 新表 Table S7 的每一格（核对用）。

   Table S7 = 2D GCN / GIN / 3D GCN 三个模型在同一个 13,556 测试集上的逐性质 R² 与 MAE。
   数据源：server_package/server_baselines.py 的输出（新域 90,377，单模型 seed 42）。
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, 'baseline_results_new.json'), encoding='utf-8'))
PROPS = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole moment', 'Metal charge',
         'E_elec', 'E_disp', 'polarizability']

COLS = [('2D GCN', 'gcn'), ('GIN', 'gin'), ('3D GCN', 'gcn3d')]

print(f'{"Property":22s}' + ''.join(f'{n:>10s}' for n, _ in COLS))
for i, p in enumerate(PROPS):
    print(f'{p:22s}' + ''.join(f'{B[k]["r2"][i]:10.3f}' for _, k in COLS))
print(f'{"Mean R2 (intensive-5)":22s}' +
      ''.join(f'{sum(B[k]["r2"][:5])/5:10.3f}' for _, k in COLS))
print(f'{"Mean R2 (all 8)":22s}' + ''.join(f'{B[k]["mean_r2"]:10.3f}' for _, k in COLS))
print('-' * 52)
for i, lab in ((5, 'E_elec MAE (Ha)'), (6, 'E_disp MAE (Ha)'), (7, 'Polarizability MAE (a.u.)')):
    print(f'{lab:22s}' + ''.join(f'{B[k]["mae"][i]:10.4f}' for _, k in COLS))
