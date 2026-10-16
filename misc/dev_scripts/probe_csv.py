# -*- coding: utf-8 -*-
"""探针: 打印 csv 与 ensemble_predictions.npz 的指纹与抽样数值 (本地/服务器同一脚本对照)
用法: python probe_csv.py [csv路径] [npz路径]
"""
import sys, os, hashlib
import numpy as np, pandas as pd

csv_p = sys.argv[1] if len(sys.argv) > 1 else 'tmQM_y.csv'
ep_p  = sys.argv[2] if len(sys.argv) > 2 else 'ensemble_predictions.npz'

print(f'csv: {csv_p} | size={os.path.getsize(csv_p)} | mtime={os.path.getmtime(csv_p):.0f}')
print(f'csv md5: {hashlib.md5(open(csv_p, "rb").read()).hexdigest()}')

df_y = pd.read_csv(csv_p, sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
print(f'csv rows: {len(df_y)}')
for k in [0, 1, 2, 861, 862, 863, 983, 984, 985, 33761, 33762, 33763]:
    r = df_y.iloc[k]
    vals = ' '.join(f'{r[t]:.10g}' for t in targets)
    print(f'iloc[{k}] {r["CSD_code"]}: {vals}')

ep = np.load(ep_p)
print(f'ep md5: {hashlib.md5(open(ep_p, "rb").read()).hexdigest()}')
print('ep keys:', list(ep.keys()))
print('ep te_idx[:5]:', ep['te_idx'][:5].tolist())
for i in range(3):
    print(f'ep y_true[{i}] (te={ep["te_idx"][i]}):', ' '.join(f'{v:.10g}' for v in ep['y_true'][i]))
