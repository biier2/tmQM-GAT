# -*- coding: utf-8 -*-
"""本地生成 vA_domain_smiles.pt (已验证版) —— 绕开服务器 csv 漂移
输入: kept_cache.npz (90,335 kept 的 csv_idx+y) + vA_removed_kept.npy (18 个删除位置)
     + 本地 tmQM_y.csv (3-17 版, md5 f8eeb967...f043) + ensemble_predictions.npz
输出: vA_domain_smiles.pt {smiles, labels(float32), labels64, te_idx, tr_idx, csv_idx, csd_code}
端到端自检: te_idx 全等 + y_true 字节级 0/13,547 才保存
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, hashlib
import torch

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
smiles_by_idx = df_y['SMILES'].tolist()
codes = df_y['CSD_code'].tolist()
y_csv = df_y[targets].astype(float).values  # float64 原值

z = np.load('kept_cache.npz')
mseq = z['csv_idx'].copy()          # 90,335 行 csv_idx (mol_idx)
rem = np.load('vA_removed_kept.npy')  # 18 个删除位置 (原 90,335 空间)
print(f'cache: {len(mseq)} 行; 删除 {len(rem)} 个位置: {rem.tolist()}')

keep = np.ones(len(mseq), dtype=bool)
keep[rem] = False
mseq = mseq[keep]
assert len(mseq) == 90317, f'期望 90317, 得到 {len(mseq)}'

smiles_list = [smiles_by_idx[c] if c < len(smiles_by_idx) else '' for c in mseq]
code_list = [codes[c] if c < len(codes) else '' for c in mseq]
labels64 = y_csv[mseq]
labels32 = labels64.astype(np.float32)

n = 90317
idx_all = np.random.RandomState(42).permutation(n)
te_idx = idx_all[:13547].copy()
tr_idx = idx_all[13547:].copy()

ep = np.load('ensemble_predictions.npz')
te_ref = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)
print('te_idx 全等:', np.array_equal(te_idx, te_ref))
d = np.abs(labels32[te_idx] - yt_ref).max()
print(f'y_true max|diff| (float32): {d:.3e}')
miss = sum(1 for i in range(13547)
           if labels32[te_idx[i]].tobytes() != yt_ref[i].tobytes())
print(f'y_true 字节级不一致: {miss} / 13547')

assert np.array_equal(te_idx, te_ref) and miss == 0, '自检失败, 不保存'

torch.save({
    'smiles': smiles_list,
    'labels': labels32,
    'labels64': labels64,
    'te_idx': te_idx.astype(np.int64),
    'tr_idx': tr_idx.astype(np.int64),
    'csv_idx': mseq.astype(np.int64),
    'csd_code': code_list,
}, 'vA_domain_smiles.pt')
print('SAVED vA_domain_smiles.pt')
print('md5:', hashlib.md5(open('vA_domain_smiles.pt', 'rb').read()).hexdigest())
print('前 3 条 SMILES:', [s[:60] for s in smiles_list[:3]])
print('labels32[0]:', ['%.6g' % v for v in labels32[0]])
print('labels64[0]:', ['%.10g' % v for v in labels64[0]])
