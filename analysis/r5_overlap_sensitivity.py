# -*- coding: utf-8 -*-
"""P116 重叠敏感性对 R5 (SMILES 9 成员集成) 重做 — 与 m1_overlap_sensitivity.py 同逻辑
SMILES 重复: te 行 canonical SMILES ∈ 训练池 (cv_idx) canonical SMILES 集
输出: full / nonoverlap / overlap 的 R5-ens 与 3D-ens dipole (clip) R²
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
smiles_by_idx = df_y['SMILES'].tolist()

z = np.load('kept_cache.npz')
mseq = z['csv_idx'].copy()
rem = np.load('vA_removed_kept.npy')
keep = np.ones(len(mseq), dtype=bool); keep[rem] = False
mseq = mseq[keep]
assert len(mseq) == 90317

sp = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_idx = sp['te_idx']; cv_idx = sp['cv_idx']
te_rows = mseq[te_idx]      # 13,547 行的 csv_idx
rest_rows = mseq[cv_idx]    # 76,770 训练池 csv_idx

def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    return Chem.MolToSmiles(m) if m is not None else None

rest_set = set()
for c in rest_rows:
    cs = canon(smiles_by_idx[c])
    if cs is not None: rest_set.add(cs)
te_canon = [canon(smiles_by_idx[c]) for c in te_rows]
smi_overlap = [i for i in range(len(te_rows)) if te_canon[i] is not None and te_canon[i] in rest_set]
print(f'SMILES-overlap: {len(smi_overlap)} / {len(te_rows)} ({len(smi_overlap)/len(te_rows)*100:.2f}%)', flush=True)
ov = np.zeros(len(te_rows), dtype=bool); ov[smi_overlap] = True

r5 = np.load('r5_smiles_va_ensemble.npz')
ep = np.load('ensemble_predictions.npz')
yt = ep['y_true']
i_d = 3  # dipole

def r2(y, p): return 1 - ((y-p)**2).sum()/((y-y.mean())**2).sum()

p5 = np.maximum(r5['mean'][:, i_d], 0)   # clip 口径 (意见 7)
p3 = np.maximum(ep['mean'][:, i_d], 0)
y = yt[:, i_d]

print('\ndipole R² (clip):')
print(f'  full        R5={r2(y, p5):.4f}  3D={r2(y, p3):.4f}')
print(f'  nonoverlap  R5={r2(y[~ov], p5[~ov]):.4f}  3D={r2(y[~ov], p3[~ov]):.4f}')
print(f'  overlap     R5={r2(y[ov], p5[ov]):.4f}  3D={r2(y[ov], p3[ov]):.4f}')
print(f'\n8 性质 mean R² 全/剔除后:')
mr5_f = np.mean([r2(yt[:, i], np.maximum(r5['mean'][:, i], 0) if i in (0, 3) else r5['mean'][:, i]) for i in range(8)])
mr5_n = np.mean([r2(yt[~ov, i], np.maximum(r5['mean'][~ov, i], 0) if i in (0, 3) else r5['mean'][~ov, i]) for i in range(8)])
m3_f = np.mean([r2(yt[:, i], np.maximum(ep['mean'][:, i], 0) if i in (0, 3) else ep['mean'][:, i]) for i in range(8)])
m3_n = np.mean([r2(yt[~ov, i], np.maximum(ep['mean'][~ov, i], 0) if i in (0, 3) else ep['mean'][~ov, i]) for i in range(8)])
print(f'  R5 mean: full {mr5_f:.4f} -> nonoverlap {mr5_n:.4f}')
print(f'  3D mean: full {m3_f:.4f} -> nonoverlap {m3_n:.4f}')
np.save('r5_overlap_idx.npy', smi_overlap)
print('\nSAVED r5_overlap_idx.npy')
