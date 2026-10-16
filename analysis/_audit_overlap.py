# -*- coding: utf-8 -*-
"""审计 P048 重叠句：在当前 13,556 划分上重算 SMILES 重复数与 R² 变化。
正文：removing all 787 SMILES-duplicate test complexes, 5.8%, changes dipole R² by at most
      -0.005, to 0.734 for the 3D ensemble
"""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.metrics import r2_score

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
TARGETS = ['HL_Gap', 'HOMO_Energy', 'LUMO_Energy', 'Dipole_M', 'Metal_q',
           'Electronic_E', 'Dispersion_E', 'Polarizability']

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df = df.dropna(subset=TARGETS).reset_index(drop=True)
print('clean CSV rows =', len(df))
sp = np.load('ensemble_split_new.npz')
te, cv = sp['te_idx'], sp['cv_idx']
print('te=%d cv=%d sum=%d' % (len(te), len(cv), len(te) + len(cv)))

smis = df['SMILES'].tolist()

def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    return Chem.MolToSmiles(m) if m is not None else None

cv_set = set()
bad = 0
for r in cv:
    c = canon(smis[r])
    if c is None:
        bad += 1
    else:
        cv_set.add(c)
print('cv canonical SMILES set: %d (unparsable %d)' % (len(cv_set), bad))

te_c = [canon(smis[r]) for r in te]
ov = np.array([c is not None and c in cv_set for c in te_c])
n = len(te)
print('★ SMILES-overlap: %d / %d  (%.2f%%)' % (ov.sum(), n, ov.sum() / n * 100))

# 也按 y_true 行序对齐（npz 的行序即 te 的顺序?）
zA = np.load('test_preds_A_8d6d.npz', allow_pickle=True)
print('npz te_idx 与 split te_idx 顺序一致:', np.array_equal(zA['te_idx'], te))

TAGS = {'A': 'A_8d6d', 'B': 'B_8d22d', 'C': 'C_11d6d', 'D': 'D_11d23d'}
res = {}
for k, t in TAGS.items():
    z = np.load('test_preds_%s.npz' % t, allow_pickle=True)
    assert np.array_equal(z['te_idx'], te)
    yt, en = z['y_true'], z['ens_mean']
    full = np.array([r2_score(yt[:, j], en[:, j]) for j in range(8)])
    no = np.array([r2_score(yt[~ov, j], en[~ov, j]) for j in range(8)])
    res[k] = (full, no)
    print('\n变体 %s   (印 Table 1 dipole: A .707 B .718 C .734 D .739)' % k)
    print('   %-14s full   nonov   Δ' % 'prop')
    for j, p in enumerate(P8):
        print('   %-14s %.4f %.4f %+.4f' % (p, full[j], no[j], no[j] - full[j]))
    print('   最大跌幅 = %+.4f （%s）;  dipole: %.4f -> %.4f' % (
        (no - full).min(), P8[int((no - full).argmin())], full[3], no[3]))
    print('   8 性质均值 full %.4f -> nonov %.4f (%+.4f)' % (full.mean(), no.mean(), no.mean() - full.mean()))

# 重叠子集本身的表现（供参考）
for k in ('A', 'D'):
    z = np.load('test_preds_%s.npz' % TAGS[k], allow_pickle=True)
    yt, en = z['y_true'], z['ens_mean']
    print('  %s 重叠子集 dipole R² = %.4f (n=%d)' % (k, r2_score(yt[ov, 3], en[ov, 3]), ov.sum()))
