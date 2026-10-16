# -*- coding: utf-8 -*-
"""重叠审计 v2 —— 用 new_domain_iso.pt 的 csv_idx 正确映射 SMILES"""
import sys
import numpy as np, pandas as pd, torch
sys.stdout.reconfigure(encoding='utf-8')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.metrics import r2_score

P8 = ["HL-Gap", "HOMO", "LUMO", "Dipole", "MetalCharge", "E_elec", "E_disp", "Polarizability"]
TARGETS = ['HL_Gap', 'HOMO_Energy', 'LUMO_Energy', 'Dipole_M', 'Metal_q',
           'Electronic_E', 'Dispersion_E', 'Polarizability']
df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';').dropna(subset=TARGETS).reset_index(drop=True)
D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
csv_idx = np.asarray(D['csv_idx'])
y_all = D['y_all'].numpy()
print('csv_idx max %d  (<%d)' % (csv_idx.max(), len(df)))
chk = df.iloc[csv_idx][TARGETS].values.astype(np.float32)
print('y_all 与 CSV 对应:', np.allclose(chk, y_all, atol=1e-4), ' 最大偏差 %.4g'
      % np.abs(chk - y_all).max())

smis = df['SMILES'].values
sp = np.load('ensemble_split_new.npz')
te, cv = sp['te_idx'], sp['cv_idx']
assert np.array_equal(np.sort(te), np.sort(D['te_idx']))
print('te=%d cv=%d' % (len(te), len(cv)))

def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    return Chem.MolToSmiles(m) if m is not None else None

te_smi = [smis[csv_idx[i]] for i in te]
cv_smi = [smis[csv_idx[i]] for i in cv]
cv_set = set(filter(None, (canon(s) for s in cv_smi)))
te_c = [canon(s) for s in te_smi]
nfail = sum(1 for c in te_c if c is None)
ov = np.array([c is not None and c in cv_set for c in te_c])
print('te 解析失败 %d;  ★ SMILES 重叠 %d / %d (%.2f%%)' % (nfail, ov.sum(), len(te), ov.sum() / len(te) * 100))

# CSD 码口径（正文/SI 称 CSD 一致率 100%）
te_csd = [D['csd_codes'][i] for i in te]
cv_csd = set(D['csd_codes'][i] for i in cv)
print('CSD 码重叠: %d' % sum(1 for c in te_csd if c in cv_csd))

TAGS = {'A': 'A_8d6d', 'B': 'B_8d22d', 'C': 'C_11d6d', 'D': 'D_11d23d'}
print()
for k, t in TAGS.items():
    z = np.load('test_preds_%s.npz' % t, allow_pickle=True)
    yt, en = z['y_true'], z['ens_mean']
    full = np.array([r2_score(yt[:, j], en[:, j]) for j in range(8)])
    no = np.array([r2_score(yt[~ov, j], en[~ov, j]) for j in range(8)])
    d = no - full
    print('变体 %s: 最大跌幅 %+.4f (%s) | dipole %.4f -> %.4f (%+.4f) | 8性质均值 %.4f -> %.4f'
          % (k, d.min(), P8[int(d.argmin())], full[3], no[3], d[3], full.mean(), no.mean()))
    if k in ('B', 'D'):
        print('        各性质 Δ: %s' % '  '.join('%s %+.4f' % (P8[j][:6], d[j]) for j in range(8)))
np.save('_audit/ov_mask.npy', ov)
