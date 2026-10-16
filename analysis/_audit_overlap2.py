# -*- coding: utf-8 -*-
"""重叠口径对比：找出 787 的来源口径"""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

TARGETS = ['HL_Gap', 'HOMO_Energy', 'LUMO_Energy', 'Dipole_M', 'Metal_q',
           'Electronic_E', 'Dispersion_E', 'Polarizability']
df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';').dropna(subset=TARGETS).reset_index(drop=True)
sp = np.load('ensemble_split_new.npz')
te, cv = sp['te_idx'], sp['cv_idx']
smis = df['SMILES'].tolist()

def c_san(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    return Chem.MolToSmiles(m) if m is not None else None
def c_soft(s):
    if not (isinstance(s, str) and s): return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        m = Chem.MolFromSmiles(s, sanitize=False)
    if m is None:
        try:
            m = Chem.MolFromSmiles(s, sanitize=False)
            m.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
        except Exception:
            return None
    if m is None: return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None

for name, fn in [('sanitize=True', c_san), ('soft', c_soft)]:
    cv_set, nfail = set(), 0
    for r in cv:
        c = fn(smis[r])
        if c is None: nfail += 1
        else: cv_set.add(c)
    te_c = [fn(smis[r]) for r in te]
    nfailte = sum(1 for c in te_c if c is None)
    pop = sum(1 for c in te_c if c is not None and c in cv_set)
    print('%-14s cv 解析失败 %5d  set %6d | te 解析失败 %4d | 重叠 %4d / %d (%.2f%%)'
          % (name, nfail, len(cv_set), nfailte, pop, len(te), pop / len(te) * 100))

# 原始字符串口径
cv_raw = set(smis[r] for r in cv if isinstance(smis[r], str))
pop_raw = sum(1 for r in te if isinstance(smis[r], str) and smis[r] in cv_raw)
print('%-14s 重叠 %4d / %d (%.2f%%)' % ('raw string', pop_raw, len(te), pop_raw / len(te) * 100))
# 去掉立体化学的骨架口径
def c_skel(s):
    c = c_san(s)
    if c is None: return None
    m = Chem.MolFromSmiles(c)
    if m is None: return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)
cv_sk = set(filter(None, (c_skel(smis[r]) for r in cv)))
pop_sk = sum(1 for r in te if c_skel(smis[r]) in cv_sk)
print('%-14s 重叠 %4d / %d (%.2f%%)' % ('no-stereo', pop_sk, len(te), pop_sk / len(te) * 100))
