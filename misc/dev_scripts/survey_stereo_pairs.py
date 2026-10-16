# -*- coding: utf-8 -*-
"""意见 13 摸底: te 13,547 中「相同连接表、不同立体构型」matched pairs 数量统计
定义: canonical SMILES (isomericSmiles=False) 相同 → 相同连接表
      组内 isomeric SMILES 不同 → 立体构型差异 (cis/trans, fac/mer 等)
另: 组内 y_true 差分布 vs 全体行间差 → 判断 matched 集是否有区分度
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
smiles_by_idx = df_y['SMILES'].tolist()

z = np.load('kept_cache.npz'); mseq = z['csv_idx'].copy()
rem = np.load('vA_removed_kept.npy')
keep = np.ones(len(mseq), dtype=bool); keep[rem] = False
mseq = mseq[keep]
sp = np.load('data_tmQM/paper_domain_vA/ensemble_split.npz')
te_rows = mseq[sp['te_idx']]
print(f'te rows: {len(te_rows)}', flush=True)

canon, isomer, ok = [], [], []
for c in te_rows:
    s = smiles_by_idx[c]
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    if m is None:
        ok.append(False); canon.append(None); isomer.append(None)
    else:
        ok.append(True)
        canon.append(Chem.MolToSmiles(m, isomericSmiles=False))
        isomer.append(Chem.MolToSmiles(m, isomericSmiles=True))
print(f'parse ok: {sum(ok)}', flush=True)

groups = defaultdict(list)
for i, c in enumerate(canon):
    if c is not None: groups[c].append(i)

multi = {c: v for c, v in groups.items() if len(v) > 1}
n_in_multi = sum(len(v) for v in multi.values())
stereo_groups = {c: v for c, v in multi.items() if len(set(isomer[i] for i in v)) > 1}
n_in_stereo = sum(len(v) for v in stereo_groups.values())

sizes = sorted((len(v) for v in multi.values()), reverse=True)
print(f'\n相同连接表组: {len(multi)} 组 / 覆盖 {n_in_multi} 行 ({n_in_multi/len(te_rows)*100:.2f}%)')
print(f'  组大小分布 (top20): {sizes[:20]}')
print(f'其中组内含立体构型差异: {len(stereo_groups)} 组 / 覆盖 {n_in_stereo} 行 ({n_in_stereo/len(te_rows)*100:.2f}%)')
print(f'  立体组大小分布 (top20): {sorted((len(v) for v in stereo_groups.values()), reverse=True)[:20]}', flush=True)

# 立体组内构型种类
kinds = defaultdict(int)
for c, v in stereo_groups.items():
    kinds[len(set(isomer[i] for i in v))] += 1
print('  组内不同构型数分布:', dict(sorted(kinds.items())), flush=True)

# 组内 y_true 差分布 (8 性质) vs 全体
yt_all = np.load('ensemble_predictions.npz')['y_true']
pairs = []
for c, v in stereo_groups.items():
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            pairs.append((v[a], v[b]))
print(f'\n组内配对总数: {len(pairs)}')
if pairs:
    pd_ = np.array([np.abs(yt_all[i] - yt_all[j]) for i, j in pairs])  # (P, 8)
    # 全体行间差参考 (随机 5000 对)
    rng = np.random.RandomState(0)
    ii = rng.randint(0, len(te_rows), 5000); jj = rng.randint(0, len(te_rows), 5000)
    ref = np.abs(yt_all[ii] - yt_all[jj])
    names = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
    for k, n in enumerate(names):
        print(f'  {n:14s} stereo-pair median |Δy|={np.median(pd_[:, k]):.4g}   random-pair median={np.median(ref[:, k]):.4g}')
np.save('stereo_pairs_idx.npy', np.array(pairs))
print('SAVED stereo_pairs_idx.npy')
