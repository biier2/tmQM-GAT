# -*- coding: utf-8 -*-
"""Comment 13 scoping: count the "same connectivity table, different stereochemistry" matched pairs in te 13,547
Definition: identical canonical SMILES (isomericSmiles=False) -> identical connectivity table
      differing isomeric SMILES within a group -> stereochemical difference (cis/trans, fac/mer, etc.)
Also: within-group y_true difference distribution vs all-row differences -> judge whether the matched set is discriminative
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
print(f'\nGroups with identical connectivity: {len(multi)} groups / covering {n_in_multi} rows ({n_in_multi/len(te_rows)*100:.2f}%)')
print(f'  Group size distribution (top 20): {sizes[:20]}')
print(f'Of these, groups containing stereochemical differences: {len(stereo_groups)} groups / covering {n_in_stereo} rows ({n_in_stereo/len(te_rows)*100:.2f}%)')
print(f'  Stereo group size distribution (top 20): {sorted((len(v) for v in stereo_groups.values()), reverse=True)[:20]}', flush=True)

# Number of distinct configurations within the stereo groups
kinds = defaultdict(int)
for c, v in stereo_groups.items():
    kinds[len(set(isomer[i] for i in v))] += 1
print('  Distribution of the number of distinct configurations per group:', dict(sorted(kinds.items())), flush=True)

# Within-group y_true difference distribution (8 properties) vs all rows
yt_all = np.load('ensemble_predictions.npz')['y_true']
pairs = []
for c, v in stereo_groups.items():
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            pairs.append((v[a], v[b]))
print(f'\nTotal within-group pairs: {len(pairs)}')
if pairs:
    pd_ = np.array([np.abs(yt_all[i] - yt_all[j]) for i, j in pairs])  # (P, 8)
    # All-row difference reference (5000 random pairs)
    rng = np.random.RandomState(0)
    ii = rng.randint(0, len(te_rows), 5000); jj = rng.randint(0, len(te_rows), 5000)
    ref = np.abs(yt_all[ii] - yt_all[jj])
    names = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
    for k, n in enumerate(names):
        print(f'  {n:14s} stereo-pair median |Δy|={np.median(pd_[:, k]):.4g}   random-pair median={np.median(ref[:, k]):.4g}')
np.save('stereo_pairs_idx.npy', np.array(pairs))
print('SAVED stereo_pairs_idx.npy')
