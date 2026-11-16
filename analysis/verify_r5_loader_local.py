# -*- coding: utf-8 -*-
"""Local simulation of the R5 (r5_smiles_va_ensemble.py) loading section -- verbatim rehearsal of the data gate
i.e.: torch.load(vA_domain_smiles.pt) → ensemble_split.npz split → ep alignment self-check
If everything passes here before running R5_SMOKE=1 on the server, the data gate is guaranteed to pass
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, torch

# ---- R5 loading section verbatim (first 120 lines) ----
data = torch.load('vA_domain_smiles.pt', weights_only=False)
graphs = data['graphs']; y_all = data['y_all']; atoms_filt = data['atoms_filt']
print(f'graphs={len(graphs):,}  y_all={tuple(y_all.shape)}  atoms_filt={len(atoms_filt):,}', flush=True)

sp_path = 'ensemble_split.npz'
if not os.path.exists(sp_path):
    sp_path = 'data_tmQM/paper_domain_vA/ensemble_split.npz'
sp = np.load(sp_path)
te_idx = sp['te_idx']; cv_idx = sp['cv_idx']
print('ensemble_split.npz keys:', list(sp.keys()), '| te_idx:', len(te_idx), '| cv_idx:', len(cv_idx), flush=True)
assert len(te_idx) == 13547, 'assert len(te_idx)==13547 failed!'

for k in sp.keys():
    if k.startswith('fold_'):
        a = sp[k]
        print(f'  {k}: shape={a.shape}, dtype={a.dtype}', flush=True)

# ---- Metal weight section verbatim ----
mc = {}
for atoms in atoms_filt:
    for z in set(atoms):
        if z >= 21: mc[z] = mc.get(z, 0) + 1
total_c = sum(mc.values())
weight_map = {z: max(0.5, min(3.0, (total_c / c) ** 0.5)) for z, c in mc.items()}
print(f'metal weights: {len(mc)} species, total_c={total_c}, weight range [{min(weight_map.values()):.3f}, {max(weight_map.values()):.3f}]', flush=True)

# ---- Alignment self-check section verbatim ----
ep = np.load('ensemble_predictions.npz')
ok1 = np.array_equal(ep['te_idx'], te_idx)
ok2 = np.allclose(y_all[te_idx].numpy(), ep['y_true'], atol=1e-4)
print(f'R5 self-check: np.array_equal(ep te_idx, split te_idx) = {ok1} | np.allclose(y_all[te_idx], ep y_true, atol=1e-4) = {ok2}', flush=True)
if not (ok1 and ok2):
    sys.exit(1)
print('★ R5 data gate fully passed -- can upload to the server and run the R5_SMOKE=1 smoke test', flush=True)
