# -*- coding: utf-8 -*-
"""本地模拟 R5 (r5_smiles_va_ensemble.py) 加载段 — 数据门逐字预演
即: torch.load(vA_domain_smiles.pt) → ensemble_split.npz 划分 → ep 对齐自检
服务器跑 R5_SMOKE=1 前, 此处全过 = 数据门必过
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, numpy as np, torch

# ---- R5 加载段逐字 (前 120 行) ----
data = torch.load('vA_domain_smiles.pt', weights_only=False)
graphs = data['graphs']; y_all = data['y_all']; atoms_filt = data['atoms_filt']
print(f'graphs={len(graphs):,}  y_all={tuple(y_all.shape)}  atoms_filt={len(atoms_filt):,}', flush=True)

sp_path = 'ensemble_split.npz'
if not os.path.exists(sp_path):
    sp_path = 'data_tmQM/paper_domain_vA/ensemble_split.npz'
sp = np.load(sp_path)
te_idx = sp['te_idx']; cv_idx = sp['cv_idx']
print('ensemble_split.npz keys:', list(sp.keys()), '| te_idx:', len(te_idx), '| cv_idx:', len(cv_idx), flush=True)
assert len(te_idx) == 13547, 'assert len(te_idx)==13547 失败!'

for k in sp.keys():
    if k.startswith('fold_'):
        a = sp[k]
        print(f'  {k}: shape={a.shape}, dtype={a.dtype}', flush=True)

# ---- 金属权重段逐字 ----
mc = {}
for atoms in atoms_filt:
    for z in set(atoms):
        if z >= 21: mc[z] = mc.get(z, 0) + 1
total_c = sum(mc.values())
weight_map = {z: max(0.5, min(3.0, (total_c / c) ** 0.5)) for z, c in mc.items()}
print(f'金属权重: 种类 {len(mc)}, total_c={total_c}, 权重范围 [{min(weight_map.values()):.3f}, {max(weight_map.values()):.3f}]', flush=True)

# ---- 对齐自检段逐字 ----
ep = np.load('ensemble_predictions.npz')
ok1 = np.array_equal(ep['te_idx'], te_idx)
ok2 = np.allclose(y_all[te_idx].numpy(), ep['y_true'], atol=1e-4)
print(f'R5 自检: np.array_equal(ep te_idx, split te_idx) = {ok1} | np.allclose(y_all[te_idx], ep y_true, atol=1e-4) = {ok2}', flush=True)
if not (ok1 and ok2):
    sys.exit(1)
print('★ R5 数据门全过 — 可上传服务器跑 R5_SMOKE=1 冒烟', flush=True)
