# -*- coding: utf-8 -*-
"""R5 (SMILES 9-member ensemble) vs 3D ensemble -- paired test under the same protocol (vA-domain 13,547 test set)
1. Integrity: byte-for-byte te_idx/y_true of the R5 npz vs ensemble_predictions.npz
2. Per-property R²/MAE under matched conventions: raw + clipped (Comment 7 reporting convention: dipole/gap max(0,·))
3. Paired test: per-molecule |err| Wilcoxon signed-rank (two-sided)
4. Ensemble convention: 9-member mean prediction
Output: r5_vs_3d_paired.json (all numbers)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from scipy import stats

PROPS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
CLIP = {'gap': True, 'dipole': True}  # Comment 7 reporting convention: clip non-negative properties

r5 = np.load('r5_smiles_va_ensemble.npz')
ep = np.load('ensemble_predictions.npz')

# ---- 1. Integrity ----
te_eq = np.array_equal(r5['te_idx'], ep['te_idx'])
yt_miss = sum(1 for i in range(13547)
              if r5['y_true'][i].tobytes() != ep['y_true'][i].tobytes())
print(f'Integrity: te_idx identical={te_eq} | y_true byte-level mismatches {yt_miss}/13547', flush=True)
assert te_eq and yt_miss == 0, 'returned npz is not from the same source as ep, stopping!'

yt = ep['y_true'].astype(np.float64)
pred_sm = r5['mean'].astype(np.float64)
pred_3d = ep['mean'].astype(np.float64)

def r2_mae(y, p):
    return float(r2(y, p)), float(np.abs(y - p).mean())

def r2(y, p):
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

out = {'per_prop': {}}
print(f'\n{"Property":14s} {"3D R²(raw)":>10s} {"SM R²(raw)":>10s} {"Δ":>7s} | {"3D R²(clip)":>10s} {"SM R²(clip)":>10s} {"Δ":>7s} | Wilcoxon(paired |err|, two-sided p)', flush=True)
for i, pn in enumerate(PROPS):
    y = yt[:, i]
    p3_raw, ps_raw = pred_3d[:, i], pred_sm[:, i]
    r3_raw, m3_raw = r2_mae(y, p3_raw)
    rs_raw, ms_raw = r2_mae(y, ps_raw)
    clip = CLIP.get(pn, False)
    if clip:
        p3, ps = np.maximum(p3_raw, 0), np.maximum(ps_raw, 0)
    else:
        p3, ps = p3_raw, ps_raw
    r3, m3 = r2_mae(y, p3)
    rs, ms = r2_mae(y, ps)
    w = stats.wilcoxon(np.abs(y - p3), np.abs(y - ps), alternative='two-sided')
    sm_win = rs > r3
    out['per_prop'][pn] = {
        'r2_3d_raw': r3_raw, 'r2_sm_raw': rs_raw,
        'r2_3d_clip': r3, 'r2_sm_clip': rs,
        'mae_3d_clip': m3, 'mae_sm_clip': ms,
        'wilcoxon_p': float(w.pvalue), 'wilcoxon_stat': float(w.statistic),
        'sm_wins': sm_win, 'delta_clip': rs - r3}
    print(f'{pn:14s} {r3_raw:10.4f} {rs_raw:10.4f} {rs_raw-r3_raw:+7.4f} | {r3:10.4f} {rs:10.4f} {rs-r3:+7.4f} | p={w.pvalue:.3g} {"SMILES wins" if sm_win else "3D wins"}', flush=True)

mr3 = float(np.mean([v['r2_3d_clip'] for v in out['per_prop'].values()]))
mrs = float(np.mean([v['r2_sm_clip'] for v in out['per_prop'].values()]))
n_win = sum(1 for v in out['per_prop'].values() if v['sm_wins'])
out['mean_r2_3d_clip'] = mr3
out['mean_r2_sm_clip'] = mrs
out['sm_wins_n'] = n_win
print(f'\nMean R² (clip convention): 3D={mr3:.4f}  SMILES={mrs:.4f}  Δ={mrs-mr3:+.4f}  SMILES wins {n_win}/8', flush=True)

# ---- checkpoint-convention cross-check (context of the deployed checkpoint 0.932) ----
print(f'\ncheckpoint-convention cross-check: deployed model 0.932 (paper) vs R5 same-protocol SMILES ensemble {mrs:.4f} (Δ {(0.932-mrs):+.4f})', flush=True)
print(f'3D ensemble paper value 0.903 vs recomputed by this script {mr3:.4f} (Δ {(0.903-mr3):+.4f})', flush=True)

with open('r5_vs_3d_paired.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSAVED r5_vs_3d_paired.json', flush=True)
