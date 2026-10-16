# -*- coding: utf-8 -*-
"""R5 (SMILES 9 成员) vs 3D 集成 —— 同协议配对检验 (vA 域 13,547 测试集)
1. 完整性: R5 npz 的 te_idx/y_true vs ensemble_predictions.npz 逐字节
2. 同口径逐性质 R²/MAE: 原始 + 截断 (意见 7 报告口径: dipole/gap max(0,·))
3. 配对检验: per-molecule |err| Wilcoxon signed-rank (双尾)
4. 集成口径: 9 成员 mean 预测
输出: r5_vs_3d_paired.json (所有数字)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from scipy import stats

PROPS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
CLIP = {'gap': True, 'dipole': True}  # 意见 7 报告口径: 非负性质截断

r5 = np.load('r5_smiles_va_ensemble.npz')
ep = np.load('ensemble_predictions.npz')

# ---- 1. 完整性 ----
te_eq = np.array_equal(r5['te_idx'], ep['te_idx'])
yt_miss = sum(1 for i in range(13547)
              if r5['y_true'][i].tobytes() != ep['y_true'][i].tobytes())
print(f'完整性: te_idx 全等={te_eq} | y_true 字节级不一致 {yt_miss}/13547', flush=True)
assert te_eq and yt_miss == 0, '回传 npz 与 ep 不同源, 停!'

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
print(f'\n{"性质":14s} {"3D R²(raw)":>10s} {"SM R²(raw)":>10s} {"Δ":>7s} | {"3D R²(clip)":>10s} {"SM R²(clip)":>10s} {"Δ":>7s} | Wilcoxon(配对|err|, 双尾 p)', flush=True)
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
    print(f'{pn:14s} {r3_raw:10.4f} {rs_raw:10.4f} {rs_raw-r3_raw:+7.4f} | {r3:10.4f} {rs:10.4f} {rs-r3:+7.4f} | p={w.pvalue:.3g} {"SMILES胜" if sm_win else "3D胜"}', flush=True)

mr3 = float(np.mean([v['r2_3d_clip'] for v in out['per_prop'].values()]))
mrs = float(np.mean([v['r2_sm_clip'] for v in out['per_prop'].values()]))
n_win = sum(1 for v in out['per_prop'].values() if v['sm_wins'])
out['mean_r2_3d_clip'] = mr3
out['mean_r2_sm_clip'] = mrs
out['sm_wins_n'] = n_win
print(f'\nMean R² (clip 口径): 3D={mr3:.4f}  SMILES={mrs:.4f}  Δ={mrs-mr3:+.4f}  SMILES 胜 {n_win}/8', flush=True)

# ---- checkpoint 口径对照 (部署 checkpoint 0.932 的语境) ----
print(f'\ncheckpoint 口径对照: 部署模型 0.932 (论文) vs R5 同协议 SMILES 集成 {mrs:.4f} (Δ {(0.932-mrs):+.4f})', flush=True)
print(f'3D 集成论文值 0.903 vs 本脚本重算 {mr3:.4f} (Δ {(0.903-mr3):+.4f})', flush=True)

with open('r5_vs_3d_paired.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSAVED r5_vs_3d_paired.json', flush=True)
