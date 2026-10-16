# -*- coding: utf-8 -*-
"""R2a: MDES 功效分析 — 30k 消融 3-seed 配对设计的最小可检测效应量
配对 t 检验: MDES = (t_{1-a/2} + t_{1-b}) * s / sqrt(n), n=3 seeds (df=2)
per-seed 配对差从 ablation_full.ipynb cell outputs 重建"""
import json, sys
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
alpha, power = 0.05, 0.80
t_a = stats.t.ppf(1 - alpha / 2, df=2)   # 4.303
t_b = stats.t.ppf(power, df=2)           # 0.816
print(f't_crit(α/2, df=2)={t_a:.3f}, t_power(df=2)={t_b:.3f}')

# per-seed 配对差 (treat - ctrl 按论文口径; mean R² across 8 props, 3 seeds)
# 来源: ablation_full.ipynb cell 3-8 outputs
deltas = {
    'A_MetalPool': [0.8382-0.8162, 0.8362-0.8257, 0.8338-0.8220],
    'B_Heads':     [0.8347-0.8382, 0.8470-0.8362, 0.8408-0.8338],
    'C_Weight':    [0.8410-0.8382, 0.8415-0.8362, 0.8364-0.8338],
    'D_DFTEdges':  [0.8285-0.8382, 0.8464-0.8362, 0.8417-0.8338],
    'E_RBF':       [0.8444-0.8382, 0.8328-0.8362, 0.8384-0.8338],
    'F_MayerBO':   [0.8430-0.8382, 0.8440-0.8362, 0.8435-0.8338],
}
# F 消融的性质级 3-seed std (cell 8 输出: full_BO / BO_free 的 ±seed std)
F_prop_std = {
    'gap':          (0.0084, 0.0066),
    'homo':         (0.0051, 0.0068),
    'lumo':         (0.0071, 0.0076),
    'dipole':       (0.0061, 0.0182),
    'metal_charge': (0.0019, 0.0068),
    'E_elec':       (0.0097, 0.0033),
    'E_disp':       (0.0066, 0.0114),
    'polarizability': (0.0081, 0.0149),
}

out = {}
print(f'\n{"Exp":15s} {"Δmean":>8s} {"s(3seed)":>9s} {"t":>7s} {"p":>7s} {"MDES80":>8s}')
for k, d in deltas.items():
    d = np.array(d)
    s = d.std(ddof=1)
    t = d.mean() / (s / np.sqrt(3))
    p = 2 * (1 - stats.t.cdf(abs(t), df=2))
    mdes = (t_a + t_b) * s / np.sqrt(3)
    out[k] = {'delta_mean': float(d.mean()), 'seed_std': float(s),
              't': float(t), 'p': float(p), 'mdes_80': float(mdes),
              'per_seed': [float(x) for x in d]}
    print(f'{k:15s} {d.mean():+8.4f} {s:9.4f} {t:7.2f} {p:7.3f} {mdes:8.4f}')

# F 性质级 MDES (配对差 seed std ≈ sqrt(s1²+s2²), 保守)
print(f'\nF_MayerBO 性质级 (3 seeds):')
for prop, (s1, s2) in F_prop_std.items():
    sd = np.sqrt(s1**2 + s2**2)
    mdes = (t_a + t_b) * sd / np.sqrt(3)
    out[f'F_prop_{prop}'] = {'paired_std': float(sd), 'mdes_80': float(mdes)}
    print(f'  {prop:14s} s={sd:.4f}  MDES80={mdes:.4f}')

# 前瞻: 全尺度 90k, n=9 配对单元 (3 folds × 3 seeds), df=8
t_a8 = stats.t.ppf(1 - alpha / 2, df=8)  # 2.306
t_b8 = stats.t.ppf(power, df=8)          # 0.889
print(f'\n前瞻 (3 folds × 3 seeds = 9 单元, df=8): t_crit={t_a8:.3f}, t_pow={t_b8:.3f}')
for prop, (s1, s2) in F_prop_std.items():
    sd = np.sqrt(s1**2 + s2**2)
    mdes = (t_a8 + t_b8) * sd / np.sqrt(9)
    out[f'F9_prop_{prop}'] = {'paired_std': float(sd), 'mdes_80': float(mdes)}
    print(f'  {prop:14s} s={sd:.4f}  MDES80={mdes:.4f}')

with open('d:/ml/chem_ml5/r2a_mdes.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved -> r2a_mdes.json')
