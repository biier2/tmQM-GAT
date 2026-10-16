# -*- coding: utf-8 -*-
"""M6: XAI attention-perturbation top-5 重叠 (18.1%) 的零基线
协议重建 (xai_rigorous.ipynb):
  - XAI 子集: rng=RandomState(42), permutation(108,541) 顺序建图 (MolFromSmiles) 至 20,000 个
  - 扰动子集: perturb_idx = RandomState(123).choice(20,000, 2,000, replace=False)
  - 重叠率 = |attn_top5 ∩ pert_top5| / 5, 实测均值 0.181 (std 0.184)
零基线: 若 pert top-5 与 attn 独立且为均匀随机 5-子集, E[率] = 5/N_i (N_i=原子数)
  逐配合物 |A∩B| ~ Hypergeometric(N_i, 5, 5): E=25/N_i, Var=5*(5/N)*(1-5/N)*(N-5)/(N-1)
  汇总 z 检验 (2000 个独立配合物)。无需任何模型推理。
"""
import sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

df_y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
smiles_by_idx = df_y['SMILES'].tolist()
print(f'csv rows: {len(smiles_by_idx):,}')

# 重建 20,000 XAI 子集 (与 notebook 相同的建图判据: SMILES 非空且 MolFromSmiles 可解析)
N_XAI = 20000
rng = np.random.RandomState(42)
indices = rng.permutation(len(df_y))
nums, csv_idx_list, built = [], [], 0
for csv_i in indices:
    if built >= N_XAI: break
    smi = smiles_by_idx[csv_i]
    if not isinstance(smi, str) or smi == '': continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    nums.append(mol.GetNumAtoms()); csv_idx_list.append(int(csv_i)); built += 1
print(f'built {built:,} graphs; mean atoms {np.mean(nums):.1f} +/- {np.std(nums):.1f}')

perturb_idx = np.random.RandomState(123).choice(len(nums), 2000, replace=False)
N_p = np.array(nums)[perturb_idx]
print(f'perturb subset: n={len(N_p)}, mean N={N_p.mean():.2f}, median N={np.median(N_p):.1f}, range [{N_p.min()},{N_p.max()}]')

# 零基线
mu_i = 5.0 / N_p                      # E[rate_i]
mu0 = mu_i.mean()                     # 零假设下的期望平均重叠率
var_i = 5.0 * (5.0/N_p) * (1 - 5.0/N_p) * (N_p - 5) / (N_p - 1)  # Var[|A∩B|], rate 的方差 = var_i/25
var_rate_i = var_i / 25.0
se = np.sqrt(var_rate_i.sum()) / len(N_p)
obs = 0.181
z = (obs - mu0) / se
from scipy.stats import norm
p = 1 - norm.cdf(z)
print(f'\n零基线 (独立均匀随机 5-子集, 同 2,000 图):')
print(f'  E[overlap rate] = {mu0*100:.2f}%  (5/N_i, N_i={N_p.mean():.1f} 均值)')
print(f'  实测 18.1% vs 零基线 {mu0*100:.2f}%  → 超出 {100*(obs-mu0):.1f} pp')
print(f'  z = {z:.2f}, p = {p:.4f}')

# 更严基线: 碳占比匹配 (attn top-5 73.4% 为 C) —— 仅作讨论用, 不写入正文
out = {'N_xai': built, 'N_perturb': len(N_p), 'mean_atoms': float(N_p.mean()),
       'baseline_rate': float(mu0), 'observed_rate': obs, 'z': float(z), 'p': float(p),
       'caveat_rdkit': '当前 RDKit 2026.03.3 重建; notebook 原环境若为 2024.03 则个别 SMILES 解析差异可忽略'}
with open('m6_overlap_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('\nSaved -> m6_overlap_baseline.json')
