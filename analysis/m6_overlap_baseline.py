# -*- coding: utf-8 -*-
"""M6: Zero baseline for the XAI attention-perturbation top-5 overlap (18.1%)
Protocol reconstruction (xai_rigorous.ipynb):
  - XAI subset: rng=RandomState(42), graphs built in permutation(108,541) order (MolFromSmiles) up to 20,000
  - Perturbation subset: perturb_idx = RandomState(123).choice(20,000, 2,000, replace=False)
  - Overlap rate = |attn_top5 ∩ pert_top5| / 5, observed mean 0.181 (std 0.184)
Zero baseline: if pert top-5 is independent of attn and is a uniform random 5-subset, E[rate] = 5/N_i (N_i = number of atoms)
  Per complex |A∩B| ~ Hypergeometric(N_i, 5, 5): E=25/N_i, Var=5*(5/N)*(1-5/N)*(N-5)/(N-1)
  Pooled z-test (2000 independent complexes). No model inference required.
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

# Reconstruct the 20,000 XAI subset (same graph-building criterion as the notebook: SMILES non-empty and parseable by MolFromSmiles)
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

# Zero baseline
mu_i = 5.0 / N_p                      # E[rate_i]
mu0 = mu_i.mean()                     # Expected mean overlap rate under the null hypothesis
var_i = 5.0 * (5.0/N_p) * (1 - 5.0/N_p) * (N_p - 5) / (N_p - 1)  # Var[|A∩B|], variance of the rate = var_i/25
var_rate_i = var_i / 25.0
se = np.sqrt(var_rate_i.sum()) / len(N_p)
obs = 0.181
z = (obs - mu0) / se
from scipy.stats import norm
p = 1 - norm.cdf(z)
print(f'\nZero baseline (independent uniform random 5-subsets, same 2,000 graphs):')
print(f'  E[overlap rate] = {mu0*100:.2f}%  (5/N_i, mean N_i={N_p.mean():.1f})')
print(f'  observed 18.1% vs zero baseline {mu0*100:.2f}%  → exceeds by {100*(obs-mu0):.1f} pp')
print(f'  z = {z:.2f}, p = {p:.4f}')

# Stricter baseline: carbon-fraction matched (73.4% of attn top-5 are C) -- for discussion only, not written into the main text
out = {'N_xai': built, 'N_perturb': len(N_p), 'mean_atoms': float(N_p.mean()),
       'baseline_rate': float(mu0), 'observed_rate': obs, 'z': float(z), 'p': float(p),
       'caveat_rdkit': 'Rebuilt with the current RDKit 2026.03.3; if the original notebook environment was 2024.03, the odd SMILES parsing difference is negligible'}
with open('m6_overlap_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('\nSaved -> m6_overlap_baseline.json')
