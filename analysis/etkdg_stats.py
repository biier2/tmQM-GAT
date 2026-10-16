"""ETKDG/MMFF statistics for tmQM complexes — CPU only, ~30min"""
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from tqdm import tqdm
import json

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
smiles_list = df['SMILES'].tolist()
N = len(smiles_list)

# ===== 1. Conformer generation success rate =====
print(f'Testing ETKDG conformer generation on {N:,} SMILES...')
success, fail = 0, 0
fail_reasons = []
for smi in tqdm(smiles_list, desc='ETKDG'):
    if not isinstance(smi, str) or smi == '':
        fail += 1; fail_reasons.append('invalid_smiles')
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        fail += 1; fail_reasons.append('parse_failed')
        continue
    mol = Chem.AddHs(mol)
    try:
        status = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200)
        if status != 0:
            fail += 1; fail_reasons.append('embed_failed')
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        success += 1
    except:
        fail += 1; fail_reasons.append('mmff_failed')

pct = 100 * success / N
print(f'\nETKDG+MMFF Results:')
print(f'  Success: {success:,} ({pct:.1f}%)')
print(f'  Fail: {fail:,} ({100-pct:.1f}%)')

# ===== 2. MMFF metal coverage =====
tm_metals = set([21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80])
metals_found = set()
for smi in smiles_list[:1000]:
    if not isinstance(smi, str) or smi == '': continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z in tm_metals:
            metals_found.add(z)
print(f'\nTransition metals found in dataset: {sorted(metals_found)}')
print(f'MMFF94 covers: H,C,N,O,F,P,S,Cl,Br,I')
print(f'MMFF94 does NOT cover: transition metals (treated as fixed points during optimization)')

# ===== 3. Quick M-L bond length check (on 100 random successes) =====
# This requires XYZ files — skip for now
print(f'\nFor full M-L bond length comparison, run with XYZ files loaded.')

with open('etkdg_stats.json', 'w') as f:
    json.dump({'total': N, 'success': success, 'fail': fail, 'pct': pct}, f, indent=2)
print('Saved etkdg_stats.json')
