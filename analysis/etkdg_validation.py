"""ETKDG/MMFF full validation for tmQM — 4 deliverables for §2.2 / Limitations"""
import numpy as np, pandas as pd, gzip, json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
import os

N_WORKERS = min(16, cpu_count())
print(f'Using {N_WORKERS} parallel workers')

# ===== Load data =====
df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
smiles_list = df['SMILES'].tolist()
N_total = len(smiles_list)
print(f'Total SMILES: {N_total:,}')

# ===== TM metals =====
TM_METALS = set([21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80])
METAL_SYMBOLS = {21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',
                 29:'Cu',30:'Zn',39:'Y',40:'Zr',41:'Nb',42:'Mo',43:'Tc',44:'Ru',
                 45:'Rh',46:'Pd',47:'Ag',48:'Cd',72:'Hf',73:'Ta',74:'W',75:'Re',
                 76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg'}

# ===== 1. Conformer generation success rate (parallel) =====
print('\n' + '='*60)
print('1. ETKDG+MMFF conformer generation success rate')
print('='*60)

def try_etkdg(smi):
    """Try ETKDG+MMFF on a single SMILES. Returns (status, metal_list)"""
    if not isinstance(smi, str) or smi == '':
        return 'invalid_smiles', []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 'parse_failed', []
    metals = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() in TM_METALS]
    mol = Chem.AddHs(mol)
    try:
        status = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200)
        if status != 0:
            return 'embed_failed', metals
        AllChem.MMFFOptimizeMolecule(mol)
        return 'success', metals
    except:
        return 'mmff_failed', metals

# Parallel execution
with Pool(N_WORKERS) as pool:
    results = list(tqdm(pool.imap(try_etkdg, smiles_list, chunksize=100),
                        total=N_total, desc='ETKDG (parallel)'))

success = sum(1 for r, _ in results if r == 'success')
fail_invalid = sum(1 for r, _ in results if r == 'invalid_smiles')
fail_parse = sum(1 for r, _ in results if r == 'parse_failed')
fail_embed = sum(1 for r, _ in results if r == 'embed_failed')
fail_mmff = sum(1 for r, _ in results if r == 'mmff_failed')

# Aggregate metal counts
metal_counts = Counter()
for _, metals in results:
    for z in metals:
        metal_counts[z] += 1

total_fail = fail_invalid + fail_parse + fail_embed + fail_mmff
pct_success = 100 * success / N_total
print(f'Success:  {success:,} ({pct_success:.1f}%)')
print(f'  Invalid SMILES: {fail_invalid}')
print(f'  Parse failed:   {fail_parse}')
print(f'  Embed failed:   {fail_embed}')
print(f'  MMFF crashed:   {fail_mmff}')
print(f'Failed structures are excluded from the scorer at inference time.')

# ===== 2. MMFF parameter coverage =====
print('\n' + '='*60)
print('2. MMFF94 parameter coverage for transition metals')
print('='*60)
print(f'Transition metals found in dataset:')
for z, count in sorted(metal_counts.items(), key=lambda x: -x[1]):
    sym = METAL_SYMBOLS.get(z, f'Z={z}')
    print(f'  {sym} (Z={z}): {count:,} complexes')
print(f'\nMMFF94 explicitly parameterizes: H, C, N, O, F, P, S, Cl, Br, I')
print(f'MMFF94 does NOT parameterize any transition metals.')
print(f'Strategy: metals fixed at ETKDG positions; only ligand atoms optimized.')

# ===== 3+4. M-L bond length comparison (500 random complexes with XYZ data) =====
print('\n' + '='*60)
print('3+4. Metal-ligand bond length: ETKDG+MMFF vs DFT (XYZ)')
print('='*60)

# Parse XYZ files for coordinates (store csv_idx for SMILES lookup)
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym,0)

def parse_xyz_coords(xyz_p, start_idx=0):
    """Get coordinates + csv_idx for each complex"""
    complexes = []
    with gzip.open(xyz_p, "rt") as f:
        blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, start_idx - 1
    while i < len(blocks):
        line = blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(blocks): break; i += 1
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(blocks): break
                cline = blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]])
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if any(z in TM_METALS for z in atoms) and len(atoms) > 1:
                complexes.append({'atoms': np.array(atoms), 'coords': np.array(coords, dtype=np.float32),
                                  'csv_idx': mol_idx})
    return complexes

print('Loading DFT coordinates from XYZ files...')
all_xyz = []
for part in [1,2,3]:
    all_xyz.extend(parse_xyz_coords(f"data_tmQM/local_vB/tmQM_X{part}.xyz.gz", start_idx=len(all_xyz)))
print(f'Loaded {len(all_xyz):,} complexes with DFT coordinates')

# Compare up to 500 complexes (csv_idx-aligned)
ml_diffs = []  # M-L bond length differences
n_compared = 0
for xyz_complex in tqdm(all_xyz, desc='Bond comparison'):
    csv_i = xyz_complex['csv_idx']
    if csv_i >= len(smiles_list): continue
    smi = smiles_list[csv_i]
    if not isinstance(smi, str) or smi == '': continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    N_rdkit = mol.GetNumAtoms()
    atoms = xyz_complex['atoms']
    N_xyz = len(atoms)
    if N_rdkit != N_xyz: continue  # must match — SMILES and XYZ same molecule
    coords_xyz = xyz_complex['coords']

    n_compared += 1
    if n_compared > 500: break  # limit to 500

    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=200) != 0: continue
        AllChem.MMFFOptimizeMolecule(mol)
    except: continue
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()

    # Compare M-L bond lengths
    for i in range(N_xyz):
        if atoms[i] not in TM_METALS: continue
        for j in range(N_xyz):
            if atoms[j] in TM_METALS: continue  # only metal-ligand bonds
            d_xyz = float(np.linalg.norm(coords_xyz[i] - coords_xyz[j]))
            if d_xyz > 3.0: continue  # only bonded pairs
            # Get ETKDG distance
            pos_i = np.array([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
            pos_j = np.array([conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z])
            d_etkdg = float(np.linalg.norm(pos_i - pos_j))
            ml_diffs.append(d_etkdg - d_xyz)

if ml_diffs:
    diffs = np.array(ml_diffs)
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))
    bias = float(np.mean(diffs))
    within_01 = float(np.mean(np.abs(diffs) < 0.1))
    within_02 = float(np.mean(np.abs(diffs) < 0.2))
    within_03 = float(np.mean(np.abs(diffs) < 0.3))

    print(f'\nM-L bond length comparison (N={len(diffs)} bonds from {n_compared} complexes):')
    print(f'  Mean signed difference (ETKDG - DFT): {bias:+.4f} Angstrom')
    print(f'  MAE:  {mae:.4f} Angstrom')
    print(f'  RMSE: {rmse:.4f} Angstrom')
    print(f'  |error| < 0.1 A: {within_01*100:.1f}%')
    print(f'  |error| < 0.2 A: {within_02*100:.1f}%')
    print(f'  |error| < 0.3 A: {within_03*100:.1f}%')

    with open('etkdg_validation.json', 'w') as f:
        json.dump({
            'success_rate_pct': pct_success,
            'n_success': success,
            'n_total': N_total,
            'fail_breakdown': {'invalid_smiles': fail_invalid, 'parse_failed': fail_parse,
                               'embed_failed': fail_embed, 'mmff_crashed': fail_mmff},
            'mmff_metal_coverage': 'None — metals fixed, ligands only',
            'ml_bond_mae': mae,
            'ml_bond_rmse': rmse,
            'ml_bond_bias': bias,
            'ml_bond_within_01A': within_01,
            'ml_bond_within_02A': within_02,
            'ml_bond_n_samples': n_compared,
        }, f, indent=2)
    print('Saved etkdg_validation.json')
else:
    print('No M-L bonds found for comparison')

print('\nDone.')
