# -*- coding: utf-8 -*-
"""Enumerate and rebuild the paper B block order: B = A with the interval [s,e] extracted and inserted after q (2 places) — self-check to pin the boundaries
f(i) observation (diag_va_align2): jump 1 = 34,967->27,345 (part 1, mol_idx coordinates), jump 2 = 63,380->61,386 (part 2 buggy mol_idx)
Hypothesis 1: B = A with [s1..e1] extracted and inserted after 34,967 (s1≈27,345, e1≈34,966) + [s2..e2] extracted and inserted after 63,380 (s2≈61,386, e2≈63,379)
Simulate the paper pipeline (buggy table + start_idx bug + mol_idx position pairing) -> first 90,317 of the kept sequence -> y[te_idx] vs ep self-check
Enumerate boundary combinations and find the one with the fewest mismatching rows (expected 0).
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
smi = df['SMILES'].tolist()
n = len(df)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,
           "Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,
           "Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

# ===== 1. Block counts per part + TM mask (A block position = row position, cnt continuous globally) =====
N_parts, tm_mask = [], np.zeros(n, bool)
cnt = 0
for part in [1, 2, 3]:
    tm_cnt = 0
    with gzip.open(f'{ROOT}/tmQM_X{part}.xyz.gz', 'rt') as f:
        it = iter(f)
        while True:
            try:
                ln = next(it).strip()
            except StopIteration:
                break
            if not ln or not ln.isdigit():
                continue
            n_at = int(ln)
            next(it)
            atoms = []
            for _ in range(n_at):
                ln2 = next(it, None)
                if ln2 is None: break
                cparts = ln2.strip().split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        [float(x) for x in cparts[1:4]]
                        atoms.append(z)
                    except ValueError: continue
            tm_mask[cnt] = any(is_TM(z) for z in atoms) and len(atoms) > 1
            cnt += 1
            if tm_mask[cnt - 1]: tm_cnt += 1
    N_parts.append(cnt)
    print(f'Part {part}: blocks {N_parts[-1] - (N_parts[-2] if len(N_parts) > 1 else 0)} ({tm_cnt} TM)', flush=True)

# ===== 2. SMILES/y masks =====
parse_ok = np.zeros(n, bool)
for i, s in enumerate(smi):
    if isinstance(s, str) and s != '':
        parse_ok[i] = Chem.MolFromSmiles(s) is not None
finite_ok = np.isfinite(y).all(axis=1)
print(f'TM {tm_mask.sum():,}  parse {parse_ok.sum():,}  finite {finite_ok.sum():,}', flush=True)

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)

# ===== 3. Pipeline simulation: mol_idx (buggy version) sequence =====
# part 1: mol_idx = block position (0..N1-1)
# part 2: mol_idx = 28737 + (block position - N1)   (start_idx=28737 -> initial mol_idx=28736, 28737 after the first block is read)
# part 3: mol_idx = 64571 + (block position - N1 - N2)  (start_idx=28737+35834=64571)
N1, N2, N3 = N_parts
def mol_idx_of(blk):
    """A block position -> paper-pipeline mol_idx (buggy version)"""
    if blk < N1:
        return blk
    if blk < N1 + N2:
        return 28737 + (blk - N1)
    return 64571 + (blk - N1 - N2)

# ===== 4. Enumeration: B csv row order = A row order with [s1..e1] extracted and inserted after 34,967 + [s2..e2] inserted after 63,380 =====
# c[j] = content of B csv row j = A csv row number (label index)
def build_c(s1, e1, a1, s2, e2, a2):
    c = (list(range(s1)) + list(range(e1 + 1, a1 + 1)) + list(range(s1, e1 + 1))
         + list(range(a1 + 1, s2)) + list(range(e2 + 1, a2 + 1)) + list(range(s2, e2 + 1))
         + list(range(a2 + 1, n)))
    return c

def simulate(c):
    """Paper pipeline: A xyz block order -> buggy mol_idx -> label = B csv row mol_idx = A row c[mol_idx] -> first 90,317 -> self-check"""
    kept = []
    for blk in range(n):
        if not tm_mask[blk]: continue
        mi = mol_idx_of(blk)
        a_row = c[mi] if mi < n else -1
        if a_row < 0 or not (parse_ok[a_row] and finite_ok[a_row]): continue
        kept.append(a_row)
        if len(kept) == 90317: break
    if len(kept) < 90317: return -1
    yt = y[np.array(kept)][te_idx]
    d = np.abs(yt - yt_ref).max(axis=1)
    return int((d > 1e-4).sum())

print('\nCoarse enumeration (±3)...', flush=True)
best = []
for s1 in range(27342, 27349):
    for e1 in range(34962, 34967):   # e1 < a1
        for s2 in range(61383, 61390):
            for e2 in range(63376, 63380):   # e2 < a2
                c = build_c(s1, e1, 34967, s2, e2, 63380)
                bad = simulate(c)
                if 0 <= bad < 10000:   # only record clear improvements (excluding -1 and near-total failures)
                    best.append((bad, s1, e1, s2, e2))
                    print(f'  mismatches {bad:>5} / 13,547  (s1={s1}, e1={e1}, s2={s2}, e2={e2})', flush=True)
best.sort()
print(f'\nBest 5 combinations:')
for b in best[:5]:
    print(f'  {b}', flush=True)
