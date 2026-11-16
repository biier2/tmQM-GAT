# -*- coding: utf-8 -*-
"""Generate vA_domain_smiles.pt locally — the format R5 expects {graphs, y_all, atoms_filt}
Input: kept_cache.npz (csv_idx+y of the 90,335 kept) + vA_removed_kept.npy (18 removed positions)
     + local tmQM_y.csv (3-17 version, md5 f8eeb967...f043) + ensemble_predictions.npz
Graph building: node 8d / edge 6d — SMILES featurisation bit-identical to build_vA_domain.py (R3/R5 protocol)
atoms_filt: per-molecule heavy-atom z sequence (metal weights use only z>=21, equivalent to the z sequence from the xyz parsing)
End-to-end self-check: saved only if te_idx is exactly equal + y_true bit-identical 0/13,547 + graph keys/shapes + the R5 allclose gate all pass
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, hashlib
import torch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
print('rdkit version:', Chem.rdBase.rdkitVersion, flush=True)

ROOT = 'data_tmQM/local_vB'
df_y = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df_y = df_y.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
    'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
smiles_by_idx = df_y['SMILES'].tolist()
y_csv = df_y[targets].astype(float).values  # float64 original values

# ---- Verified kept sequence: 90,335 - 18 = 90,317 ----
z = np.load('kept_cache.npz')
mseq = z['csv_idx'].copy()          # csv_idx (mol_idx) for the 90,335 rows (notebook kept sequence)
rem = np.load('vA_removed_kept.npy')  # 18 removed positions (in the original 90,335 space)
keep = np.ones(len(mseq), dtype=bool)
keep[rem] = False
mseq = mseq[keep]
assert len(mseq) == 90317, f'expected 90317, got {len(mseq)}'
print(f'kept sequence: {len(mseq):,} rows ({len(rem)} RDKit-version-difference rows removed)', flush=True)

# ---- SMILES graph construction (node 8d / edge 6d) — verbatim = build_vA_domain.py ----
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}

def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

print('Building SMILES graphs (node 8d / edge 6d)...', flush=True)
graphs, y_vals, atoms_filt, failed = [], [], [], 0
for pos, c in enumerate(mseq):
    smi = smiles_by_idx[c]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        failed += 1; continue   # all 90,317 are notebook kept rows, so all should succeed locally; stop on failure
    N = mol.GetNumAtoms()
    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    sm_z = [xf[i][0] for i in range(N)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za >= 21) != (zb >= 21))
        is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * 4
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 6)
    graphs.append({'x': x, 'edge_index': ei, 'edge_attr': ea, 'num_nodes': N})
    y_vals.append(y_csv[c])
    atoms_filt.append(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64))
    if (pos + 1) % 20000 == 0:
        print(f'  {pos+1:,}/90,317', flush=True)

assert failed == 0, f'{failed} SMILES failed to parse locally — contradicts the notebook kept sequence, stopping'
y_all = torch.tensor(y_vals, dtype=torch.float32)
assert y_all.shape == (90317, 8), y_all.shape
mask = ~(torch.isnan(y_all).any(dim=1) | torch.isinf(y_all).any(dim=1))
assert mask.all(), f'{(~mask).sum()} rows have non-finite y — contradicts the notebook NaN removal of 0, stopping'
print(f'Built {len(graphs):,} graphs / y_all {tuple(y_all.shape)} / atoms_filt {len(atoms_filt):,}', flush=True)

# ---- Graph key/shape checks (item by item, R5 consumer side) ----
for i, g in enumerate(graphs):
    if g['x'].shape[1] != 8 or g['edge_attr'].shape[1] != 6 or g['edge_index'].shape[0] != 2:
        raise SystemExit(f'graph[{i}] shape anomaly: x{g["x"].shape} ea{g["edge_attr"].shape} ei{g["edge_index"].shape}')
    if g['edge_index'].shape[1] != g['edge_attr'].shape[0]:
        raise SystemExit(f'graph[{i}] edge count mismatch')
    if g['num_nodes'] != g['x'].shape[0]:
        raise SystemExit(f'graph[{i}] num_nodes mismatch')
print('graph key/shape checks passed (x:8d, ea:6d, ei:2xE, num_nodes)', flush=True)

# ---- Alignment self-check vs ensemble_predictions.npz (same gate as build_vA_pt_local + R5 allclose gate) ----
n = 90317
idx_all = np.random.RandomState(42).permutation(n)
te_idx = idx_all[:13547].copy()
ep = np.load('ensemble_predictions.npz')
te_ref = ep['te_idx']; yt_ref = ep['y_true'].astype(np.float32)
assert np.array_equal(te_idx, te_ref), 'te_idx not equal!'
miss = sum(1 for i in range(13547)
           if y_all[te_idx[i]].numpy().tobytes() != yt_ref[i].tobytes())
print(f'te_idx exactly equal: True | y_true bit-identical mismatches: {miss} / 13,547', flush=True)
assert miss == 0, 'bit-level mismatch — stopping'
assert np.allclose(y_all[te_idx].numpy(), yt_ref, atol=1e-4), 'R5 allclose(atol=1e-4) gate failed!'
print('R5 internal self-check rehearsal: np.array_equal(ep te_idx) + np.allclose(y_all[te_idx], y_true, atol=1e-4) all pass ★', flush=True)

# ---- Metal weight statistics rehearsal (R5: for atoms in atoms_filt: count z>=21) ----
mc = {}
for atoms in atoms_filt:
    for zz in set(atoms.tolist()):
        if zz >= 21: mc[zz] = mc.get(zz, 0) + 1
print('metal counts (used for R5 weights):', {k: mc[k] for k in sorted(mc)}, flush=True)

torch.save({'graphs': graphs, 'y_all': y_all, 'atoms_filt': atoms_filt}, 'vA_domain_smiles.pt')
print('SAVED vA_domain_smiles.pt (R5 format {graphs, y_all, atoms_filt})', flush=True)
print('md5:', hashlib.md5(open('vA_domain_smiles.pt', 'rb').read()).hexdigest(), flush=True)
print('R5 load: d = torch.load("vA_domain_smiles.pt", weights_only=False); graphs=d["graphs"]; y_all=d["y_all"]; atoms_filt=d["atoms_filt"]', flush=True)
