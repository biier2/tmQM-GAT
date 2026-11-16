# -*- coding: utf-8 -*-
"""DimeNet++ controlled baseline — data preprocessing (run on the server)

The parsing logic is aligned line by line with the 3D GAT pipeline (shared by tmQM_Fe_XYZ / 3D_GCN / build_ensemble):
  - xyz block-level traversal: mol_idx +1 for every block (including non-TM blocks), csv_idx = block index (0-based, aligned with df_y rows)
  - retention condition (paper mask, reproduced item by item): contains TM and >1 atom → SMILES non-empty and parseable → all 8 targets finite
  - domain alignment: paper main domain 90,317 = the first 90,317 of the retained sequence of the original tmQM file (102,077 TM) (truncated)
  - split: train/val = ensemble_split.npz paper fold 0 split (fold_0_train 51,180 / fold_0_val 25,590,
          the same set as paper 3D GAT fold 0); test set = te_idx (13,547, unchanged)
  - radius graph: r=5.0 Å per molecule, at most 32 neighbours per node (pure numpy, no torch_cluster needed)
  - z-score: 8 properties, training-set statistics
  - 3D input: tmQM original DFT XYZ (same source as the 3D GAT training)

⚠️ Data version: the original tmQM files under /home/chen_jianqi/zw/ml/chem_ml5/ on the server
(102,077 TM blocks, 18 July notebook anchor Total 102077 TM / Built 95,152) are the parent copy of the paper main domain;
the 96,984-TM xyz used in the 2-4 August main experiments was derived from it by deleting blocks, and the first 90,317 of the retained
sequence are bit-for-bit identical (evidence: local eval_smiles_gat.py reproduces the paper's 0.932 using the local original file + te_idx).
The script truncates to 90,317 automatically; run it in the same (tmqm) environment as 18 July / 4 August.

Output: dimenet_data.npz  (z/pos/edge_index/y sharded by train/val/test + split indices + normalisation statistics)
Dependencies: numpy pandas scipy rdkit (torch not required)
"""
import os, sys, gzip, time
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Fallback for the Windows GBK console; harmless on Linux
except Exception:
    pass
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# ============ Configurable options ============
CUTOFF = 5.0            # DimeNet++ default cutoff
MAX_NEIGHBORS = 32      # DimeNet++ default neighbour cap
# Split: ensemble_split.npz paper fold 0 (fold_0_train 51,180 / fold_0_val 25,590) + te_idx 13,547
# — the same train/val sets as paper 3D GAT fold 0 (the 30k sampling scheme was abandoned: the first-round 30k without regularisation overfitted severely, val R2≈0.09)
# ================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = ['tmQM_y.csv', 'ensemble_split.npz',
                  'tmQM_X1.xyz.gz', 'tmQM_X2.xyz.gz', 'tmQM_X3.xyz.gz']

def resolve_data_dir():
    """Data directory: 1st command-line argument > parent directory of the script (old layout inside chem_ml5) > chem_ml5 beside the parent
    (script at the same level as chem_ml5, e.g. server /home/.../zw/ml/dimenetpp_baseline/ + data in /home/.../zw/ml/chem_ml5/)
    > local data_tmQM/paper_domain_vA. Requires all 5 files in the same directory; if not found, SystemExit with clear guidance."""
    cands = []
    if len(sys.argv) > 1:
        cands.append(sys.argv[1])
    cands += [os.path.dirname(SCRIPT_DIR),
              os.path.join(os.path.dirname(SCRIPT_DIR), 'chem_ml5'),
              os.path.join(SCRIPT_DIR, '..', 'data_tmQM', 'paper_domain_vA')]
    for d in cands:
        d = os.path.normpath(d)
        if all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES):
            return d
    raise SystemExit(
        'Complete data files not found, checked:\n  ' + '\n  '.join(os.path.normpath(c) for c in cands) +
        '\nRequired in the same directory: ' + ', '.join(REQUIRED_FILES) +
        '\nUse the original tmQM files under /home/chen_jianqi/zw/ml/chem_ml5/ on the server directly'
        ' (102,077 TM version, retained sequence truncated to the first 90,317 = paper main domain);'
        '\nor specify the data directory with the 1st argument, e.g.: python preprocess_dimenet.py /home/chen_jianqi/zw/ml/chem_ml5')

DATA_DIR = resolve_data_dir()
print(f'Data directory: {DATA_DIR}', flush=True)
t0 = time.time()

# ===== Line-by-line aligned with build_ensemble.py Cell 3: element table / TM test =====
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

# ===== XYZ parsing (reproduces build_ensemble.parse_one, BO matrix omitted) =====
def parse_one(xyz_p, start_idx=0):
    mols, labels = [], []
    with gzip.open(os.path.join(DATA_DIR, xyz_p), "rt") as f:
        xyz_blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break
            i += 1  # comment line
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]])
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1                      # +1 per block, including non-TM blocks (consistent with the paper)
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"atoms": atoms, "coords": np.array(coords, dtype=np.float32),
                         "csv_idx": mol_idx})
            labels.append(df_y.iloc[mol_idx][targets].astype(float).values)
    return mols, labels

# ===== Labels =====
df_y = pd.read_csv(os.path.join(DATA_DIR, 'tmQM_y.csv'), sep=';')
df_y = df_y.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                            'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                            'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                            'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
smiles_by_idx = df_y['SMILES'].tolist()

all_m, all_l = [], []
for part in [1, 2, 3]:
    m, l = parse_one(f"tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l)
    print(f'  Part {part}: {len(m)} TM complexes', flush=True)
print(f'Total parsed: {len(all_m)} TM complexes  ({time.time()-t0:.0f}s)', flush=True)

# ===== Paper 90,317 mask: SMILES non-empty and parseable + all y finite =====
# The 90,317 graph-building set of the paper main domain = the first 90,317 of the parse-retained sequence of the original tmQM file (102,077 TM blocks).
# The xyz used in the 2-4 August main experiments was derived from the original file by deleting blocks (96,984 TM blocks); its retained sequence is
# bit-for-bit identical to the first 90,317 of the original file's retained sequence — evidence: local eval_smiles_gat.py re-runs the
# SMILES checkpoint using the local original file + te_idx and obtains 0.932, exactly matching the final paper number.
kept_z, kept_pos, kept_y, kept_tmpos = [], [], [], []
n_failed_smiles, n_nan = 0, 0
for tm_pos, (m, y) in enumerate(zip(all_m, all_l), start=1):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': n_failed_smiles += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: n_failed_smiles += 1; continue
    if not (np.isfinite(y).all()): n_nan += 1; continue
    kept_z.append(np.asarray(m['atoms'], dtype=np.int64))
    kept_pos.append(m['coords'])
    kept_y.append(y)
    kept_tmpos.append(tm_pos)

print(f'Parse-retained: {len(kept_z)}  (SMILES failures {n_failed_smiles}, NaN {n_nan})  ({time.time()-t0:.0f}s)', flush=True)
# --- Align with the paper main domain (90,317) ---
# Original-file anchors (18 July server tmQM_Fe_XYZ.ipynb): Total 102077 TM / Built 95,152 (6925 failed)
# Truncation anchor (measured on the server, 2026-08-28): TM block index of the 90,317th retained molecule = 97,045
SERVER_TM = 102077
SERVER_KEPT = 95152
SERVER_ANCHOR_TMPOS = 97045
if len(kept_z) == 90317:
    print('[verify] OK: retained count is exactly 90,317 — directly aligned with ensemble_split.npz', flush=True)
elif len(kept_z) > 90317:
    print(f'[verify] retained {len(kept_z)} > 90,317; truncating to the prefix (paper main domain = the first 90,317 of the original file retained sequence). '
          f'TM block index of the 90,317th retained molecule = {kept_tmpos[90316]} (expected {SERVER_ANCHOR_TMPOS})', flush=True)
    assert len(all_m) == SERVER_TM and len(kept_z) == SERVER_KEPT and kept_tmpos[90316] == SERVER_ANCHOR_TMPOS, (
        f'Data/environment inconsistent with the paper main-domain parent copy: total TM {len(all_m)} (expected {SERVER_TM}), '
        f'retained {len(kept_z)} (expected {SERVER_KEPT}), anchor TM block index {kept_tmpos[90316]} (expected {SERVER_ANCHOR_TMPOS}).\n'
        f'If the difference is only a single digit, it is most likely an rdkit version difference (local rdkit2024 differs from the server by about 19 molecules of parsing results); '
        f'please re-run in the same (tmqm) environment as the 18 July / 4 August server runs.\n'
        f'Use the tmQM_X*.xyz.gz + tmQM_y.csv under /home/chen_jianqi/zw/ml/chem_ml5/ on the server directly.')
    kept_z = kept_z[:90317]
    kept_pos = kept_pos[:90317]
    kept_y = kept_y[:90317]
    print('[verify] OK: set size after prefix truncation = 90,317 (paper graph-building set)', flush=True)
else:
    raise RuntimeError(f'retained count {len(kept_z)} < 90,317 — data files incomplete or wrong version.')

# ===== Split alignment: ensemble_split.npz indices are relative to this 90,317 sequence =====
# Train/val use the paper fold 0 split directly (same set as 3D GAT fold 0); the test set te_idx 13,547 is unchanged
split = np.load(os.path.join(DATA_DIR, 'ensemble_split.npz'))
te_idx = split['te_idx'].astype(np.int64)
train_sel = split['fold_0_train'].astype(np.int64)
val_sel = split['fold_0_val'].astype(np.int64)
test_sel = te_idx
assert te_idx.max() < len(kept_z) and train_sel.max() < len(kept_z) and val_sel.max() < len(kept_z)
assert len(train_sel) == 51180 and len(val_sel) == 25590, (
    f'abnormal fold_0 split size: train {len(train_sel)} (expected 51,180) / val {len(val_sel)} (expected 25,590)')
print(f'Train {len(train_sel)} / Val {len(val_sel)} / Test {len(test_sel)} (paper fold 0 split, te_idx unchanged)', flush=True)
print('[verify] OK: train/val = same sets as paper 3D GAT fold 0 (51,180/25,590); test set = 13,547', flush=True)

y_all = np.array(kept_y, dtype=np.float32)
y_mean = y_all[train_sel].mean(axis=0).astype(np.float32)
y_std = (y_all[train_sel].std(axis=0) + 1e-8).astype(np.float32)

# ===== Radius graph (pure numpy: r=CUTOFF, at most MAX_NEIGHBORS neighbours per node, consistent with PyG radius_graph) =====
def radius_graph_np(pos, r, max_n):
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    src, dst = [], []
    for i in range(n):
        row = d[i]
        cand = np.where(row < r)[0]
        if len(cand) > max_n:
            cand = cand[np.argsort(row[cand])[:max_n]]
        for j in cand:
            src.append(i); dst.append(j)
    return np.array([src, dst], dtype=np.int32) if src else np.zeros((2, 0), dtype=np.int32)

def build_split(sel, tag):
    z_arr = np.empty(len(sel), dtype=object)
    pos_arr = np.empty(len(sel), dtype=object)
    ei_arr = np.empty(len(sel), dtype=object)
    n_z0 = n_iso = 0
    for k, i in enumerate(sel):
        z = kept_z[i]; pos = kept_pos[i]
        if (z == 0).any(): n_z0 += 1
        ei = radius_graph_np(pos, CUTOFF, MAX_NEIGHBORS)
        if ei.shape[1] > 0:
            deg = np.bincount(ei[0], minlength=len(z))
            if (deg == 0).any(): n_iso += 1
        else:
            n_iso += 1
        z_arr[k] = z; pos_arr[k] = pos; ei_arr[k] = ei
    print(f'  [{tag}] {len(sel)} molecules; molecules with z=0 atoms = {n_z0}; molecules with isolated nodes = {n_iso}', flush=True)
    return z_arr, pos_arr, ei_arr

print('Building radius graphs (only the molecules needed for train/val/test)...', flush=True)
z_tr, pos_tr, ei_tr = build_split(train_sel, 'train')
z_va, pos_va, ei_va = build_split(val_sel, 'val')
z_te, pos_te, ei_te = build_split(test_sel, 'test')

np.savez_compressed(
    os.path.join(SCRIPT_DIR, 'dimenet_data.npz'),
    z_train=z_tr, pos_train=pos_tr, edge_train=ei_tr, y_train=y_all[train_sel],
    z_val=z_va, pos_val=pos_va, edge_val=ei_va, y_val=y_all[val_sel],
    z_test=z_te, pos_test=pos_te, edge_test=ei_te, y_test=y_all[test_sel],
    train_sel=train_sel, val_sel=val_sel, test_sel=test_sel,
    y_mean=y_mean, y_std=y_std, targets=np.array(targets),
    cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS, train_n=len(train_sel), val_n=len(val_sel),
)
print(f'saved: dimenet_data.npz  ({time.time()-t0:.0f}s total)', flush=True)
print(f'[verify] set size = 90,317 | test set = {len(test_sel)} | train/val = {len(train_sel)}/{len(val_sel)}', flush=True)
