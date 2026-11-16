# -*- coding: utf-8 -*-
"""RF / XGBoost baseline re-evaluation — paper-standard 13,547 test-set basis (run on the server, do not run locally)

Background (round-4 supervisor comment 5): the old RF/XGB baselines were evaluated on the 15,126 test
set of their own 70/15/15 split of the 100,849 domain, a different basis from the GNN 13,547 test set.
Hard evidence from baseline_results.pt (gcn/gin preds
shape=(13547,8)) and build_3dgcn.py (reads the same ensemble_split.npz) confirms: GCN/GIN/3D GCN were all already evaluated on
13,547, so no re-evaluation is needed for them; only RF/XGB need re-evaluation.

New basis (consistent with the new Table 2 caption):
  train = cv_idx of ensemble_split.npz (76,770 = complement of the 13,547 test set within the 90,317 domain)
  test = te_idx of ensemble_split.npz (13,547, exactly the same as all GNNs)
  features = ECFP4 (radius=2, 2048-bit) — consistent with the old baselines
  model = original configuration of the old baselines, trained independently per property:
        RF  = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=5,
                                    max_features='sqrt', random_state=42, n_jobs=16)
        XGB = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                           reg_lambda=1.0, random_state=42, n_jobs=16)

Domain alignment: line-by-line identical to preprocess_dimenet.py (102,077 TM blocks → retained 95,152 → truncated 90,317,
anchor: TM block index of the 90,317th retained molecule = 97,045). Coordinates do not enter the retention decision
(it depends only on atom types / TM test / block index / SMILES / y), so coordinate storage is skipped during parsing; the domain is exactly identical.

Output (script directory): benchmark_reval_results.json / .csv / benchmark_reval_preds.npz
Dependencies: numpy pandas rdkit scikit-learn xgboost (torch not required)
Estimated runtime: single-property RF ≈ 15-30 minutes (500 trees); 8 properties × 2 models about 2-6 hours in total
Usage: python reval_benchmark_rf_xgb.py [data directory]
      the data directory defaults to the resolve_data_dir candidate order of preprocess_dimenet.py
      (original tmQM files under /home/chen_jianqi/zw/ml/chem_ml5/ on the server, 102,077 TM block version)
"""
import os, sys, gzip, time, json
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Fallback for the Windows GBK console; harmless on Linux
except Exception:
    pass
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

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
        '\nor specify the data directory with the 1st argument, e.g.: python reval_benchmark_rf_xgb.py /home/chen_jianqi/zw/ml/chem_ml5')

DATA_DIR = resolve_data_dir()
print(f'Data directory: {DATA_DIR}', flush=True)
t0 = time.time()

# ===== Line-by-line aligned with preprocess_dimenet.py: element table / TM test =====
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

# ===== XYZ parsing (reproduces preprocess_dimenet.parse_one; coordinates do not enter the retention decision and are not stored) =====
def parse_one(xyz_p, start_idx=0):
    mols = []
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
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        # Coordinate parsing only for line-by-line alignment (skip the atom on ValueError, consistent with preprocess)
                        [float(x) for x in cparts[1:4]]
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1                      # +1 per block, including non-TM blocks (consistent with the paper)
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"atoms": atoms, "csv_idx": mol_idx})
    return mols

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
    m = parse_one(f"tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m)
    all_l.extend([df_y.iloc[x["csv_idx"]][targets].astype(float).values for x in m])
    print(f'  Part {part}: {len(m)} TM complexes', flush=True)
print(f'Total parsed: {len(all_m)} TM complexes  ({time.time()-t0:.0f}s)', flush=True)

# ===== Paper 90,317 mask: SMILES non-empty and parseable + all y finite =====
# (line-by-line identical to preprocess_dimenet.py; same anchor checks, with clear guidance on failure)
kept_smiles, kept_y, kept_tmpos = [], [], []
n_failed_smiles, n_nan = 0, 0
for tm_pos, (m, y) in enumerate(zip(all_m, all_l), start=1):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': n_failed_smiles += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: n_failed_smiles += 1; continue
    if not (np.isfinite(y).all()): n_nan += 1; continue
    kept_smiles.append(smi)
    kept_y.append(y)
    kept_tmpos.append(tm_pos)

print(f'Parse-retained: {len(kept_smiles)}  (SMILES failures {n_failed_smiles}, NaN {n_nan})  ({time.time()-t0:.0f}s)', flush=True)
# --- Align with the paper main domain (90,317) ---
# Original-file anchors (18 July server tmQM_Fe_XYZ.ipynb): Total 102077 TM / Built 95,152 (6925 failed)
# Truncation anchor (measured on the server, 2026-08-28): TM block index of the 90,317th retained molecule = 97,045
SERVER_TM = 102077
SERVER_KEPT = 95152
SERVER_ANCHOR_TMPOS = 97045
if len(kept_smiles) == 90317:
    print('[verify] OK: retained count is exactly 90,317 — directly aligned with ensemble_split.npz', flush=True)
elif len(kept_smiles) > 90317:
    print(f'[verify] retained {len(kept_smiles)} > 90,317; truncating to the prefix (paper main domain = the first 90,317 of the original file retained sequence). '
          f'TM block index of the 90,317th retained molecule = {kept_tmpos[90316]} (expected {SERVER_ANCHOR_TMPOS})', flush=True)
    assert len(all_m) == SERVER_TM and len(kept_smiles) == SERVER_KEPT and kept_tmpos[90316] == SERVER_ANCHOR_TMPOS, (
        f'Data/environment inconsistent with the paper main-domain parent copy: total TM {len(all_m)} (expected {SERVER_TM}), '
        f'retained {len(kept_smiles)} (expected {SERVER_KEPT}), anchor TM block index {kept_tmpos[90316]} (expected {SERVER_ANCHOR_TMPOS}).\n'
        f'If the difference is only a single digit, it is most likely an rdkit version difference (local rdkit2024 differs from the server by about 19 molecules of parsing results); '
        f'please re-run in the same (tmqm) environment as the 18 July / 4 August server runs.\n'
        f'Use the tmQM_X*.xyz.gz + tmQM_y.csv under /home/chen_jianqi/zw/ml/chem_ml5/ on the server directly.')
    kept_smiles = kept_smiles[:90317]
    kept_y = kept_y[:90317]
    print('[verify] OK: set size after prefix truncation = 90,317 (paper graph-building set)', flush=True)
else:
    raise RuntimeError(f'retained count {len(kept_smiles)} < 90,317 — data files incomplete or wrong version.')

y_all = np.array(kept_y, dtype=np.float32)
del all_m, all_l, kept_y, kept_tmpos

# ===== Split alignment: ensemble_split.npz indices are relative to this 90,317 sequence =====
split = np.load(os.path.join(DATA_DIR, 'ensemble_split.npz'))
cv_idx = split['cv_idx'].astype(np.int64)
te_idx = split['te_idx'].astype(np.int64)
assert cv_idx.max() < len(kept_smiles) and te_idx.max() < len(kept_smiles)
assert len(cv_idx) == 76770 and len(te_idx) == 13547, (
    f'abnormal split size: cv_idx {len(cv_idx)} (expected 76,770) / te_idx {len(te_idx)} (expected 13,547)')
# Verify that cv_idx is exactly the complement of te_idx within the 90,317 domain (Table 2 caption basis: "the 76,770 non-test complexes")
union = np.sort(np.concatenate([cv_idx, te_idx]))
assert len(union) == 90317 and (union == np.arange(90317)).all(), (
    'cv_idx is not the complement of te_idx — ensemble_split.npz is not aligned with the 90,317 domain, stopping the re-evaluation')
print(f'Train(cv_idx) {len(cv_idx)} / Test(te_idx) {len(te_idx)} — complement check passed (union=90,317)', flush=True)
print(f'te_idx first 5: {te_idx[:5].tolist()}  last 5: {te_idx[-5:].tolist()}  (can be cross-checked against the 3D GAT test set)', flush=True)

# ===== ECFP4 fingerprints (radius=2, 2048-bit; consistent with the old baseline build_benchmark.py) =====
def compute_ecfp(smiles_list, radius=2, nBits=2048):
    fps = np.zeros((len(smiles_list), nBits), dtype=np.float32)
    failed = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fps[i])
    if failed:
        print(f'  Warning: {failed} SMILES fingerprint failures, zero vectors filled in (should theoretically be 0: the retention mask already checked MolFromSmiles)')
    return fps

print('Generating ECFP4 fingerprints (radius=2, 2048-bit)...', flush=True)
X_ecfp = compute_ecfp(kept_smiles)
print(f'ECFP matrix: {X_ecfp.shape}  ({time.time()-t0:.0f}s)', flush=True)

X_tr = X_ecfp[cv_idx]
X_te = X_ecfp[te_idx]
y_tr_all = y_all[cv_idx]
y_te_all = y_all[te_idx]

# ===== Models (original configuration of the old baselines) =====
RF_KW = dict(n_estimators=500, max_depth=15, min_samples_split=5,
             max_features='sqrt', random_state=42, n_jobs=16)
XGB_KW = dict(n_estimators=500, max_depth=8, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
              reg_lambda=1.0, random_state=42, n_jobs=16, verbosity=0)

results = {
    'config': {'features': 'ECFP4 radius=2 nBits=2048',
               'rf': {k: v for k, v in RF_KW.items() if k != 'n_jobs'},
               'xgb': {k: v for k, v in XGB_KW.items() if k not in ('n_jobs', 'verbosity')},
               'n_jobs': 16, 'random_state': 42},
    'split': {'n_train': int(len(cv_idx)), 'n_test': int(len(te_idx)),
              'train': 'cv_idx', 'test': 'te_idx'},
    'targets': targets, 'properties': {},
}
RESULT_JSON = os.path.join(SCRIPT_DIR, 'benchmark_reval_results.json')

def save_results():
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print('=' * 70, flush=True)
print(f'Starting training: {len(cv_idx):,} train / {len(te_idx):,} test × 8 properties × 2 models (RF/XGB)', flush=True)
print(f'Results are written incrementally to {os.path.basename(RESULT_JSON)} after each model; the run can be interrupted and resumed (finished ones are skipped).', flush=True)
print('=' * 70, flush=True)

preds_rf = np.zeros((len(te_idx), 8), dtype=np.float32)
preds_xgb = np.zeros((len(te_idx), 8), dtype=np.float32)

for t, name in enumerate(targets):
    y_tr, y_te = y_tr_all[:, t], y_te_all[:, t]
    for tag, cls, kw in [('rf', RandomForestRegressor, RF_KW),
                         ('xgb', XGBRegressor, XGB_KW)]:
        key = f'{name}/{tag}'
        if key in results['properties']:  # resume from checkpoint
            print(f'[skip] {key} already done, skipping', flush=True)
            continue
        t1 = time.time()
        model = cls(**kw)
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        m = {'r2': float(r2_score(y_te, p)),
             'mae': float(mean_absolute_error(y_te, p)),
             'rmse': float(np.sqrt(mean_squared_error(y_te, p)))}
        results['properties'].setdefault(name, {})[tag] = m
        save_results()
        (preds_rf if tag == 'rf' else preds_xgb)[:, t] = p
        print(f'[done] {key}: R²={m["r2"]:.4f}  MAE={m["mae"]:.4g}  RMSE={m["rmse"]:.4g}'
              f'  ({time.time()-t1:.0f}s, cumulative {time.time()-t0:.0f}s)', flush=True)

# ===== Summary output =====
np.savez_compressed(os.path.join(SCRIPT_DIR, 'benchmark_reval_preds.npz'),
                    y_true=y_te_all, rf_preds=preds_rf, xgb_preds=preds_xgb,
                    te_idx=te_idx, targets=np.array(targets))

rows = []
for name in targets:
    for tag in ('rf', 'xgb'):
        m = results['properties'][name][tag]
        rows.append({'property': name, 'model': tag, **m})
pd.DataFrame(rows).to_csv(os.path.join(SCRIPT_DIR, 'benchmark_reval_results.csv'), index=False)

r2_rf = [results['properties'][p]['rf']['r2'] for p in targets]
r2_xgb = [results['properties'][p]['xgb']['r2'] for p in targets]
print('\n' + '=' * 70, flush=True)
print('=== New Table 2 numbers (per-property R²) ===', flush=True)
print('RF  = [' + ', '.join(f'{v:.3f}' for v in r2_rf) + f']  mean {np.mean(r2_rf):.3f}', flush=True)
print('XGB = [' + ', '.join(f'{v:.3f}' for v in r2_xgb) + f']  mean {np.mean(r2_xgb):.3f}', flush=True)
print('\n=== Old-basis reference (15,126 domain, not used in the computation, for comparison only) ===', flush=True)
print('RF  = [0.292, 0.139, 0.186, 0.170, 0.639, 0.405, 0.463, 0.475]  mean 0.346', flush=True)
print('XGB = [0.578, 0.411, 0.428, 0.435, 0.870, 0.785, 0.730, 0.742]  mean 0.623', flush=True)
print(f'\nAll done, total time {(time.time()-t0)/60:.1f} minutes; outputs: '
      f'{os.path.basename(RESULT_JSON)} / benchmark_reval_results.csv / benchmark_reval_preds.npz', flush=True)
