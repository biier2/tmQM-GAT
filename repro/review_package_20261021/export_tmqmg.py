# -*- coding: utf-8 -*-
"""Export the row-level map of the tmQMg external-validation set.

Replicates the filter chain of baselines/server_package/server_crossfunct_A.py
(the same chain as external_validation.ipynb cell 1):

    tmQMg_properties_and_targets.csv   74,547 rows
      keep smiles is str, non-empty, not 'nan'         -> 73,443
      drop rows with NaN/Inf in the 7 tzvp columns     -> 73,443 (no NaN rows)
      drop rows whose SMILES RDKit cannot parse        -> 73,438
                                                          = rows of
                                                          crossfunct_A_preds.npz

Writes tmqmg_pairs.csv, one row per prediction row, so that an index into
crossfunct_A_preds.npz can be joined to a molecule:

    pred_row   0-based row in crossfunct_A_preds.npz arrays (graph order)
    csv_row    0-based row in tmQMg_properties_and_targets.csv (the join key)
    id, metal_center, charge, n_atoms, smiles

Also records the 5 graph-parse failures in tmqmg_reconciliation.json and
asserts that y_true of the npz equals the seven tzvp columns of the CSV at the
mapped rows (float32), which is the end-to-end check of the row mapping.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data_indices')
CSV = r'D:\ml\chem_ml5\tmQMg_properties_and_targets.csv'
NPZ = r'D:\ml\chem_ml5\crossfunct_A_preds.npz'
PROPS = ['tzvp_homo_lumo_gap', 'tzvp_homo_energy', 'tzvp_lumo_energy',
         'tzvp_dipole_moment', 'tzvp_electronic_energy', 'tzvp_dispersion_energy',
         'polarisability']


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV)
    n_total = len(df)
    print(f'csv rows: {n_total:,}', flush=True)

    valid_mask = df['smiles'].apply(lambda s: isinstance(s, str) and s != '' and s != 'nan')
    df_valid = df[valid_mask].copy()
    print(f'valid smiles: {len(df_valid):,}', flush=True)

    y = df_valid[PROPS].to_numpy(dtype=np.float32)
    nan_mask = ~(np.isnan(y).any(axis=1) | np.isinf(y).any(axis=1))
    print(f'after NaN/Inf drop: {int(nan_mask.sum()):,}', flush=True)
    df_valid = df_valid[nan_mask]
    y = y[nan_mask]

    csv_rows = df_valid.index.to_numpy(dtype=np.int64)   # original CSV row numbers
    df_valid = df_valid.reset_index(drop=True)

    keep_pos, failed = [], []
    for i, smi in enumerate(df_valid['smiles']):
        if Chem.MolFromSmiles(smi) is None:
            failed.append(i)
        else:
            keep_pos.append(i)
    keep_pos = np.asarray(keep_pos, dtype=np.int64)
    print(f'graphs: {len(keep_pos):,}  (RDKit {Chem.rdBase.rdkitVersion})', flush=True)

    z = np.load(NPZ)
    y_true = z['y_true']
    if y_true.shape != (len(keep_pos), 7):
        raise RuntimeError(f'npz y_true shape {y_true.shape} != {(len(keep_pos), 7)}')

    pulled = y[keep_pos]
    if not np.array_equal(pulled, y_true):
        diff = np.abs(pulled.astype(np.float64) - y_true.astype(np.float64))
        raise RuntimeError(f'y_true mismatch: max abs diff {diff.max():.6g}, '
                           f'n cells {int((diff > 0).sum())}')

    pairs = pd.DataFrame({
        'pred_row': np.arange(len(keep_pos), dtype=np.int64),
        'csv_row': csv_rows[keep_pos],
        'id': df_valid['id'].to_numpy()[keep_pos],
        'metal_center': df_valid['metal_center'].to_numpy()[keep_pos],
        'charge': df_valid['charge'].to_numpy()[keep_pos],
        'n_atoms': df_valid['n_atoms'].to_numpy()[keep_pos],
        'smiles': df_valid['smiles'].to_numpy()[keep_pos],
    })
    path = os.path.join(OUT, 'tmqmg_pairs.csv')
    pairs.to_csv(path, index=False)
    print(f'wrote {path}: {len(pairs)} rows', flush=True)

    failed_rows = [{'df_valid_row': int(i), 'csv_row': int(csv_rows[i]),
                    'id': str(df_valid['id'].iloc[i]),
                    'smiles': str(df_valid['smiles'].iloc[i])} for i in failed]
    recon = {
        'source_csv': CSV,
        'npz': NPZ,
        'n_csv_rows': n_total,
        'n_valid_smiles': int(valid_mask.sum()),
        'n_after_nan_drop': int(nan_mask.sum()),
        'n_graphs': int(len(keep_pos)),
        'npz_shapes': {k: list(z[k].shape) for k in z.files},
        'rdkit_version': Chem.rdBase.rdkitVersion,
        'parse_failures': failed_rows,
        'y_true_matches_csv': True,
        'props': PROPS,
        'metal_center_counts': df_valid['metal_center'].to_numpy()[keep_pos].tolist() and
                               {str(k): int(v) for k, v in
                                pd.Series(df_valid['metal_center'].to_numpy()[keep_pos]).value_counts().items()},
    }
    with open(os.path.join(OUT, 'tmqmg_reconciliation.json'), 'w', encoding='utf-8') as f:
        json.dump(recon, f, ensure_ascii=False, indent=2, default=int)
    print(f'parse failures: {len(failed_rows)}', flush=True)
    for r in failed_rows:
        print('  ', r['csv_row'], r['id'], r['smiles'][:80], flush=True)
    print('y_true == CSV tzvp columns at mapped rows: OK', flush=True)


if __name__ == '__main__':
    main()
