# -*- coding: utf-8 -*-
"""Verify the metal-frequency weights of the main 90,377-complex cohort.

The manuscript uses w_m = sqrt(N / n_m), clipped to [0.5, 3.0] and normalised
to mean 1 over the training sample.  In the main domain the rarest transition
metal is Pd (n_m = 9,456 of 90,377), so the smallest raw weight is 3.0915 and
every element weight is clipped to 3.0.  Every complex contains at least one
transition metal, hence every per-sample weight is 3.0 and every normalised
weight is 1.0: the weighting is degenerate by construction.

This script re-derives the released table from its counts and, when the public
tmQM CSV is available, recounts the metals from the SMILES strings for the
retained csv_idx values.  CPU only; no GPU and no XYZ parsing.

Usage
-----
    python analysis/verify_metal_weight.py
    python analysis/verify_metal_weight.py --tmqm-y data_tmQM/local_vB/tmQM_y.csv

The optional recount uses RDKit and requires the public tmQM release; it is
skipped with a notice when the CSV is absent.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data_indices')

TM_Z = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
CLIP_LO, CLIP_HI = 0.5, 3.0
TOL = 1e-9


def fail(msg):
    print(f'FAIL: {msg}')
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tmqm-y', default=None,
                    help='path to the public tmQM_y.csv (semicolon-separated)')
    ap.add_argument('--table', default=os.path.join(DATA, 'metal_weight_table.csv'))
    ap.add_argument('--recon', default=os.path.join(DATA, 'metal_weight_reconciliation.json'))
    ap.add_argument('--retained', default=os.path.join(DATA, 'retained_ids.csv'))
    args = ap.parse_args()

    table = pd.read_csv(args.table)
    recon = json.load(open(args.recon, encoding='utf-8'))
    n_total = int(recon['main']['n'])
    print(f'main cohort: {n_total:,} complexes, {len(table)} transition metals')

    for col in ('Z', 'element', 'n_m', 'raw', 'clipped', 'normalized'):
        if col not in table.columns:
            fail(f'column {col!r} missing from {args.table}')
    if len(table) != recon['main']['n_metals']:
        fail(f'{len(table)} metals in table, {recon["main"]["n_metals"]} expected')
    if set(table['Z'].astype(int)) - TM_Z:
        fail('non-transition-metal Z in table')

    # --- re-derive the weights from the counts -----------------------------
    raw = np.sqrt(n_total / table['n_m'].to_numpy(dtype=float))
    clipped = np.clip(raw, CLIP_LO, CLIP_HI)
    if not np.allclose(raw, table['raw'], atol=TOL, rtol=0):
        fail('raw weights do not reproduce from sqrt(N / n_m)')
    if not np.allclose(clipped, table['clipped'], atol=TOL, rtol=0):
        fail('clipped weights do not reproduce with clip [0.5, 3.0]')

    # Every element weight is clipped to the ceiling, so every sample weight
    # equals the ceiling and the mean-normalised weight is 1.0.
    if not np.all(clipped == CLIP_HI):
        fail('not all element weights are clipped to 3.0; cohort is no longer degenerate')
    mean_sample_weight = CLIP_HI
    normalized = clipped / mean_sample_weight
    if not np.allclose(normalized, table['normalized'], atol=TOL, rtol=0):
        fail('normalised weights do not reproduce')
    if not np.allclose(table['normalized'], 1.0, atol=TOL, rtol=0):
        fail('normalised weights are not all 1.0')

    # --- cross-check the reconciliation record ----------------------------
    main_rec = recon['main']
    checks = {
        'pd_n': int(table.loc[table['Z'] == 46, 'n_m'].iloc[0]),
        'raw_min': float(raw.min()),
        'raw_max': float(raw.max()),
        'mean_sample_weight': mean_sample_weight,
        'sample_weight_min': float(normalized.min()),
        'sample_weight_max': float(normalized.max()),
    }
    for key, value in checks.items():
        recorded = main_rec[key]
        if isinstance(value, int):
            ok = value == recorded
        else:
            ok = abs(value - recorded) <= 1e-9
        if not ok:
            fail(f'{key}: recomputed {value!r} != recorded {recorded!r}')

    print(f'  Pd count: {checks["pd_n"]:,} ({checks["pd_n"] / n_total * 100:.2f}%)')
    print(f'  raw weight range: {raw.min():.4f} - {raw.max():.4f}')
    print(f'  clipped weight: all {CLIP_HI:.1f} -> per-sample weight all '
          f'{mean_sample_weight:.1f} -> normalised weight all 1.0')
    print('  DEGENERATE BY CONSTRUCTION: the weighting is a constant factor and '
          'cannot change relative sample losses.')

    # --- optional independent recount from the public SMILES --------------
    if args.tmqm_y:
        if not os.path.exists(args.tmqm_y):
            fail(f'tmQM CSV not found: {args.tmqm_y}')
        from rdkit import Chem, RDLogger, rdBase
        RDLogger.DisableLog('rdApp.*')
        y = pd.read_csv(args.tmqm_y, sep=';')
        if 'SMILES' not in y.columns:
            fail("'SMILES' column missing from the tmQM CSV")
        retained = pd.read_csv(args.retained)
        idx = retained['csv_idx'].to_numpy(dtype=int)
        print(f'  recounting metals from {len(idx):,} retained SMILES ...')

        counts = {z: 0 for z in sorted(set(table['Z'].astype(int)))}
        n_parsed = 0
        n_unsanitized = 0
        n_multi = 0
        failures = 0
        for csv_idx in idx:
            smi = y['SMILES'].iloc[csv_idx]
            if not isinstance(smi, str) or not smi:
                failures += 1
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Some records parsed under the training stack (RDKit 2024.03.6)
                # fail the stricter valence checks of newer builds; the atom
                # graph is still readable and is all we need for the counts.
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                if mol is None:
                    failures += 1
                    continue
                n_unsanitized += 1
            n_parsed += 1
            metals = {a.GetAtomicNum() for a in mol.GetAtoms()
                      if a.GetAtomicNum() in TM_Z}
            if len(metals) > 1:
                n_multi += 1
            for z in metals:
                if z not in counts:
                    fail(f'metal Z={z} in SMILES but not in the released table')
                counts[z] += 1
        if failures:
            fail(f'{failures} retained SMILES could not be read by this RDKit build')
        if n_unsanitized:
            print(f'  {n_unsanitized} SMILES required sanitize=False '
                  f'(RDKit {rdBase.rdkitVersion}; stricter than the training stack)')
        table_counts = dict(zip(table['Z'].astype(int), table['n_m'].astype(int)))
        for z in sorted(counts):
            if counts[z] != table_counts[z]:
                fail(f'Z={z}: SMILES recount {counts[z]:,} != table {table_counts[z]:,}')
        print(f'  SMILES recount matches the table for all {len(counts)} metals '
              f'({n_parsed:,} complexes; {n_multi:,} contain more than one metal)')

    # --- ablation cohort, for context -------------------------------------
    summary = pd.read_csv(os.path.join(DATA, 'metal_weight_summary.csv'))
    abl = summary[summary['cohort'] == 'ablation_27993'].set_index('quantity')['value']
    print(f'ablation cohort (27,993 complexes): raw {abl["raw_min"]:.4f} - '
          f'{abl["raw_max"]:.4f}, normalised {abl["normalized_min"]:.4f} - '
          f'{abl["normalized_max"]:.4f}; this is the run in which the weighting '
          'is non-trivial.')

    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
