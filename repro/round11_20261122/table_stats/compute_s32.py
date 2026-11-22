# -*- coding: utf-8 -*-
"""Recompute SI Table S32: per-property MAE and RMSE of the tmQMg transfer
predictions, split into the three CSD-code overlap groups of Table S25.

Inputs:
    preds/crossfunct_A_preds.npz            (9-member variant-A ensemble on the pairs)
    data_indices/tmqmg_pairs.csv            (pred_row, CSD id)
    data_indices/ensemble_split_new.npz     (te_idx / cv_idx over the 90,377 cohort)
    data_indices/retained_ids.csv           (csv_idx of each retained row)
    <tmQM>/tmQM_y.csv                       (target table; --tmqm-csv)

Usage:
    python compute_s32.py --repo ../../.. --tmqm-csv /path/to/tmQM_y.csv
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

T = ['HL-Gap', 'HOMO', 'LUMO', 'dipole', 'E_elec', 'E_disp', 'polarizability']
COLS = ['HL_Gap', 'HOMO_Energy', 'LUMO_Energy', 'Dipole_M', 'Electronic_E',
        'Dispersion_E', 'Polarizability']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   '..', '..', '..'))
    ap.add_argument('--tmqm-csv', required=True)
    args = ap.parse_args()

    code_rows = {}
    rows = []
    with io.open(args.tmqm_csv, encoding='utf-8', errors='replace') as f:
        for i, row in enumerate(csv.DictReader(f, delimiter=';')):
            rows.append(row)
            code_rows.setdefault(row['CSD_code'], []).append(i)

    pairs = []
    with io.open(os.path.join(args.repo, 'data_indices', 'tmqmg_pairs.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pairs.append((int(row['pred_row']), row['id']))

    z = np.load(os.path.join(args.repo, 'preds', 'crossfunct_A_preds.npz'))
    yq = z['y_true'].astype(np.float64)
    pred = z['ens_mean'].astype(np.float64)

    sp = np.load(os.path.join(args.repo, 'data_indices', 'ensemble_split_new.npz'))
    retained = []
    with io.open(os.path.join(args.repo, 'data_indices', 'retained_ids.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            retained.append(int(r['csv_idx']))
    rc = np.array(retained)
    te = set(rc[sp['te_idx']].astype(int).tolist())
    cv = set(rc[sp['cv_idx']].astype(int).tolist())

    keep, group = [], []
    for pred_row, csd in pairs:
        idxs = code_rows.get(csd)
        if not idxs:
            continue
        ok = None
        for i in idxs:
            try:
                [float(rows[i][c]) for c in COLS]
            except Exception:
                continue
            ok = i
            break
        if ok is None:
            continue
        keep.append(pred_row)
        s = set(idxs)
        if s & te:
            group.append('test')
        elif s & cv:
            group.append('train')
        else:
            group.append('outside')

    keep = np.array(keep)
    group = np.array(group)
    p2, y2 = pred[keep], yq[keep]
    print('aligned %d rows' % len(keep))

    print('\n%-10s %7s %8s | per-property MAE' % ('group', 'n', 'mean R2'))
    for g in ('train', 'test', 'outside'):
        m = group == g
        r2s = [1 - ((p2[m, j] - y2[m, j]) ** 2).sum()
               / ((y2[m, j] - y2[m, j].mean()) ** 2).sum() for j in range(7)]
        maes = [np.abs(p2[m, j] - y2[m, j]).mean() for j in range(7)]
        rmses = [np.sqrt(((p2[m, j] - y2[m, j]) ** 2).mean()) for j in range(7)]
        print('%-10s %7d %8.4f' % (g, m.sum(), np.mean(r2s)))
        print('   MAE : ' + '  '.join('%s=%.4g' % (t, v) for t, v in zip(T, maes)))
        print('   RMSE: ' + '  '.join('%s=%.4g' % (t, v) for t, v in zip(T, rmses)))


if __name__ == '__main__':
    main()
