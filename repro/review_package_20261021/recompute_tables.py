# -*- coding: utf-8 -*-
"""Recompute the ensemble metrics of Tables 2 and 3 from the archived
predictions, and check them against the released result JSONs.

Inputs (all inside the repository):
    preds/test_preds_<variant>.npz   member predictions, ensemble mean,
                                     reference values and test indices
    results_tables/results_<variant>.json
                                     per-property ensemble R2 / MAE as printed
    data_indices/ensemble_split_new.npz
    data_indices/baseline_splits.npz

The script is CPU-only, needs only NumPy, and writes nothing. It exits with
status 1 if any metric, split index or prediction array fails its check.

Usage:
    python repro/review_package_20261021/recompute_tables.py
    python repro/review_package_20261021/recompute_tables.py --repo /path/to/repo
"""
import argparse
import json
import os
import sys

import numpy as np

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge',
           'E_elec', 'E_disp', 'polarizability']
# GAT variants use the 13,556-complex paper split; the two geometry-aware
# baselines use the 13,557-complex baseline split (one extra complex).
GAT_VARIANTS = ['A_8d6d', 'B_8d22d', 'C_11d6d', 'D_11d23d',
                'E_9d6d', 'F_10d6d', 'G_7d6d', 'H_7d22d']
BASE_VARIANTS = ['SchNet', 'DimeNet']


def r2_score(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, np.float64) - np.asarray(p, np.float64))))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_repo = os.path.abspath(os.path.join(here, '..', '..'))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repo', default=default_repo, help='repository root')
    ap.add_argument('--tol-r2', type=float, default=1e-6,
                    help='absolute tolerance on R2 (default 1e-6)')
    ap.add_argument('--tol-mae', type=float, default=1e-6,
                    help='relative tolerance on MAE (default 1e-6)')
    args = ap.parse_args()

    pred_dir = os.path.join(args.repo, 'preds')
    res_dir = os.path.join(args.repo, 'results_tables')
    idx_dir = os.path.join(args.repo, 'data_indices')

    split_new = np.load(os.path.join(idx_dir, 'ensemble_split_new.npz'))
    split_base = np.load(os.path.join(idx_dir, 'baseline_splits.npz'))
    te_gat = split_new['te_idx']
    te_base = split_base['te_idx']

    failures = []
    checks = 0
    ref_yt = None
    ref_te = None

    print(f'repository: {args.repo}')
    print(f'{"variant":10s} {"mean R2 json":>13s} {"recomputed":>12s} '
          f'{"max |dR2|":>10s} {"max rel dMAE":>14s}  status')
    print('-' * 78)

    for variant in GAT_VARIANTS + BASE_VARIANTS:
        is_gat = variant in GAT_VARIANTS
        npz_path = os.path.join(pred_dir, f'test_preds_{variant}.npz')
        json_path = os.path.join(res_dir, f'results_{variant}.json')
        for path in (npz_path, json_path):
            if not os.path.exists(path):
                failures.append(f'{variant}: missing {os.path.relpath(path, args.repo)}')
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            continue

        z = np.load(npz_path)
        with open(json_path, encoding='utf-8') as f:
            ref = json.load(f)

        preds = np.asarray(z['preds'], dtype=np.float64)
        ens = np.asarray(z['ens_mean'], dtype=np.float64)
        yt = np.asarray(z['y_true'], dtype=np.float64)
        te = np.asarray(z['te_idx'])
        n = len(yt)

        # --- prediction-array consistency -------------------------------
        expected_members = 9 if variant != 'DimeNet' else 3
        checks += 1
        if preds.shape != (expected_members, n, len(TARGETS)):
            failures.append(
                f'{variant}: preds shape {preds.shape}, expected '
                f'{(expected_members, n, len(TARGETS))}')
        checks += 1
        # ens_mean was accumulated in float32; allow float32 round-off scaled
        # to the magnitude of the property (E_elec is O(1e3)).
        member_mean = preds.mean(axis=0)
        if not np.allclose(member_mean, ens, rtol=1e-5, atol=1e-6):
            failures.append(
                f'{variant}: ens_mean differs from mean(preds) '
                f'(max rel {np.max(np.abs(member_mean - ens) / np.maximum(np.abs(ens), 1.0)):.2e})')

        # --- split ------------------------------------------------------
        checks += 1
        if is_gat:
            want_te = te_gat
            want_n = 13556
            if te.shape != want_te.shape or not np.array_equal(te, want_te):
                failures.append(
                    f'{variant}: te_idx does not equal ensemble_split_new.npz')
            if n != want_n:
                failures.append(f'{variant}: test rows {n}, expected {want_n}')
            if ref_te is None:
                ref_te = te
                ref_yt = yt
        else:
            want_n = 13557
            if n != want_n:
                failures.append(f'{variant}: test rows {n}, expected {want_n}')
            checks += 1
            if te.shape != te_base.shape or not np.array_equal(te, te_base):
                failures.append(
                    f'{variant}: te_idx does not equal baseline_splits.npz')
            # the GAT split is the first 13,556 rows of the baseline split
            checks += 1
            if not np.array_equal(te[:13556], te_gat):
                failures.append(
                    f'{variant}: first 13,556 te_idx entries differ from the GAT split')

        # --- metrics ----------------------------------------------------
        r2s = []
        max_dr2 = 0.0
        max_dmae = 0.0
        for k, target in enumerate(TARGETS):
            r2 = r2_score(yt[:, k], ens[:, k])
            m = mae(yt[:, k], ens[:, k])
            r2s.append(r2)
            ref_r2 = float(ref[target]['ens_r2'])
            ref_mae = float(ref[target]['ens_mae'])
            dr2 = abs(r2 - ref_r2)
            dmae = abs(m - ref_mae) / max(abs(ref_mae), 1e-12)
            max_dr2 = max(max_dr2, dr2)
            max_dmae = max(max_dmae, dmae)
            checks += 1
            if dr2 > args.tol_r2:
                failures.append(
                    f'{variant}/{target}: R2 {r2:.9f} vs json {ref_r2:.9f} '
                    f'(|d|={dr2:.2e})')
            if dmae > args.tol_mae:
                failures.append(
                    f'{variant}/{target}: MAE {m:.9f} vs json {ref_mae:.9f} '
                    f'(rel d={dmae:.2e})')

        mean_r2 = float(np.mean(r2s))
        ref_mean = float(ref['mean_ens_r2'])
        checks += 1
        if abs(mean_r2 - ref_mean) > args.tol_r2:
            failures.append(
                f'{variant}: mean R2 {mean_r2:.9f} vs json {ref_mean:.9f}')

        ok = (max_dr2 <= args.tol_r2 and max_dmae <= args.tol_mae)
        print(f'{variant:10s} {ref_mean:13.9f} {mean_r2:12.9f} '
              f'{max_dr2:10.2e} {max_dmae:14.2e}  {"OK" if ok else "FAIL"}')

    # all eight GAT variants must share the same reference rows
    if ref_yt is not None:
        for variant in GAT_VARIANTS:
            path = os.path.join(pred_dir, f'test_preds_{variant}.npz')
            if not os.path.exists(path):
                continue
            checks += 1
            if not np.array_equal(np.asarray(np.load(path)['y_true']), ref_yt):
                failures.append(f'{variant}: y_true differs from the other GAT variants')

    print('-' * 78)
    if failures:
        print(f'{len(failures)} FAILURE(S) after {checks} checks:')
        for f in failures:
            print('  -', f)
        return 1
    print(f'ALL CHECKS PASSED ({checks} checks).')
    print('Tables 2 and 3 reproduce from preds/ within float32 round-off.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
