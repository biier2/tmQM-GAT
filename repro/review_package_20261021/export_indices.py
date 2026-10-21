# -*- coding: utf-8 -*-
"""Export reviewer-package index files for the 90,377-complex tmQM domain.

Outputs (to ./data_indices/):
  retained_ids.csv    90,377 rows: csv_idx, csd_code
  excluded_ids.csv    18,164 rows: csv_idx, csd_code, reason
                      reason in {no_TM, unparseable_SMILES, isomorphism_failed}
  baseline_splits.npz te_idx (13,557, SchNet/DimeNet++), cv_idx (76,820),
                      fold_{0,1,2}_{train,val} (KFold(3, shuffle=True, random_state=42))
  index_reconciliation.json  counters + cross-checks

The exclusion chain reproduces analysis/rebuild_match_iso.py on the same
13 Aug 2024 source files (md5 pinned below); it does not re-run the VF2++
matching, but derives the 10,356 isomorphism failures as the complement of
new_domain_iso.pt['csv_idx'] inside the 100,733-strong parseable set.
Row order of new_domain_iso.pt equals ascending csv_idx, hence domain index
= row number in retained_ids.csv.
"""
import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = r'D:\ml\chem_ml5\data_tmQM\local_vB'
PT_FILE = r'D:\ml\chem_ml5\new_domain_iso.pt'
PARSE_NPZ = r'D:\ml\paper\draft\round8\_recon_20261016\parse_rd2024.npz'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_indices')
N_TOTAL = 108541

SRC_MD5 = {
    'tmQM_y.csv': 'f8eeb967b71938f1a8c1d44be8f1f043',
    'tmQM_X1.xyz.gz': 'f273275f288b375ab20b2f9d5c5c227f',
    'tmQM_X2.xyz.gz': '563be6efff4d99b64e3590caeb5c5699',
    'tmQM_X3.xyz.gz': 'c4a9f3c48c47792086c0532d31eb6d6a',
    'tmQM_X1.BO.gz': 'cf64d4f145f649bc68c18e6f8704c8d1',
    'tmQM_X2.BO.gz': '00b561bf31e2c967e3d919059701b936',
    'tmQM_X3.BO.gz': 'ea4bd75955d596224145b8e4cf4ebbe5',
}
TM_SET = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def xyz_blocks(path):
    """Yield (code, [atomic numbers]) for every XYZ block, in file order."""
    with gzip.open(path, 'rt') as f:
        lines = f.read().split('\n')
    i, n_lines = 0, len(lines)
    while i < n_lines:
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if not line.isdigit():
            raise RuntimeError(f'unexpected XYZ line: {line!r}')
        n = int(line)
        comment = lines[i].strip() if i < n_lines else ''
        i += 1
        code = '?'
        if 'CSD_code' in comment:
            code = comment.split('CSD_code')[1].lstrip('= ').split('|')[0].strip() or '?'
        syms = []
        for _ in range(n):
            if i >= n_lines:
                break
            parts = lines[i].split()
            i += 1
            if parts:
                syms.append(parts[0])
        yield code, syms


def bo_codes(path):
    codes = set()
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('CSD_code'):
                codes.add(line.split('=', 1)[1].split('|')[0].strip())
    return codes


def main():
    os.makedirs(OUT, exist_ok=True)
    print('checking source md5 ...', flush=True)
    md5_ok = {}
    for name, want in SRC_MD5.items():
        got = md5(os.path.join(ROOT, name))
        md5_ok[name] = {'expected': want, 'actual': got, 'ok': got == want}
        print(f'  {name}: {"OK" if got == want else "MISMATCH " + got}', flush=True)

    print('loading tmQM_y.csv ...', flush=True)
    y = pd.read_csv(os.path.join(ROOT, 'tmQM_y.csv'), sep=';')
    csd_csv = y['CSD_code'].astype(str).tolist()
    smi_csv = y['SMILES'].tolist()
    assert len(csd_csv) == N_TOTAL, len(csd_csv)

    print('parsing BO headers ...', flush=True)
    bo_all = set()
    for p in (1, 2, 3):
        bo_all |= bo_codes(os.path.join(ROOT, f'tmQM_X{p}.BO.gz'))
    print(f'  BO codes: {len(bo_all):,}', flush=True)

    print('scanning XYZ blocks ...', flush=True)
    has_tm = np.zeros(N_TOTAL, dtype=bool)
    n_atoms = np.zeros(N_TOTAL, dtype=np.int32)
    csd_xyz = []
    k = 0
    for p in (1, 2, 3):
        for code, syms in xyz_blocks(os.path.join(ROOT, f'tmQM_X{p}.xyz.gz')):
            if k >= N_TOTAL:
                raise RuntimeError('more XYZ blocks than csv rows')
            zs = [Chem.GetPeriodicTable().GetAtomicNumber(s) for s in syms]
            n_atoms[k] = len(zs)
            has_tm[k] = any(z in TM_SET for z in zs)
            csd_xyz.append(code)
            k += 1
        print(f'  part {p}: cum {k:,}', flush=True)
    if k != N_TOTAL:
        raise RuntimeError(f'XYZ blocks {k} != csv rows {N_TOTAL}')
    csd_xyz = np.array(csd_xyz, dtype=object)
    csd_csv_arr = np.array(csd_csv, dtype=object)
    n_code_diff = int((csd_xyz != csd_csv_arr).sum())
    print(f'  CSD code XYZ-vs-CSV mismatches: {n_code_diff}', flush=True)

    print('loading training-stack SMILES parse status (RDKit 2024.03.6) ...', flush=True)
    pz = np.load(PARSE_NPZ)
    status = pz['status']
    if len(status) != N_TOTAL:
        raise RuntimeError(f'parse status length {len(status)} != {N_TOTAL}')
    # status: 0 ok, 1 empty/nan, 2 parse-none, 3 sanitize error, 4 exception
    parse_ok = status == 0
    print(f'  parse_ok={int(parse_ok.sum()):,} (expected 100,849)', flush=True)

    print('loading new_domain_iso.pt ...', flush=True)
    d = torch.load(PT_FILE, map_location='cpu', weights_only=False)
    csv_idx = np.asarray(d['csv_idx']).astype(np.int64)
    csd_pt = np.array(d['csd_codes'], dtype=object)
    assert len(csv_idx) == 90377 and len(np.unique(csv_idx)) == 90377
    assert np.all(np.diff(csv_idx) > 0), 'csv_idx not ascending; domain index != row order'

    # ---- retained ----
    ret = pd.DataFrame({'csv_idx': csv_idx, 'csd_code': csd_csv_arr[csv_idx]})
    mism = int((ret['csd_code'].to_numpy() != csd_pt).sum())
    csv_mism = int((ret['csd_code'].to_numpy() != csd_xyz[csv_idx]).sum())
    print(f'  csd_code mismatches pt-vs-xyz={mism}, pt-vs-csv={csv_mism}', flush=True)
    ret.to_csv(os.path.join(OUT, 'retained_ids.csv'), index=False)

    # ---- excluded ----
    in_bo = np.array([c in bo_all for c in csd_xyz], dtype=bool)
    retained_set = np.zeros(N_TOTAL, dtype=bool)
    retained_set[csv_idx] = True
    single_atom = n_atoms <= 1
    no_tm = (~single_atom) & (~has_tm)
    unparseable = (~single_atom) & has_tm & (~parse_ok)
    parseable = (~single_atom) & has_tm & parse_ok
    no_bo = parseable & (~in_bo)
    iso_fail = parseable & in_bo & (~retained_set)

    reasons = np.array([''] * N_TOTAL, dtype=object)
    reasons[single_atom] = 'single_atom'
    reasons[no_tm] = 'no_TM'
    reasons[unparseable] = 'unparseable_SMILES'
    reasons[no_bo] = 'no_BO'
    reasons[iso_fail] = 'isomorphism_failed'
    excl = np.where(reasons != '')[0]
    if len(excl) != N_TOTAL - len(csv_idx):
        raise RuntimeError(f'excluded {len(excl)} != {N_TOTAL - len(csv_idx)}')
    ex = pd.DataFrame({'csv_idx': excl, 'csd_code': csd_csv_arr[excl], 'reason': reasons[excl]})
    ex.to_csv(os.path.join(OUT, 'excluded_ids.csv'), index=False)

    counts = {r: int((reasons == r).sum()) for r in sorted(set(reasons) - {''})}
    print('  exclusion counts:', counts, flush=True)

    # ---- baseline splits (SchNet / DimeNet++, 13,557 test) ----
    from sklearn.model_selection import KFold
    N = len(csv_idx)
    perm = np.random.RandomState(42).permutation(N)
    n_test = 13557
    te = perm[:n_test].copy()
    cv = perm[n_test:].copy()
    sch_te = np.load(r'D:\ml\chem_ml5\test_preds_SchNet.npz')['te_idx']
    assert len(sch_te) == n_test and np.array_equal(np.sort(sch_te), np.sort(te))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = {'te_idx': te.astype(np.int32), 'cv_idx': cv.astype(np.int32),
              'domain_n': np.array([N], dtype=np.int32),
              'n_test_source': np.array(['RandomState(42).permutation(90377)[:13557]'], dtype=object)}
    for fi, (tr, va) in enumerate(kf.split(cv)):
        splits[f'fold_{fi}_train'] = cv[tr].astype(np.int32)
        splits[f'fold_{fi}_val'] = cv[va].astype(np.int32)
    np.savez_compressed(os.path.join(OUT, 'baseline_splits.npz'), **splits)
    print('  folds:', [(len(splits[f'fold_{i}_train']), len(splits[f'fold_{i}_val'])) for i in range(3)], flush=True)

    recon = {
        'source_md5': md5_ok,
        'n_csv_rows': N_TOTAL,
        'n_has_tm': int(has_tm.sum()),
        'n_parseable': int(parse_ok.sum()),
        'n_parseable_with_bo': int((parse_ok & in_bo).sum()),
        'n_retained': int(len(csv_idx)),
        'counts_by_reason': counts,
        'csd_code_mismatch_retained_vs_pt': mism,
        'csd_code_mismatch_xyz_vs_csv_all': n_code_diff,
        'rdkit_training_stack': '2024.03.6',
        'parse_status_counts': {str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))},
        'rest_chain': {'108541_minus_no_tm': N_TOTAL - int(no_tm.sum()),
                       'minus_unparseable': int(parseable.sum())},
        'baseline_split': {'te': n_test, 'cv': len(cv),
                           'folds': [[len(splits[f'fold_{i}_train']), len(splits[f'fold_{i}_val'])] for i in range(3)]},
    }
    with open(os.path.join(OUT, 'index_reconciliation.json'), 'w', encoding='utf-8') as f:
        json.dump(recon, f, ensure_ascii=False, indent=2)
    print(json.dumps(recon, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
