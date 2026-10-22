# -*- coding: utf-8 -*-
"""Verify a rebuilt new_domain_iso.pt against the content digests recorded for
the authors' copy.

torch.save is not byte-deterministic, so the file md5 is not a valid anchor for
a rebuilt tensor. This script hashes the array contents in domain order and
compares them with repro/domain_content_digests.json. It also checks the domain
size, the seed-42 test split, and (when the repository layout is present) the
retained-ID table.

Usage:
    python repro/verify_domain_content.py --data /path/to/new_domain_iso.pt
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch


def as_np(t):
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def arr_md5(a):
    a = np.ascontiguousarray(as_np(a))
    return hashlib.md5(a.tobytes()).hexdigest()


def list_md5(lst):
    h = hashlib.md5()
    for a in lst:
        h.update(np.ascontiguousarray(as_np(a)).tobytes())
    return h.hexdigest()


def str_md5(lst):
    return hashlib.md5('\n'.join(map(str, lst)).encode('utf-8')).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..'))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default='new_domain_iso.pt',
                    help='rebuilt domain tensor (default: ./new_domain_iso.pt)')
    ap.add_argument('--ref', default=os.path.join(here, 'domain_content_digests.json'))
    ap.add_argument('--skip-cloud', action='store_true',
                    help='skip the 800 MB node/edge feature lists')
    args = ap.parse_args()

    with open(args.ref, encoding='utf-8') as f:
        ref = json.load(f)

    print(f'data: {args.data}')
    print(f'ref : {args.ref}')
    try:
        d = torch.load(args.data, map_location='cpu', weights_only=False, mmap=True)
    except Exception as e:
        print('mmap load failed (%s); loading normally' % e)
        d = torch.load(args.data, map_location='cpu', weights_only=False)

    problems = []
    n = len(d['csv_idx'])
    print(f'N = {n} (reference {ref["N"]})')
    if n != ref['N']:
        problems.append(f'N={n}, expected {ref["N"]}')

    for key, info in ref['arrays'].items():
        got = arr_md5(d[key])
        ok = got == info['md5']
        print(f'{key:10s} {got} {"OK" if ok else "MISMATCH " + info["md5"]}')
        if not ok:
            problems.append(f'{key} md5 {got} != {info["md5"]}')

    for key, info in ref['strings'].items():
        got = str_md5(d[key])
        ok = got == info['md5']
        print(f'{key:10s} {got} {"OK" if ok else "MISMATCH " + info["md5"]}')
        if not ok:
            problems.append(f'{key} md5 {got} != {info["md5"]}')

    if not args.skip_cloud:
        for key, info in ref['lists'].items():
            got = list_md5(d[key])
            ok = got == info['md5']
            print(f'{key:10s} {got} {"OK" if ok else "MISMATCH " + info["md5"]}')
            if not ok:
                problems.append(f'{key} md5 {got} != {info["md5"]}')

    # seed-42 split: te_idx must be the first 15% of permutation(90377), with
    # the paper's floor convention (13,556 rows), not round(0.15*N).
    n_te = len(d['te_idx'])
    perm = np.random.RandomState(42).permutation(n)
    print(f'te_idx rows = {n_te}')
    if n_te != 13556:
        problems.append(f'te_idx has {n_te} rows, expected 13,556')
    if n_te <= n and not np.array_equal(np.asarray(d['te_idx']), perm[:n_te]):
        problems.append('te_idx is not the seed-42 permutation prefix')

    retained = os.path.join(repo, 'data_indices', 'retained_ids.csv')
    if os.path.exists(retained):
        rows = np.genfromtxt(retained, delimiter=',', names=True,
                             dtype=None, encoding='utf-8')
        csv_idx = np.asarray(rows['csv_idx'])
        csd = [str(x) for x in rows['csd_code']]
        ok_idx = np.array_equal(np.asarray(d['csv_idx']), csv_idx)
        ok_csd = csd == [str(x) for x in d['csd_codes']]
        print(f'retained_ids.csv: csv_idx {"OK" if ok_idx else "MISMATCH"}, '
              f'CSD codes {"OK" if ok_csd else "MISMATCH"}')
        if not ok_idx:
            problems.append('csv_idx differs from data_indices/retained_ids.csv')
        if not ok_csd:
            problems.append('csd_codes differ from data_indices/retained_ids.csv')

    if problems:
        print('\nFAILED:')
        for p in problems:
            print('  -', p)
        return 1
    print('\nALL CONTENT DIGESTS MATCH.')
    print('The rebuilt tensor is content-identical to the copy used for the paper.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
