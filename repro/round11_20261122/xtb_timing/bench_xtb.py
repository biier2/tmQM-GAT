# -*- coding: utf-8 -*-
"""Time one GFN2-xTB single point and the WBO file parse on representative
held-out complexes (the timing quoted in SI Section S3.3 and Table S24).

The XYZ inputs are taken from the released tmQM geometries (X1/X2/X3 archives),
sampled at the 10th/25th/50th/75th/90th percentiles of the test-set atom count.
A manifest (name, atom count) is expected in the XYZ directory.

Usage:
    python bench_xtb.py --xtb /path/to/xtb --xyz-dir xyz --runs 5

Each run uses a fresh directory so that xtb cannot reuse a restart file.
"""
import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')


def read_manifest(xyz_dir):
    out = []
    with open(os.path.join(xyz_dir, 'manifest.txt')) as f:
        for line in f:
            name, n = line.split()
            out.append((name, int(n)))
    return out


def parse_wbo(path, n):
    """Featuriser-style WBO parse: dense n x n matrix from the pair list."""
    import numpy as np
    m = np.zeros((n, n), dtype=np.float64)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            i, j, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            m[i, j] = m[j, i] = w
    return m


def run_once(xtb, xyz_dir, xyz_name, tag, env=None):
    d = os.path.join(xyz_dir, '_run_' + tag)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    shutil.copy2(os.path.join(xyz_dir, xyz_name), d)
    t0 = time.perf_counter()
    p = subprocess.run([xtb, xyz_name, '--gfn', '2', '--sp'], cwd=d,
                       capture_output=True, text=True, env=env)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return dt, os.path.join(d, 'wbo')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xtb', required=True, help='path to the xtb executable')
    ap.add_argument('--xyz-dir', required=True, help='directory with xyz files and manifest.txt')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--out', default='xtb_timing.json')
    args = ap.parse_args()

    manifest = read_manifest(args.xyz_dir)
    print('%-16s %5s | %-22s | %-22s | %s'
          % ('complex', 'atoms', 'default threads (s)', '1 thread (s)', 'wbo parse (ms)'))
    rows = []
    for name, n in manifest:
        dts = [run_once(args.xtb, args.xyz_dir, name, 'def%d' % i)[0] for i in range(args.runs)]
        env1 = dict(os.environ)
        env1['OMP_NUM_THREADS'] = '1'
        dt1s = [run_once(args.xtb, args.xyz_dir, name, 'one%d' % i, env=env1)[0]
                for i in range(args.runs)]
        _, wbo = run_once(args.xtb, args.xyz_dir, name, 'p')
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            parse_wbo(wbo, n)
            ts.append((time.perf_counter() - t0) * 1000)
        rows.append((name, n, statistics.median(dts), statistics.median(dt1s),
                     statistics.median(ts)))
        print('%-16s %5d | %8.3f  (min %.3f max %.3f) | %8.3f  (min %.3f max %.3f) | %.3f'
              % (name, n, statistics.median(dts), min(dts), max(dts),
                 statistics.median(dt1s), min(dt1s), max(dt1s), statistics.median(ts)))

    print('\nmedian over complexes: default %.3f s, 1 thread %.3f s'
          % (statistics.median([r[2] for r in rows]),
             statistics.median([r[3] for r in rows])))
    import json
    with open(args.out, 'w') as f:
        json.dump([{'complex': r[0], 'atoms': r[1], 'default_s': r[2],
                    'one_thread_s': r[3], 'wbo_parse_ms': r[4]} for r in rows], f, indent=1)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
