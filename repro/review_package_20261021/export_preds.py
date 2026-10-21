# -*- coding: utf-8 -*-
"""Copy the test-set and external-validation prediction archives into the
review package (preds/) as compressed npz, and verify every array round-trips.

Source files live in D:\\ml\\chem_ml5:
    test_preds_{A_8d6d,B_8d22d,C_11d6d,D_11d23d,E_9d6d,F_10d6d,G_7d6d,H_7d22d,
                SchNet,DimeNet}.npz          (rows: 13,556 GAT split / 13,557
                                              SchNet-DimeNet split)
    crossfunct_A_preds.npz                   (73,438 tmQMg rows; preds = the nine
                                              member predictions, ens_mean = their mean)

Checks
  * te_idx of the nine-member GAT files equals RandomState(42).permutation(90377)[:13556]
    element for element (order included).
  * SchNet/DimeNet te_idx equals RandomState(42).permutation(90377)[:13557].
  * every array in the copy is equal to the source array (`equal_nan=True`).
Writes preds/manifest.json with shapes, dtypes, sizes and md5 of both files.
"""
import hashlib
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'preds')
SRC_DIR = r'D:\ml\chem_ml5'
FILES = ['test_preds_A_8d6d.npz', 'test_preds_B_8d22d.npz', 'test_preds_C_11d6d.npz',
         'test_preds_D_11d23d.npz', 'test_preds_E_9d6d.npz', 'test_preds_F_10d6d.npz',
         'test_preds_G_7d6d.npz', 'test_preds_H_7d22d.npz', 'test_preds_SchNet.npz',
         'test_preds_DimeNet.npz', 'crossfunct_A_preds.npz']
N_DOMAIN = 90377


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    perm = np.random.RandomState(42).permutation(N_DOMAIN)
    expect = {13556: perm[:13556], 13557: perm[:13557]}
    manifest = {'source_dir': SRC_DIR, 'files': {}}
    for name in FILES:
        src = os.path.join(SRC_DIR, name)
        z = np.load(src, allow_pickle=True)
        arrays = {k: z[k] for k in z.files}
        z.close()
        if 'te_idx' in arrays:
            te = arrays['te_idx']
            want = expect.get(len(te))
            if want is None or not np.array_equal(te, want):
                raise RuntimeError(f'{name}: te_idx does not match the expected split')
        dst = os.path.join(OUT, name)
        np.savez_compressed(dst, **arrays)
        w = np.load(dst, allow_pickle=False)
        for k, a in arrays.items():
            b = w[k]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError(f'{name}:{k} shape/dtype changed on round-trip')
            if a.dtype.kind in 'fc' and not np.array_equal(a, b, equal_nan=True):
                raise RuntimeError(f'{name}:{k} values changed on round-trip')
            if a.dtype.kind not in 'fc' and not np.array_equal(a, b):
                raise RuntimeError(f'{name}:{k} values changed on round-trip')
        w.close()
        manifest['files'][name] = {
            'arrays': {k: {'shape': list(a.shape), 'dtype': str(a.dtype)} for k, a in arrays.items()},
            'src_bytes': os.path.getsize(src),
            'dst_bytes': os.path.getsize(dst),
            'src_md5': md5(src),
            'dst_md5': md5(dst),
        }
        print(f'{name}: {os.path.getsize(src):,} -> {os.path.getsize(dst):,} B, '
              f'arrays {list(arrays)} OK', flush=True)

    with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'wrote {os.path.join(OUT, "manifest.json")}')


if __name__ == '__main__':
    main()
