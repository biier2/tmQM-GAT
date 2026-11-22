# -*- coding: utf-8 -*-
"""Derive the total molecular charge of every complex in the fixed DimeNet++ domain.

The charge is read from the 'q = ...' field of the tmQM XYZ title lines, in block
order, and mapped onto the training domain through the ``csv_idx`` array stored in
``new_domain_iso.pt``.  The result is ``fixed_q.npz`` with a single array ``q_all``
(90,377 entries, values -1/0/+1) used by ``dimenetpp_q_train.py``.

The csv_idx array can be dumped once from the training file:

    python -c "import numpy as np, torch; \
D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False); \
np.save('csv_idx.npy', D['csv_idx'])"

Usage:
    python build_charge_vector.py --data-dir /path/to/tmQM --csv-idx csv_idx.npy \
        --out fixed_q.npz
"""
import argparse
import gzip
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')


def read_block_charges(data_dir):
    """Charge of every XYZ block, in file order (X1, X2, X3)."""
    q = []
    for fn in ('tmQM_X1.xyz.gz', 'tmQM_X2.xyz.gz', 'tmQM_X3.xyz.gz'):
        with gzip.open(os.path.join(data_dir, fn), 'rt', errors='replace') as h:
            while True:
                line = h.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                n = int(line)
                title = h.readline()
                for _ in range(n):
                    h.readline()
                try:
                    value = int(title.split('q =')[1].split('|')[0].strip())
                except Exception:
                    value = 0
                q.append(value)
    return np.asarray(q, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True,
                    help='directory containing tmQM_X1/X2/X3.xyz.gz')
    ap.add_argument('--csv-idx', required=True,
                    help='npy file with the csv_idx array of new_domain_iso.pt')
    ap.add_argument('--out', default='fixed_q.npz')
    args = ap.parse_args()

    q_block = read_block_charges(args.data_dir)
    print('blocks: %d  charge counts: %s'
          % (len(q_block), dict(zip(*np.unique(q_block, return_counts=True)))))

    csv_idx = np.load(args.csv_idx)
    q_all = q_block[csv_idx.astype(np.int64)]
    print('domain: %d  charge counts: %s'
          % (len(q_all), dict(zip(*np.unique(q_all, return_counts=True)))))

    np.savez(args.out, q_all=q_all)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
