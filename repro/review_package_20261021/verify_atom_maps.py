# -*- coding: utf-8 -*-
"""Structural verification of data_indices/atom_maps.npz.

Checks, for every one of the 90,377 records:
  * n_atoms equals the number of heavy atoms in the SMILES and the number of
    non-hydrogen atoms in the XYZ block;
  * sm2xyz entries are unique, in range, and point to atoms of the same element;
  * the mapping preserves every bond of the SMILES heavy-atom graph.
The stage counts must also equal the `match_src` counts stored in
new_domain_iso.pt.
"""
import gzip
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

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'data_indices', 'atom_maps.npz')
ROOT = r'D:\ml\chem_ml5\data_tmQM\local_vB'
PT_FILE = r'D:\ml\chem_ml5\new_domain_iso.pt'
PERIODIC = Chem.GetPeriodicTable()


def parse_xyz(path):
    blocks = []
    with gzip.open(path, 'rt') as f:
        lines = f.read().strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line:
            continue
        if line.isdigit():
            n = int(line)
            i += 1  # comment
            syms, coords = [], []
            for _ in range(n):
                if i >= len(lines):
                    break
                cp = lines[i].strip().split(); i += 1
                if len(cp) >= 4:
                    syms.append(cp[0])
                    coords.append([float(x) for x in cp[1:4]])
            blocks.append((syms, np.asarray(coords, dtype=np.float32)))
    return blocks


def main():
    z = np.load(NPZ)
    csv_idx = z['csv_idx']; sm2xyz = z['sm2xyz']; n_atoms = z['n_atoms']
    match_src = z['match_src']
    n = len(csv_idx)
    print(f'{n:,} maps, width {sm2xyz.shape[1]}, stages '
          f'{dict(zip(*np.unique(match_src, return_counts=True)))}', flush=True)

    d = torch.load(PT_FILE, map_location='cpu', weights_only=False)
    ref = np.asarray(d['csv_idx'], dtype=np.int64)
    if not np.array_equal(csv_idx, ref):
        raise RuntimeError('csv_idx differs from new_domain_iso.pt')
    import collections
    stored = collections.Counter(d['match_src'])
    got = collections.Counter(match_src.tolist())
    if stored != got:
        raise RuntimeError(f'stage counts {got} != stored {stored}')

    y = pd.read_csv(os.path.join(ROOT, 'tmQM_y.csv'), sep=';')
    smiles = y['SMILES'].tolist()
    xyz = []
    for p in (1, 2, 3):
        xyz.extend(parse_xyz(os.path.join(ROOT, f'tmQM_X{p}.xyz.gz')))
    print(f'parsed {len(xyz):,} XYZ blocks', flush=True)

    bad = 0
    for k in range(n):
        bidx = int(csv_idx[k])
        mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles[bidx]))
        syms, coords = xyz[bidx]
        heavy = [i for i, s in enumerate(syms) if s != 'H']
        na = int(n_atoms[k])
        if na != mol.GetNumAtoms() or na != len(heavy):
            bad += 1; print(f'  atom count mismatch at {k}', flush=True); continue
        mp = sm2xyz[k, :na].astype(int)
        if len(set(mp.tolist())) != na or mp.min() < 0 or mp.max() >= len(syms):
            bad += 1; print(f'  map not injective/in range at {k}', flush=True); continue
        for i, j in enumerate(mp):
            if PERIODIC.GetAtomicNumber(syms[j]) != mol.GetAtomWithIdx(i).GetAtomicNum():
                bad += 1; print(f'  element mismatch at {k} atom {i}', flush=True); break
        else:
            long_bond = False
            for b in mol.GetBonds():
                j1, j2 = mp[b.GetBeginAtomIdx()], mp[b.GetEndAtomIdx()]
                d = float(np.linalg.norm(coords[j1] - coords[j2]))
                if d > 4.0:
                    long_bond = True
                    break
            if long_bond:
                bad += 1; print(f'  mapped bond too long at {k}', flush=True)
        if (k + 1) % 20000 == 0:
            print(f'  checked {k + 1:,}', flush=True)

    print(f'\nbad records: {bad}')
    print('RESULT:', 'ALL PASS' if bad == 0 else 'FAILED')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
