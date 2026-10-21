# -*- coding: utf-8 -*-
"""Recover the SMILES -> XYZ atom mapping for the 90,377 retained complexes.

The mapping is not stored in new_domain_iso.pt, but it is the missing link
between the public SMILES in tmQM_y.csv and the atom order of the XYZ files.
This script reruns the exact matching chain of
analysis/rebuild_match_iso.py for the retained records only:

    BO graph (bond order >= 0.2) -> distance graph (tol 0.45) -> distance graph (tol 0.60)
    each matched with networkx vf2pp on the heavy-atom element-labelled graph

and then verifies every map against the arrays that were built from it in
new_domain_iso.pt:
    * the fallback stage equals the stored `match_src`;
    * the recomputed node3s row ([BO neighbour count, sum, max]) equals the stored row;
    * every directed bond's RBF vector and BO value equal the stored values.
A record is only written if all three checks pass.

Output data_indices/atom_maps.npz:
    csv_idx       (90377,) int64   row in tmQM_y.csv / XYZ file order
    csd_codes     (90377,) <U6
    match_src     (90377,) <U8
    n_atoms       (90377,) int32   heavy atoms in SMILES order
    sm2xyz        (90377, 101) int32 padded; sm2xyz[i, k] = XYZ atom index of
                                     SMILES atom k, -1 beyond n_atoms[i]
                                     (101 = the largest heavy-atom count in the domain)

Run with the RDKit 2024.03.6 environment, which is the training stack:
    C:/Users/AS/miniconda3/envs/rdkit2024/python.exe export_atom_maps.py [--limit N]
"""
import argparse
import gzip
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.algorithms.isomorphism import vf2pp_isomorphism
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data_indices', 'atom_maps.npz')
ROOT = r'D:\ml\chem_ml5\data_tmQM\local_vB'
PT_FILE = r'D:\ml\chem_ml5\new_domain_iso.pt'
BO_THRESH = 0.2
COV = {1: 0.31, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
       11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
       19: 2.03, 20: 1.76, 21: 1.44, 22: 1.47, 23: 1.53, 24: 1.39, 25: 1.50,
       26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 33: 1.19, 34: 1.16,
       35: 1.20, 39: 1.76, 40: 1.75, 41: 1.64, 42: 1.47, 43: 1.56, 44: 1.46,
       45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 53: 1.39, 72: 1.75, 73: 1.70,
       74: 1.46, 75: 1.44, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32}
TM_SET = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
RBF_C = np.linspace(1.0, 5.0, 16).astype(np.float32)
RBF_SIGMA = 0.2
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
            comment = lines[i].strip(); i += 1
            code = comment.split('=')[1].split('|')[0].strip() if 'CSD_code' in comment else '?'
            syms, coords = [], []
            for _ in range(n):
                if i >= len(lines):
                    break
                cp = lines[i].strip().split(); i += 1
                if len(cp) >= 4:
                    try:
                        syms.append(cp[0]); coords.append([float(x) for x in cp[1:4]])
                    except ValueError:
                        pass
            blocks.append((code, syms, np.array(coords, dtype=np.float32)))
    return blocks


def parse_bo(path):
    blocks = {}
    with gzip.open(path, 'rt') as f:
        lines = [l.strip() for l in f.readlines()]
    cur_code, cur = None, []
    for line in lines:
        if line.startswith('CSD_code'):
            if cur_code is not None and cur:
                blocks[cur_code] = cur
            cur_code = line.split('=')[1].split('|')[0].strip(); cur = []
        elif line and cur_code is not None:
            cur.append(line)
    if cur_code is not None and cur:
        blocks[cur_code] = cur
    out = {}
    for code, blk in blocks.items():
        srcs, nbrs, bos = [], [], []
        for bline in blk:
            p = bline.split()
            if len(p) < 4:
                continue
            src = int(p[0]) - 1
            for t in range(3, len(p) - 2, 3):
                try:
                    srcs.append(src); nbrs.append(int(p[t + 1]) - 1); bos.append(float(p[t + 2]))
                except (ValueError, IndexError):
                    continue
        out[code] = (np.array(srcs, dtype=np.int64), np.array(nbrs, dtype=np.int64),
                     np.array(bos, dtype=np.float32))
    return out


def build_sm_heavy(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    ms = Chem.RWMol(mol)
    for b in ms.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
    for a in ms.GetAtoms():
        a.SetIsAromatic(False)
    G = nx.Graph()
    for a in ms.GetAtoms():
        G.add_node(a.GetIdx(), z=a.GetAtomicNum())
    for b in ms.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return G


def build_xyz_bo_graph(syms, code, bo_by_code):
    zs = [PERIODIC.GetAtomicNumber(s) for s in syms]
    heavy = [k for k in range(len(zs)) if zs[k] != 1]
    G = nx.Graph()
    for k in heavy:
        G.add_node(k, z=zs[k])
    entry = bo_by_code.get(code)
    if entry is not None:
        bos_src, bos_nbr, bos_val = entry
        for s, nb, bv in zip(bos_src, bos_nbr, bos_val):
            s, nb = int(s), int(nb)
            if float(bv) >= BO_THRESH and zs[s] != 1 and zs[nb] != 1 and s < nb:
                G.add_edge(s, nb)
    return G, zs, heavy


def build_xyz_dist_graph(syms, coords, tol):
    zs = [PERIODIC.GetAtomicNumber(s) for s in syms]
    heavy = [k for k in range(len(zs)) if zs[k] != 1]
    G = nx.Graph()
    for k in heavy:
        G.add_node(k, z=zs[k])
    for ii in range(len(heavy)):
        for jj in range(ii + 1, len(heavy)):
            a, b = heavy[ii], heavy[jj]
            d = float(np.linalg.norm(coords[a] - coords[b]))
            if d < COV.get(zs[a], 1.5) + COV.get(zs[b], 1.5) + tol:
                G.add_edge(a, b)
    return G, zs, heavy


def exact_iso(Gx, Gs):
    if len(Gx) != len(Gs):
        return None
    m = vf2pp_isomorphism(Gx, Gs, node_label='z')
    if m is None:
        return None
    sm2xyz = [-1] * len(Gs)
    for x_i, s_i in m.items():
        sm2xyz[s_i] = x_i
    for i, j in Gs.edges():
        if not Gx.has_edge(sm2xyz[i], sm2xyz[j]):
            return None
    return sm2xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='only the first N retained records')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    d = torch.load(PT_FILE, map_location='cpu', weights_only=False)
    csv_idx_all = np.asarray(d['csv_idx'], dtype=np.int64)
    csd_pt = list(d['csd_codes'])
    match_src_pt = list(d['match_src'])
    node3s_pt = d['node3s']
    rbf_pt = d['rbf_ls']
    bo1_pt = d['bo1_ls']
    n_total = len(csv_idx_all)
    n_run = n_total if args.limit <= 0 else min(args.limit, n_total)
    print(f'retained {n_total:,}; processing {n_run:,}', flush=True)

    y_csv = pd.read_csv(os.path.join(ROOT, 'tmQM_y.csv'), sep=';')
    smiles = y_csv['SMILES'].tolist()

    print('parsing XYZ ...', flush=True)
    xyz = []
    for p in (1, 2, 3):
        xyz.extend(parse_xyz(os.path.join(ROOT, f'tmQM_X{p}.xyz.gz')))
        print(f'  part {p}: cum {len(xyz):,}', flush=True)
    print('parsing BO ...', flush=True)
    bo = {}
    for p in (1, 2, 3):
        bo.update(parse_bo(os.path.join(ROOT, f'tmQM_X{p}.BO.gz')))
    print(f'  BO codes {len(bo):,}', flush=True)

    max_n = max(len(a) for a in d['node8s'])
    sm2xyz = np.full((n_run, max_n), -1, dtype=np.int32)
    n_atoms = np.zeros(n_run, dtype=np.int32)
    match_src = np.empty(n_run, dtype='U8')
    t0 = time.time()
    for k in range(n_run):
        bidx = int(csv_idx_all[k])
        code, syms, coords = xyz[bidx]
        smi = smiles[bidx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f'record {k} (csv {bidx}) SMILES failed to parse')
        mol = Chem.RemoveHs(mol)
        Gs = build_sm_heavy(smi)
        if Gs is None:
            raise RuntimeError(f'record {k} (csv {bidx}) graph build failed')
        Gx, _, _ = build_xyz_bo_graph(syms, code, bo)
        mp = exact_iso(Gx, Gs)
        src = 'BO0.2'
        if mp is None:
            Gx2, _, _ = build_xyz_dist_graph(syms, coords, 0.45)
            mp = exact_iso(Gx2, Gs)
            src = 'dist0.45'
            if mp is None:
                Gx3, _, _ = build_xyz_dist_graph(syms, coords, 0.60)
                mp = exact_iso(Gx3, Gs)
                src = 'dist0.60'
        if mp is None:
            raise RuntimeError(f'record {k} (csv {bidx}) matching failed')
        if src != match_src_pt[k]:
            raise RuntimeError(f'record {k} (csv {bidx}): stage {src} != stored {match_src_pt[k]}')

        # verify against the arrays that were built from this map
        bos_src, bos_nbr, bos_val = bo[code]
        n_full = len(syms)
        bm = np.zeros((n_full, n_full), dtype=np.float32)
        for s, nb, bv in zip(bos_src, bos_nbr, bos_val):
            if 0 <= s < n_full and 0 <= nb < n_full:
                bm[s, nb] = bv
        xf3 = []
        for i in range(mol.GetNumAtoms()):
            j = mp[i]
            if j < 0 or j >= n_full:
                xf3.append([0, 0.0, 0.0]); continue
            row = bm[j]
            xf3.append([max(0, int((row > 0.05).sum()) - 1),
                        max(0.0, float(row.sum()) - float(row[j])),
                        max(0.0, float(row.max()))])
        if not np.allclose(np.asarray(xf3, dtype=np.float32), node3s_pt[k], atol=1e-6):
            raise RuntimeError(f'record {k} (csv {bidx}): node3s mismatch')

        rbf_rows, bo_rows = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            fi, fj = mp[i], mp[j]
            dd = float(np.linalg.norm(coords[fi] - coords[fj]))
            rv = np.exp(-(dd - RBF_C) ** 2 / RBF_SIGMA).astype(np.float32)
            bv = float(bm[fi, fj])
            for _ in range(2):
                rbf_rows.append(rv); bo_rows.append(bv)
        if rbf_rows:
            if not np.allclose(np.stack(rbf_rows), rbf_pt[k].numpy(), atol=1e-6, rtol=0):
                raise RuntimeError(f'record {k} (csv {bidx}): RBF mismatch')
            if not np.allclose(np.array(bo_rows, dtype=np.float32).reshape(-1, 1),
                               bo1_pt[k].numpy(), atol=1e-6, rtol=0):
                raise RuntimeError(f'record {k} (csv {bidx}): BO mismatch')

        sm2xyz[k, :len(mp)] = np.asarray(mp, dtype=np.int32)
        n_atoms[k] = len(mp)
        match_src[k] = src
        if (k + 1) % 5000 == 0:
            print(f'  {k + 1:,}/{n_run:,} [{time.time() - t0:.0f}s]', flush=True)

    np.savez_compressed(args.out,
                        csv_idx=csv_idx_all[:n_run],
                        csd_codes=np.asarray(csd_pt, dtype='U6')[:n_run],
                        match_src=match_src,
                        n_atoms=n_atoms,
                        sm2xyz=sm2xyz)
    print(f'wrote {args.out}: {n_run:,} maps, max atoms {int(n_atoms.max())}, '
          f'[{time.time() - t0:.0f}s]', flush=True)
    print(f'stage counts: {dict(zip(*np.unique(match_src, return_counts=True)))}', flush=True)


if __name__ == '__main__':
    main()
