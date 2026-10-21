# -*- coding: utf-8 -*-
"""Re-evaluate the SMILES-only GAT on exactly the 5,000 held-out complexes used
by eval_fig4_zeroed.py, so that revised Fig. 4 panels (a)-(f) share one common
subset.

Panel (c) of the published figure used the deployed checkpoint
models/tmqm_gat_best_backup.pt (8d/6d) with whole-CSV mean/std for
inverse-transformation.  The same convention is used here, so the new number is
directly comparable with the published 0.20 mHa, only the complex subset changes
(from a dataset-wide sample to a held-out-test sample).

Input : fig4_zeroed_results.npz produced by eval_fig4_zeroed.py
Output: fig4_smiles_results_subset.npz with the keys fig4_plot.py expects
        (gap_pred, gap_derived, deviation) plus sample_global_idx / csv_idx.

Usage:
  python eval_smiles_same_subset.py --results fig4_zeroed_results.npz \
      --data new_domain_iso.pt --csv tmQM_y.csv --ckpt tmqm_gat_best_backup.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_zeroed_lib import XYZ_GNN, collate_graphs, set_seed  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ELEMENT_DATA = {1: (1, 1), 5: (2, 13), 6: (2, 14), 7: (2, 15), 8: (2, 16), 9: (2, 17),
                14: (3, 14), 15: (3, 15), 16: (3, 16), 17: (3, 17), 20: (4, 2),
                21: (4, 3), 22: (4, 4), 23: (4, 5), 24: (4, 6), 25: (4, 7), 26: (4, 8),
                27: (4, 9), 28: (4, 10), 29: (4, 11), 30: (4, 12), 35: (4, 17),
                42: (5, 6), 44: (5, 8), 45: (5, 9), 46: (5, 10), 47: (5, 11), 48: (5, 12),
                53: (5, 17), 72: (6, 4), 73: (6, 5), 74: (6, 6), 75: (6, 7), 76: (6, 8),
                77: (6, 9), 78: (6, 10), 79: (6, 11), 80: (6, 12)}
BOND_TYPES = [1, 2, 3, 12]
B2I = {b: i for i, b in enumerate(BOND_TYPES)}
N_BT = len(BOND_TYPES)
TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']


def atom_features(a):
    z = a.GetAtomicNum()
    p, g = ELEMENT_DATA.get(z, (0, 0))
    return [z, a.GetDegree(), a.GetFormalCharge(), p, g,
            int(a.GetIsAromatic()), a.GetTotalNumHs(), int(z >= 21)]


def smiles_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, battr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(b.GetBondType())
        za = mol.GetAtomWithIdx(i).GetAtomicNum()
        zb = mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21))
        is_ring = float(b.IsInRing())
        src += [i, j]
        dst += [j, i]
        bf = [0.0] * N_BT
        if bt in B2I:
            bf[B2I[bt]] = 1.0
        bf += [is_ring, is_ml]
        battr += [bf, bf]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, N_BT + 2)
    return {'x': x, 'edge_index': ei, 'edge_attr': ea}


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(here, 'fig4_zeroed_results.npz'))
    ap.add_argument('--data', default=os.path.join(here, 'new_domain_iso.pt'))
    ap.add_argument('--csv', default=os.path.join(here, 'tmQM_y.csv'))
    ap.add_argument('--ckpt', default=os.path.join(here, 'tmqm_gat_best_backup.pt'))
    ap.add_argument('--out', default=os.path.join(here, 'fig4_smiles_results_subset.npz'))
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='auto')
    return ap.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device if args.device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
    set_seed(42)
    res = np.load(args.results)
    global_idx = res['sample_global_idx'].astype(np.int64)
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    csv_rows = data['csv_idx'][global_idx]
    print(f'Subset: {len(global_idx):,} complexes from {args.results}', flush=True)

    df = pd.read_csv(args.csv, sep=';').rename(columns={
        'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
        'Polarizability': 'polarizability'})

    graphs, keep = [], []
    for k, row in enumerate(csv_rows):
        g = smiles_graph(df.iloc[int(row)]['SMILES'])
        if g is None:
            print(f'  [warn] SMILES parse failed at csv row {row}; skipped')
            continue
        graphs.append(g)
        keep.append(k)
    keep = np.asarray(keep)
    print(f'Built {len(graphs):,} SMILES graphs', flush=True)

    y_mean = df[TARGETS].mean().values.astype(np.float32)
    y_std = df[TARGETS].std().values.astype(np.float32)

    model = XYZ_GNN(node_dim=8, edge_dim=6).to(dev)
    state = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Loaded {args.ckpt}', flush=True)

    preds = []
    with torch.no_grad():
        for s in range(0, len(graphs), args.batch_size):
            bg, bi = collate_graphs([{k: v.to(dev) for k, v in g.items()}
                                     for g in graphs[s:s + args.batch_size]])
            preds.append(model(bg, bi).float().cpu())
    pred = torch.cat(preds, dim=0).numpy() * y_std + y_mean
    gap_pred, homo_pred, lumo_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    gap_derived = lumo_pred - homo_pred
    deviation = np.abs(gap_pred - gap_derived)
    print(f'SMILES GAT on held-out subset: mean={deviation.mean() * 1000:.3f} mHa  '
          f'<0.001={np.mean(deviation < 0.001) * 100:.2f}%  '
          f'<0.01={np.mean(deviation < 0.01) * 100:.2f}%  '
          f'R2={np.corrcoef(gap_pred, gap_derived)[0, 1] ** 2:.4f}', flush=True)

    np.savez_compressed(args.out,
                        gap_pred=gap_pred, gap_derived=gap_derived,
                        deviation=deviation, homo_pred=homo_pred, lumo_pred=lumo_pred,
                        sample_global_idx=global_idx[keep], csv_idx=csv_rows[keep])
    print(f'Saved {args.out} ({os.path.getsize(args.out):,} bytes)')


if __name__ == '__main__':
    main()
