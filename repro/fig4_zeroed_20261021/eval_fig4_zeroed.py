# -*- coding: utf-8 -*-
"""Evaluate the retrained zero-padded Fig. 4 model on the held-out test split.

The model was trained on fold_0_train (51,214) and validated on fold_0_val
(25,607); the 13,556 test complexes in ensemble_split_new.npz are never seen
during training.  Predictions are made with the same zeroed 11d/23d layout used
in training.

Outputs (-o):
  fig4_zeroed_results.npz   arrays for replotting Fig. 4 plus full-test arrays
Printout: mean |gap - (LUMO-HOMO)| in mHa, compliance rates, R^2, violation rate,
          on both the 5,000-complex figure subset (seed 42) and all 13,556 test complexes.

Usage:
  python eval_fig4_zeroed.py --data new_domain_iso.pt --split ensemble_split_new.npz \
      --ckpt fig4_zeroed_best.pt --meta fig4_zeroed_meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_zeroed_lib import (  # noqa: E402
    EDGE_DIM, NODE_DIM, GraphDataset, XYZ_GNN, build_graphs, check_layout,
    collate_pairs, set_seed,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(here, 'new_domain_iso.pt'))
    ap.add_argument('--split', default=os.path.join(here, 'ensemble_split_new.npz'))
    ap.add_argument('--ckpt', default=os.path.join(here, 'fig4_zeroed_best.pt'))
    ap.add_argument('--meta', default=os.path.join(here, 'fig4_zeroed_meta.json'))
    ap.add_argument('--out', default=os.path.join(here, 'fig4_zeroed_results.npz'))
    ap.add_argument('--sample-size', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='auto')
    return ap.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device if args.device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = dev.type == 'cuda'
    set_seed(args.seed)
    print(f'Device: {dev}', flush=True)

    with open(args.meta, encoding='utf-8') as f:
        meta = json.load(f)
    if meta.get('model', {}).get('node_dim') != NODE_DIM or \
       meta.get('model', {}).get('edge_dim') != EDGE_DIM:
        raise ValueError('meta.json does not describe an 11d/23d model')
    y_mean = torch.tensor(meta['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(meta['y_std'], dtype=torch.float32)

    data = torch.load(args.data, map_location='cpu', weights_only=False)
    te_idx = np.asarray(data['te_idx']).astype(np.int64)
    if os.path.exists(args.split):
        npz_te = np.load(args.split)['te_idx'].astype(np.int64)
        if set(npz_te.tolist()) != set(te_idx.tolist()):
            print(f'[warn] {args.split} te_idx differs from new_domain_iso.pt; '
                  f'using the latter ({len(te_idx):,} complexes)', flush=True)
    y_gap = data['y_all'][te_idx, 0].float()
    print(f'Test complexes: {len(te_idx):,}', flush=True)

    graphs = build_graphs(data, te_idx)
    check_layout(graphs, spots=(0, len(graphs) // 2, len(graphs) - 1))
    ds = GraphDataset(graphs, te_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_pairs, pin_memory=(dev.type == 'cuda'))

    model = XYZ_GNN().to(dev)
    state = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'Loaded {args.ckpt} ({sum(p.numel() for p in model.parameters()):,} params)', flush=True)

    preds = []
    with torch.no_grad():
        for _, batched, batch_idx in loader:
            batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
            batch_idx = batch_idx.to(dev, non_blocking=True)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                preds.append(model(batched, batch_idx).float().cpu())
    pred = torch.cat(preds, dim=0) * y_std + y_mean
    gap_pred = pred[:, 0].numpy()
    homo_pred = pred[:, 1].numpy()
    lumo_pred = pred[:, 2].numpy()
    gap_derived = lumo_pred - homo_pred
    deviation = np.abs(gap_pred - gap_derived)
    gap_true = y_gap.numpy()

    rs = np.random.RandomState(args.seed)
    n_sample = min(args.sample_size, len(te_idx))
    sample_pos = rs.choice(len(te_idx), size=n_sample, replace=False)
    sample_global = te_idx[sample_pos]

    def report(tag, d, gp, gd):
        r2 = np.corrcoef(gp, gd)[0, 1] ** 2
        print(f'[{tag}] n={len(d):,}  mean={d.mean() * 1000:.3f} mHa  '
              f'compliance<0.01Ha={(d < 0.01).mean() * 100:.2f}%  '
              f'<0.001Ha={(d < 0.001).mean() * 100:.2f}%  '
              f'R2(gap vs derived)={r2:.4f}  '
              f'violation={(gd < 0).mean() * 100:.2f}%', flush=True)

    report('test all', deviation, gap_pred, gap_derived)
    report(f'sample n={n_sample}', deviation[sample_pos],
           gap_pred[sample_pos], gap_derived[sample_pos])

    # Per-target test MAE (sanity check only; Fig. 4 uses gap/homo/lumo)
    targets = meta.get('targets', ['gap', 'homo', 'lumo', 'dipole', 'metal_charge',
                                   'E_elec', 'E_disp', 'polarizability'])
    y_true_all = data['y_all'][te_idx].float().numpy()
    mae = np.abs(pred.numpy() - y_true_all).mean(axis=0)
    print('Test MAE per target: ' + '  '.join(f'{t}={mae[i]:.4f}' for i, t in enumerate(targets)),
          flush=True)

    np.savez_compressed(
        args.out,
        test_idx=te_idx,
        sample_pos=sample_pos,
        sample_global_idx=sample_global,
        gap_pred=gap_pred[sample_pos],
        gap_derived=gap_derived[sample_pos],
        deviation=deviation[sample_pos],
        homo_pred=homo_pred[sample_pos],
        lumo_pred=lumo_pred[sample_pos],
        gap_true=gap_true[sample_pos],
        full_gap_pred=gap_pred,
        full_gap_derived=gap_derived,
        full_deviation=deviation,
        full_homo_pred=homo_pred,
        full_lumo_pred=lumo_pred,
        full_gap_true=gap_true,
        y_mean=meta['y_mean'],
        y_std=meta['y_std'],
        test_mae=mae,
    )
    print(f'Saved {args.out} ({os.path.getsize(args.out):,} bytes)')


if __name__ == '__main__':
    main()
