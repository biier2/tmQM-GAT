# -*- coding: utf-8 -*-
"""Evaluate the R2-3e gap-identity-penalty runs on the held-out 13,556 test set.

For every checkpoint produced by train_fig4_physL1.py this reports
  * internal consistency  |gap - (LUMO - HOMO)|  (mean / median / max, mHa)
  * accuracy against DFT  gap / HOMO / LUMO MAE (mHa) and R^2
  * all eight per-target MAEs (the other five should be untouched)
and then aggregates across seeds and runs a paired comparison of each lambda
against the lambda = 0 arm complex by complex.

Outputs
  --out-json   metrics per run, per-lambda mean/sd, paired statistics
  --out-npz    test predictions for every run (for replotting)

Usage:
  python eval_r23e.py --data new_domain_iso.pt --split ensemble_split_new.npz \
      --ckpt-dir out --glob 'physL1_lam*_s*_best.pt' --out-json r23e_metrics.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_zeroed_lib import (  # noqa: E402
    GraphDataset, XYZ_GNN, build_graphs, check_layout, collate_pairs, set_seed,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
IG, IH, IL = 0, 1, 2


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(here, 'new_domain_iso.pt'))
    ap.add_argument('--split', default=os.path.join(here, 'ensemble_split_new.npz'))
    ap.add_argument('--ckpt-dir', default=os.path.join(here, 'out'))
    ap.add_argument('--glob', default='physL1_lam*_s*_best.pt')
    ap.add_argument('--out-json', default=os.path.join(here, 'out', 'r23e_metrics.json'))
    ap.add_argument('--out-npz', default=os.path.join(here, 'out', 'r23e_test_preds.npz'))
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--boot-seed', type=int, default=20261115)
    return ap.parse_args()


def infer(model, loader, dev, use_amp, y_mean, y_std):
    preds = []
    with torch.no_grad():
        for _, batched, batch_idx in loader:
            batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
            batch_idx = batch_idx.to(dev, non_blocking=True)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                preds.append(model(batched, batch_idx).float().cpu())
    return torch.cat(preds, dim=0) * y_std + y_mean


def metrics(pred, y_true, idx):
    out = {}
    g, h, l = pred[idx, IG], pred[idx, IH], pred[idx, IL]
    dev = np.abs(g - (l - h))
    out['internal_mean_mha'] = float(dev.mean() * 1000)
    out['internal_median_mha'] = float(np.median(dev) * 1000)
    out['internal_max_mha'] = float(dev.max() * 1000)
    out['internal_p99_mha'] = float(np.percentile(dev, 99) * 1000)
    for i, t in enumerate(TARGETS):
        e = np.abs(pred[idx, i] - y_true[idx, i])
        out[f'mae_{t}'] = float(e.mean())
        if i < 3:                                  # gap / HOMO / LUMO are in Ha
            out[f'mae_{t}_mha'] = float(e.mean() * 1000)
        ss_res = float(((pred[idx, i] - y_true[idx, i]) ** 2).sum())
        ss_tot = float(((y_true[idx, i] - y_true[idx, i].mean()) ** 2).sum())
        out[f'r2_{t}'] = 1.0 - ss_res / ss_tot
    out['r2_mean8'] = float(np.mean([out[f'r2_{t}'] for t in TARGETS]))
    return out


def parse_tag(tag):
    """physL1_lam1e4_s123 -> (1e4, 123); physL1_lam0_s42 -> (0.0, 42).

    Any other tag (e.g. a reference checkpoint) gets lambda NaN and seed -1 so
    that it is reported but excluded from the per-lambda aggregation.
    """
    if not tag.startswith('physL1_lam'):
        return float('nan'), -1
    body = tag[len('physL1_lam'):]
    lam_s, seed_s = body.rsplit('_s', 1)
    return float(lam_s), int(seed_s)


def main():
    args = parse_args()
    dev = torch.device(args.device if args.device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = dev.type == 'cuda'
    set_seed(0)
    print(f'Device: {dev}', flush=True)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.glob)))
    if not ckpts:
        raise SystemExit(f'no checkpoints matched {args.glob} in {args.ckpt_dir}')
    print(f'{len(ckpts)} checkpoints', flush=True)

    data = torch.load(args.data, map_location='cpu', weights_only=False)
    te_idx = np.asarray(data['te_idx']).astype(np.int64)
    if os.path.exists(args.split):
        npz_te = np.load(args.split)['te_idx'].astype(np.int64)
        if not np.array_equal(npz_te, te_idx):
            raise SystemExit('split file and data tensor disagree on te_idx')
    n = len(data['y_all'])
    perm = np.random.RandomState(42).permutation(n)
    if not np.array_equal(te_idx, perm[:13556]):
        raise SystemExit('te_idx is not RandomState(42).permutation(N)[:13556]')
    y_true = data['y_all'].float().numpy()
    te32 = torch.as_tensor(te_idx, dtype=torch.long)

    # DFT-side identity residual, for reference
    dft_dev = np.abs(y_true[te_idx, IG] - (y_true[te_idx, IL] - y_true[te_idx, IH]))
    print(f'DFT reference |gap-(LUMO-HOMO)|: mean {dft_dev.mean() * 1e9:.1f} nHa, '
          f'max {dft_dev.max() * 1e9:.1f} nHa', flush=True)

    graphs = build_graphs(data, te_idx)
    check_layout(graphs, spots=(0, len(graphs) // 2, len(graphs) - 1))
    ds = GraphDataset(graphs, te_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_pairs, pin_memory=(dev.type == 'cuda'))

    runs, preds_by_tag = {}, {}
    for ck in ckpts:
        tag = os.path.basename(ck)[:-len('_best.pt')]
        meta_path = os.path.join(os.path.dirname(ck), f'{tag}_meta.json')
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        y_mean = torch.tensor(meta['y_mean'], dtype=torch.float32)
        y_std = torch.tensor(meta['y_std'], dtype=torch.float32)
        model = XYZ_GNN().to(dev)
        state = torch.load(ck, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        model.eval()
        pred = infer(model, loader, dev, use_amp, y_mean, y_std).numpy()
        lam, seed = parse_tag(tag)
        m = metrics(pred, y_true[te_idx], np.arange(len(te_idx)))
        m.update({'tag': tag, 'lam_sigma': lam, 'seed': seed,
                  'lam_ha': meta.get('penalty', {}).get('lam_ha'),
                  'best_epoch': meta.get('best_epoch'),
                  'epochs_run': meta.get('epochs_run'),
                  'val_internal_dev_mha': meta.get('val_internal_dev_mha')})
        runs[tag] = m
        preds_by_tag[tag] = pred
        print(f'{tag:22s} lam={lam:<8g} seed={seed}  internal {m["internal_mean_mha"]:8.4f} mHa  '
              f'gap MAE {m["mae_gap_mha"]:7.4f} mHa  HOMO {m["mae_homo_mha"]:7.4f}  '
              f'LUMO {m["mae_lumo_mha"]:7.4f}  R2gap {m["r2_gap"]:.4f}', flush=True)
        del model

    # ---------------------------------------------------------------- aggregate
    lams = sorted({r['lam_sigma'] for r in runs.values() if np.isfinite(r['lam_sigma'])})
    agg = {}
    for lam in lams:
        rs = [r for r in runs.values() if r['lam_sigma'] == lam]
        a = {'n_runs': len(rs), 'seeds': sorted(r['seed'] for r in rs)}
        for k in ('internal_mean_mha', 'internal_median_mha', 'internal_max_mha',
                  'mae_gap_mha', 'mae_homo_mha', 'mae_lumo_mha',
                  'r2_gap', 'r2_homo', 'r2_lumo', 'r2_mean8'):
            v = np.array([r[k] for r in rs], dtype=float)
            a[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    'values': [float(x) for x in v]}
        agg[f'{lam:g}'] = a

    print('\n== per-lambda summary (mean +- sd over seeds) ==')
    print(f'{"lam_sigma":>10s} {"n":>2s}  {"internal mHa":>16s}  {"gap MAE mHa":>16s}  '
          f'{"R2gap":>14s}')
    for lam in lams:
        a = agg[f'{lam:g}']
        print(f'{lam:>10g} {a["n_runs"]:2d}  '
              f'{a["internal_mean_mha"]["mean"]:8.4f} +- {a["internal_mean_mha"]["sd"]:6.4f}  '
              f'{a["mae_gap_mha"]["mean"]:8.4f} +- {a["mae_gap_mha"]["sd"]:6.4f}  '
              f'{a["r2_gap"]["mean"]:7.4f} +- {a["r2_gap"]["sd"]:6.4f}', flush=True)

    # ------------------------------------------------- paired vs lambda = 0
    base_tags = sorted(t for t, r in runs.items() if r['lam_sigma'] == 0.0)
    paired = {}
    if base_tags:
        rs = np.random.RandomState(args.boot_seed)
        boot = rs.randint(0, len(te_idx), size=(args.n_boot, len(te_idx)))
        y_te_gap = y_true[te_idx, IG]
        base = np.mean([np.abs(preds_by_tag[t][:, IG] - y_te_gap)
                        for t in base_tags], axis=0)
        for lam in lams:
            if lam == 0:
                continue
            tags = sorted(t for t, r in runs.items() if r['lam_sigma'] == lam)
            cur = np.mean([np.abs(preds_by_tag[t][:, IG] - y_te_gap)
                           for t in tags], axis=0)
            d = cur - base                       # per-complex gap AE change, Ha
            bs = np.array([d[b].mean() for b in boot])
            paired[f'{lam:g}'] = {
                'delta_mae_gap_mha': float(d.mean() * 1000),
                'ci95_mha': [float(np.percentile(bs, 2.5) * 1000),
                             float(np.percentile(bs, 97.5) * 1000)],
                'frac_worse': float((d > 0).mean()),
                'n_test': int(len(te_idx)),
            }
            p = paired[f'{lam:g}']
            print(f'lam={lam:<8g} vs lam=0: d(gap MAE) = {p["delta_mae_gap_mha"]:+.4f} mHa '
                  f'[{p["ci95_mha"][0]:+.4f}, {p["ci95_mha"][1]:+.4f}]  '
                  f'worse on {p["frac_worse"] * 100:.1f}% of complexes', flush=True)

    out = {'test_n': int(len(te_idx)),
           'dft_identity_residual_nha': {'mean': float(dft_dev.mean() * 1e9),
                                         'max': float(dft_dev.max() * 1e9)},
           'runs': runs, 'per_lambda': agg, 'paired_vs_lam0': paired}
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.savez_compressed(args.out_npz,
                        test_idx=te_idx, y_true=y_true[te_idx],
                        **{f'pred__{t}': preds_by_tag[t] for t in preds_by_tag})
    print(f'\nSaved {args.out_json}\nSaved {args.out_npz}', flush=True)


if __name__ == '__main__':
    main()
