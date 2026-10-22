# -*- coding: utf-8 -*-
"""Retrain the Fig. 4 11d/23d GAT with geometry channels zeroed (review comment id57).

Why: the manuscript states that the 11-node/23-edge single GAT used in Fig. 4 was
zero-padded "during both training and evaluation".  The original checkpoint
(models/xyz_best.pt) was in fact trained on real DFT distances/WBO, so the claim is
false.  This script trains a fresh single model from scratch with those channels
identically zero in training, validation and test, on the paper split (fold-0
train/val + the 13,556 held-out test complexes).  Nothing is overwritten: a new
checkpoint file is written to --out-dir.

Protocol mirrors the paper GAT training (baselines/server_package/server_train_variants.py):
  AdamW lr=1e-3 wd=1e-4, CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6),
  AMP, batch=256, max 200 epochs, early stopping patience=40, grad clip 1.0,
  sqrt metal weights clipped to [0.5, 3.0] and mean-normalised to 1,
  per-model standardisation using training-set mean/std.

Outputs in --out-dir:
  <tag>_best.pt          raw state_dict (drop-in for XYZ_GNN_Eval in fig4_infer.py)
  <tag>_meta.json        config, normalisation stats, best epoch/loss, environment
  <tag>_history.json     train/val loss per epoch

Usage (server):
  python train_fig4_zeroed.py --data new_domain_iso.pt --split ensemble_split_new.npz
  python train_fig4_zeroed.py --smoke          # 2 epochs on a small CPU subset, no save
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4_zeroed_lib import (  # noqa: E402
    DROPOUT, EDGE_DIM, GAT_HEADS, HIDDEN, N_LAYERS, N_TARGETS, NODE_DIM,
    GraphDataset, XYZ_GNN, build_graphs, check_layout, collate_pairs, set_seed,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
EXPECTED_DATA_MD5 = '3035057aeb8638680d676283ace2194b'  # 90,377-complex paper domain, te=13,556


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(here, 'new_domain_iso.pt'))
    ap.add_argument('--split', default=os.path.join(here, 'ensemble_split_new.npz'))
    ap.add_argument('--rebuild-split', action='store_true',
                    help='reconstruct the paper split from new_domain_iso.pt te_idx '
                         '(RandomState(42) permutation + KFold(3, shuffle=True, random_state=42)); '
                         'use when the split npz is missing or from another data version')
    ap.add_argument('--out-dir', default=here)
    ap.add_argument('--tag', default='fig4_zeroed')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny CPU run (2 epochs), checks the pipeline, saves nothing')
    return ap.parse_args()


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rebuild_split(n_total, data_te):
    """Reproduce ensemble_split_new.npz exactly from the data file's te_idx.

    Paper split recipe (baselines/server_package/server_train_variants.py:60-70):
      idx_all = RandomState(42).permutation(N)
      te_idx  = idx_all[:n_test]; cv_idx = idx_all[n_test:]
      KFold(3, shuffle=True, random_state=42) on cv_idx -> fold_0 train/val
    """
    from sklearn.model_selection import KFold
    perm = np.random.RandomState(42).permutation(n_total)
    n_te = len(data_te)
    te = perm[:n_te].astype(np.int64)
    if set(te.tolist()) != set(np.asarray(data_te).tolist()):
        raise ValueError(
            'new_domain_iso.pt te_idx is not the seed-42 permutation; cannot rebuild the paper split. '
            'Check that the data file is the 90,377-complex new_domain_iso.pt and verify its '
            'content with repro/verify_domain_content.py.')
    cv = perm[n_te:].astype(np.int64)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    tr_pos, va_pos = next(iter(kf.split(cv)))
    return cv[tr_pos].astype(np.int64), cv[va_pos].astype(np.int64), te


def main():
    args = parse_args()
    dev = pick_device(args.device)
    use_amp = (dev.type == 'cuda')
    if args.smoke:
        args.max_epochs = 2
        args.workers = 0
        args.out_dir = os.path.join(args.out_dir, '_smoke')
        args.tag = 'fig4_zeroed_smoke'
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f'Device: {dev} | AMP: {use_amp}')
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM {p.total_memory / 1024 ** 3:.1f} GB')

    # ---------------- data ----------------
    t0 = time.time()
    print(f'Loading {args.data} ...', flush=True)
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    need = ['node8s', 'node3s', 'ei_ls', 'e6_ls', 'rbf_ls', 'bo1_ls',
            'y_all', 'atoms_filt']
    missing = [k for k in need if k not in data]
    if missing:
        raise KeyError(f'missing keys in {args.data}: {missing}')
    n_total = len(data['node8s'])
    for k in ['node3s', 'ei_ls', 'e6_ls', 'rbf_ls', 'bo1_ls', 'y_all', 'atoms_filt']:
        if len(data[k]) != n_total:
            raise ValueError(f'length mismatch: {k}={len(data[k])} vs node8s={n_total}')
    y_all = data['y_all'].float()
    if y_all.shape != (n_total, N_TARGETS):
        raise ValueError(f'y_all shape {tuple(y_all.shape)} != ({n_total}, {N_TARGETS})')
    if not torch.isfinite(y_all).all():
        raise ValueError('y_all contains non-finite values')
    data_md5 = file_md5(args.data)
    print(f'N={n_total:,}  loaded in {time.time() - t0:.1f}s', flush=True)
    print(f'Data md5: {data_md5}', flush=True)
    if data_md5 != EXPECTED_DATA_MD5:
        print(f'[warn] file md5 differs from the authors\' copy {EXPECTED_DATA_MD5}. '
              'This is expected for a rebuilt tensor (torch.save is not byte-deterministic); '
              'verify the contents with repro/verify_domain_content.py.', flush=True)

    # ---------------- split ----------------
    data_te = np.asarray(data['te_idx']).astype(np.int64)
    if n_total == 90377 and len(data_te) != 13556:
        raise ValueError(
            f'new_domain_iso.pt has {len(data_te):,} test indices; the paper split uses 13,556. '
            'This is the old 13,557-index data version (md5 17d34723...). Use the correct file: '
            f'md5 {EXPECTED_DATA_MD5}, 1,107,170,889 bytes.')
    split_ok = False
    if os.path.exists(args.split):
        z = np.load(args.split)
        npz_te = z['te_idx'].astype(np.int64)
        if set(npz_te.tolist()) == set(data_te.tolist()):
            split_ok = True
            tr_idx = z['fold_0_train'].astype(np.int64)
            va_idx = z['fold_0_val'].astype(np.int64)
            te_idx = npz_te
            print(f'Split file: {args.split} (md5 {file_md5(args.split)})', flush=True)
        else:
            print('[warn] split te_idx differs from new_domain_iso.pt te_idx: '
                  f'npz n={len(npz_te):,}, data n={len(data_te):,}, '
                  f'intersection={len(set(npz_te.tolist()) & set(data_te.tolist())):,}',
                  flush=True)
    else:
        print(f'[warn] split file not found: {args.split}', flush=True)

    if not split_ok:
        if not args.rebuild_split:
            print('[hint] the split npz is missing or from another data version. Re-run with '
                  '--rebuild-split to reconstruct the exact paper split from '
                  'new_domain_iso.pt te_idx (seed 42).', flush=True)
            raise ValueError('split file missing or inconsistent; use --rebuild-split')
        tr_idx, va_idx, te_idx = rebuild_split(n_total, data_te)
        print(f'[rebuild] paper split reconstructed from new_domain_iso.pt te_idx: '
              f'train={len(tr_idx):,} val={len(va_idx):,} test={len(te_idx):,}', flush=True)

    for name, arr in [('fold_0_train', tr_idx), ('fold_0_val', va_idx), ('te_idx', te_idx)]:
        if arr.min() < 0 or arr.max() >= n_total:
            raise ValueError(f'{name} out of range')
    if len(set(tr_idx.tolist()) & set(va_idx.tolist())) or \
       len(set(tr_idx.tolist()) & set(te_idx.tolist())) or \
       len(set(va_idx.tolist()) & set(te_idx.tolist())):
        raise ValueError('splits overlap')
    print(f'Split: train={len(tr_idx):,}  val={len(va_idx):,}  test={len(te_idx):,}', flush=True)

    if args.smoke:
        rs = np.random.RandomState(0)
        tr_idx = np.sort(rs.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False))
        va_idx = np.sort(rs.choice(va_idx, size=min(1000, len(va_idx)), replace=False))

    # ---------------- zeroed graphs ----------------
    t0 = time.time()
    build_idx = np.union1d(tr_idx, va_idx)
    graphs = build_graphs(data, build_idx)
    check_layout(graphs, spots=(0, len(graphs) // 2, len(graphs) - 1))
    pos = {int(g): k for k, g in enumerate(build_idx)}
    src_pts = [int(i) for i in np.asarray(build_idx)[[0, len(build_idx) // 2, len(build_idx) - 1]]]
    for i in src_pts:
        g = graphs[pos[i]]
        assert g['x'].size(1) == NODE_DIM and g['edge_attr'].size(1) == EDGE_DIM
        assert float(g['x'][:, 8:11].abs().max()) == 0.0
        if g['edge_attr'].numel():
            assert float(g['edge_attr'][:, 6:23].abs().max()) == 0.0
    print(f'Built {len(graphs):,} zeroed 11d/23d graphs in {time.time() - t0:.1f}s', flush=True)

    # ---------------- metal weights (paper protocol) ----------------
    mc = Counter()
    for atoms in data['atoms_filt']:
        for zz in set(np.asarray(atoms).tolist()):
            if zz >= 21:
                mc[zz] += 1
    weight_map = {zz: max(0.5, min(3.0, (n_total / c) ** 0.5)) for zz, c in mc.items()}
    sample_w = torch.ones(n_total, dtype=torch.float32)
    for i, atoms in enumerate(data['atoms_filt']):
        sample_w[i] = max((weight_map.get(zz, 1.0) for zz in set(np.asarray(atoms).tolist())
                           if zz >= 21), default=1.0)
    sample_w = sample_w / sample_w.mean()
    print(f'Metal weight map: {dict(sorted(weight_map.items()))}')
    print(f'Sample weights: mean={sample_w.mean():.3f} max={sample_w.max():.2f}', flush=True)

    # ---------------- loaders ----------------
    tr_ds = GraphDataset([graphs[pos[int(i)]] for i in tr_idx], tr_idx)
    va_ds = GraphDataset([graphs[pos[int(i)]] for i in va_idx], va_idx)
    bs = args.batch_size
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=args.workers,
                           collate_fn=collate_pairs, pin_memory=(dev.type == 'cuda'),
                           drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=args.workers,
                           collate_fn=collate_pairs, pin_memory=(dev.type == 'cuda'),
                           drop_last=False)

    # ---------------- model ----------------
    model = XYZ_GNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: 11d/23d GAT x{N_LAYERS} x{HIDDEN}d x{GAT_HEADS} heads, '
          f'dropout={DROPOUT}, params={n_params:,}', flush=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    crit = nn.MSELoss(reduction='none')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.float16

    y_tr = y_all[tr_idx]
    y_mean_local = y_tr.mean(dim=0)
    y_std_local = y_tr.std(dim=0).clamp(min=1e-8)
    y_norm = ((y_all - y_mean_local) / y_std_local).to(dev)
    sw_dev = sample_w.to(dev)

    best_va = float('inf')
    best_state = None
    best_ep = -1
    patience = 0
    history = []
    t_train = time.time()

    for ep in range(args.max_epochs):
        model.train()
        tr_loss, n_batches = 0.0, 0
        for gidx, batched, batch_idx in tr_loader:
            batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
            batch_idx = batch_idx.to(dev, non_blocking=True)
            gidx = gidx.to(dev)
            by = y_norm[gidx]
            w = sw_dev[gidx]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(batched, batch_idx)
                loss = (crit(preds, by).mean(dim=1) * w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item()
            n_batches += 1
        sched.step(ep + tr_loss / max(1, n_batches))

        model.eval()
        va_preds = []
        with torch.no_grad():
            for gidx, batched, batch_idx in va_loader:
                batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
                batch_idx = batch_idx.to(dev, non_blocking=True)
                with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    va_preds.append(model(batched, batch_idx).float())
        vp = torch.cat(va_preds, dim=0)
        va_loss = nn.MSELoss()(vp, y_norm[torch.as_tensor(va_idx, device=dev)]).item()

        history.append({'epoch': ep + 1, 'train_loss': tr_loss / max(1, n_batches),
                        'val_loss': va_loss, 'lr': opt.param_groups[0]['lr']})
        if va_loss < best_va:
            best_va = va_loss
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if (ep + 1) % 10 == 0 or patience == 0 or ep == 0:
            print(f'Ep {ep + 1:3d}/{args.max_epochs} | tr {tr_loss / max(1, n_batches):.5f} | '
                  f'va {va_loss:.5f} | lr {opt.param_groups[0]["lr"]:.2e} | pat {patience}', flush=True)
        if patience >= args.patience:
            print(f'Early stop at epoch {ep + 1}', flush=True)
            break

    elapsed_min = (time.time() - t_train) / 60
    print(f'Training done in {elapsed_min:.1f} min; best epoch {best_ep + 1}, '
          f'best val loss {best_va:.6f}', flush=True)

    # ---------------- validation metrics with best weights ----------------
    model.load_state_dict(best_state)
    model.eval()
    vp_all = []
    with torch.no_grad():
        for gidx, batched, batch_idx in va_loader:
            batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
            batch_idx = batch_idx.to(dev, non_blocking=True)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                vp_all.append(model(batched, batch_idx).float().cpu())
    vp_orig = torch.cat(vp_all, dim=0) * y_std_local + y_mean_local
    vt_orig = y_all[va_idx]
    mae = (vp_orig - vt_orig).abs().mean(dim=0)
    ss_res = ((vp_orig - vt_orig) ** 2).sum(dim=0)
    ss_tot = ((vt_orig - vt_orig.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot
    print('Validation per-target MAE / R2:')
    for i, t in enumerate(TARGETS):
        print(f'  {t:14s} MAE {mae[i]:9.4f}  R2 {r2[i]:8.4f}')
    print(f'  mean R2 = {r2.mean():.4f}', flush=True)

    # ---------------- save ----------------
    if args.smoke:
        print('[smoke] not saving any checkpoint')
        return

    ckpt_path = os.path.join(args.out_dir, f'{args.tag}_best.pt')
    torch.save(best_state, ckpt_path)
    meta = {
        'tag': args.tag,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Fig.4 11d/23d single GAT; geometry channels zeroed during training, validation and test (id57)',
        'data': os.path.abspath(args.data),
        'split': os.path.abspath(args.split),
        'split_sizes': {'train': int(len(tr_idx)), 'val': int(len(va_idx)), 'test': int(len(te_idx))},
        'zeroed': {'node_dims': [8, 11], 'edge_dims': [6, 23],
                   'meaning': 'BO node stats (8:11) and 16 RBF + 1 WBO edge channels (6:23) set to 0'},
        'seed': args.seed,
        'epochs_run': len(history),
        'best_epoch': best_ep + 1,
        'best_val_loss': best_va,
        'model': {'node_dim': NODE_DIM, 'edge_dim': EDGE_DIM, 'hidden': HIDDEN,
                  'n_layers': N_LAYERS, 'heads': GAT_HEADS, 'dropout': DROPOUT,
                  'output_dim': N_TARGETS, 'params': int(n_params)},
        'optim': {'name': 'AdamW', 'lr': args.lr, 'weight_decay': args.weight_decay,
                  'scheduler': 'CosineAnnealingWarmRestarts(T_0=50,T_mult=2,eta_min=1e-6)',
                  'batch_size': bs, 'max_epochs': args.max_epochs,
                  'early_stopping_patience': args.patience, 'grad_clip': 1.0,
                  'amp': use_amp, 'metal_weight': 'max(0.5,min(3.0,sqrt(total_c/c))), mean-normalised'},
        'targets': TARGETS,
        'y_mean': [float(x) for x in y_mean_local],
        'y_std': [float(x) for x in y_std_local],
        'val_mae': {t: float(mae[i]) for i, t in enumerate(TARGETS)},
        'val_r2': {t: float(r2[i]) for i, t in enumerate(TARGETS)},
        'env': {'python': sys.version.split()[0], 'torch': torch.__version__,
                'numpy': np.__version__,
                'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'},
    }
    with open(os.path.join(args.out_dir, f'{args.tag}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, f'{args.tag}_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'Saved {ckpt_path} ({os.path.getsize(ckpt_path):,} bytes)')
    print(f'Saved {args.tag}_meta.json / {args.tag}_history.json')


if __name__ == '__main__':
    main()
