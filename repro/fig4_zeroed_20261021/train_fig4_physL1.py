# -*- coding: utf-8 -*-
"""R2-3e: retrain the zero-padded Fig. 4 GAT with an explicit gap-identity penalty.

The reviewer (R2-3) notes that the internal consistency of Fig. 4 is an algebraic
identity that any model satisfies approximately and that a linear composition
model satisfies exactly; internal consistency therefore says nothing about
accuracy.  This script tests the complementary direction: if the identity is
imposed as a soft constraint during training, does the DFT-gap accuracy change?

Total objective (per batch):

    L = L_MTL + lam_sigma * mean( ((gap_hat - (lumo_hat - homo_hat)) / sigma_gap)^2 )

where the residual is evaluated in Hartree on the de-standardised predictions
(sigma_gap = training-set sd of the gap target, 0.03292 Ha), so lam_sigma = 1
means "a violation of one gap standard deviation costs one loss unit" and the
equivalent raw-unit weight is lam_ha = lam_sigma / sigma_gap^2 (~923 Ha^-2).

Why the scale has to be defined this way: y_gap = y_lumo - y_homo holds row by
row in the reference table, so the three per-target normalisation means satisfy
the identity to 7.5e-9 Ha.  A randomly initialised network therefore already
satisfies the constraint almost exactly, and any raw-unit lambda with an
O(1) magnitude is inert (the penalty at the observed 1e-4 Ha deviation is
1e-8 Ha^2).  lam_sigma is fixed a priori by the sd rule above, not tuned on
validation performance.

Everything else (data, split, zeroed layout, seed, optimiser, schedule, batch
size, metal weighting, early stopping) is identical to train_fig4_zeroed.py, so
lam_sigma = 0 is a bit-for-bit replicate of the released Fig. 4 model.

Usage (server):
  python train_fig4_physL1.py --lam-sigma 1   --tag physL1_l1
  python train_fig4_physL1.py --lam-sigma 0   --tag physL1_l0     # replicate
  python train_fig4_physL1.py --smoke                              # 2 epochs, CPU
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
IDX_GAP, IDX_HOMO, IDX_LUMO = 0, 1, 2


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(here, 'new_domain_iso.pt'))
    ap.add_argument('--split', default=os.path.join(here, 'ensemble_split_new.npz'))
    ap.add_argument('--rebuild-split', action='store_true')
    ap.add_argument('--out-dir', default=here)
    ap.add_argument('--tag', default='physL1')
    ap.add_argument('--lam-sigma', type=float, default=1.0,
                    help='weight of the gap-identity penalty in units of sigma_gap^2; '
                         '0 = baseline replicate (released Fig. 4 protocol)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--smoke', action='store_true')
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
    from sklearn.model_selection import KFold
    perm = np.random.RandomState(42).permutation(n_total)
    n_te = len(data_te)
    te = perm[:n_te].astype(np.int64)
    if set(te.tolist()) != set(np.asarray(data_te).tolist()):
        raise ValueError('new_domain_iso.pt te_idx is not the seed-42 permutation')
    cv = perm[n_te:].astype(np.int64)
    tr_pos, va_pos = next(iter(KFold(n_splits=3, shuffle=True, random_state=42).split(cv)))
    return cv[tr_pos].astype(np.int64), cv[va_pos].astype(np.int64), te


def identity_residual_ha(pred_norm, y_mean, y_std):
    """gap - (LUMO - HOMO) in Hartree, from standardised predictions."""
    gap = pred_norm[:, IDX_GAP] * y_std[IDX_GAP] + y_mean[IDX_GAP]
    homo = pred_norm[:, IDX_HOMO] * y_std[IDX_HOMO] + y_mean[IDX_HOMO]
    lumo = pred_norm[:, IDX_LUMO] * y_std[IDX_LUMO] + y_mean[IDX_LUMO]
    return gap - (lumo - homo)


def main():
    args = parse_args()
    dev = pick_device(args.device)
    use_amp = (dev.type == 'cuda')
    if args.smoke:
        args.max_epochs = 2
        args.workers = 0
        args.out_dir = os.path.join(args.out_dir, '_smoke')
        args.tag = f'physL1_smoke_lam{args.lam_sigma:g}'
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f'Device: {dev} | AMP: {use_amp} | lam_sigma={args.lam_sigma:g} | tag={args.tag}', flush=True)
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM {p.total_memory / 1024 ** 3:.1f} GB', flush=True)

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
        print(f'[warn] file md5 differs from the authors\' copy {EXPECTED_DATA_MD5}.', flush=True)

    # ---------------- split ----------------
    data_te = np.asarray(data['te_idx']).astype(np.int64)
    if n_total == 90377 and len(data_te) != 13556:
        raise ValueError('this is the old 13,557-index data version')
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
    if not split_ok:
        if not args.rebuild_split:
            raise ValueError('split file missing or inconsistent; use --rebuild-split')
        tr_idx, va_idx, te_idx = rebuild_split(n_total, data_te)
        print('[rebuild] paper split reconstructed', flush=True)
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

    # penalty constants (float32, de-normalisation kept on device)
    ym_d = y_mean_local.to(dev)
    ys_d = y_std_local.to(dev)
    sigma_gap = float(y_std_local[IDX_GAP])
    lam_ha = float(args.lam_sigma / sigma_gap ** 2) if args.lam_sigma else 0.0
    print(f'sigma_gap = {sigma_gap:.8f} Ha | lam_sigma = {args.lam_sigma:g} | '
          f'lam_ha = {lam_ha:.4g} Ha^-2', flush=True)

    def penalty_ha(pred_norm):
        """(gap - (LUMO - HOMO))^2 / sigma_gap^2, mean over the batch."""
        if not args.lam_sigma:
            return pred_norm.new_zeros(())
        resid = identity_residual_ha(pred_norm.float(), ym_d, ys_d)
        return ((resid / sigma_gap) ** 2).mean()

    best_va = float('inf')
    best_state = None
    best_ep = -1
    patience = 0
    history = []
    t_train = time.time()

    for ep in range(args.max_epochs):
        model.train()
        tr_mtl, tr_pen, n_batches = 0.0, 0.0, 0
        for gidx, batched, batch_idx in tr_loader:
            batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
            batch_idx = batch_idx.to(dev, non_blocking=True)
            gidx = gidx.to(dev)
            by = y_norm[gidx]
            w = sw_dev[gidx]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(batched, batch_idx)
                mtl = (crit(preds, by).mean(dim=1) * w).mean()
            pen = penalty_ha(preds)          # float32, outside autocast
            loss = mtl + args.lam_sigma * pen
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_mtl += mtl.item()
            tr_pen += pen.detach().item()
            n_batches += 1
        sched.step(ep + tr_mtl / max(1, n_batches))

        model.eval()
        va_preds = []
        with torch.no_grad():
            for gidx, batched, batch_idx in va_loader:
                batched = {k: v.to(dev, non_blocking=True) for k, v in batched.items()}
                batch_idx = batch_idx.to(dev, non_blocking=True)
                with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    va_preds.append(model(batched, batch_idx).float())
        vp = torch.cat(va_preds, dim=0)
        va_mtl = nn.MSELoss()(vp, y_norm[torch.as_tensor(va_idx, device=dev)]).item()
        va_pen = float(penalty_ha(vp))
        va_total = va_mtl + args.lam_sigma * va_pen

        # selection on the objective that is actually optimised
        history.append({'epoch': ep + 1, 'train_mtl': tr_mtl / max(1, n_batches),
                        'train_penalty': tr_pen / max(1, n_batches),
                        'val_mtl': va_mtl, 'val_penalty': va_pen, 'val_total': va_total,
                        'lr': opt.param_groups[0]['lr']})
        if va_total < best_va:
            best_va = va_total
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if (ep + 1) % 10 == 0 or patience == 0 or ep == 0:
            print(f'Ep {ep + 1:3d}/{args.max_epochs} | tr {tr_mtl / max(1, n_batches):.5f} '
                  f'pen {tr_pen / max(1, n_batches):.3e} | va {va_mtl:.5f} '
                  f'vpen {va_pen:.3e} | lr {opt.param_groups[0]["lr"]:.2e} | pat {patience}',
                  flush=True)
        if patience >= args.patience:
            print(f'Early stop at epoch {ep + 1}', flush=True)
            break

    elapsed_min = (time.time() - t_train) / 60
    print(f'Training done in {elapsed_min:.1f} min; best epoch {best_ep + 1}, '
          f'best val total {best_va:.6f}', flush=True)
    if dev.type == 'cuda':
        print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB '
              f'(reserved {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB)', flush=True)

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
    vp_norm = torch.cat(vp_all, dim=0)
    vp_orig = vp_norm * y_std_local + y_mean_local
    vt_orig = y_all[va_idx]
    mae = (vp_orig - vt_orig).abs().mean(dim=0)
    ss_res = ((vp_orig - vt_orig) ** 2).sum(dim=0)
    ss_tot = ((vt_orig - vt_orig.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot
    dev_ha = identity_residual_ha(vp_norm, y_mean_local, y_std_local).abs()
    print('Validation per-target MAE / R2:')
    for i, t in enumerate(TARGETS):
        print(f'  {t:14s} MAE {mae[i]:9.4f}  R2 {r2[i]:8.4f}')
    print(f'  mean R2 = {r2.mean():.4f}')
    print(f'  val internal |gap-(LUMO-HOMO)| mean = {float(dev_ha.mean()) * 1000:.4f} mHa', flush=True)

    # ---------------- save ----------------
    if args.smoke:
        print('[smoke] not saving any checkpoint')
        return

    ckpt_path = os.path.join(args.out_dir, f'{args.tag}_best.pt')
    torch.save(best_state, ckpt_path)
    meta = {
        'tag': args.tag,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'R2-3e: gap-identity penalty (gap = LUMO - HOMO) added to the Fig.4 '
                   '11d/23d zero-padded GAT objective; same protocol as train_fig4_zeroed.py',
        'data': os.path.abspath(args.data),
        'split': os.path.abspath(args.split),
        'split_sizes': {'train': int(len(tr_idx)), 'val': int(len(va_idx)), 'test': int(len(te_idx))},
        'zeroed': {'node_dims': [8, 11], 'edge_dims': [6, 23]},
        'penalty': {'form': 'lam_sigma * mean(((gap_hat-(lumo_hat-homo_hat))/sigma_gap)^2)',
                    'residual_units': 'Hartree',
                    'lam_sigma': args.lam_sigma, 'lam_ha': lam_ha, 'sigma_gap': sigma_gap,
                    'selection': 'val_mtl + lam_sigma*val_penalty'},
        'seed': args.seed,
        'epochs_run': len(history),
        'best_epoch': best_ep + 1,
        'best_val_total': best_va,
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
        'val_internal_dev_mha': float(dev_ha.mean()) * 1000,
        'elapsed_min': elapsed_min,
        'env': {'python': sys.version.split()[0], 'torch': torch.__version__,
                'numpy': np.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'},
    }
    with open(os.path.join(args.out_dir, f'{args.tag}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, f'{args.tag}_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'Saved {ckpt_path} ({os.path.getsize(ckpt_path):,} bytes)')
    print(f'Saved {args.tag}_meta.json / {args.tag}_history.json', flush=True)


if __name__ == '__main__':
    main()
