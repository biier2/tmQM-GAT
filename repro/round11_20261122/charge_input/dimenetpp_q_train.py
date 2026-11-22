# -*- coding: utf-8 -*-
"""
DimeNet++ with total-charge input (DimeNet++-Q, data-fixed version)

Required files in the working directory:
    new_domain_iso.pt           labels, atom lists, te_idx (fixed domain, 90,377 complexes)
    schnet_domain_coords.npz    coordinates by global block index
    fixed_q.npz                 q_all from build_charge_vector.py

Usage:
    python dimenetpp_q_train.py --folds 0 --seeds 42 123 456

Root cause identical to SchNet: the original preprocess_dimenet.py used
    parse_one(..., start_idx=len(all_m))      # ✗ all_m is the "cumulative number of retained TM complexes", not the "cumulative number of blocks"
so csv_idx for parts 2/3 (about 2/3 of the data) was systematically offset and labels were taken from other complexes.
This is exactly the typical label-noise signature of the old train log: train_rmse fell all the way to 0.35 (the model was
memorising the random labels of 2/3 of the data), while val_rmse stayed pinned at 0.956 for 260 epochs (≈ mean prediction).

This script: architecture and hyperparameters exactly identical to the original DimeNet++ (PyG DimeNetPlusPlus: hidden 128, 4 interaction
        blocks, int_emb 64, basis_emb 8, out_emb 256, num_spherical 7, num_radial 6,
        cutoff 5.0 Å, envelope 5, 3-layer output MLP; radius graph r=5.0 / ≤32 neighbours per node;
        bs=8, AdamW wd=1e-4, clip 5.0), doing only two things:
        (1) use the correct coordinates (schnet_domain_coords.npz, extracted by global block index, elements 100% aligned)
            and the correct labels (y_all of new_domain_iso.pt)
        (2) the same domain and split as the SchNet fixed version / four GAT variants

⚠️ Runtime: measured at about 220 s/epoch (51,214 training + validation). The default --epochs 300 (= the original
   protocol of the retracted pilot, warmup 10 + cosine 1e-3→1e-5 + no early stopping) gives ≈ 18.3 h per model.
   The default --folds 0 --seeds 42 123 456 = 3 models ≈ 55 hours.
   For 9 models (3 folds × 3 seeds, consistent with the SchNet/GAT variants) use --folds 0 1 2, about 165 hours.
   ★ Can be run in two stages: first --folds 0 (3 models), then re-running with --folds 0 1 2 automatically skips
     the completed (fold, seed) pairs and only trains the remaining 6, and finally evaluates all models together — no repeated training.

Usage:
    python server_dimenet_fixed.py --smoke                 # 2-minute smoke test
    python server_dimenet_fixed.py --subset 2000 --epochs 60   # ~40-minute small-sample fit diagnostic
    nohup python server_dimenet_fixed.py > train_dimenet_fixed.log 2>&1 &
"""
import os, sys, time, json, argparse, contextlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import scatter

class DimeNetPlusPlusQ(DimeNetPlusPlus):
    """DimeNet++ with the total molecular charge as a global input.

    The charge index (0 -> -1, 1 -> 0, 2 -> +1) is embedded, projected to the
    output width, and added to the pooled graph representation.  The projection
    is initialized to zero so training starts from the uncharged model.
    """

    def __init__(self, *args, num_charge_states=3, **kwargs):
        hidden = kwargs.get('hidden_channels')
        out = kwargs.get('out_channels')
        if hidden is None or out is None:
            raise ValueError('hidden_channels and out_channels must be keyword arguments')
        super().__init__(*args, **kwargs)
        self.q_emb = nn.Embedding(num_charge_states, hidden)
        self.q_out = nn.Linear(hidden, out, bias=False)
        nn.init.zeros_(self.q_out.weight)

    def forward(self, z, pos, batch=None, q=None):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))
        out = scatter(P, batch, dim=0, reduce='sum') if batch is not None else P.sum(dim=0)
        if q is not None:
            out = out + self.q_out(self.q_emb(q))
        return out



os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=300,
                help='default 300 = the original protocol of the retracted pilot; the cosine schedule is expanded over this value, lowering it truncates the annealing')
ap.add_argument('--batch-size', type=int, default=8, help='the original DimeNet++ used 8; a 16GB card can try 16')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--warmup', type=int, default=10)
ap.add_argument('--min-lr', type=float, default=1e-5)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--blocks', type=int, default=4)
ap.add_argument('--folds', type=int, nargs='+', default=[0], help='which folds to use (default is only 0)')
ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
ap.add_argument('--val-every', type=int, default=5, help='evaluate the validation set every N epochs (saves time)')
ap.add_argument('--workers', type=int, default=0, help='0=load in the main process, avoiding CUDA fork deadlock')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--subset', type=int, default=0, help='diagnostic: use only the first N training molecules')
args = ap.parse_args()
# Full FP32 + TF32 throughout: combining PyG sparse operators with autocast is risky (the SchNet fp16 overflow was the lesson),
# and the original DimeNet++ was measured in FP32 (220 s/epoch); the benefit of switching to AMP is uncertain.

if args.smoke:
    args.epochs, args.batch_size, args.val_every = 2, 4, 1
    args.folds, args.seeds = [0], [42]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM {p.total_memory/1024**3:.1f}GB')
print(f'Config: epochs={args.epochs} bs={args.batch_size} lr={args.lr} folds={args.folds} seeds={args.seeds}')

# ---------------- Data (line-for-line same source as server_schnet_fixed.py)----------------
D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
y_all = D['y_all'].float()
N = y_all.size(0)
atoms_filt = D['atoms_filt']
coords_np = np.load('schnet_domain_coords.npz')
COORDS = coords_np['coords']; LENS = coords_np['lens']
assert len(LENS) == N and LENS.sum() == len(COORDS), f'{len(LENS)} {LENS.sum()} {len(COORDS)}'
print(f'Domain N={N:,}, total atoms {len(COORDS):,}, mean {len(COORDS)/N:.1f} atoms/complex')

# Slice out pos / z for each complex once (avoids re-slicing every epoch)
POS, ZS = [], []
off = 0
for i in range(N):
    n = int(LENS[i])
    POS.append(torch.as_tensor(COORDS[off:off + n], dtype=torch.float32))
    off += n
    ZS.append(torch.as_tensor(np.asarray(atoms_filt[i]), dtype=torch.long))
print(f'pos/z slicing done ({time.time():.0f})')

QALL = np.load('fixed_q.npz')['q_all']
assert len(QALL) == N, f'charge vector length {len(QALL)} != {N}'
print('charge counts:', dict(zip(*np.unique(QALL, return_counts=True))))

# ---------------- Split (same as the GAT variants / SchNet)----------------
te_stored = np.asarray(D['te_idx'])
idx_all = np.random.RandomState(42).permutation(N)
n_test = len(te_stored)
te_idx = idx_all[:n_test].copy()
cv_idx = idx_all[n_test:].copy()
assert np.array_equal(te_idx, te_stored), 'te_idx does not match the one stored in new_domain_iso.pt'
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_splits = list(kf.split(cv_idx))
print(f'Test {len(te_idx):,} | CV {len(cv_idx):,}')

# ---------------- Model ----------------
def build_model():
    return DimeNetPlusPlusQ(
        hidden_channels=args.hidden, out_channels=len(TARGETS), num_blocks=args.blocks,
        int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, cutoff=5.0, max_num_neighbors=32,
        envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3)

_m = build_model()
print(f'DimeNetPlusPlus: hidden={args.hidden} blocks={args.blocks} out=8  '
      f'params={sum(p.numel() for p in _m.parameters())/1e6:.2f}M')
del _m

def make_loader(sel, tag, shuffle, bs):
    """Build the Data list from indices. z-score uses training-set statistics (de-normalised at evaluation)."""
    data = []
    for k in sel:
        data.append(Data(z=ZS[k], pos=POS[k],
                         y=((y_all[k] - ym_cpu) / ys_cpu).view(1, -1),
                         q=torch.tensor([int(QALL[k]) + 1], dtype=torch.long)))
    print(f'  [{tag}] {len(data):,} molecules', flush=True)
    return DataLoader(data, batch_size=bs, shuffle=shuffle, num_workers=args.workers)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    for d in loader:
        d = d.to(device)
        preds.append(model(d.z, d.pos, d.batch, d.q.squeeze(-1))); ys.append(d.y)
    return torch.cat(preds), torch.cat(ys)

def train_one(tr_abs, va_abs, seed, name):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - args.warmup), eta_min=args.min_lr)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])

    tr_loader = make_loader(tr_abs, f'{name}/train', True, args.batch_size)
    va_loader = make_loader(va_abs, f'{name}/val', False, args.batch_size)

    best_va, best_state = float('inf'), None
    for ep in range(args.epochs):
        t_ep = time.time()
        model.train(); tot, n = 0.0, 0
        for d in tr_loader:
            d = d.to(device)
            opt.zero_grad()
            # DimeNet++ builds the radius graph internally in the forward pass (consistent with the PyG reference implementation)
            loss = F.mse_loss(model(d.z, d.pos, d.batch, d.q.squeeze(-1)), d.y)
            if not torch.isfinite(loss):
                for gp in opt.param_groups: gp['lr'] *= 0.5
                print(f'  [{name}] WARN: non-finite loss ep{ep+1}; batch skipped, '
                      f'lr -> {opt.param_groups[0]["lr"]:.2e}', flush=True)
                opt.zero_grad(); continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                for gp in opt.param_groups: gp['lr'] *= 0.5
                print(f'  [{name}] WARN: non-finite grad ep{ep+1}; step skipped', flush=True)
                opt.zero_grad(); continue
            opt.step()
            tot += loss.item() * d.num_graphs; n += d.num_graphs
        sched.step()
        tr_rmse = (tot / max(1, n)) ** 0.5
        if (ep + 1) % args.val_every == 0 or ep + 1 == args.epochs:
            vp, vy = evaluate(model, va_loader)
            va = float(F.mse_loss(vp, vy).sqrt())
            if np.isfinite(va) and va < best_va:
                best_va = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep < 3:
                p = vp.cpu().numpy() * ys_cpu.numpy() + ym_cpu.numpy()
                t = vy.cpu().numpy() * ys_cpu.numpy() + ym_cpu.numpy()
                vr2 = float(np.mean([r2_score(t[:, j], p[:, j]) for j in range(8)])) \
                    if np.isfinite(p).all() else float('nan')
                print(f'  [{name}] ep{ep+1:3d} tr={tr_rmse:.4f} va={va:.4f} valR2={vr2:+.4f} '
                      f'{time.time()-t_ep:.0f}s', flush=True)
        elif (ep + 1) % 10 == 0:
            print(f'  [{name}] ep{ep+1:3d} tr={tr_rmse:.4f} {time.time()-t_ep:.0f}s', flush=True)
    if best_state is None:
        print(f'  [{name}] WARN: no finite validation epoch; using final weights', flush=True)
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    del model, tr_loader, va_loader
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_state

# ---------------- Main loop ----------------
ym_cpu = y_all[cv_idx].mean(0)          # placeholder; recomputed inside each model from training-set statistics
ys_cpu = y_all[cv_idx].std(0).clamp(min=1e-8)

# ⚠️ Smoke tests must use a separate checkpoint: --smoke trains a junk model of "128 molecules × 2 epochs", and if it is written into the official
# checkpoint, the official run immediately does "Restored 1 models / SKIP Fold0_Seed42" and treats this junk model as
# a genuine ensemble member (actually hit on 2026-09-18; the 3-model ensemble was dragged down to 0.832).
CKPT = 'smoke_checkpoint_DimeNetQ.pt' if args.smoke else 'ensemble_checkpoint_DimeNetQ.pt'
all_models, log = [], []
done = set()
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    all_models, log = ck['models'], ck.get('log', [])
    done = {(m['fold'], m['seed']) for m in all_models}
    print(f'Restored {len(all_models)} models')

t_start = time.time()
for fi in args.folds:
    tr_full = cv_idx[fold_splits[fi][0]]; va_full = cv_idx[fold_splits[fi][1]]
    if args.smoke:
        tr_full, va_full = tr_full[:128], va_full[:32]
    elif args.subset:
        tr_full = tr_full[:args.subset]; va_full = va_full[:max(64, args.subset // 8)]
    for seed in args.seeds:
        if (fi, seed) in done:
            print(f'  SKIP Fold{fi}_Seed{seed}'); continue
        name = f'Fold{fi}_Seed{seed}'
        # z-score statistics: each model uses its own training set (consistent with SchNet/GAT)
        ym_cpu = y_all[tr_full].mean(0)
        ys_cpu = y_all[tr_full].std(0).clamp(min=1e-8)
        t1 = time.time()
        print(f'  Training {name} (train={len(tr_full):,} val={len(va_full):,})...', flush=True)
        st = train_one(tr_full, va_full, seed, name)
        el = (time.time() - t1) / 60
        all_models.append({'state': st, 'y_mean': ym_cpu.clone(), 'y_std': ys_cpu.clone(),
                           'fold': fi, 'seed': seed})
        log.append({'fold': fi, 'seed': seed, 'min': round(el, 1)})
        done.add((fi, seed))
        print(f'  Done {name} in {el:.1f} min', flush=True)
        torch.save({'models': all_models, 'log': log}, CKPT)

# ---------------- Test-set evaluation ----------------
print('Test eval...', flush=True)
preds = []
for m in all_models:
    model = build_model().to(device)
    model.load_state_dict(m['state']); model.eval()
    data = [Data(z=ZS[i], pos=POS[i],
                 q=torch.tensor([int(QALL[i]) + 1], dtype=torch.long)) for i in te_idx]
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    out = []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            out.append(model(d.z, d.pos, d.batch, d.q.squeeze(-1)).float().cpu())
    out = torch.cat(out).numpy() * m['y_std'].numpy() + m['y_mean'].numpy()
    preds.append(out)
    del model, loader, data
    if torch.cuda.is_available(): torch.cuda.empty_cache()
P = np.stack(preds, 0)
yt = y_all[te_idx].numpy()
ens = P.mean(0)
res = {'variant': 'DimeNet_Q_fixed', 'n_models': len(P), 'folds': args.folds, 'seeds': args.seeds}
for i, p in enumerate(TARGETS):
    res[p] = {'ens_r2': float(r2_score(yt[:, i], ens[:, i])),
              'mean_r2': float(np.mean([r2_score(yt[:, i], P[m, :, i]) for m in range(len(P))])),
              'std_r2': float(np.std([r2_score(yt[:, i], P[m, :, i]) for m in range(len(P))])),
              'ens_mae': float(mean_absolute_error(yt[:, i], ens[:, i]))}
res['mean_ens_r2'] = float(np.mean([res[p]['ens_r2'] for p in TARGETS]))
np.savez('test_preds_DimeNetQ.npz', preds=P, ens_mean=ens, y_true=yt, te_idx=te_idx)
json.dump(res, open('results_DimeNetQ.json', 'w'), indent=2)
print(f"\nDimeNet++-Q mean ensemble R2 = {res['mean_ens_r2']:.4f}")
print(json.dumps(res, indent=2)[:900])
print(f'\nALL DONE in {(time.time()-t_start)/60:.0f} min')
