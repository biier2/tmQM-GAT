# -*- coding: utf-8 -*-
"""DimeNet++ controlled baseline — training/evaluation (server GPU)

Protocol (strictly aligned with the paper, see README.md):
  - test set: ensemble_split.npz te_idx (13,547, unchanged) — the same test split as the GNNs
  - train/val = paper fold 0 split (fold_0_train 51,180 / fold_0_val 25,590, the same set as 3D GAT fold 0)
  - 8-property multi-task (DimeNetPlusPlus out_channels=8), z-score with training-set statistics
  - 3D input: tmQM original DFT XYZ; radius graph r=5.0 Å / 32 neighbours per node

Usage:
  python train_dimenet.py --smoke          # 2-epoch smoke test (about 2 minutes, verifies the pipeline)
  python train_dimenet.py --subset 2000 --epochs 150 --patience 200   # small-sample fitting diagnostic (about 40 minutes)
  python train_dimenet.py --epochs 300 --val-every 5   # R4 full training (cosine schedule, batch 8)
  python train_dimenet.py --resume         # resume training (from dimenetpp_last.pt)
  python train_dimenet.py --schedule plateau --epochs 200 --batch-size 16 --lr 5e-4

Output: dimenetpp_best.pt / dimenetpp_last.pt / dimenet_train_log.csv / dimenet_test_preds.npz
"""
import os, sys, csv, time, argparse
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=os.path.join(SCRIPT_DIR, 'dimenet_data.npz'))
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--blocks', type=int, default=4)
parser.add_argument('--workers', type=int, default=0, help='0=load in the main process (recommended, avoids CUDA fork deadlock; data loading is not the bottleneck)')
parser.add_argument('--patience', type=int, default=40, help='early-stopping patience for the plateau schedule (no early stopping under cosine)')
parser.add_argument('--schedule', choices=['cosine', 'plateau'], default='cosine',
                    help='cosine: full-run cosine annealing (R4, unaffected by val plateaus); plateau: legacy decay on val')
parser.add_argument('--warmup', type=int, default=10, help='number of cosine warmup epochs')
parser.add_argument('--min-lr', type=float, default=1e-5, help='final learning rate of the cosine schedule')
parser.add_argument('--val-every', type=int, default=1, help='evaluate the validation set every N epochs (N>1 saves about 100s/epoch)')
parser.add_argument('--smoke', action='store_true', help='2-epoch smoke test')
parser.add_argument('--subset', type=int, default=0, help='diagnostic: use only the first N training molecules (0=full), to quickly judge whether the model can fit a small sample')
parser.add_argument('--resume', action='store_true', help='resume training from dimenetpp_last.pt')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ===== Data loading =====
print('Loading dimenet_data.npz ...')
D = np.load(args.data, allow_pickle=True)
y_mean_cpu = torch.from_numpy(D['y_mean'].astype(np.float32))
y_std_cpu = torch.from_numpy(D['y_std'].astype(np.float32))
y_mean = y_mean_cpu.to(device)
y_std = y_std_cpu.to(device)
print(f'y_mean={y_mean.cpu().numpy().round(2)}', flush=True)
print(f'y_std ={y_std.cpu().numpy().round(2)}', flush=True)

def make_dataset(sel, tag):
    """Load z/pos/edge_index/y by index sequence and assemble a list of PyG Data objects.

    ⚠️ NpzFile (the object returned by np.load) does not cache members: every access to D['z_train']
    re-decompresses + deserialises the entire object array column. It must be fetched once outside the loop,
    otherwise the whole column is decompressed O(N) times and assembly becomes hours slower."""
    z_arr = D[f'z_{tag}']
    pos_arr = D[f'pos_{tag}']
    edge_arr = D[f'edge_{tag}']
    y_arr = D[f'y_{tag}']
    data = []
    for k in sel:
        data.append(Data(
            z=torch.from_numpy(z_arr[k].astype(np.int64)),
            pos=torch.from_numpy(pos_arr[k]),
            edge_index=torch.from_numpy(edge_arr[k].astype(np.int64)),
            # z-score (training-set statistics) — consistent with the paper GNN pipeline; de-normalised at evaluation by multiplying by y_std and adding y_mean.
            # (1, 8): PyG treats y as a node-level attribute and concatenates along dim0 by default; (8,) would be concatenated into (batch*8,), which does not match the graph-level output (batch, 8)
            y=((torch.from_numpy(y_arr[k]) - y_mean_cpu) / y_std_cpu).view(1, -1)))
    return data

sel_tr, sel_va, sel_te = range(D['train_sel'].shape[0]), range(D['val_sel'].shape[0]), range(D['test_sel'].shape[0])
if args.smoke:
    print('[smoke] reduced datasets: train 128 / val 32 / test 64, epochs=2, batch=4')
    args.epochs, args.batch_size, args.patience = 2, 4, 100
    sel_tr, sel_va, sel_te = range(128), range(32), range(64)
    args.lr = 5e-4       # small learning rate for fast convergence in the smoke test
    args.schedule = 'plateau'   # cosine warmup is not applicable to a 2-epoch smoke test
if args.subset and not args.smoke:
    n_sub = args.subset
    print(f'[subset] diagnostic mode: train {n_sub} / val {max(64, n_sub // 8)} / test 64, epochs={args.epochs}, '
          f'criterion: train_rmse < 0.6 = healthy architecture (the problem is training scale/schedule); stuck at 0.9+ = geometry pathway broken')
    sel_tr, sel_va, sel_te = range(n_sub), range(max(64, n_sub // 8)), range(64)
t_ds = time.time()
train_ds = make_dataset(sel_tr, 'train')
val_ds = make_dataset(sel_va, 'val')
test_ds = make_dataset(sel_te, 'test')
print(f'Train {len(train_ds)} / Val {len(val_ds)} / Test {len(test_ds)}  (dataset assembly {time.time()-t_ds:.0f}s)', flush=True)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# ===== Model =====
model = DimeNetPlusPlus(
    hidden_channels=args.hidden, out_channels=len(TARGETS), num_blocks=args.blocks,
    int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
    num_spherical=7, num_radial=6, cutoff=float(D['cutoff']),
    max_num_neighbors=int(D['max_neighbors']),
    envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'DimeNetPlusPlus: hidden={args.hidden} blocks={args.blocks} out=8  params={n_params/1e6:.2f}M', flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.schedule == 'cosine':
    # R4 key fix: R3 used ReduceLROnPlateau; val stayed for a long time on the 0.95 mean-prediction floor → lr was
    # decayed all the way to 1e-5 and training was frozen before fitting even began (epoch 66 train≈val≈0.95 is exactly this symptom).
    # The small-sample diagnostic (--subset 2000 → train_rmse 0.026) already proved the architecture is healthy, so full-run cosine annealing is used instead.
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup),
                                                     eta_min=args.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[args.warmup])
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
start_epoch, best_val = 0, float('inf')

if args.resume and os.path.exists(os.path.join(SCRIPT_DIR, 'dimenetpp_last.pt')):
    ck = torch.load(os.path.join(SCRIPT_DIR, 'dimenetpp_last.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    start_epoch = ck['epoch'] + 1
    best_val = ck['best_val']
    print(f'[resume] continuing from epoch {start_epoch} (best_val={best_val:.5f})')

# ===== Evaluation =====
@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, ys = [], []
    for data in loader:
        data = data.to(device)
        preds.append(model(data.z, data.pos, data.batch))
        ys.append(data.y)
    return torch.cat(preds), torch.cat(ys)

# ===== Training =====
log_path = os.path.join(SCRIPT_DIR, 'dimenet_train_log.csv')
with open(log_path, 'a' if args.resume else 'w', newline='') as f:
    w = csv.writer(f)
    if not args.resume:
        w.writerow(['epoch', 'lr', 'train_rmse', 'val_rmse', 'best_val', 'sec'])

patience_left = args.patience if args.schedule == 'plateau' else args.epochs  # no early stopping under cosine; run to completion
print(f'Starting training: {args.epochs} epochs, batch {args.batch_size}, workers {args.workers}', flush=True)
for epoch in range(start_epoch, args.epochs):
    t_ep = time.time()
    model.train()
    total, n = 0.0, 0
    for bi, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.z, data.pos, data.batch)
        loss = F.mse_loss(pred, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * data.num_graphs
        n += data.num_graphs
        if (bi + 1) % 500 == 0:
            print(f'  epoch {epoch+1}: {bi+1}/{len(train_loader)} steps, elapsed {time.time()-t_ep:.0f}s, loss {loss.item():.4f}', flush=True)
    train_rmse = (total / n) ** 0.5

    if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
        val_pred, val_y = evaluate(val_loader)
        val_rmse = float(F.mse_loss(val_pred, val_y).sqrt())
    else:
        val_rmse = float('nan')
    if args.schedule == 'cosine':
        scheduler.step()
    elif not np.isnan(val_rmse):
        scheduler.step(val_rmse)
    lr = optimizer.param_groups[0]['lr']

    improved = val_rmse < best_val - 1e-6
    if improved:
        best_val = val_rmse
        patience_left = args.patience
        torch.save({'model': model.state_dict(), 'best_val': best_val, 'epoch': epoch},
                   os.path.join(SCRIPT_DIR, 'dimenetpp_best.pt'))
    elif args.schedule == 'plateau':
        patience_left -= 1
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'best_val': best_val, 'epoch': epoch},
               os.path.join(SCRIPT_DIR, 'dimenetpp_last.pt'))

    sec = time.time() - t_ep
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([epoch + 1, f'{lr:.2e}', f'{train_rmse:.5f}', f'{val_rmse:.5f}',
                                f'{best_val:.5f}', f'{sec:.0f}'])
    print(f'Epoch {epoch+1:3d}/{args.epochs}  lr={lr:.1e}  train_rmse={train_rmse:.4f}  '
          f'val_rmse={val_rmse:.4f}  best={best_val:.4f}  {sec:.0f}s/epoch', flush=True)
    if patience_left <= 0:
        print(f'Early stopping: no val improvement for {args.patience} consecutive epochs')
        break

# ===== Test-set evaluation (de-normalised, per-property R²) =====
print('=' * 70)
print('Test-set evaluation (13,547, de-normalised)')
print('=' * 70)
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'dimenetpp_best.pt'), map_location=device, weights_only=False)['model'])
test_pred, test_y = evaluate(test_loader)
pred_unstd = test_pred.cpu().numpy() * y_std.cpu().numpy() + y_mean.cpu().numpy()
true_unstd = test_y.cpu().numpy() * y_std.cpu().numpy() + y_mean.cpu().numpy()

from sklearn.metrics import r2_score, mean_absolute_error
r2s, maes = [], []
print(f'{"property":<15} {"R2":>8} {"MAE":>10}')
for i, t in enumerate(TARGETS):
    r2 = r2_score(true_unstd[:, i], pred_unstd[:, i])
    mae = mean_absolute_error(true_unstd[:, i], pred_unstd[:, i])
    r2s.append(r2); maes.append(mae)
    print(f'{t:<15} {r2:>8.4f} {mae:>10.4f}')
print(f'{"mean":<15} {np.mean(r2s):>8.4f} {np.mean(maes):>10.4f}')
np.savez(os.path.join(SCRIPT_DIR, 'dimenet_test_preds.npz'),
         y_true=true_unstd, y_pred=pred_unstd, targets=np.array(TARGETS))
print('saved: dimenet_test_preds.npz')
print(f'[verify] mean R2 = {np.mean(r2s):.4f}   (reference: 3D GAT ensemble 0.903 / SMILES GAT 0.932, same 13,547 test set)')
